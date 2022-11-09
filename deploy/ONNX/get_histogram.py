#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
# from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch.nn as nn
from torch import Tensor

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from io import BytesIO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    
    
    # summary_writer = SummaryWriter(log_dir='log' ,comment='test tensorboard histogram', filename_suffix='test_1')
    count = 0
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            kernel3x3, bias3x3 = layer._fuse_bn_tensor(layer.rbr_dense)
            kernel1x1, bias1x1 = layer._fuse_bn_tensor(layer.rbr_1x1)
            kernelid, biasid = layer._fuse_bn_tensor(layer.rbr_identity)
            kernel, bias = kernel3x3 + layer._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
            layer.rbr_reparam = nn.Conv2d(in_channels=layer.rbr_dense.conv.in_channels, out_channels=layer.rbr_dense.conv.out_channels,
                                        kernel_size=layer.rbr_dense.conv.kernel_size, stride=layer.rbr_dense.conv.stride,
                                        padding=layer.rbr_dense.conv.padding, dilation=layer.rbr_dense.conv.dilation, groups=layer.rbr_dense.conv.groups, bias=True)
            layer.rbr_reparam.weight.data = kernel
            layer.rbr_reparam.bias.data = bias
            for para in layer.parameters():
                para.detach_()
            layer.__delattr__('rbr_dense')
            layer.__delattr__('rbr_1x1')
            if hasattr(layer, 'rbr_identity'):
                layer.__delattr__('rbr_identity')
            if hasattr(layer, 'id_tensor'):
                layer.__delattr__('id_tensor')
            layer.deploy = True
            # 这里插入model三分支合并前的权重分布可视化函数
            # summary_writer.add_histogram(tag='kernel3x3-{}'.format(count), values=kernel3x3)
            # summary_writer.add_histogram(tag='bias3x3-{}'.format(count), values=bias3x3)
            # summary_writer.add_histogram(tag='kernel1x1-{}'.format(count), values=kernel1x1)
            # summary_writer.add_histogram(tag='bias1x1-{}'.format(count), values=bias1x1)
            # summary_writer.add_histogram(tag='kernelid-{}'.format(count), values=kernelid)
            # summary_writer.add_histogram(tag='biasid-{}'.format(count), values=biasid)
            
            fig = plt.figure(figsize=(32,16),dpi=300)
            # margin_float = 0.05
            
            kernel3x3_plt = plt.subplot(2, 4, 1)
            # kernel3x3_plt.margins(margin_float)
            kernel3x3_cpu = torch.flatten(kernel3x3).cpu().detach().numpy()
            kernel3x3_plt.hist(kernel3x3_cpu, bins=1024, range=(kernel3x3_cpu.min(), kernel3x3_cpu.max()))
            kernel3x3_plt.set_title('kernel3x3-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            bias3x3_plt = plt.subplot(2, 4, 5)
            # bias3x3_plt.margins(margin_float)
            bias3x3_cpu = torch.flatten(bias3x3).cpu().detach().numpy()
            bias3x3_plt.hist(bias3x3_cpu, bins=1024, range=(bias3x3_cpu.min(), bias3x3_cpu.max()))
            bias3x3_plt.set_title('bias3x3-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            kernel1x1_plt = plt.subplot(2, 4, 2)
            # kernel1x1_plt.margins(margin_float)
            kernel1x1_cpu = torch.flatten(kernel1x1).cpu().detach().numpy()
            kernel1x1_plt.hist(kernel1x1_cpu, bins=1024, range=(kernel1x1_cpu.min(), kernel1x1_cpu.max()))
            kernel1x1_plt.set_title('kernel1x1-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            bias1x1_plt = plt.subplot(2, 4, 6)
            # bias1x1_plt.margins(margin_float)
            bias1x1_cpu = torch.flatten(bias1x1).cpu().detach().numpy()
            bias1x1_plt.hist(bias1x1_cpu, bins=1024, range=(bias1x1_cpu.min(), bias1x1_cpu.max()))
            bias1x1_plt.set_title('bias1x1-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            kernelid_plt = plt.subplot(2, 4, 3)
            # kernelid_plt.margins(margin_float)
            if isinstance(kernelid, Tensor):
                kernelid_cpu = torch.flatten(kernelid).cpu().detach().numpy()
                kernelid_plt.hist(kernelid_cpu, bins=1024, range=(kernelid_cpu.min(), kernelid_cpu.max()))
            else:
                kernelid_plt.hist(kernelid, bins=1024, range=(kernelid * -3, kernelid * 3))
            kernelid_plt.set_title('kernelid-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。

            biasid_plt = plt.subplot(2, 4, 7)
            # biasid_plt.margins(margin_float)
            if isinstance(biasid, Tensor):
                biasid_cpu = torch.flatten(biasid).cpu().detach().numpy()
                biasid_plt.hist(biasid_cpu, bins=1024, range=(biasid_cpu.min(), biasid_cpu.max()))
            else:
                biasid_plt.hist(biasid, bins=1024, range=(biasid * -3, biasid * 3))
            biasid_plt.set_title('biasid-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。

            # 这里插入model三分支合并后的权重分布可视化函数
            
            kernel_plt = plt.subplot(2, 4, 4)
            # kernel_plt.margins(margin_float)
            kernel_cpu = torch.flatten(kernel).cpu().detach().numpy()
            kernel_plt.hist(kernel_cpu, bins=1024, range=(kernel_cpu.min(), kernel_cpu.max()))
            kernel_plt.set_title('kernel-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            bias_plt = plt.subplot(2, 4, 8)
            # bias_plt.margins(margin_float)
            bias_cpu = torch.flatten(bias).cpu().detach().numpy()
            bias_plt.hist(bias_cpu, bins=1024, range=(bias_cpu.min(), bias_cpu.max()))
            bias_plt.set_title('bias-{}'.format(count))
            plt.yscale('log')  # 设置纵坐标为10的n次方，以直观显示0-10之间的分布，兼顾高峰。
            
            if not os.path.exists("./log"):
                os.mkdir("log")
                
            plt.tight_layout()
            plt.savefig("./log/{}.png".format(count))
            plt.close()
            count += 1
    # summary_writer.close()
    
    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace
    dynamic_axes = None
    

    print("===================")
    print(model)
    print("===================")
