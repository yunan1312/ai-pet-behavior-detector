"""RTMPose AP-10K 模型架构实现

独立实现 RTMPose-M 的核心架构（CSPNeXt backbone + SimCC head），
无需依赖 MMPose 框架即可加载预训练权重并进行推理。
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class ConvBnSiLU(nn.Module):
    """Conv + BatchNorm + SiLU 激活函数"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class CSPBlock(nn.Module):
    """CSP (Cross Stage Partial) Block"""
    def __init__(self, in_channels, out_channels, expand_ratio=0.5):
        super().__init__()
        mid_channels = int(out_channels * expand_ratio)

        self.conv1 = ConvBnSiLU(in_channels, mid_channels, 1)
        self.conv2 = ConvBnSiLU(in_channels, mid_channels, 1)
        self.conv3 = ConvBnSiLU(mid_channels * 2, out_channels, 1)

        # 深度可分离卷积块
        self.dw_conv = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x2 = self.dw_conv(x2)
        return self.conv3(torch.cat([x1, x2], dim=1))


class CSPNeXtBackbone(nn.Module):
    """CSPNeXt Backbone (P5 架构)"""
    def __init__(self, widen_factor=0.75, deepen_factor=0.67):
        super().__init__()

        # 计算各阶段通道数
        base_channels = [int(c * widen_factor) for c in [64, 128, 256, 512]]
        depths = [int(d * deepen_factor) for d in [3, 6, 6, 3]]

        # Stem
        self.stem = nn.Sequential(
            ConvBnSiLU(3, base_channels[0] // 2, 3, 2, 1),
            ConvBnSiLU(base_channels[0] // 2, base_channels[0], 3, 2, 1),
        )

        # Stages
        self.stages = nn.ModuleList()
        in_ch = base_channels[0]

        for i in range(4):
            out_ch = base_channels[i]
            stage_layers = []

            # 下采样
            if i > 0:
                stage_layers.append(ConvBnSiLU(in_ch, out_ch, 3, 2, 1))
                in_ch = out_ch

            # CSP 块
            for _ in range(depths[i]):
                stage_layers.append(CSPBlock(in_ch, out_ch))
                in_ch = out_ch

            self.stages.append(nn.Sequential(*stage_layers))

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        return x


class SimCCHead(nn.Module):
    """SimCC (Simple Coordinate Classification) Head

    将关键点预测转化为 x/y 坐标的分类问题。
    """
    def __init__(self, in_channels=576, num_keypoints=17, input_size=(256, 256), simcc_split_ratio=2.0):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.input_size = input_size
        self.simcc_split_ratio = simcc_split_ratio

        feat_h = input_size[0] // 32
        feat_w = input_size[1] // 32

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 分类头
        x_channels = num_keypoints * int(input_size[1] * simcc_split_ratio)
        y_channels = num_keypoints * int(input_size[0] * simcc_split_ratio)

        self.fc_x = nn.Linear(in_channels, x_channels)
        self.fc_y = nn.Linear(in_channels, y_channels)

    def forward(self, x):
        # x: [B, C, H, W]
        B = x.shape[0]

        # 全局特征
        feat = self.gap(x).flatten(1)  # [B, C]

        # 预测 x/y 坐标分布
        pred_x = self.fc_x(feat)  # [B, K * W']
        pred_y = self.fc_y(feat)  # [B, K * H']

        # 重塑为 [B, K, W'] 和 [B, K, H']
        w_out = int(self.input_size[1] * self.simcc_split_ratio)
        h_out = int(self.input_size[0] * self.simcc_split_ratio)

        pred_x = pred_x.view(B, self.num_keypoints, w_out)
        pred_y = pred_y.view(B, self.num_keypoints, h_out)

        return pred_x, pred_y


class RTMPoseModel(nn.Module):
    """RTMPose-M 模型（用于 AP-10K 动物姿态估计）"""
    def __init__(self, num_keypoints=17, input_size=(256, 256)):
        super().__init__()
        self.backbone = CSPNeXtBackbone(widen_factor=0.75, deepen_factor=0.67)
        self.head = SimCCHead(
            in_channels=int(512 * 0.75),  # 384
            num_keypoints=num_keypoints,
            input_size=input_size,
            simcc_split_ratio=2.0
        )
        self.input_size = input_size

    def forward(self, x):
        feat = self.backbone(x)
        pred_x, pred_y = self.head(feat)
        return pred_x, pred_y


def load_rtmpose_weights(model: RTMPoseModel, checkpoint_path: str) -> RTMPoseModel:
    """加载 RTMPose 预训练权重"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 尝试加载权重（允许部分匹配）
    model_dict = model.state_dict()
    pretrained_dict = {}

    matched = 0
    skipped = 0
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
                matched += 1
            else:
                skipped += 1
        else:
            skipped += 1

    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"  权重加载: 匹配 {matched} 层, 跳过 {skipped} 层")
    return model


def convert_to_onnx(checkpoint_path: str, onnx_path: str, input_size=(256, 256)) -> bool:
    """将 RTMPose PyTorch 模型转换为 ONNX 格式"""
    try:
        print("  创建 RTMPose-M 模型...")
        model = RTMPoseModel(num_keypoints=17, input_size=input_size)

        print("  加载预训练权重...")
        model = load_rtmpose_weights(model, checkpoint_path)
        model.eval()

        print("  导出 ONNX 模型...")
        dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

        with torch.no_grad():
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                opset_version=11,
                input_names=['input'],
                output_names=['pred_x', 'pred_y'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'pred_x': {0: 'batch_size'},
                    'pred_y': {0: 'batch_size'}
                }
            )

        file_size = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  ONNX 导出成功: {onnx_path} ({file_size:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ONNX 导出失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def decode_simcc(pred_x: np.ndarray, pred_y: np.ndarray, input_size=(256, 256)) -> np.ndarray:
    """解码 SimCC 输出为关键点坐标

    Args:
        pred_x: [K, W'] x 坐标分布
        pred_y: [K, H'] y 坐标分布
        input_size: 输入图像尺寸

    Returns:
        keypoints: [K, 3] 关键点坐标 (x, y, confidence)
    """
    num_kpts = pred_x.shape[0]

    # 获取最大响应位置
    x_idx = np.argmax(pred_x, axis=1)
    y_idx = np.argmax(pred_y, axis=1)

    # 获取置信度
    x_conf = np.max(pred_x, axis=1)
    y_conf = np.max(pred_y, axis=1)
    conf = (x_conf + y_conf) / 2

    # 转换为原始坐标
    simcc_split_ratio = pred_x.shape[1] / input_size[1]
    x = x_idx / simcc_split_ratio
    y = y_idx / simcc_split_ratio

    keypoints = np.stack([x, y, conf], axis=1)
    return keypoints


import os

if __name__ == "__main__":
    # 测试模型创建和转换
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    pth_path = os.path.join(models_dir, "rtmpose_ap10k.pth")
    onnx_path = os.path.join(models_dir, "rtmpose_ap10k.onnx")

    if os.path.exists(pth_path):
        print(f"Found PyTorch model: {pth_path}")
        convert_to_onnx(pth_path, onnx_path)
    else:
        print(f"PyTorch model not found: {pth_path}")
