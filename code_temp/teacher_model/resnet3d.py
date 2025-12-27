# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Tuple, Callable
import torch
import torch.nn as nn
import torchvision.models as models
from collections import OrderedDict
import os

# -------------------------
# Pluggable 3D Normalization Factory
# -------------------------
NormLayer3d = Callable[[int], nn.Module]

def bn_factory() -> NormLayer3d:
    return lambda c: nn.BatchNorm3d(c)

def in_factory(track_running_stats: bool = False) -> NormLayer3d:
    return lambda c: nn.InstanceNorm3d(c, affine=True, track_running_stats=track_running_stats)

def gn_factory(groups: int = 32) -> NormLayer3d:
    def _gn(c: int) -> nn.Module:
        g = min(groups, c)
        while g > 1 and (c % g) != 0:
            g -= 1
        return nn.GroupNorm(g, c, affine=True)
    return _gn

def _default_norm_layer(num_features: int) -> nn.Module:
    return nn.BatchNorm3d(num_features)

# -------------------------
# Basic 3D building blocks
# -------------------------
class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
        downsample: nn.Module | None = None,
        norm_layer: NormLayer3d = _default_norm_layer,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: Tuple[int, int, int] = (1, 1, 1),
        downsample: nn.Module | None = None,
        norm_layer: NormLayer3d = _default_norm_layer,
    ) -> None:
        super().__init__()
        width = planes
        self.conv1 = nn.Conv3d(in_planes, width, kernel_size=1, stride=(1, 1, 1), bias=False)
        self.bn1 = norm_layer(width)
        self.conv2 = nn.Conv3d(width, width, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out)
        out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

# -------------------------
# ResNet3D backbone
# -------------------------
class ResNet3D(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock3D],
        layers: List[int],
        num_classes: int = 2,
        in_channels: int = 1,
        stem_channels: int = 64,
        downsample_depth_in_layer4: bool = True,
        norm_layer: NormLayer3d = _default_norm_layer,
    ) -> None:
        super().__init__()
        self.inplanes = stem_channels
        self.norm_layer = norm_layer

        # [关键修改] 回归标准 3D ResNet 设置，完美适配 MedicalNet 权重
        # Kernel: (7,7,7), Stride: (2,2,2), Padding: (3,3,3)
        self.conv1 = nn.Conv3d(
            in_channels, stem_channels,
            kernel_size=(7, 7, 7),  # 这里的形状现在变成了 [C_out, C_in, 7, 7, 7]
            stride=(2, 2, 2),       # 下采样更快，但保留了预训练的感受野逻辑
            padding=(3, 3, 3),
            bias=False
        )
        self.bn1 = norm_layer(stem_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # MaxPool 保持一致性，也可以设为 stride=(2,2,2)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))

        self.layer1 = self._make_layer(block, 64,  layers[0], stride=(1, 1, 1))
        self.layer2 = self._make_layer(block, 128, layers[1], stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], stride=(2, 2, 2))
        l4_stride = (2, 2, 2) if downsample_depth_in_layer4 else (1, 2, 2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=l4_stride)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._initialize_weights()
        self._zero_init_residual()

    def _make_layer(self, block, planes, blocks, stride=(1, 1, 1)):
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride=stride, downsample=downsample, norm_layer=self.norm_layer)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, norm_layer=self.norm_layer))
        return nn.Sequential(*layers)

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm3d, nn.InstanceNorm3d, nn.GroupNorm)):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.constant_(m.weight, 1.0)
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def _zero_init_residual(self) -> None:
        for m in self.modules():
            if isinstance(m, Bottleneck3D):
                if hasattr(m.bn3, "weight") and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)
            elif isinstance(m, BasicBlock3D):
                if hasattr(m.bn2, "weight") and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x); x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# -------------------------
# Weight Loading Utilities
# -------------------------

def load_pretrained_2d_weights_to_3d(model_3d: nn.Module, arch: str = 'resnet18') -> None:
    print(f"[Init] Loading 2D ImageNet weights for {arch} and inflating to 3D...")
    if arch == 'resnet18':
        model_2d = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif arch == 'resnet34':
        model_2d = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
    elif arch == 'resnet50':
        model_2d = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        return

    model_2d_dict = model_2d.state_dict()
    model_3d_dict = model_3d.state_dict()
    loaded_layers = 0
    for name_3d, param_3d in model_3d_dict.items():
        if 'fc' in name_3d: continue
        if name_3d in model_2d_dict:
            param_2d = model_2d_dict[name_3d]
            if param_3d.shape == param_2d.shape:
                model_3d_dict[name_3d] = param_2d
                loaded_layers += 1
            elif param_3d.dim() == 5 and param_2d.dim() == 4:
                depth = param_3d.shape[2]
                new_weight = param_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1)
                new_weight = new_weight / depth 
                model_3d_dict[name_3d] = new_weight
                loaded_layers += 1
    model_3d.load_state_dict(model_3d_dict, strict=False)
    print(f"[Init] Successfully loaded/inflated {loaded_layers} layers from 2D ImageNet.")


def load_medicalnet_weights(model: nn.Module, weight_path: str) -> None:
    """
    智能加载 MedicalNet 权重 (64, 1, 7, 7, 7)
    适配双通道输入: (64, 2, 7, 7, 7)
    """
    if not os.path.exists(weight_path):
        print(f"[Init] Pretrain path not found: {weight_path}. Skipping.")
        return

    print(f"[Init] Loading MedicalNet weights from: {weight_path}")
    try:
        state_dict = torch.load(weight_path, map_location='cpu')
    except Exception as e:
        print(f"[Error] Failed to load file: {e}")
        return

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k 
        new_state_dict[name] = v

    model_dict = model.state_dict()
    loaded_layers = 0
    skipped_layers = 0
    ready_to_load = {}

    for k, v in new_state_dict.items():
        if k in model_dict:
            target_shape = model_dict[k].shape
            source_shape = v.shape
            
            # Case A: 形状完全匹配
            if source_shape == target_shape:
                ready_to_load[k] = v
                loaded_layers += 1
            
            # Case B: Conv1 通道不匹配 (MedicalNet=1 vs Model=N)
            # 现在深度都是 7，不需要裁剪，只需要扩展通道
            elif k == "conv1.weight" and source_shape[1] != target_shape[1]:
                print(f"[Adapt] Adapting {k} from {source_shape} to {target_shape} (Channel Expansion)")
                
                # 检查深度维度是否匹配 (现在应该都是 7)
                if source_shape[2] != target_shape[2]:
                    # 如果仍然不匹配，说明你可能没改好模型定义，这里打印个警告
                    print(f"[Warn] Depth mismatch still exists! Src:{source_shape[2]} Tgt:{target_shape[2]}")
                
                # 复制通道 (1 -> 2)
                repeat_times = target_shape[1]
                v_adapted = v.repeat(1, repeat_times, 1, 1, 1)
                v_adapted = v_adapted / repeat_times # 归一化能量
                
                if v_adapted.shape == target_shape:
                    ready_to_load[k] = v_adapted
                    loaded_layers += 1
                else:
                    print(f"[Fail] Shape mismatch after adapt: {v_adapted.shape} != {target_shape}")
                    skipped_layers += 1
            else:
                skipped_layers += 1
        else:
            skipped_layers += 1

    model_dict.update(ready_to_load)
    model.load_state_dict(model_dict)
    
    print(f"[Init] MedicalNet weights loaded! ({loaded_layers} layers loaded, {skipped_layers} skipped)")


# -------------------------
# Factory
# -------------------------
def generate_resnet18(num_classes: int = 2, in_channels: int = 1,
                      downsample_depth_in_layer4: bool = True,
                      norm_layer: NormLayer3d = _default_norm_layer) -> ResNet3D:
    return ResNet3D(
        block=BasicBlock3D,
        layers=[2, 2, 2, 2],
        num_classes=num_classes,
        in_channels=in_channels,
        downsample_depth_in_layer4=downsample_depth_in_layer4,
        norm_layer=norm_layer,
    )

def generate_resnet34(num_classes: int = 2, in_channels: int = 1,
                      downsample_depth_in_layer4: bool = True,
                      norm_layer: NormLayer3d = _default_norm_layer) -> ResNet3D:
    return ResNet3D(
        block=BasicBlock3D,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        downsample_depth_in_layer4=downsample_depth_in_layer4,
        norm_layer=norm_layer,
    )

def generate_resnet50(num_classes: int = 2, in_channels: int = 1,
                      downsample_depth_in_layer4: bool = True,
                      norm_layer: NormLayer3d = _default_norm_layer) -> ResNet3D:
    return ResNet3D(
        block=Bottleneck3D,
        layers=[3, 4, 6, 3],
        num_classes=num_classes,
        in_channels=in_channels,
        downsample_depth_in_layer4=downsample_depth_in_layer4,
        norm_layer=norm_layer,
    )