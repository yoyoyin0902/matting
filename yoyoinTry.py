import os
import sys
import cv2
import uuid
import glob
import time
import math
import shutil
import random
import torch
import darknet
import logging
import shutil
import datetime
import argparse
import torchvision
import numpy as np
import pyzed.sl as sl
import ogl_viewer.viewer as gl

from pylsd.lsd import lsd
from ctypes import *
# from random import randintre
from tqdm import tqdm
from PIL import Image
from queue import Queue
from shutil import copyfile
from threading import Thread, enumerate 
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision.transforms.functional import to_pil_image
from threading import Thread

from torch.utils.data import Dataset
from typing import Callable, Optional, List, Tuple
from torch import nn
from torchvision.models.resnet import ResNet, Bottleneck
from torch import Tensor

# Get the top-level logger object
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# torch.backends.cudnn.enabled = False

# --------------- hy ---------------
class HomographicAlignment:
    """
    Apply homographic alignment on background to match with the source image.
    """
  
    def __init__(self):
        self.detector = cv2.ORB_create()
        self.matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE)
  
    def __call__(self, src, bgr):
        src = np.asarray(src)
        bgr = np.asarray(bgr)
  
        keypoints_src, descriptors_src = self.detector.detectAndCompute(src, None)
        keypoints_bgr, descriptors_bgr = self.detector.detectAndCompute(bgr, None)
  
        matches = self.matcher.match(descriptors_bgr, descriptors_src, None)
        matches.sort(key=lambda x: x.distance, reverse=False)
        num_good_matches = int(len(matches) * 0.15)
        matches = matches[:num_good_matches]
  
        points_src = np.zeros((len(matches), 2), dtype=np.float32)
        points_bgr = np.zeros((len(matches), 2), dtype=np.float32)
        for i, match in enumerate(matches):
            points_src[i, :] = keypoints_src[match.trainIdx].pt
            points_bgr[i, :] = keypoints_bgr[match.queryIdx].pt
  
        H, _ = cv2.findHomography(points_bgr, points_src, cv2.RANSAC)
  
        h, w = src.shape[:2]
        bgr = cv2.warpPerspective(bgr, H, (w, h))
        msk = cv2.warpPerspective(np.ones((h, w)), H, (w, h))
  
        # For areas that is outside of the background,
        # We just copy pixels from the source.
        bgr[msk != 1] = src[msk != 1]
  
        src = Image.fromarray(src)
        bgr = Image.fromarray(bgr)
  
        return src, bgr
  
  
class Refiner(nn.Module):
    # For TorchScript export optimization.
    __constants__ = ['kernel_size', 'patch_crop_method', 'patch_replace_method']
  
    def __init__(self,
                 mode: str,
                 sample_pixels: int,
                 threshold: float,
                 kernel_size: int = 3,
                 prevent_oversampling: bool = True,
                 patch_crop_method: str = 'unfold',
                 patch_replace_method: str = 'scatter_nd'):
        super().__init__()
        assert mode in ['full', 'sampling', 'thresholding']
        assert kernel_size in [1, 3]
        assert patch_crop_method in ['unfold', 'roi_align', 'gather']
        assert patch_replace_method in ['scatter_nd', 'scatter_element']
  
        self.mode = mode
        self.sample_pixels = sample_pixels
        self.threshold = threshold
        self.kernel_size = kernel_size
        self.prevent_oversampling = prevent_oversampling
        self.patch_crop_method = patch_crop_method
        self.patch_replace_method = patch_replace_method
  
        channels = [32, 24, 16, 12, 4]
        self.conv1 = nn.Conv2d(channels[0] + 6 + 4, channels[1], kernel_size, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernel_size, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(channels[2] + 6, channels[3], kernel_size, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(channels[3], channels[4], kernel_size, bias=True)
        self.relu = nn.ReLU(True)
  
    def forward(self,
                src: torch.Tensor,
                bgr: torch.Tensor,
                pha: torch.Tensor,
                fgr: torch.Tensor,
                err: torch.Tensor,
                hid: torch.Tensor):
        H_full, W_full = src.shape[2:]
        H_half, W_half = H_full // 2, W_full // 2
        H_quat, W_quat = H_full // 4, W_full // 4
  
        src_bgr = torch.cat([src, bgr], dim=1)
  
        if self.mode != 'full':
            err = F.interpolate(err, (H_quat, W_quat), mode='bilinear', align_corners=False)
            ref = self.select_refinement_regions(err)
            idx = torch.nonzero(ref.squeeze(1))
            idx = idx[:, 0], idx[:, 1], idx[:, 2]
  
            if idx[0].size(0) > 0:
                x = torch.cat([hid, pha, fgr], dim=1)
                x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
                x = self.crop_patch(x, idx, 2, 3 if self.kernel_size == 3 else 0)
  
                y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
                y = self.crop_patch(y, idx, 2, 3 if self.kernel_size == 3 else 0)
  
                x = self.conv1(torch.cat([x, y], dim=1))
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
  
                x = F.interpolate(x, 8 if self.kernel_size == 3 else 4, mode='nearest')
                y = self.crop_patch(src_bgr, idx, 4, 2 if self.kernel_size == 3 else 0)
  
                x = self.conv3(torch.cat([x, y], dim=1))
                x = self.bn3(x)
                x = self.relu(x)
                x = self.conv4(x)
  
                out = torch.cat([pha, fgr], dim=1)
                out = F.interpolate(out, (H_full, W_full), mode='bilinear', align_corners=False)
                out = self.replace_patch(out, x, idx)
                pha = out[:, :1]
                fgr = out[:, 1:]
            else:
                pha = F.interpolate(pha, (H_full, W_full), mode='bilinear', align_corners=False)
                fgr = F.interpolate(fgr, (H_full, W_full), mode='bilinear', align_corners=False)
        else:
            x = torch.cat([hid, pha, fgr], dim=1)
            x = F.interpolate(x, (H_half, W_half), mode='bilinear', align_corners=False)
            y = F.interpolate(src_bgr, (H_half, W_half), mode='bilinear', align_corners=False)
            if self.kernel_size == 3:
                x = F.pad(x, (3, 3, 3, 3))
                y = F.pad(y, (3, 3, 3, 3))
  
            x = self.conv1(torch.cat([x, y], dim=1))
            x = self.bn1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = self.relu(x)
  
            if self.kernel_size == 3:
                x = F.interpolate(x, (H_full + 4, W_full + 4))
                y = F.pad(src_bgr, (2, 2, 2, 2))
            else:
                x = F.interpolate(x, (H_full, W_full), mode='nearest')
                y = src_bgr
  
            x = self.conv3(torch.cat([x, y], dim=1))
            x = self.bn3(x)
            x = self.relu(x)
            x = self.conv4(x)
  
            pha = x[:, :1]
            fgr = x[:, 1:]
            ref = torch.ones((src.size(0), 1, H_quat, W_quat), device=src.device, dtype=src.dtype)
  
        return pha, fgr, ref
  
    def select_refinement_regions(self, err: torch.Tensor):
        """
        Select refinement regions.
        Input:
            err: error map (B, 1, H, W)
        Output:
            ref: refinement regions (B, 1, H, W). FloatTensor. 1 is selected, 0 is not.
        """
        if self.mode == 'sampling':
            # Sampling mode.
            b, _, h, w = err.shape
            err = err.view(b, -1)
            idx = err.topk(self.sample_pixels // 16, dim=1, sorted=False).indices
            ref = torch.zeros_like(err)
            ref.scatter_(1, idx, 1.)
            if self.prevent_oversampling:
                ref.mul_(err.gt(0).float())
            ref = ref.view(b, 1, h, w)
        else:
            # Thresholding mode.
            ref = err.gt(self.threshold).float()
        return ref
  
    def crop_patch(self,
                   x: torch.Tensor,
                   idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                   size: int,
                   padding: int):
        """
        Crops selected patches from image given indices.
        Inputs:
            x: image (B, C, H, W).
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            patch: (P, C, h, w), where h = w = size + 2 * padding.
        """
        if padding != 0:
            x = F.pad(x, (padding,) * 4)
  
        if self.patch_crop_method == 'unfold':
            # Use unfold. Best performance for PyTorch and TorchScript.
            return x.permute(0, 2, 3, 1) \
                .unfold(1, size + 2 * padding, size) \
                .unfold(2, size + 2 * padding, size)[idx[0], idx[1], idx[2]]
        elif self.patch_crop_method == 'roi_align':
            # Use roi_align. Best compatibility for ONNX.
            idx = idx[0].type_as(x), idx[1].type_as(x), idx[2].type_as(x)
            b = idx[0]
            x1 = idx[2] * size - 0.5
            y1 = idx[1] * size - 0.5
            x2 = idx[2] * size + size + 2 * padding - 0.5
            y2 = idx[1] * size + size + 2 * padding - 0.5
            boxes = torch.stack([b, x1, y1, x2, y2], dim=1)
            return torchvision.ops.roi_align(x, boxes, size + 2 * padding, sampling_ratio=1)
        else:
            # Use gather. Crops out patches pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size, padding)
            pat = torch.gather(x.view(-1), 0, idx_pix.view(-1))
            pat = pat.view(-1, x.size(1), size + 2 * padding, size + 2 * padding)
            return pat
  
    def replace_patch(self,
                      x: torch.Tensor,
                      y: torch.Tensor,
                      idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Replaces patches back into image given index.
        Inputs:
            x: image (B, C, H, W)
            y: patches (P, C, h, w)
            idx: selection indices Tuple[(P,), (P,), (P,)] where the 3 values are (B, H, W) index.
        Output:
            image: (B, C, H, W), where patches at idx locations are replaced with y.
        """
        xB, xC, xH, xW = x.shape
        yB, yC, yH, yW = y.shape
        if self.patch_replace_method == 'scatter_nd':
            # Use scatter_nd. Best performance for PyTorch and TorchScript. Replacing patch by patch.
            x = x.view(xB, xC, xH // yH, yH, xW // yW, yW).permute(0, 2, 4, 1, 3, 5)
            x[idx[0], idx[1], idx[2]] = y
            x = x.permute(0, 3, 1, 4, 2, 5).view(xB, xC, xH, xW)
            return x
        else:
            # Use scatter_element. Best compatibility for ONNX. Replacing pixel by pixel.
            idx_pix = self.compute_pixel_indices(x, idx, size=4, padding=0)
            return x.view(-1).scatter_(0, idx_pix.view(-1), y.view(-1)).view(x.shape)
  
    def compute_pixel_indices(self,
                              x: torch.Tensor,
                              idx: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                              size: int,
                              padding: int):
        """
        Compute selected pixel indices in the tensor.
        Used for crop_method == 'gather' and replace_method == 'scatter_element', which crop and replace pixel by pixel.
        Input:
            x: image: (B, C, H, W)
            idx: selection indices Tuple[(P,), (P,), (P,),], where the 3 values are (B, H, W) index.
            size: center size of the patch, also stride of the crop.
            padding: expansion size of the patch.
        Output:
            idx: (P, C, O, O) long tensor where O is the output size: size + 2 * padding, P is number of patches.
                 the element are indices pointing to the input x.view(-1).
        """
        B, C, H, W = x.shape
        S, P = size, padding
        O = S + 2 * P
        b, y, x = idx
        n = b.size(0)
        c = torch.arange(C)
        o = torch.arange(O)
        idx_pat = (c * H * W).view(C, 1, 1).expand([C, O, O]) + (o * W).view(1, O, 1).expand([C, O, O]) + o.view(1, 1,
                                                                                                                 O).expand(
            [C, O, O])
        idx_loc = b * W * H + y * W * S + x * S
        idx_pix = idx_loc.view(-1, 1, 1, 1).expand([n, C, O, O]) + idx_pat.view(1, C, O, O).expand([n, C, O, O])
        return idx_pix
  
  
def load_matched_state_dict(model, state_dict, print_stats=True):
    """
    Only loads weights that matched in key and shape. Ignore other weights.
    """
    num_matched, num_total = 0, 0
    curr_state_dict = model.state_dict()
    for key in curr_state_dict.keys():
        num_total += 1
        if key in state_dict and curr_state_dict[key].shape == state_dict[key].shape:
            curr_state_dict[key] = state_dict[key]
            num_matched += 1
    model.load_state_dict(curr_state_dict)
    if print_stats:
        print(f'Loaded state_dict: {num_matched}/{num_total} matched')
  
  
def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
  
  
class ConvNormActivation(torch.nn.Sequential):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: Optional[int] = None,
            groups: int = 1,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.BatchNorm2d,
            activation_layer: Optional[Callable[..., torch.nn.Module]] = torch.nn.ReLU,
            dilation: int = 1,
            inplace: bool = True,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        layers = [torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,
                                  dilation=dilation, groups=groups, bias=norm_layer is None)]
        if norm_layer is not None:
            layers.append(norm_layer(out_channels))
        if activation_layer is not None:
            layers.append(activation_layer(inplace=inplace))
        super().__init__(*layers)
        self.out_channels = out_channels
  
  
class InvertedResidual(nn.Module):
    def __init__(
            self,
            inp: int,
            oup: int,
            stride: int,
            expand_ratio: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
  
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
  
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup
  
        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvNormActivation(inp, hidden_dim, kernel_size=1, norm_layer=norm_layer,
                                             activation_layer=nn.ReLU6))
        layers.extend([
            # dw
            ConvNormActivation(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim, norm_layer=norm_layer,
                               activation_layer=nn.ReLU6),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        ])
        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1
  
    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
  
  
class MobileNetV2(nn.Module):
    def __init__(
            self,
            num_classes: int = 1000,
            width_mult: float = 1.0,
            inverted_residual_setting: Optional[List[List[int]]] = None,
            round_nearest: int = 8,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        """
        MobileNet V2 main class
        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
        """
        super(MobileNetV2, self).__init__()
  
        if block is None:
            block = InvertedResidual
  
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
  
        input_channel = 32
        last_channel = 1280
  
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]
  
        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))
  
        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        features: List[nn.Module] = [ConvNormActivation(3, input_channel, stride=2, norm_layer=norm_layer,
                                                        activation_layer=nn.ReLU6)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t, norm_layer=norm_layer))
                input_channel = output_channel
        # building last several layers
        features.append(ConvNormActivation(input_channel, self.last_channel, kernel_size=1, norm_layer=norm_layer,
                                           activation_layer=nn.ReLU6))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
  
        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )
  
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
  
    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        x = self.features(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
  
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
  
  
class MobileNetV2Encoder(MobileNetV2):
    """
    MobileNetV2Encoder inherits from torchvision's official MobileNetV2. It is modified to
    use dilation on the last block to maintain output stride 16, and deleted the
    classifier block that was originally used for classification. The forward method
    additionally returns the feature maps at all resolutions for decoder's use.
    """
  
    def __init__(self, in_channels, norm_layer=None):
        super().__init__()
  
        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.features[0][0] = nn.Conv2d(in_channels, 32, 3, 2, 1, bias=False)
  
        # Remove last block
        self.features = self.features[:-1]
  
        # Change to use dilation to maintain output stride = 16
        self.features[14].conv[1][0].stride = (1, 1)
        for feature in self.features[15:]:
            feature.conv[1][0].dilation = (2, 2)
            feature.conv[1][0].padding = (2, 2)
  
        # Delete classifier
        del self.classifier
  
    def forward(self, x):
        x0 = x  # 1/1
        x = self.features[0](x)
        x = self.features[1](x)
        x1 = x  # 1/2
        x = self.features[2](x)
        x = self.features[3](x)
        x2 = x  # 1/4
        x = self.features[4](x)
        x = self.features[5](x)
        x = self.features[6](x)
        x3 = x  # 1/8
        x = self.features[7](x)
        x = self.features[8](x)
        x = self.features[9](x)
        x = self.features[10](x)
        x = self.features[11](x)
        x = self.features[12](x)
        x = self.features[13](x)
        x = self.features[14](x)
        x = self.features[15](x)
        x = self.features[16](x)
        x = self.features[17](x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0
  
  
class Decoder(nn.Module):
  
    def __init__(self, channels, feature_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(feature_channels[0] + channels[0], channels[1], 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.conv2 = nn.Conv2d(feature_channels[1] + channels[1], channels[2], 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels[2])
        self.conv3 = nn.Conv2d(feature_channels[2] + channels[2], channels[3], 3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels[3])
        self.conv4 = nn.Conv2d(feature_channels[3] + channels[3], channels[4], 3, padding=1)
        self.relu = nn.ReLU(True)
  
    def forward(self, x4, x3, x2, x1, x0):
        x = F.interpolate(x4, size=x3.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x3], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x2.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x1.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = F.interpolate(x, size=x0.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, x0], dim=1)
        x = self.conv4(x)
        return x
  
  
class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)
  
  
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)
  
  
class ASPP(nn.Module):
    def __init__(self, in_channels: int, atrous_rates: List[int], out_channels: int = 256) -> None:
        super(ASPP, self).__init__()
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()))
  
        rates = tuple(atrous_rates)
        for rate in rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))
  
        modules.append(ASPPPooling(in_channels, out_channels))
  
        self.convs = nn.ModuleList(modules)
  
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)
  
  
class ResNetEncoder(ResNet):
    layers = {
        'resnet50': [3, 4, 6, 3],
        'resnet101': [3, 4, 23, 3],
    }
  
    def __init__(self, in_channels, variant='resnet101', norm_layer=None):
        super().__init__(
            block=Bottleneck,
            layers=self.layers[variant],
            replace_stride_with_dilation=[False, False, True],
            norm_layer=norm_layer)
  
        # Replace first conv layer if in_channels doesn't match.
        if in_channels != 3:
            self.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
  
        # Delete fully-connected layer
        del self.avgpool
        del self.fc
  
    def forward(self, x):
        x0 = x  # 1/1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = x  # 1/2
        x = self.maxpool(x)
        x = self.layer1(x)
        x2 = x  # 1/4
        x = self.layer2(x)
        x3 = x  # 1/8
        x = self.layer3(x)
        x = self.layer4(x)
        x4 = x  # 1/16
        return x4, x3, x2, x1, x0
  
  
class Base(nn.Module):
    """
    A generic implementation of the base encoder-decoder network inspired by DeepLab.
    Accepts arbitrary channels for input and output.
    """
  
    def __init__(self, backbone: str, in_channels: int, out_channels: int):
        super().__init__()
        assert backbone in ["resnet50", "resnet101", "mobilenetv2"]
        if backbone in ['resnet50', 'resnet101']:
            self.backbone = ResNetEncoder(in_channels, variant=backbone)
            self.aspp = ASPP(2048, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [512, 256, 64, in_channels])
        else:
            self.backbone = MobileNetV2Encoder(in_channels)
            self.aspp = ASPP(320, [3, 6, 9])
            self.decoder = Decoder([256, 128, 64, 48, out_channels], [32, 24, 16, in_channels])
  
    def forward(self, x):
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        return x
  
    def load_pretrained_deeplabv3_state_dict(self, state_dict, print_stats=True):
        # Pretrained DeepLabV3 models are provided by <https://github.com/VainF/DeepLabV3Plus-Pytorch>.
        # This method converts and loads their pretrained state_dict to match with our model structure.
        # This method is not needed if you are not planning to train from deeplab weights.
        # Use load_state_dict() for normal weight loading.
  
        # Convert state_dict naming for aspp module
        state_dict = {k.replace('classifier.classifier.0', 'aspp'): v for k, v in state_dict.items()}
  
        if isinstance(self.backbone, ResNetEncoder):
            # ResNet backbone does not need change.
            load_matched_state_dict(self, state_dict, print_stats)
        else:
            # Change MobileNetV2 backbone to state_dict format, then change back after loading.
            backbone_features = self.backbone.features
            self.backbone.low_level_features = backbone_features[:4]
            self.backbone.high_level_features = backbone_features[4:]
            del self.backbone.features
            load_matched_state_dict(self, state_dict, print_stats)
            self.backbone.features = backbone_features
            del self.backbone.low_level_features
            del self.backbone.high_level_features
  
  
class MattingBase(Base):
  
    def __init__(self, backbone: str):
        super().__init__(backbone, in_channels=6, out_channels=(1 + 3 + 1 + 32))
  
    def forward(self, src, bgr):
        x = torch.cat([src, bgr], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha = x[:, 0:1].clamp_(0., 1.)
        fgr = x[:, 1:4].add(src).clamp_(0., 1.)
        err = x[:, 4:5].clamp_(0., 1.)
        hid = x[:, 5:].relu_()
        return pha, fgr, err, hid
  
  
class MattingRefine(MattingBase):
  
    def __init__(self,
                 backbone: str,
                 backbone_scale: float = 1 / 4,
                 refine_mode: str = 'sampling',
                 refine_sample_pixels: int = 80_000,
                 refine_threshold: float = 0.1,
                 refine_kernel_size: int = 3,
                 refine_prevent_oversampling: bool = True,
                 refine_patch_crop_method: str = 'unfold',
                 refine_patch_replace_method: str = 'scatter_nd'):
        assert backbone_scale <= 1 / 2, 'backbone_scale should not be greater than 1/2'
        super().__init__(backbone)
        self.backbone_scale = backbone_scale
        self.refiner = Refiner(refine_mode,
                               refine_sample_pixels,
                               refine_threshold,
                               refine_kernel_size,
                               refine_prevent_oversampling,
                               refine_patch_crop_method,
                               refine_patch_replace_method)
  
    def forward(self, src, bgr):
        assert src.size() == bgr.size(), 'src and bgr must have the same shape'
        assert src.size(2) // 4 * 4 == src.size(2) and src.size(3) // 4 * 4 == src.size(3), \
            'src and bgr must have width and height that are divisible by 4'
  
        # Downsample src and bgr for backbone
        src_sm = F.interpolate(src,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
        bgr_sm = F.interpolate(bgr,
                               scale_factor=self.backbone_scale,
                               mode='bilinear',
                               align_corners=False,
                               recompute_scale_factor=True)
  
        # Base
        x = torch.cat([src_sm, bgr_sm], dim=1)
        x, *shortcuts = self.backbone(x)
        x = self.aspp(x)
        x = self.decoder(x, *shortcuts)
        pha_sm = x[:, 0:1].clamp_(0., 1.)
        fgr_sm = x[:, 1:4]
        err_sm = x[:, 4:5].clamp_(0., 1.)
        hid_sm = x[:, 5:].relu_()
  
        # Refiner
        pha, fgr, ref_sm = self.refiner(src, bgr, pha_sm, fgr_sm, err_sm, hid_sm)
  
        # Clamp outputs
        pha = pha.clamp_(0., 1.)
        fgr = fgr.add_(src).clamp_(0., 1.)
        fgr_sm = src_sm.add_(fgr_sm).clamp_(0., 1.)
  
        return pha, fgr, pha_sm, fgr_sm, err_sm, ref_sm
  
  
class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])
        print(self.filenames)                        
  
    def __len__(self):
        return len(self.filenames)
  
    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
        if self.transforms:
            img = self.transforms(img)
  
        return img
  
  
class NewImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = [root]
        # print(self.filenames)
  
    def __len__(self):
        return len(self.filenames)
  
    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = img.convert(self.mode)
  
        if self.transforms:
            img = self.transforms(img)
  
        return img
  
  
class ZipDataset(Dataset):
    def __init__(self, datasets: List[Dataset], transforms=None, assert_equal_length=False):
        self.datasets = datasets
        self.transforms = transforms
  
        if assert_equal_length:
            for i in range(1, len(datasets)):
                assert len(datasets[i]) == len(datasets[i - 1]), 'Datasets are not equal in length.'
  
    def __len__(self):
        return max(len(d) for d in self.datasets)
  
    def __getitem__(self, idx):
        x = tuple(d[idx % len(d)] for d in self.datasets)
        print(x)
        if self.transforms:
            x = self.transforms(*x)
        return x
  
  
class PairCompose(T.Compose):
    def __call__(self, *x):
        for transform in self.transforms:
            x = transform(*x)
        return x
  
  
class PairApply:
    def __init__(self, transforms):
        self.transforms = transforms
  
    def __call__(self, *x):
        return [self.transforms(xi) for xi in x]

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            # 文件名列表，包含完整路径
            Filelist.append(os.path.join(home, filename))
            # # 文件名列表，只包含文件名
            # Filelist.append( filename)

# ---------------------------------------------------------Matting-------------------------------------------------------------------#
def handle(image_path, bgr_path):
    print(image_path)
    print(bgr_path)
    # set device
    device = torch.device(args.device)
    # Load model
    if args.model_type == 'mattingbase':
        model = MattingBase(args.model_backbone)
    if args.model_type == 'mattingrefine':
        model = MattingRefine(
            args.model_backbone,
            args.model_backbone_scale,
            args.model_refine_mode,
            args.model_refine_sample_pixels,
            args.model_refine_threshold,
            args.model_refine_kernel_size)

    model = model.to(device).eval()
    model.load_state_dict(torch.load(args.model_checkpoint, map_location=device), strict=False)

    assert 'err' not in args.output_types or args.model_type in ['mattingbase', 'mattingrefine'], \
        'Only mattingbase and mattingrefine support err output'
    assert 'ref' not in args.output_types or args.model_type in ['mattingrefine'], \
        'Only mattingrefine support ref output'

        # --------------- Main ---------------  
#region
    # set imgfile
#ednregion
    dataset = ZipDataset([NewImagesDataset(image_path),NewImagesDataset(bgr_path),], assert_equal_length=True, transforms=PairCompose([
        HomographicAlignment() if args.preprocess_alignment else PairApply(nn.Identity()),PairApply(T.ToTensor())]))
        
    dataloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
             
    # print(dataloader)
    with torch.no_grad():
        for (src, bgr) in dataloader:
            src = src.to(device, non_blocking=True)
            bgr = bgr.to(device, non_blocking=True)

            if args.model_type == 'mattingbase':
                pha, fgr, err, _ = model(src, bgr)
            elif args.model_type == 'mattingrefine':
                pha, fgr, _, _, err, ref = model(src, bgr)
            
            com = torch.cat([fgr * pha.ne(0), pha], dim=1)
            
    return com,fgr,pha

#----------------------------------------------------Procseeing--------------------------------------------------------------------#
#tensor to numpy
def tensor_to_np(tensor):
    img = tensor.mul(255).byte()
    img = img.cpu().numpy().squeeze(0).transpose((1, 2, 0))
    return img

#after matting processing
def post_processing(pha):
    pha_img = tensor_to_np(pha)

    
    imgBlur = cv2.GaussianBlur(pha_img, (5, 5), 0)
    # gray_img = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

    # kernel = np.ones((5, 5),np.uint8)
    # imgErode = cv2.erode(imgBlur, kernel, iterations = 5)
    # imgDil = cv2.dilate(imgErode, kernel, iterations = 5)

    ret,bin_img = cv2.threshold(imgBlur,110,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return bin_img

#----------------------------------------------------Segment--------------------------------------------------------------------#
def getLinearEquation(p1x, p1y, p2x, p2y):
    sign = 1
    a = p2y - p1y
    if a < 0:
        sign = -1
        a = sign * a
    b = sign * (p1x - p2x)
    c = sign * (p1y * p2x - p1x * p2y)
    return [a, b, c]


def distance(x0,y0,x1,y1):
    result = math.sqrt((x0-x1)**2+(y0-y1)**2)
    return result

def max2(x):
    m1 = max(x)
    x2 = x.copy()
    x2.remove(m1)
    m2 = max(x2)
    x3 = x2.copy()
    x3.remove(m2)
    m3 = max(x3)

    x4 = x3.copy()
    x3.remove(m3)
    m4 = max(x4)
    return m1,m2,m3,m4

def angle(x1, y1, x2, y2):

    if x1 == x2: #線是豎直的
        return 90

    if y1 == y2: #線是水平的
        return 0

    k = -(y2 - y1) / (x2 - x1)
    result = np.arctan(k) * 180 / np.pi 
    # if x1 > x2 and y1 > y2:
    #     result += 180
    # elif x1 > x2 and y1 < y2:
    #     result += 180
    # elif x1 < x2 and y1 < y2:
    #     result += 360
    print("angle: " + str(result) + "度")
    return result
#line segment
def rota_rect(box, theta, x, y):
    """
    :param box: 正矩形的四个顶点
    :param theta: 旋转角度
    :param x: 旋转中心(x,y)
    :param y: 旋转中心(x,y)
    :return: 旋转矩形的四个顶点坐标
    """
    # 旋转矩形
    box_matrix = np.array(box) - np.repeat(np.array([[x, y]]), len(box), 0)
    theta = theta / 180. * np.pi
    rota_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]], np.float)

    new_box = box_matrix.dot(rota_matrix) + np.repeat(np.array([[x, y]]), len(box), 0)
    return new_box

def line_Segment(mat,orig):
    mat_img = cv2.imread(str(mat),0)
    orig_img = cv2.imread(orig)

    #這是舊版本 opencv不在支援此library
    # lsd = cv2.createLineSegmentDetector(0)
    # dlines = lsd.detect(mat_img)

    dlines = lsd(mat_img) #dlines = [point1.x, point1.y, point2.x, point2.y, width]
    
    ver_lines = []
    coordinate = []
    angle1 = []

    for dline in dlines:
        # print(dline[i])
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        print(distance)
        ver_lines.append(distance)

    maxIndex = max2(ver_lines)

    for dline in dlines:
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
    
        if(distance >= int(maxIndex[1])):
            # cv2.line(orig_img,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
            coordinate.append(((x0,y0),(x1,y1)))

            result = angle(x0,y0,x1,y1)
            angle1.append(result)

    line1 = math.sqrt((coordinate[0][1][0]-coordinate[1][1][0])**2+(coordinate[0][1][1]-coordinate[1][1][1])**2)
    line2 = math.sqrt((coordinate[0][0][0]-coordinate[1][0][0])**2+(coordinate[0][0][1]-coordinate[1][0][1])**2)

    # cv2.line(orig_img,(coordinate[0][0][0],coordinate[0][0][1]),(coordinate[1][1][0],coordinate[1][1][1]),(255,0,0),2,cv2.LINE_AA)
    # cv2.line(orig_img,(coordinate[0][1][0],coordinate[0][1][1]),(coordinate[1][0][0],coordinate[1][1][1]),(255,0,0),2,cv2.LINE_AA)
    if (line1 > line2):
        # cv2.line(orig_img,coordinate[0][1],coordinate[1][1],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][1][0] + coordinate[1][1][0])/2
        circle_y = (coordinate[0][1][1] + coordinate[1][1][1])/2
    
    else:
        # cv2.line(orig_img,coordinate[0][0],coordinate[1][0],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][0][0] + coordinate[1][0][0])/2
        circle_y = (coordinate[0][0][1] + coordinate[1][0][1])/2

    cv2.circle(orig_img,(int(circle_x),int(circle_y)),2,(0,0,255),2)

    #計算物體長寬用
    object_Image_mat = mat_img.copy()
    object_Image = orig_img.copy()
    object_edges = cv2.Canny(object_Image_mat, 70, 210)
    contours_object, hierarchy = cv2.findContours(object_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    for cnt in contours_object:
        # 最小外接矩形框，有方向角
        rect = cv2.minAreaRect(cnt)
        print(rect)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        # cv2.drawContours(object_Image, [box], 0, (0, 0, 255), 2)
        # cv2.imshow("1111",object_Image)

    width = 20
    height = 60

    grasp_left_x = int(circle_x - (width/2.0))
    grasp_left_y = int(circle_y - (height/2.0))
    grasp_right_x = int(circle_x + (width/2.0))
    grasp_right_y = int(circle_y + (height/2.0))
    
    # cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)

    box = [(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
    
    
    print(angle1[0], angle1[1])
    print("222222222222222222222222222222")
    
    
    #手臂旋轉角度(以90度為界)
    # if (angle1[0] and angle1[1])<0 and (angle1[0] and angle1[1])>-90:
    #     real_angle = (angle1[0] + angle1[1])/2.0 -90
    # if (angle1[0] and angle1[1]) > 0 and (angle1[0] and angle1[1]) < 90:

    if (angle1[0]>0 and angle1[1]<0):
        real_angle = angle1[0]
    elif(angle1[0]<0 and angle1[1]>0):
        real_angle = angle1[1]
    else:
        real_angle = (angle1[0] + angle1[1])/2.0

    aa = rota_rect(box,real_angle,circle_x,circle_y)

    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[2][0]),int(aa[2][1])),(255, 0, 0),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[1][0]),int(aa[1][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 0),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[1][0]),int(aa[1][1])),(255, 0, 0),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[2][0]),int(aa[2][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 0),2,cv2.LINE_AA)

    return orig_img ,circle_x,circle_y,real_angle

def line_Segment_cup(mat,orig):
    mat_img = cv2.imread(str(mat),0)
    # cv2.imshow("mat",mat_img)
    orig_img = cv2.imread(orig)

    # lsd = cv2.createLineSegmentDetector(0)
    # dlines = lsd.detect(mat_img)
    
    dlines = lsd(mat_img)
    
    ver_lines = []
    coordinate = []
    angle1 = []

    for dline in dlines:
        # print(dline[i])
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        ver_lines.append(distance)

    maxIndex = max2(ver_lines)

    for dline in dlines:
        x0 = int(round(dline[0]))
        y0 = int(round(dline[1]))
        x1 = int(round(dline[2]))
        y1 = int(round(dline[3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
    
        if(distance >= int(maxIndex[1])):
            # cv2.line(orig_img,(x0,y0),(x1,y1),(0,255,0),2,cv2.LINE_AA)
            coordinate.append(((x0,y0),(x1,y1)))

            result = angle(x0,y0,x1,y1)
            angle1.append(result)

    line1 = math.sqrt((coordinate[0][1][0]-coordinate[1][1][0])**2+(coordinate[0][1][1]-coordinate[1][1][1])**2)
    line2 = math.sqrt((coordinate[0][0][0]-coordinate[1][0][0])**2+(coordinate[0][0][1]-coordinate[1][0][1])**2)
    # cv2.line(orig_img,(coordinate[0][0][0],coordinate[0][0][1]),(coordinate[1][1][0],coordinate[1][1][1]),(255,0,0),2,cv2.LINE_AA)
    # cv2.line(orig_img,(coordinate[0][1][0],coordinate[0][1][1]),(coordinate[1][0][0],coordinate[1][1][1]),(255,0,0),2,cv2.LINE_AA)
    if (line1 > line2):
        # cv2.line(orig_img,coordinate[0][1],coordinate[1][1],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][1][0] + coordinate[1][1][0])/2
        circle_y = (coordinate[0][1][1] + coordinate[1][1][1])/2
    
    else:
        # cv2.line(orig_img,coordinate[0][0],coordinate[1][0],(255,0,0),2,cv2.LINE_AA)
        circle_x = (coordinate[0][0][0] + coordinate[1][0][0])/2
        circle_y = (coordinate[0][0][1] + coordinate[1][0][1])/2

    # cv2.circle(orig_img,(int(circle_x),int(circle_y)),2,(0,0,255),2)

    #計算物體長寬用
    object_Image_mat = mat_img.copy()
    object_Image = orig_img.copy()
    object_edges = cv2.Canny(object_Image_mat, 70, 210)
    contours_object, hierarchy = cv2.findContours(object_edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)  #cv2.RETR_EXTERNAL 定义只检测外围轮廓
    for cnt in contours_object:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(object_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        center_x1 = int(x + w/2) #外接矩形中心
        center_y1 = int(y+h/2)
        cv2.circle(object_Image,(int(center_x1),int(center_y1)),2,(255, 128, 0),2)
    
    circle_X = center_x1 
    circle_Y = center_y1 +20

    # 直线拟合
    rows, cols = object_Image.shape[:2]
    [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    lefty = int((-x * vy / vx) + y)
    righty = int(((cols - x) * vy / vx) + y)
    cv2.line(object_Image, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
    # cv2.imshow("test",object_Image)
    roct_result = angle(cols - 1, righty,0, lefty)
    print(roct_result)

    width = 20
    height = w+20
    grasp_left_x = int(circle_X - (width/2.0))
    grasp_left_y = int(circle_Y - (height/2.0))
    grasp_right_x = int(circle_X + (width/2.0))
    grasp_right_y = int(circle_Y + (height/2.0))


    # cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)

    box = [(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
    
    print(angle1[0],angle1[1])
    print("123456789")

    aa = rota_rect(box,roct_result,circle_X,circle_Y)

    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[2][0]),int(aa[2][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[1][0]),int(aa[1][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[1][0]),int(aa[1][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[2][0]),int(aa[2][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)

   
    circle_centerX = (aa[0][0] + aa[2][0])/2.0
    circle_centerY = (aa[0][1] + aa[2][1])/2.0
    cv2.circle(orig_img,(int(circle_X),int(circle_Y)),2,(255, 128, 0),2)

    return orig_img ,circle_centerX,circle_centerY,roct_result

# global circle_centerX, circle_centerY
#circle detection
def circle_transform(mat,orig,detections_number):
    mat_img = cv2.imread(mat)
    orig_img = cv2.imread(orig)

    gray = cv2.cvtColor(mat_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 210)
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # print(len(contours))

    areas = []
    for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))

    max_id = areas.index(max(areas))
    cnt = contours[max_id] #max contours

    #計算物體長寬用
    object_Image = orig_img.copy()
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(object_Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    center_x = int(x + w/2)
    center_y = int(y + h/2 - 10)
    cv2.circle(object_Image,(int(center_x),int(center_y)),2,(255, 128, 0),2)
    cv2.imshow("11111",object_Image)

    if w < 100: #小於一定大小直接抓全部
        width = 60
        height = 20
        grasp_left_x = int(center_x - (width/2.0))
        grasp_left_y = int(center_y - (height/2.0))
        grasp_right_x = int(center_x + (width/2.0))
        grasp_right_y = int(center_y + (height/2.0))
        # cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)
        # cv2.imshow("22222",orig_img)

        # 直线拟合
        rows, cols = object_Image.shape[:2]
        [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
        lefty = int((-x * vy / vx) + y)
        righty = int(((cols - x) * vy / vx) + y)
        cv2.line(object_Image, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
        cv2.imshow("area",object_Image)
        roct_result = angle(cols - 1, righty,0, lefty)
       
        box = [(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_left_y),
                (grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
        # box = [(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_left_y),
        #             (grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
        rota = rota_rect(box,roct_result,int(center_x),int(center_y))

        cv2.line(orig_img,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 0),2,cv2.LINE_AA)
        cv2.line(orig_img,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 0),2,cv2.LINE_AA)
        cv2.line(orig_img,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 0),2,cv2.LINE_AA)
        cv2.line(orig_img,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 0),2,cv2.LINE_AA)

        circle_centerX = (rota[0][0] + rota[2][0])/2.0
        circle_centerY = (rota[0][1] + rota[2][1])/2.0
        cv2.circle(orig_img,(int(circle_centerX),int(circle_centerY)),2,(0,0,255),2)
        # # cv2.imshow("orig_img",orig_img)
        result = roct_result

    else:
        if detections_num == 2:
            print("Hollow") #中空
             
            # 直线拟合
            rows, cols = object_Image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)

            cv2.line(object_Image, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
            cv2.imshow("area",object_Image)
            # img = cv2.line(object_Image, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
            
            width =  60
            height = 20
            # cv2.rectangle(object_Image,(int(center_x - width/2.0),int(center_y)),(int(center_x + width/2.0),int(center_y + height)),(255,255,0),2)
            roct_result = angle(cols - 1, righty,0, lefty)
            print(roct_result)
            box = [(center_x- width,center_y - height/2.0),(center_x - width,center_y + height/2.0),
                    (center_x ,center_y + height/2.0),(center_x ,center_y - height/2.0)]
            rota = rota_rect(box,roct_result,int(center_x),int(center_y))
            
            # if roct_result > 0:
            #     result = -(90 - roct_result)
            # else:
            #     result = (90 + roct_result)

            # print(result)
            result = roct_result

            cv2.line(orig_img,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0,0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 0),2,cv2.LINE_AA)

            circle_centerX = (rota[0][0] + rota[2][0])/2.0
            circle_centerY = (rota[0][1] + rota[2][1])/2.0
            cv2.circle(orig_img,(int(circle_centerX),int(circle_centerY)),2,(0, 0, 255),2)

        elif detections_num == 1:
            print("Solid") #實心圓
            width = 20
            height = 60
            grasp_left_x = int(center_x - (width/2.0))
            grasp_left_y = int(center_y - (height/2.0))
            grasp_right_x = int(center_x + (width/2.0))
            grasp_right_y = int(center_y + (height/2.0))

            cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)
            result = 0

            # 直线拟合
            rows, cols = object_Image.shape[:2]
            [vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
            lefty = int((-x * vy / vx) + y)
            righty = int(((cols - x) * vy / vx) + y)
            cv2.line(object_Image, (cols - 1, righty), (0, lefty), (0, 255, 255), 2)
            cv2.imshow("area",object_Image)

            roct_result = angle(cols - 1, righty,0, lefty)
            print(roct_result)
            box = [(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_left_y),
                        (grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
            rota = rota_rect(box,roct_result,int(center_x),int(center_y))

            cv2.line(orig_img,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 0),2,cv2.LINE_AA)
            cv2.line(orig_img,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 0),2,cv2.LINE_AA)

            circle_centerX = (rota[0][0] + rota[2][0])/2.0
            circle_centerY = (rota[0][1] + rota[2][1])/2.0

            cv2.circle(orig_img,(int(circle_centerX),int(circle_centerY)),2,(255, 0, 0),2)

            result = roct_result


    return orig_img,circle_centerX,circle_centerY,result

#grip and columnar detection
def calculate_center(left_x,left_y,right_x,right_y):
    width = abs(right_x -left_x) 
    height = abs(right_y - left_y)
    center_x = left_x + (width/2.0)
    center_y = left_y + (height/2.0)
    return center_x,center_y

def get_object_depth(depth, bounds):
    '''
    Calculates the median x, y, z position of top slice(area_div) of point cloud
    in camera frame.
    Arguments:
        depth: Point cloud data of whole frame.
        bounds: Bounding box for object in pixels.
            bounds[0]: x-center
            bounds[1]: y-center
            bounds[2]: width of bounding box.
            bounds[3]: height of bounding box.

    Return:
        x, y, z: Location of object in meters.
    '''
    area_div = 2

    x_vect = []
    y_vect = []
    z_vect = []

    for j in range(int(bounds[0] - area_div), int(bounds[0] + area_div)):
        for i in range(int(bounds[1] - area_div), int(bounds[1] + area_div)):
            z = depth[i, j, 2]
            if not np.isnan(z) and not np.isinf(z):
                x_vect.append(depth[i, j, 0])
                y_vect.append(depth[i, j, 1])
                z_vect.append(z)
    try:
        x_median = statistics.median(x_vect)
        y_median = statistics.median(y_vect)
        z_median = statistics.median(z_vect)
    except Exception:
        x_median = -1
        y_median = -1
        z_median = -1
        pass

    return x_median, y_median, z_median



# ------------------------------------------------------- Matting Arguments -------------------------------------------------
parser = argparse.ArgumentParser(description='Inference images')
parser.add_argument('--model-type', type=str, required=False, choices=['mattingbase', 'mattingrefine'],default='mattingrefine')
parser.add_argument('--model-backbone', type=str, required=False, choices=['resnet101', 'resnet50', 'mobilenetv2'],default='resnet101')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=False, default='./model_pth/pytorch_resnet101.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.9)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0,help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')
parser.add_argument('--output-dir', type=str, required=False, default='/home/user/shape_detection/circle/')
parser.add_argument('--output-types', type=str, required=False, nargs='+',choices=['com', 'pha', 'fgr', 'err', 'ref', 'new'])
parser.add_argument('-y', action='store_true')
args = parser.parse_args()
# --------------------------------------------------------------------Main--------------------------------------------------------#
#yolo
weights = "./yolo_data/yolov4-obj_best.weights"
config = "./yolo_data/yolov4-obj.cfg"
classes = "./yolo_data/obj.names"
data = "./yolo_data/obj.data"
thresh = 0.7
show_coordinates = True
zed_id = 0

#original_img left
save_path_columnar_left = "/home/user/shape_detection/columnar/orig/orig_left/"
save_path_long_left = "/home/user/shape_detection/long/orig/orig_left/"
save_path_circle_left = "/home/user/shape_detection/circle/orig/orig_left/"
save_path_blade_left = "/home/user/shape_detection/blade/orig/orig_left/"

#original_img right
save_path_columnar_right = "/home/user/shape_detection/columnar/orig/orig_right/"
save_path_long_right = "/home/user/shape_detection/long/orig/orig_right/"
save_path_circle_right = "/home/user/shape_detection/circle/orig/orig_right/"
save_path_blade_right = "/home/user/shape_detection/blade/orig/orig_right/"

#process
save_process_columnar = "/home/user/shape_detection/columnar/process/" 
save_process_circle = "/home/user/shape_detection/circle/process/"
save_process_long = "/home/user/shape_detection/long/process/"
save_process_blade = "/home/user/shape_detection/blade/process/"

#matting left
save_mat_columnar_left = "/home/user/shape_detection/columnar/mat/mat_left/"
save_mat_long_left = "/home/user/shape_detection/long/mat/mat_left/"
save_mat_circler_left = "/home/user/shape_detection/circle/mat/mat_left/"
save_mat_blader_left = "/home/user/shape_detection/blade/mat/mat_left/"

#matting right
save_mat_columnar_right = "/home/user/shape_detection/columnar/mat/mat_right/"
save_mat_long_right = "/home/user/shape_detection/long/mat/mat_right/"
save_mat_circle_right = "/home/user/shape_detection/circle/mat/mat_right/"
save_mat_blade_right = "/home/user/shape_detection/blade/mat/mat_right/"

dataset_root_path = r"/home/user/matting/imagedata"
img_floder = os.path.join(dataset_root_path,"img")
bgr_floder = os.path.join(dataset_root_path,"bgr")

#first_bgr
local_img_name_left=r'/home/user/shape_detection/bgr_left/1.jpg'
local_img_name_right=r'/home/user/shape_detection/bgr_right/1.jpg'
save_bgr_left = "/home/user/shape_detection/bgr_left/"
save_bgr_right = "/home/user/shape_detection/bgr_right/"

#second_bgr left
bgrcircle_path_left = "/home/user/shape_detection/circle/bgr/bgr_left/"
bgrblade_path_left = "/home/user/shape_detection/blade/bgr/bgr_left/"
bgrlong_path_left = "/home/user/shape_detection/long/bgr/bgr_left/"
bgrcolumnar_path_left = "/home/user/shape_detection/columnar/bgr/bgr_left/"

#second_bgr right
bgrcircle_path_right = "/home/user/shape_detection/circle/bgr/bgr_right/"
bgrblade_path_right = "/home/user/shape_detection/blade/bgr/bgr_right/"
bgrlong_path_right = "/home/user/shape_detection/long/bgr/bgr_right/"
bgrcolumnar_path_right = "/home/user/shape_detection/columnar/bgr/bgr_right/"

#find center
center_circle = "/home/user/shape_detection/circle/center/"
center_long = "/home/user/shape_detection/long/center/"
center_blade = "/home/user/shape_detection/blade/center/"
center_columnar = "/home/user/shape_detection/columnar/center/"

curr_time = datetime.datetime.now()

if __name__ == '__main__':
    #定義zed
    # zed = sl.Camera()
    # zed_pose = sl.Pose()

    # #init
    # input_type = sl.InputType() # Set configuration parameters
    # input_type.set_from_camera_id(zed_id)
    # init = sl.InitParameters(input_t=input_type)  # 初始化
    # init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
    # init.camera_fps = 15
    # init.coordinate_units = sl.UNIT.METER
    # # init.coordinate_units = sl.UNIT.MILLIMETER
    # # init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
    # init.depth_mode = sl.DEPTH_MODE.NEURAL         # 深度模式  (默认-PERFORMANCE)
    # # init.depth_minimum_distance = 300
    # # init.depth_maximum_distance = 5000 
    # init.depth_minimum_distance = 0.3
    # init.depth_maximum_distance = 5 
    # # init.coordinate_system=sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP

    # zed.set_camera_settings(sl.VIDEO_SETTINGS.SHARPNESS, 8)
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.SATURATION, 8)
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.BRIGHTNESS, 4)
    # zed.set_camera_settings(sl.VIDEO_SETTINGS.GAIN, 9)

    # #camera parameter
    # calibration_params = zed.get_camera_information().camera_configuration.calibration_parameters
    # focal_left_x = calibration_params.left_cam.fx
    # focal_left_y = calibration_params.left_cam.fy
    # focal_right_x = calibration_params.right_cam.fx
    # focal_right_y = calibration_params.right_cam.fy
    # center_point_x = calibration_params.left_cam.cx
    # center_point_y = calibration_params.left_cam.cy
    # translate = calibration_params.T
    # rotation = calibration_params.R

    # #open camera
    # if not zed.is_opened():
    #     log.info("Opening ZED Camera...")
    # status = zed.open(init)
    # if status != sl.ERROR_CODE.SUCCESS:
    #     log.error(repr(status))
    #     exit()

    # #set zed runtime value
    # runtime_parameters =sl.RuntimeParameters()
    # runtime_parameters.sensing_mode = sl.SENSING_MODE.FILL

    # #set image size
    # image_size = zed.get_camera_information().camera_resolution
    # print(image_size.width, image_size.height)
    # # exit(-1)
    # image_size.width = 960
    # image_size.height = 540

    # # # Create OpenGL viewer
    # # viewer = gl.GLViewer()
    # # viewer.init(1, sys.argv,calibration_params,image_size)

    # #turn zed to numpy(for opencv)
    # image_zed_left = sl.Mat(image_size.width, image_size.height)
    # image_zed_right = sl.Mat(image_size.width, image_size.height)
    # depth_image_zed = sl.Mat(image_size.width,image_size.height)
    # point_cloud = sl.Mat(image_size.width,image_size.height)
    # point_cloud1 = sl.Mat(image_size.width,image_size.height)
    # # point_cloud1 = sl.Mat(500,500,sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    i = -10
    a = 0
    b = 0
    c = 0
    d = 0

    #yolo modle
    network, class_names, class_colors = darknet.load_network(config,data,weights,batch_size=1)

    key = ''
    while key != 113 : 
        # # while viewer.is_available():
        # zed.grab() #開啟管道
        # zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        # zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)
        # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        # zed.retrieve_measure(point_cloud1, sl.MEASURE.XYZRGBA,sl.MEM.CPU, image_size)
        
        # color_image = image_zed_left.get_data()
        # # image_right = image_zed_left.get_data()
        # depth_image = depth_image_zed.get_data()
        # # print(np.min(depth_image), np.max(depth_image))

        # i += 1
        # if i ==1:
        #     cv2.imwrite(save_bgr_left + '1.jpg',color_image)
        
        # cv2.imshow("depth",depth_image)
        # # cv2.imshow("depth2",depth_image1)
    
        # #shape detection
        width_left = 300
        height_left = 300
        orig_circle_left = "/home/user/testimg/rgb/21.jpg"
        background_image1 = "/home/user/testimg/new_back/21.jpg"
        color_image  = cv2.imread("/home/user/testimg/rgb/21.jpg")
        background_image = cv2.imread("/home/user/testimg/new_back/21.jpg")

        frame_rgb_left = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        frame_resized_left = cv2.resize(frame_rgb_left, (width_left, height_left))

        darknet_image_left = darknet.make_image(width_left, height_left, 3)
        darknet.copy_image_from_bytes(darknet_image_left, frame_resized_left.tobytes()) 
        detections_left = darknet.detect_image(network, class_names, darknet_image_left, thresh=thresh)
        darknet.print_detections(detections_left, show_coordinates)
        print(detections_left)
        darknet.free_image(darknet_image_left)

        # draw bounding box
        image_left = darknet.draw_boxes(detections_left, frame_resized_left, class_colors)
        image_left = cv2.cvtColor(image_left, cv2.COLOR_BGR2RGB)

        cv2.imshow("image_left", image_left)

        if  len(detections_left) != 0:
            if int(float(detections_left[0][1])) >= 70:
                if detections_left[0][0] == "long" and key == 114:
                    t_prev = time.time()
                    a += 1

                    long_center_x_left = detections_left[0][2][0]
                    long_center_y_left = detections_left[0][2][1]
                    long_width_left = detections_left[0][2][2]
                    long_height_left = detections_left[0][2][3]

                    left_up_x_left = int(round((long_center_x_left - (long_width_left/2.0)),3))
                    left_up_y_left = int(round((long_center_y_left - (long_height_left/2.0)),3))
                    right_down_x_left = int(round((long_center_x_left + long_width_left/2.0),3))
                    right_down_y_left = int(round((long_center_y_left + (long_height_left/2.0)),3))
                    #matting
                    matimg_left = handle(orig_long_left,background_image1) #com,fgr,pha
                    mat_long_left = tensor_to_np(matimg_left[2])
                    matimg_mat_left = save_mat_long_left+"long_"+str(a)+'.jpg' 
                    cv2.imwrite(matimg_mat_left,mat_long_left)

                    process_left = post_processing(matimg_left[2])
                    process_long_left = save_process_long+"long_left_"+str(a)+'.jpg'
                    cv2.imwrite(process_long_left,process_left)

                    center_left = line_Segment(process_long_left,orig_long_left)


                    # t_end = time.time()
                    # cv2.imwrite(center_circle+"circle_left_no"+'.jpg',center_left[0])
                    # cv2.putText(center_left[0], "Object: " + str(detections_left[0][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    # # cv2.putText(center_left[0], "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    # cv2.putText(center_left[0], "Center: " + str(round(center_left[1],3)) +","+ str(round(center_left[2],3)), (10, 60), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    # cv2.putText(center_left[0], "Angle: " + str(round(center_left[3],3)) , (10, 90), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    # cv2.putText(center_left[0], "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    # cv2.putText(center_left[0], "time: " + str(detections_left[0][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(center_long+"long_left_"+str(a)+'.jpg',center_left[0])
                    cv2.imshow("finish_left",center_left[0])

                elif detections_left[0][0] == "circle" or detections_left[0][0] == "hollow":
                    if  key == 114:
                        b += 1
                        detections_num = int(len(detections_left))

                        matimg_left = handle(orig_circle_left,background_image1)
                        process_left = post_processing(matimg_left[2])
                        process_circle_left = save_process_circle + "circle_left_" + str(b) + '.jpg'
                        cv2.imwrite(process_circle_left,process_left)

                        center_left = circle_transform(process_circle_left,orig_circle_left,len(detections_left))

                        cv2.imwrite(center_circle+"circle_left_"+str(b)+'.jpg',center_left[0])
                        cv2.imshow("finish_left",center_left[0])

                elif detections_left[0][0] == "blade" or detections_left[0][0] == "grasp" or detections_left[0][0] == "round_grasp": 
                    if key == 114:
                        d += 1
                        if detections_left[1][0] == "round_grasp" and detections_left[0][0] == "blade":
                            round_grasp_center_x_left = round(detections_left[1][2][0])
                            round_grasp_center_y_left = round(detections_left[1][2][1])
                            round_grasp_width_left = round(detections_left[1][2][2])
                            round_grasp_height_left = round(detections_left[1][2][3])
                            blade_center_x_left = round(detections_left[0][2][0])
                            blade_center_y_left = round(detections_left[0][2][1])
                            blade_width_left = round(detections_left[0][2][2])
                            blade_height_left = round(detections_left[0][2][3])

                            left_up_x_left = round(round_grasp_center_x_left - (round_grasp_width_left/2.0))
                            left_up_y_left = round(round_grasp_center_y_left - (round_grasp_height_left/2.0))
                            right_down_x_left = round(round_grasp_center_x_left + (round_grasp_width_left/2.0))
                            right_down_y_left = round(round_grasp_center_y_left + (round_grasp_height_left/2.0))
                            blade_left_up_x_left = round(blade_center_x_left - (blade_width_left/2.0))
                            blade_left_up_y_left = round(blade_center_y_left - (blade_height_left/2.0))
                            blade_right_down_x_left = round(blade_center_x_left + blade_width_left/2.0)
                            blade_right_down_y_left = round(blade_center_y_left + (blade_height_left/2.0))

                            roct_result_left = angle(round_grasp_center_x_left, round_grasp_center_y_left,blade_center_x_left, blade_center_y_left)

                            #quadrant 1 , cut right down 
                            if(blade_center_x_left > round_grasp_center_x_left and blade_center_y_left < round_grasp_center_y_left) and (blade_center_x_right > round_grasp_center_x_right and blade_center_y_right < round_grasp_center_y_right): 
                                if(roct_result_left >=15 and roct_result_left <=70):
                                    width = 20
                                    height = 60
                                    real_grasp_center_x_left = (right_down_x_left + blade_left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (left_up_y_left + blade_right_down_y_left)/2.0

                                    result_left = angle(right_down_x_left ,left_up_y_left,left_up_x_left ,right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                                    (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                    real_rota_left = roct_result_left
                                elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left - blade_width_left/2.0
                                    real_grasp_center_y_left  = blade_center_y_left 

                                    result_left = angle(blade_left_up_x_left ,blade_left_up_y_left,blade_center_x_left ,blade_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left, roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(2, 202, 119),2)

                                    real_rota_left = roct_result_left
                                elif (roct_result_left > 70) or (roct_result_right >70):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left
                                    real_grasp_center_y_left = blade_center_y_left + blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_right_down_x_left,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,90 - result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(0,0,255),2)
                                    real_rota_left = 90 - roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)
                            
                            #quadrant 2 , cut right down 
                            elif(blade_center_x_left > round_grasp_center_x_left and blade_center_y_left > round_grasp_center_y_left) : #右下角
                                if(roct_result_left <=-15  and roct_result_left >=-70):
                                    width = 20 
                                    height = 40

                                    real_grasp_center_x_left = (right_down_x_left + blade_left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (right_down_y_left + blade_left_up_y_left)/2.0

                                    result_left = angle(left_up_x_left ,left_up_y_left,right_down_x_left ,right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                            (real_grasp_center_x_left - width/2,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_rota_left = roct_result_left
                                elif(roct_result_left >-15 and roct_result_left <=-0 ):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left - blade_width_left/2.0
                                    real_grasp_center_y_left  = blade_center_y_left

                                    result_left = angle(real_grasp_center_x_left ,real_grasp_center_y_left,blade_center_x_left ,blade_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),
                                            (real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                elif(roct_result_left < -70 and roct_result_left >=-90):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left - blade_height_left/2.0

                                    result_left = angle(blade_left_up_x_left, blade_left_up_y_left, real_grasp_center_x_left, real_grasp_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                    real_rota_left =roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)
                            
                            #quadrant 3 , cut left up
                            elif(blade_center_x_left < round_grasp_center_x_left and blade_center_y_left < round_grasp_center_y_left) : 
                                if(roct_result_left <=-15  and roct_result_left >=-70):
                                    width = 20
                                    height = 40

                                    real_grasp_center_x_left = (left_up_x_left + blade_right_down_x_left)/2.0 
                                    real_grasp_center_y_left = (left_up_y_left + blade_right_down_y_left)/2.0

                                    result_left = angle(blade_left_up_x_left ,blade_left_up_y_left ,blade_right_down_x_left  ,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                            (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                    real_rota_left = roct_result_left
                                    
                                elif(roct_result_left >-15 and roct_result_left <=-0 ):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left  + blade_width_left/2.0
                                    real_grasp_center_y_left  = blade_center_y_left

                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left= rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                    real_rota_left = roct_result_left
                                
                                elif(roct_result_left < -70 and roct_result_left >=-90):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left + blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_right_down_x_left,blade_right_down_y_left)
                                    box_left= [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                    real_rota_left = roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)
                            
                            #quadrant 4 , cut left down
                            elif(round_grasp_center_x_left > blade_center_x_left and blade_center_y_left > round_grasp_center_y_left) : 
                                if(roct_result_left >=15 and roct_result_left <=70) :
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = (blade_right_down_x_left + left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (blade_left_up_y_left + right_down_y_left)/2.0
                            
                                    result_left = angle(blade_left_up_x_left,blade_right_down_y_left,blade_right_down_x_left,blade_left_up_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                    real_rota_left = roct_result_left

                                elif(roct_result_left < 15) :
                                    width = 20
                                    height = 60
                                    
                                    real_grasp_center_x_left = blade_center_x_left + blade_width_left/2.0
                                    real_grasp_center_y_left  = blade_center_y_left 

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left ,blade_center_x_left  ,blade_center_y_left )
                                    box_left  = [(real_grasp_center_x_left  + width/2.0,real_grasp_center_y_left  - height/2.0),(real_grasp_center_x_left  - width/2.0,real_grasp_center_y_left  - height/2.0),
                                        (real_grasp_center_x_left  - width/2.0,real_grasp_center_y_left  + height/2.0),(real_grasp_center_x_left  + width/2.0,real_grasp_center_y_left  + height/2.0)]
                                    rota_left  = rota_rect(box_left ,roct_result_left ,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                    real_rota_left = roct_result_left

                                elif (roct_result_left > 70) :
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left - blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_left_up_x_left,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),
                                            (real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                    real_rota_left = roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                        elif detections_left[0][0] == "round_grasp" and detections_left[1][0] == "blade":
                            round_grasp_center_x_left = round(detections_left[0][2][0])
                            round_grasp_center_y_left = round(detections_left[0][2][1])
                            round_grasp_width_left = round(detections_left[0][2][2])
                            round_grasp_height_left = round(detections_left[0][2][3])
                            blade_center_x_left = round(detections_left[1][2][0])
                            blade_center_y_left = round(detections_left[1][2][1])
                            blade_width_left = round(detections_left[1][2][2])
                            blade_height_left = round(detections_left[1][2][3])

                            left_up_x_left = round(round_grasp_center_x_left - (round_grasp_width_left/2.0))
                            left_up_y_left = round(round_grasp_center_y_left - (round_grasp_height_left/2.0))
                            right_down_x_left = round(round_grasp_center_x_left + (round_grasp_width_left/2.0))
                            right_down_y_left = round(round_grasp_center_y_left + (round_grasp_height_left/2.0))
                            blade_left_up_x_left = round(blade_center_x_left - (blade_width_left/2.0))
                            blade_left_up_y_left = round(blade_center_y_left - (blade_height_left/2.0))
                            blade_right_down_x_left = round(blade_center_x_left + blade_width_left/2.0)
                            blade_right_down_y_left = round(blade_center_y_left + (blade_height_left/2.0))

                            roct_result_left = angle(round_grasp_center_x_left, round_grasp_center_y_left,blade_center_x_left, blade_center_y_left)

                            #quadrant 1 , cut right up
                            if(blade_center_x_left > round_grasp_center_x_left and blade_center_y_left < round_grasp_center_y_left) and (blade_center_x_right > round_grasp_center_x_right and blade_center_y_right < round_grasp_center_y_right): 
                                if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = (right_down_x_left + blade_left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (left_up_y_left + blade_right_down_y_left)/2.0

                                    result_left = angle(right_down_x_left ,left_up_y_left,left_up_x_left ,right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                                (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                
                                elif(roct_result_left < 15 ) or (roct_result_right <15):
                                    width = 20
                                    height = 60

                                    real_grasp_center_y_left  = blade_center_y_left 

                                    result_left = angle(blade_left_up_x_left ,blade_left_up_y_left,blade_center_x_left ,blade_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(2, 202, 119),2)

                                    real_rota_left = roct_result_left

                                elif (roct_result_left > 75) or (roct_result_right >75):
                                    width = 60
                                    height = 20

                                    real_grasp_center_x_left = blade_center_x_left
                                    real_grasp_center_y_left = blade_center_y_left + blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_right_down_x_left,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                    real_rota_left = 90 - roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                            #quadrant 2 , cut right down
                            elif(blade_center_x_left > round_grasp_center_x_left and blade_center_y_left > round_grasp_center_y_left) : #右下角
                                if(roct_result_left <=-15  and roct_result_left >=-70):
                                    width = 20 
                                    height = 60

                                    real_grasp_center_x_left = (right_down_x_left + blade_left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (right_down_y_left + blade_left_up_y_left)/2.0
                                    
                                    result_left = angle(left_up_x_left ,left_up_y_left,right_down_x_left ,right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                            (real_grasp_center_x_left - width/2,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                
                                elif(roct_result_left >-15 and roct_result_left <=-0 ) :
                                    width = 20
                                    height = 0
                            
                                    real_grasp_center_x_left = blade_center_x_left - blade_width_left/2.0
                                    real_grasp_center_y_left = blade_center_y_left

                                    result_left = angle(real_grasp_center_x_left ,real_grasp_center_y_left,blade_center_x_left ,blade_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),
                                            (real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)                                  
                                    real_rota_left = roct_result_left

                                elif(roct_result_left < -70 and roct_result_left >=-90) :
                                    width = 60
                                    height = 20

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left - blade_height_left/2.0

                                    result_left = angle(blade_left_up_x_left, blade_left_up_y_left, real_grasp_center_x_left, real_grasp_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    
                                    real_rota_left =roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                            #quadrant 3 , cut left up
                            elif(blade_center_x_left < round_grasp_center_x_left and blade_center_y_left < round_grasp_center_y_left) : 
                                if(roct_result_left <=-15  and roct_result_left >=-70):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = (left_up_x_left + blade_right_down_x_left)/2.0 
                                    real_grasp_center_y_left = (left_up_y_left + blade_right_down_y_left)/2.0

                                    result_left = angle(blade_left_up_x_left ,blade_left_up_y_left,blade_right_down_x_left ,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                            (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                    
                                elif(roct_result_left >-15 and roct_result_left <=-0 ):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left  + blade_width_left /2.0
                                    real_grasp_center_y_left  = blade_center_y_left 

                                    result_left = angle(real_grasp_center_x_left ,real_grasp_center_y_left,blade_center_x_left ,blade_center_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0,real_grasp_center_y_left + height/2.0)]
                                    rota_left= rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_rota_left = roct_result_left
                                    
                                elif(roct_result_left < -70 and roct_result_left >=-90):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left + blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_right_down_x_left,blade_right_down_y_left)
                                    box_left= [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)
                                
                            #quadrant 4 , cut left down
                            elif(round_grasp_center_x_left > blade_center_x_left and blade_center_y_left > round_grasp_center_y_left): 
                                if(roct_result_left >=15 and roct_result_left <=70) :
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = (blade_right_down_x_left + left_up_x_left)/2.0 
                                    real_grasp_center_y_left = (blade_left_up_y_left + right_down_y_left)/2.0

                                    result_left = angle(blade_left_up_x_left,blade_right_down_y_left,blade_right_down_x_left,blade_left_up_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),
                                        (real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left

                                elif(roct_result_left < 15) :
                                    width = 20
                                    height = 60
                                    
                                    real_grasp_center_x_left = blade_center_x_left + blade_width_left/2.0
                                    real_grasp_center_y_left = blade_center_y_left

                                    result_left = angle(real_grasp_center_x_left  ,real_grasp_center_y_left ,blade_center_x_left  ,blade_center_y_left )
                                    box_left  = [(real_grasp_center_x_left  + width/2.0,real_grasp_center_y_left  - height/2.0),(real_grasp_center_x_left  - width/2.0,real_grasp_center_y_left  - height/2.0),
                                        (real_grasp_center_x_left  - width/2.0,real_grasp_center_y_left  + height/2.0),(real_grasp_center_x_left  + width/2.0,real_grasp_center_y_left  + height/2.0)]
                                    rota_left  = rota_rect(box_left ,roct_result_left ,int(real_grasp_center_x_left),int(real_grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                    real_rota_left = roct_result_left

                                elif (roct_result_left > 70):
                                    width = 20
                                    height = 60

                                    real_grasp_center_x_left = blade_center_x_left 
                                    real_grasp_center_y_left = blade_center_y_left - blade_height_left/2.0

                                    result_left = angle(real_grasp_center_x_left,real_grasp_center_y_left,blade_left_up_x_left,blade_right_down_y_left)
                                    box_left = [(real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left - height/2.0),(real_grasp_center_x_left - width/2.0,real_grasp_center_y_left + height/2.0),
                                            (real_grasp_center_x_left + width/2.0 ,real_grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,roct_result_left,int(real_grasp_center_x_left),int(real_grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)
                                    real_rota_left = roct_result_left
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)
                
                        if detections_left[1][0] == "grasp" and detections_left[0][0] == "blade":
                            grasp_center_x_left = round(detections_left[1][2][0])
                            grasp_center_y_left = round(detections_left[1][2][1])
                            grasp_width_left = round(detections_left[1][2][2])
                            grasp_height_left = round(detections_left[1][2][3])

                            grasp_center_x_right = round(detections_right[1][2][0])
                            grasp_center_y_right = round(detections_right[1][2][1])
                            grasp_width_right = round(detections_right[1][2][2])
                            grasp_height_right = round(detections_right[1][2][3])

                            blade_center_x_left = round(detections_left[0][2][0])
                            blade_center_y_left = round(detections_left[0][2][1])
                            blade_width_left = round(detections_left[0][2][2])
                            blade_height_left = round(detections_left[0][2][3])

                            blade_center_x_right = round(detections_right[0][2][0])
                            blade_center_y_right = round(detections_right[0][2][1])
                            blade_width_right = round(detections_right[0][2][2])
                            blade_height_right = round(detections_right[0][2][3])

                            left_up_x_left = round(grasp_center_x_left - (grasp_width_left/2.0))
                            left_up_y_left = round(grasp_center_y_left - (grasp_height_left/2.0))
                            right_down_x_left = round(grasp_center_x_left + grasp_width_left/2.0)
                            right_down_y_left = round(grasp_center_y_left + (blade_height_left/2.0))

                            left_up_x_right = round(grasp_center_x_right - (grasp_width_right/2.0))
                            left_up_y_right = round(grasp_center_y_right - (grasp_height_right/2.0))
                            right_down_x_right = round(grasp_center_x_right + grasp_width_right/2.0)
                            right_down_y_right = round(grasp_center_y_right + (blade_height_right/2.0))

                            blade_left_up_x_left = round(blade_center_x_left - (blade_width_left/2.0))
                            blade_left_up_y_left = round(blade_center_y_left - (blade_height_left/2.0))
                            blade_right_down_x_left = round(blade_center_x_left + blade_width_left/2.0)
                            blade_right_down_y_left = round(blade_center_y_left + (blade_height_left/2.0))

                            blade_left_up_x_right = round(blade_center_x_right - (blade_width_right/2.0))
                            blade_left_up_y_right = round(blade_center_y_right - (blade_height_right/2.0))
                            blade_right_down_x_right = round(blade_center_x_right + blade_width_right/2.0)
                            blade_right_down_y_right = round(blade_center_y_right + (blade_height_right/2.0))

                            roct_result_left = angle(grasp_center_x_left, grasp_center_y_left,blade_center_x_left, blade_center_y_left)
                            print(roct_result_left)
                            print("55555555555555555")
                            roct_result_right = angle(grasp_center_x_right, grasp_center_y_right,blade_center_x_right, blade_center_y_right)
                            print(roct_result_right)
                            print("555555555555")

                            #quadrant 1,right up 
                            if( blade_center_x_left > grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image, "right up 0101" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "right up 0101 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                # if((blade_center_x - grasp_center_x) < 120 and(blade_center_x - grasp_center_x) > 25): 
                                    # cv2.putText(color_image, "right up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 20
                                    height = 50

                                    # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                    # cv2.circle(img_org,(int(grasp_center_x),int(grasp_center_y + height/2)),2,(255,0,0),2)
                                    result_left = angle(left_up_x_left ,right_down_y_left,right_down_x_left ,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                        (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right ,right_down_y_right,right_down_x_right ,left_up_y_right)
                                    box_right = [(grasp_center_x_right + width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                        (grasp_center_x_right - width/2.0,grasp_center_y_right+height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                # elif((blade_center_x - grasp_center_x) >= 120):
                                    # cv2.putText(color_image, "right up 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right up 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 20
                                    height = 50
                                    
                                    # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                    box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right+height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif (roct_result_left > 70) or (roct_result_right >70):
                                # elif((blade_center_x - grasp_center_x) < 25): 
                                    # cv2.putText(color_image, "right up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 50
                                    height = 20

                                    # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                    box_right= [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                        (grasp_center_x_right - width,grasp_center_y_right+height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))


                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                
                                point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                x_left = point3D_left[1][0]
                                y_left = point3D_left[1][1]
                                z_left = point3D_left[1][2]
                                color_left = point3D_left[1][3]

                                point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                x_right = point3D_right[1][0]
                                y_right = point3D_right[1][1]
                                z_right = point3D_right[1][2]
                                color_right = point3D_right[1][3]

                                # viewer.updateData(point_cloud1)
                                #depth
                                z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "angle: " + str(round(result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                                cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "angle: " + str(round(result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                cv2.imshow("finish_right",color_image1)

                            # quadrant 2 , right down
                            elif(blade_center_x_left > grasp_center_x_left and blade_center_y_left > grasp_center_y_left): #右下角
                                # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image, "right down 0101" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "right down 0101 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                # if((blade_center_x - grasp_center_x) <= 120 and(blade_center_x - grasp_center_x)>25):
                                    # cv2.putText(color_image, "right down 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right down 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 50
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left) + 90
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left - height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left + height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right,left_up_y_right,right_down_x_right,right_down_y_right) + 90
                                    box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right- height/2.0),
                                        (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                # elif((blade_center_x - grasp_center_x) >= 120): 
                                    # cv2.putText(color_image, "right down 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right down 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 20
                                    height = 50
                                    
                                    # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+ grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                        (grasp_center_x_left - width/2,grasp_center_y_left+height),(grasp_center_x_left + width/2,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                    box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2,grasp_center_y_right),
                                        (grasp_center_x_right - width/2,grasp_center_y_right+height),(grasp_center_x_right + width/2,grasp_center_y_right + height)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                # elif((blade_center_x - grasp_center_x) < 25):
                                    # cv2.putText(color_image, "right down 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right down 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 60
                                    height = 20

                                    # cv2.rectangle(color_image,(int(grasp_center_x_left-width), int(grasp_center_y_left - height/2.0)), (int(grasp_center_x_left), int(grasp_center_y_left + height/2.0)),(0, 127, 255), 2)
                                    point1_left = ((grasp_center_x_left-width) + (grasp_center_x_left)) / 2.0
                                    point2_left = ((grasp_center_y_left - height/2.0) + (grasp_center_y_left + height/2.0)) /2.0
                                    # cv2.circle(color_image,(int(c),int(z)),2,(255,0,0),2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))

                                    # cv2.rectangle(color_image1,(int(grasp_center_x_right-width), int(grasp_center_y_right - height/2.0)), (int(grasp_center_x_right), int(grasp_center_y_right + height/2.0)),(0, 127, 255), 2)
                                    point1_right = ((grasp_center_x_right-width) + (grasp_center_x_right)) / 2.0
                                    point2_right = ((grasp_center_y_right - height/2.0) + (grasp_center_y_right + height/2.0)) /2.0
                                    result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                    box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                        (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                    rota_right = rota_rect(box_right,90+roct_result_right,int(point1_right),int(point2_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                x_left = point3D_left[1][0]
                                y_left = point3D_left[1][1]
                                z_left = point3D_left[1][2]
                                color_left = point3D_left[1][3]

                                point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                x_right = point3D_right[1][0]
                                y_right = point3D_right[1][1]
                                z_right = point3D_right[1][2]
                                color_right = point3D_right[1][3]

                                # viewer.updateData(point_cloud1)
                                #depth
                                z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                                cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                cv2.imshow("finish_right",color_image1)

                            # quadrant 3 , left up 
                            elif(blade_center_x_left < grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image, "left up 0101" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                # cv2.putText(color_image1, "left up 0101 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                # if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x)>25):
                                    # cv2.putText(color_image, "left up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 20
                                    height = 70

                                    # cv2.rectangle(img_org,(int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x+width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                    result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                        (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right = angle(left_up_x_right,left_up_y_right,right_down_x_right,right_down_y_right)
                                    box_right = [(grasp_center_x_right + width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                        (grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0 ,grasp_center_y_right + height)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                # elif((grasp_center_x - blade_center_x) >= 120): 
                                    # cv2.putText(color_image, "left up 2222 1" , (10,190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left up 2222 1" , (10,190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 20
                                    height = 70
                                    
                                    # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,right_down_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left+height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    result_right= angle(left_up_x_right,left_up_y_right,left_up_x_right + grasp_width_right,right_down_y_right)
                                    box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2,grasp_center_y_right),
                                        (grasp_center_x_right - width/2,grasp_center_y_right + height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                    rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                # elif((grasp_center_x - blade_center_x) < 25): 
                                    # cv2.putText(color_image, "left up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    width = 70
                                    height = 20

                                    # cv2.rectangle(color_image,(int(grasp_center_x_left-width), int(grasp_center_y_left - height/2.0)), (int(grasp_center_x_left), int(grasp_center_y_left + height/2.0)),(0, 127, 255), 2)
                                    point1_left = ((grasp_center_x_left-width) + (grasp_center_x_left)) / 2.0
                                    point2_left = ((grasp_center_y_left - height/2.0) + (grasp_center_y_left + height/2.0)) /2.0
                                    # cv2.circle(color_image,(int(c),int(z)),2,(255,0,0),2)
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))

                                    # cv2.rectangle(color_image1,(int(grasp_center_x_right-width), int(grasp_center_y_right - height/2.0)), (int(grasp_center_x_right), int(grasp_center_y_right + height/2.0)),(0, 127, 255), 2)
                                    point1_right = ((grasp_center_x_right-width) + (grasp_center_x_right)) / 2.0
                                    point2_right = ((grasp_center_y_right - height/2.0) + (grasp_center_y_right + height/2.0)) /2.0
                                    result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                    box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                        (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                    rota_right = rota_rect(box_right,90+roct_result_right,int(point1_right),int(point2_right))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                    real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                    real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                    cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                
                                point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                x_left = point3D_left[1][0]
                                y_left = point3D_left[1][1]
                                z_left = point3D_left[1][2]
                                color_left = point3D_left[1][3]

                                point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                x_right = point3D_right[1][0]
                                y_right = point3D_right[1][1]
                                z_right = point3D_right[1][2]
                                color_right = point3D_right[1][3]

                                # viewer.updateData(point_cloud1)
                                #depth
                                z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                                cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                cv2.imshow("finish_right",color_image1)
                            
                            # quadrant 4 , left down
                            elif(grasp_center_x_left > blade_center_x_left and blade_center_y_left > grasp_center_y_left): #左下角
                                    # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left down 0101" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left down 0101 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)                        
                                    if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    # if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x) >= 25):
                                        # cv2.putText(color_image, "left down 1111 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 1111 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,right_down_y_left,right_down_x_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0 ,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                            (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,right_down_y_right,right_down_x_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right + width/2.0 ,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                            (grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0 ,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                        
                                    elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    # elif((grasp_center_x - blade_center_x) >= 120):
                                        # cv2.putText(color_image, "left down 2222 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 2222 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left + grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right+ width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif (roct_result_left > 70) or (roct_result_right >70):
                                    # elif((grasp_center_x - blade_center_x) < 25):
                                        # cv2.putText(color_image, "left down 3333 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 3333 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 60
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right+height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))
                                        
                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                    
                                    point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    x_left = point3D_left[1][0]
                                    y_left = point3D_left[1][1]
                                    z_left = point3D_left[1][2]
                                    color_left = point3D_left[1][3]

                                    point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                    x_right = point3D_right[1][0]
                                    y_right = point3D_right[1][1]
                                    z_right = point3D_right[1][2]
                                    color_right = point3D_right[1][3]

                                    # viewer.updateData(point_cloud1)
                                    #depth
                                    z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                    cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                    cv2.imshow("finish_left",color_image)

                                    cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                    cv2.imshow("finish_right",color_image1)

                            elif detections_right[0][0] == "grasp" and detections_right[1][0] == "blade": 
                                grasp_center_x_left = round(detections_left[1][2][0])
                                grasp_center_y_left = round(detections_left[1][2][1])
                                grasp_width_left = round(detections_left[1][2][2])
                                grasp_height_left = round(detections_left[1][2][3])

                                grasp_center_x_right = round(detections_right[0][2][0])
                                grasp_center_y_right = round(detections_right[0][2][1])
                                grasp_width_right = round(detections_right[0][2][2])
                                grasp_height_right = round(detections_right[0][2][3])

                                blade_center_x_left = round(detections_left[0][2][0])
                                blade_center_y_left = round(detections_left[0][2][1])
                                blade_width_left = round(detections_left[0][2][2])
                                blade_height_left = round(detections_left[0][2][3])

                                blade_center_x_right = round(detections_right[1][2][0])
                                blade_center_y_right = round(detections_right[1][2][1])
                                blade_width_right = round(detections_right[1][2][2])
                                blade_height_right = round(detections_right[1][2][3])

                                left_up_x_left = round(grasp_center_x_left - (grasp_width_left/2.0))
                                left_up_y_left = round(grasp_center_y_left - (grasp_height_left/2.0))
                                right_down_x_left = round(grasp_center_x_left + grasp_width_left/2.0)
                                right_down_y_left = round(grasp_center_y_left + (blade_height_left/2.0))

                                left_up_x_right = round(grasp_center_x_right - (grasp_width_right/2.0))
                                left_up_y_right = round(grasp_center_y_right - (grasp_height_right/2.0))
                                right_down_x_right = round(grasp_center_x_right + grasp_width_right/2.0)
                                right_down_y_right = round(grasp_center_y_right + (blade_height_right/2.0))

                                blade_left_up_x_left = round(blade_center_x_left - (blade_width_left/2.0))
                                blade_left_up_y_left = round(blade_center_y_left - (blade_height_left/2.0))
                                blade_right_down_x_left = round(blade_center_x_left + blade_width_left/2.0)
                                blade_right_down_y_left = round(blade_center_y_left + (blade_height_left/2.0))

                                blade_left_up_x_right = round(blade_center_x_right - (blade_width_right/2.0))
                                blade_left_up_y_right = round(blade_center_y_right - (blade_height_right/2.0))
                                blade_right_down_x_right = round(blade_center_x_right + blade_width_right/2.0)
                                blade_right_down_y_right = round(blade_center_y_right + (blade_height_right/2.0))

                                roct_result_left = angle(grasp_center_x_left, grasp_center_y_left,blade_center_x_left, blade_center_y_left)
                                print(roct_result_left)
                                print("666666666666666666666666666")
                                roct_result_right = angle(grasp_center_x_right, grasp_center_y_right,blade_center_x_right, blade_center_y_right)
                                print(roct_result_right)
                                print("666666666666666666666666666")

                                #quadrant 1,right up 
                                if( blade_center_x_left > grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                    # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right up 0110" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right up 0110 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    # if((blade_center_x - grasp_center_x) < 120 and(blade_center_x - grasp_center_x) > 25): 
                                        # cv2.putText(color_image, "right up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        # cv2.circle(img_org,(int(grasp_center_x),int(grasp_center_y + height/2)),2,(255,0,0),2)
                                        result_left = angle(left_up_x_left ,right_down_y_left,right_down_x_left ,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                            (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right ,right_down_y_right,right_down_x_right ,left_up_y_right)
                                        box_right = [(grasp_center_x_right + width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                            (grasp_center_x_right - width/2.0,grasp_center_y_right+height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    # elif((blade_center_x - grasp_center_x) >= 120):
                                        # cv2.putText(color_image, "right up 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right up 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right+height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif (roct_result_left > 70) or (roct_result_right >70):
                                    # elif((blade_center_x - grasp_center_x) < 25): 
                                        # cv2.putText(color_image, "right up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right= [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right+height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))


                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                    
                                    point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    x_left = point3D_left[1][0]
                                    y_left = point3D_left[1][1]
                                    z_left = point3D_left[1][2]
                                    color_left = point3D_left[1][3]

                                    point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                    x_right = point3D_right[1][0]
                                    y_right = point3D_right[1][1]
                                    z_right = point3D_right[1][2]
                                    color_right = point3D_right[1][3]

                                    # viewer.updateData(point_cloud1)
                                    #depth
                                    z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                    cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: " + str(round(result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "2left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                    cv2.imshow("finish_left",color_image)

                                    cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "angle: " + str(round(result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "2right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                    cv2.imshow("finish_right",color_image1)
                                
                                # quadrant 2 , right down
                                elif(blade_center_x_left > grasp_center_x_left and blade_center_y_left > grasp_center_y_left): #右下角
                                    # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right down 0110" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "right down 0110 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                    # if((blade_center_x - grasp_center_x) <= 120 and(blade_center_x - grasp_center_x)>25):
                                        # cv2.putText(color_image, "right down 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right down 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left) + 90
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left - height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left + height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right,left_up_y_right,right_down_x_right,right_down_y_right) + 90
                                        box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right- height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                    # elif((blade_center_x - grasp_center_x) >= 120): 
                                        # cv2.putText(color_image, "right down 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right down 2222 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+ grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                            (grasp_center_x_left - width/2,grasp_center_y_left+height),(grasp_center_x_left + width/2,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2,grasp_center_y_right),
                                            (grasp_center_x_right - width/2,grasp_center_y_right+height),(grasp_center_x_right + width/2,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                    # elif((blade_center_x - grasp_center_x) < 25):
                                        # cv2.putText(color_image, "right down 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "right down 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 60
                                        height = 20

                                        # cv2.rectangle(color_image,(int(grasp_center_x_left-width), int(grasp_center_y_left - height/2.0)), (int(grasp_center_x_left), int(grasp_center_y_left + height/2.0)),(0, 127, 255), 2)
                                        point1_left = ((grasp_center_x_left-width) + (grasp_center_x_left)) / 2.0
                                        point2_left = ((grasp_center_y_left - height/2.0) + (grasp_center_y_left + height/2.0)) /2.0
                                        # cv2.circle(color_image,(int(c),int(z)),2,(255,0,0),2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))

                                        # cv2.rectangle(color_image1,(int(grasp_center_x_right-width), int(grasp_center_y_right - height/2.0)), (int(grasp_center_x_right), int(grasp_center_y_right + height/2.0)),(0, 127, 255), 2)
                                        point1_right = ((grasp_center_x_right-width) + (grasp_center_x_right)) / 2.0
                                        point2_right = ((grasp_center_y_right - height/2.0) + (grasp_center_y_right + height/2.0)) /2.0
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,90+roct_result_right,int(point1_right),int(point2_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    x_left = point3D_left[1][0]
                                    y_left = point3D_left[1][1]
                                    z_left = point3D_left[1][2]
                                    color_left = point3D_left[1][3]

                                    point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                    x_right = point3D_right[1][0]
                                    y_right = point3D_right[1][1]
                                    z_right = point3D_right[1][2]
                                    color_right = point3D_right[1][3]

                                    # viewer.updateData(point_cloud1)
                                    #depth
                                    z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                    cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                    cv2.imshow("finish_left",color_image)

                                    cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                    cv2.imshow("finish_right",color_image1)
                                
                                # quadrant 3 , left up 
                                elif(blade_center_x_left < grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                    # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left up 0110" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left up 0110 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                    # if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x)>25):
                                        # cv2.putText(color_image, "left up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left up 1111 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 60

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x+width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                            (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right = angle(left_up_x_right,left_up_y_right,right_down_x_right,right_down_y_right)
                                        box_right = [(grasp_center_x_right + width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                            (grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0 ,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                    # elif((grasp_center_x - blade_center_x) >= 120): 
                                        cv2.putText(color_image, "left up 2222 1" , (10,190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image1, "left up 2222 1" , (10,190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 60
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,right_down_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left+height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        result_right= angle(left_up_x_right,left_up_y_right,left_up_x_right + grasp_width_right,right_down_y_right)
                                        box_right = [(grasp_center_x_right + width/2,grasp_center_y_right),(grasp_center_x_right - width/2,grasp_center_y_right),
                                            (grasp_center_x_right - width/2,grasp_center_y_right + height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                    # elif((grasp_center_x - blade_center_x) < 25): 
                                        # cv2.putText(color_image, "left up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left up 3333 1" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 70
                                        height = 20

                                        # cv2.rectangle(color_image,(int(grasp_center_x_left-width), int(grasp_center_y_left - height/2.0)), (int(grasp_center_x_left), int(grasp_center_y_left + height/2.0)),(0, 127, 255), 2)
                                        point1_left = ((grasp_center_x_left-width) + (grasp_center_x_left)) / 2.0
                                        point2_left = ((grasp_center_y_left - height/2.0) + (grasp_center_y_left + height/2.0)) /2.0
                                        # cv2.circle(color_image,(int(c),int(z)),2,(255,0,0),2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))

                                        # cv2.rectangle(color_image1,(int(grasp_center_x_right-width), int(grasp_center_y_right - height/2.0)), (int(grasp_center_x_right), int(grasp_center_y_right + height/2.0)),(0, 127, 255), 2)
                                        point1_right = ((grasp_center_x_right-width) + (grasp_center_x_right)) / 2.0
                                        point2_right = ((grasp_center_y_right - height/2.0) + (grasp_center_y_right + height/2.0)) /2.0
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right + height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,90+roct_result_right,int(point1_right),int(point2_right))
                                        
                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                    
                                    point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    x_left = point3D_left[1][0]
                                    y_left = point3D_left[1][1]
                                    z_left = point3D_left[1][2]
                                    color_left = point3D_left[1][3]

                                    point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                    x_right = point3D_right[1][0]
                                    y_right = point3D_right[1][1]
                                    z_right = point3D_right[1][2]
                                    color_right = point3D_right[1][3]

                                    # viewer.updateData(point_cloud1)
                                    #depth
                                    z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                    cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                    cv2.imshow("finish_left",color_image)

                                    cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                    cv2.imshow("finish_right",color_image1)

                                # quadrant 4 , left down
                                elif(grasp_center_x_left > blade_center_x_left and blade_center_y_left > grasp_center_y_left): #左下角
                                    # cv2.putText(color_image, "check: " + str(round(roct_result_left,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "check: " + str(round(roct_result_right,3)) , (10, 220), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left down 0110" , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image1, "left down 0110 " , (10, 250), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)                        
                                    if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    # if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x) >= 25):
                                        # cv2.putText(color_image, "left down 1111 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 1111 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,right_down_y_left,right_down_x_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0 ,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                            (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,right_down_y_right,right_down_x_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right + width/2.0 ,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),
                                            (grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0 ,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                        
                                    elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    # elif((grasp_center_x - blade_center_x) >= 120):
                                        # cv2.putText(color_image, "left down 2222 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 2222 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left + grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right+ width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right),(grasp_center_x_right - width/2.0,grasp_center_y_right + height),(grasp_center_x_right + width/2.0,grasp_center_y_right + height)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))

                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)

                                    elif (roct_result_left > 70) or (roct_result_right >70):
                                    # elif((grasp_center_x - blade_center_x) < 25):
                                        # cv2.putText(color_image, "left down 3333 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        # cv2.putText(color_image1, "left down 3333 2" , (10, 190), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 60
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                        box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                            (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                        rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result_right = angle(left_up_x_right,left_up_y_right,left_up_x_right+grasp_width_right,left_up_y_right)
                                        box_right = [(grasp_center_x_right ,grasp_center_y_right - height/2.0),(grasp_center_x_right - width,grasp_center_y_right-height/2.0),
                                            (grasp_center_x_right - width,grasp_center_y_right+height/2.0),(grasp_center_x_right ,grasp_center_y_right + height/2.0)]
                                        rota_right = rota_rect(box_right,result_right,int(grasp_center_x_right),int(grasp_center_y_right))
                                        
                                        cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        cv2.line(color_image1,(int(rota_right[0][0]),int(rota_right[0][1])),(int(rota_right[1][0]),int(rota_right[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[1][0]),int(rota_right[1][1])),(int(rota_right[2][0]),int(rota_right[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[2][0]),int(rota_right[2][1])),(int(rota_right[3][0]),int(rota_right[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image1,(int(rota_right[3][0]),int(rota_right[3][1])),(int(rota_right[0][0]),int(rota_right[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                        real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                        real_grasp_center_x_right = (rota_right[0][0] + rota_right[2][0])/2.0 
                                        real_grasp_center_y_right = (rota_right[0][1] + rota_right[2][1])/2.0
                                        cv2.circle(color_image1,(int(real_grasp_center_x_right),int(real_grasp_center_y_right)),2,(255,0,0),2)
                                    
                                    point3D_left = point_cloud.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    x_left = point3D_left[1][0]
                                    y_left = point3D_left[1][1]
                                    z_left = point3D_left[1][2]
                                    color_left = point3D_left[1][3]

                                    point3D_right = point_cloud1.get_value(real_grasp_center_x_right,real_grasp_center_y_right)
                                    x_right = point3D_right[1][0]
                                    y_right = point3D_right[1][1]
                                    z_right = point3D_right[1][2]
                                    color_right = point3D_right[1][3]

                                    # viewer.updateData(point_cloud1)
                                    #depth
                                    z_value_left = depth_image_zed.get_value(real_grasp_center_x_left,real_grasp_center_y_left)
                                    z_value_right = depth_image_zed.get_value(real_grasp_center_x_right,real_grasp_center_y_right)

                                    cv2.putText(color_image, "Object: " + str(detections_left[0][0]) +"," +str(detections_left[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value_left[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x_left,3)) +","+ str(round(real_grasp_center_y_left,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: " + str(round(roct_result_left,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "point3D_xyz: " + str(round(x_left,5))+", " + str(round(y_left,5))+", "  + str(round(z_left,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                    cv2.imshow("finish_left",color_image)

                                    cv2.putText(color_image1, "Object: " + str(detections_right[0][0]) +"," +str(detections_right[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "Depth: " + str(round(z_value_right[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "center: " + "("+ str(round(real_grasp_center_x_right,3)) +","+ str(round(real_grasp_center_y_right,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "angle: " + str(round(roct_result_right,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image1, "point3D_xyz: " + str(round(x_right,5))+", " + str(round(y_right,5))+", "  + str(round(z_right,5)) , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_blade + "right_blade_grasp_"+str(d)+'.jpg',color_image1) 
                                    cv2.imshow("finish_right",color_image1)


                        elif detections_left[0][0] == "grasp" and detections_left[1][0] == "blade":
                            grasp_center_x_left = round(detections_left[0][2][0])
                            grasp_center_y_left = round(detections_left[0][2][1])
                            grasp_width_left = round(detections_left[0][2][2])
                            grasp_height_left = round(detections_left[0][2][3])

                            blade_center_x_left = round(detections_left[1][2][0])
                            blade_center_y_left = round(detections_left[1][2][1])
                            blade_width_left = round(detections_left[1][2][2])
                            blade_height_left = round(detections_left[1][2][3])

                            left_up_x_left = round(grasp_center_x_left - (grasp_width_left/2.0))
                            left_up_y_left = round(grasp_center_y_left - (grasp_height_left/2.0))
                            right_down_x_left = round(grasp_center_x_left + grasp_width_left/2.0)
                            right_down_y_left = round(grasp_center_y_left + (blade_height_left/2.0))

                            blade_left_up_x_left = round(blade_center_x_left - (blade_width_left/2.0))
                            blade_left_up_y_left = round(blade_center_y_left - (blade_height_left/2.0))
                            blade_right_down_x_left = round(blade_center_x_left + blade_width_left/2.0)
                            blade_right_down_y_left = round(blade_center_y_left + (blade_height_left/2.0))

                            roct_result_left = angle(grasp_center_x_left, grasp_center_y_left,blade_center_x_left, blade_center_y_left)
                            print(roct_result_left)
                            print("77777777777777777777777777777777")

                            #quadrant 1,right up 
                            if( blade_center_x_left > grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    width = 20
                                    height = 60

                                    result_left = angle(left_up_x_left ,right_down_y_left,right_down_x_left ,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                        (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    width = 20
                                    height = 60
                        
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif (roct_result_left > 70) or (roct_result_right >70):
                            
                                    width = 60
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                        
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                cv2.imwrite(center_blade + "2left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                            # quadrant 2 , right down
                            elif(blade_center_x_left > grasp_center_x_left and blade_center_y_left > grasp_center_y_left): #右下角
                                if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                    width = 60
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left) + 90
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left - height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left + height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                    width = 20
                                    height = 60
                                    
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+ grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2,grasp_center_y_left),(grasp_center_x_left - width/2,grasp_center_y_left),
                                        (grasp_center_x_left - width/2,grasp_center_y_left+height),(grasp_center_x_left + width/2,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                    width = 60
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)


                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                            # quadrant 3 , left up 
                            elif(blade_center_x_left < grasp_center_x_left and blade_center_y_left < grasp_center_y_left): 
                                if(roct_result_left <=-15  and roct_result_left >=-70) or (roct_result_right <=-15 and roct_result_right >= -70 ):
                                    width = 20
                                    height = 60

                                    result_left = angle(left_up_x_left,left_up_y_left,right_down_x_left,right_down_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                        (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif(roct_result_left >-15 and roct_result_left <=-0 ) or (roct_result_right >-15 and roct_result_right <=-0):
                                    width = 20
                                    height = 60
                                    
                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,right_down_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left+height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0,0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(255,0,0),2)

                                elif(roct_result_left < -70 and roct_result_left >=-90) or (roct_result_right < -70 and roct_result_right >= -90):
                                    width = 60
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,90+roct_result_left,int(point1_left),int(point2_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)


                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)

                            # quadrant 4 , left down
                            elif(grasp_center_x_left > blade_center_x_left and blade_center_y_left > grasp_center_y_left): #左下角
                                if(roct_result_left >=15 and roct_result_left <=70) or (roct_result_right >=15 and roct_result_right <=70):
                                    width = 20
                                    height = 60

                                    result_left = angle(left_up_x_left,right_down_y_left,right_down_x_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0 ,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),
                                        (grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0 ,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)
     
                                elif(roct_result_left < 15 ) or (roct_result_right <15 ):
                                    width = 20
                                    height = 60

                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left + grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left + width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left),(grasp_center_x_left - width/2.0,grasp_center_y_left + height),(grasp_center_x_left + width/2.0,grasp_center_y_left + height)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))

                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0,0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                elif (roct_result_left > 70) or (roct_result_right >70):
                                    width = 60
                                    height = 20

                                    result_left = angle(left_up_x_left,left_up_y_left,left_up_x_left+grasp_width_left,left_up_y_left)
                                    box_left = [(grasp_center_x_left ,grasp_center_y_left - height/2.0),(grasp_center_x_left - width,grasp_center_y_left-height/2.0),
                                        (grasp_center_x_left - width,grasp_center_y_left+height/2.0),(grasp_center_x_left ,grasp_center_y_left + height/2.0)]
                                    rota_left = rota_rect(box_left,result_left,int(grasp_center_x_left),int(grasp_center_y_left))
                                    
                                    cv2.line(color_image,(int(rota_left[0][0]),int(rota_left[0][1])),(int(rota_left[1][0]),int(rota_left[1][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[1][0]),int(rota_left[1][1])),(int(rota_left[2][0]),int(rota_left[2][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[2][0]),int(rota_left[2][1])),(int(rota_left[3][0]),int(rota_left[3][1])),(255, 0, 0),2,cv2.LINE_AA)
                                    cv2.line(color_image,(int(rota_left[3][0]),int(rota_left[3][1])),(int(rota_left[0][0]),int(rota_left[0][1])),(255, 0, 0),2,cv2.LINE_AA)

                                    real_grasp_center_x_left = (rota_left[0][0] + rota_left[2][0])/2.0 
                                    real_grasp_center_y_left = (rota_left[0][1] + rota_left[2][1])/2.0
                                    cv2.circle(color_image,(int(real_grasp_center_x_left),int(real_grasp_center_y_left)),2,(0,0,255),2)

                                cv2.imwrite(center_blade + "left_blade_grasp_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish_left",color_image)




        # cv2.waitKey()
        key = cv2.waitKey(5)

        # if len(detections_left) !=0

        
        if key == 27:
            break
    # zed.close()
    # viewer.exit()
    cv2.destroyAllWindows()