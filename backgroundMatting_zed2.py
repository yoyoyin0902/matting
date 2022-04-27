import os
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

from ctypes import *
from random import randint
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
    # set imgfile
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

    ret,bin_img = cv2.threshold(imgBlur,110,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

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
    if x1 == x2:
        return 90
    if y1 == y2:
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
    # cv2.imshow("mat",mat_img)

    # gray = cv2.cvtColor(mat,cv2.COLOR_BGR2GRAY)
    lsd = cv2.createLineSegmentDetector(0)

    dlines = lsd.detect(mat_img)
    
    ver_lines = []
    coordinate = []
    angle1 = []

    for dline in dlines[0]:
        # print(dline[i])
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        ver_lines.append(distance)

    maxIndex = max2(ver_lines)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
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

    width = 40
    height = 100
    grasp_left_x = int(circle_x - (width/2.0))
    grasp_left_y = int(circle_y - (height/2.0))
    grasp_right_x = int(circle_x + (width/2.0))
    grasp_right_y = int(circle_y + (height/2.0))
    
    # cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)

    box = [(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
    real_angle = (angle1[0] + angle1[1])/2.0
    
    aa = rota_rect(box,real_angle,circle_x,circle_y)

    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[2][0]),int(aa[2][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[1][0]),int(aa[1][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[1][0]),int(aa[1][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[2][0]),int(aa[2][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)

    return orig_img ,circle_x,circle_y,real_angle

def line_Segment_cup(mat,orig):
    mat_img = cv2.imread(str(mat),0)
    orig_img = cv2.imread(orig)

    lsd = cv2.createLineSegmentDetector(0)

    dlines = lsd.detect(mat_img)
    
    ver_lines = []
    coordinate = []
    angle1 = []

    for dline in dlines[0]:
        # print(dline[i])
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
        distance = math.sqrt((x0-x1)**2+(y0-y1)**2)
        ver_lines.append(distance)

    maxIndex = max2(ver_lines)

    for dline in dlines[0]:
        x0 = int(round(dline[0][0]))
        y0 = int(round(dline[0][1]))
        x1 = int(round(dline[0][2]))
        y1 = int(round(dline[0][3]))
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

    width = 130
    height = 30
    grasp_left_x = int(circle_x - (width/2.0))
    grasp_left_y = int(circle_y - (height/2.0))
    grasp_right_x = int(circle_x + (width/2.0))
    grasp_right_y = int(circle_y + (height/2.0))

    box = [(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_left_y),(grasp_left_x,grasp_right_y),(grasp_right_x,grasp_right_y)]
    
    real_angle =angle1[0] + 90

    aa = rota_rect(box,real_angle,circle_x,circle_y)
    # print(aa)
    # cv2.rectangle(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2)

    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[2][0]),int(aa[2][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[1][0]),int(aa[1][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[0][0]),int(aa[0][1])),(int(aa[1][0]),int(aa[1][1])),(255, 0, 255),2,cv2.LINE_AA)
    cv2.line(orig_img,(int(aa[2][0]),int(aa[2][1])),(int(aa[3][0]),int(aa[3][1])),(255, 0, 255),2,cv2.LINE_AA)

    return orig_img ,circle_x,circle_y,real_angle

#circle detection
def circle_transform(mat,orig):
    mat_img = cv2.imread(mat)
    orig_img = cv2.imread(orig)

    gray = cv2.cvtColor(mat_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 210)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    areas = []
    for c in range(len(contours)):
            areas.append(cv2.contourArea(contours[c]))

    max_id = areas.index(max(areas))
    cnt = contours[max_id] #max contours

    M_point = cv2.moments(cnt)
    # cv2.drawContours(orig_img, cnt, -1, (0, 0, 255), 2)

    center_x = int(M_point['m10']/M_point['m00'])
    center_y = int(M_point['m01']/M_point['m00'])
    drawCenter = cv2.circle(orig_img,(int(center_x),int(center_y)),2,(255,0,0),2)

    contours, hierarchy = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    print(len(contours))
    center_distance = []
    
    # cv2.drawContours(orig_img, contours, -1, (255, 0, 255), 2)
    if len(contours)>2:
        print("Hollow")
        # cv2.imshow("11111",orig_img)

        # for cont in contours:
        lx, ly, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(orig_img, (lx, ly), (lx+w, ly+h), (0, 0, 255), 2)

        leftdistance = distance(lx,center_y,center_x,center_y)
        downdistance = distance(center_x,ly+h,center_x,center_y)
        rightdistance = distance(center_x,center_y,lx+w,center_y)
        updistance = distance(center_x,ly,center_x,center_y)
        # center_distance.append([leftdistance,downdistance,rightdistance,updistance])

        center_distance.extend([leftdistance,downdistance,rightdistance])
        print(center_distance)

        best_area = center_distance.index(min(center_distance))
        print(best_area)
        if (best_area == 0):
            print("left")
            cv2.rectangle(orig_img,(int(center_x-20-center_distance[0]),int(center_y-20)),(int(center_x),int(center_y+20)),(255,255,0),2)
            result = angle(lx,center_y,center_x,center_y)
            # cv2.imshow("left",orig_img)
        elif (best_area == 1):
            print("down")
            cv2.rectangle(orig_img,(int(center_x-20),int(center_y)),(int(center_x+20),int(center_y+20+center_distance[1])),(255,255,0),2)
            result = angle(center_x,ly+h,center_x,center_y)
            # cv2.imshow("down",orig_img)
        elif (best_area == 2):
            print("right")
            cv2.rectangle(orig_img,(int(center_x),int(center_y-20)),(int(center_x+20+center_distance[2]),int(center_y+20)),(255,255,0),2)
            result = angle(center_x,center_y,lx+w,center_y)
            # cv2.imshow("right",orig_img)
        # elif (best_area == 3):
        #     print("up")
        #     cv2.rectangle(orig_img,(int(center_x-20),int(center_y-20-center_distance[3])),(int(center_x+20),int(center_y)),(255,255,0),2)
        #     cv2.imshow("up",orig_img)
        else:
            print("no")
        
        # width = 30
        # height = 80
        # grasp_left_x = int(center_x - (width/2.0))
        # grasp_left_y = int(center_y - 15)
        # grasp_right_x = int(center_x + (width/2.0))
        # grasp_right_y = int(center_y + height)


        # cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)
        # result = angle(grasp_left_x,grasp_left_y,grasp_left_x,grasp_left_y + height)
        # cv2.imshow("2222",orig_img)
        # result = 90
        print(result)
        # cv2.imshow("1",orig_img)

    else:
        print("Solid")
        width = 140
        height = 30
        grasp_left_x = int(center_x - (width/2.0))
        grasp_left_y = int(center_y - (height/2.0))
        grasp_right_x = int(center_x + (width/2.0))
        grasp_right_y = int(center_y + (height/2.0))

        # ellipse= cv2.fitEllipse(contours[])
        # cv2.ellipse(orig_img,ellipse,(0,235,0),2)
        # cv2.imshow("ellipse",orig_img)
    
        cv2.rectangle(orig_img,(grasp_left_x,grasp_left_y),(grasp_right_x,grasp_right_y),(255,255,0),2)
        result = 0
        # result = angle(grasp_left_x,grasp_left_y,grasp_left_x,grasp_left_y + height)
        # cv2.imshow("2222",orig_img)
        print(result)

    return orig_img,center_x,center_y,result

#grip and columnar detection
def calculate_center(left_x,left_y,right_x,right_y):
    width = abs(right_x -left_x) 
    height = abs(right_y - left_y)
    center_x = left_x + (width/2.0)
    center_y = left_y + (height/2.0)
    return center_x,center_y
# ------------------------------------------------------- detection -------------------------------------------------
# def detect(net, meta, image, thresh=.5, hier_thresh=.5, nms=.45, debug=False):
#     """
#     Performs the detection
#     """
#     custom_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     custom_image = cv2.resize(custom_image, (lib.network_width(net), lib.network_height(net)), interpolation=cv2.INTER_LINEAR)
#     im, arr = array_to_image(custom_image)
#     num = c_int(0)
#     pnum = pointer(num)
#     predict_image(net, im)
#     dets = get_network_boxes(net, image.shape[1], image.shape[0], thresh, hier_thresh, None, 0, pnum, 0)
#     num = pnum[0]
#     if nms:
#         do_nms_sort(dets, num, meta.classes, nms)
#     res = []
#     if debug:
#         log.debug("about to range")
#     for j in range(num):
#         for i in range(meta.classes):
#             if dets[j].prob[i] > 0:
#                 b = dets[j].bbox
#                 if altNames is None:
#                     name_tag = meta.names[i]
#                 else:
#                     name_tag = altNames[i]
#                 res.append((name_tag, dets[j].prob[i], (b.x, b.y, b.w, b.h), i))
#     res = sorted(res, key=lambda x: -x[1])
#     free_detections(dets, num)
#     return res

# def array_to_image(arr):
#     import numpy as np
#     # need to return old values to avoid python freeing memory
#     arr = arr.transpose(2, 0, 1)
#     c = arr.shape[0]
#     h = arr.shape[1]
#     w = arr.shape[2]
#     arr = np.ascontiguousarray(arr.flat, dtype=np.float32) / 255.0
#     data = arr.ctypes.data_as(POINTER(c_float))
#     im = IMAGE(w, h, c, data)
#     return im, arr


# def classify(net, meta, im):
#     out = predict_image(net, im)
#     res = []
#     for i in range(meta.classes):
#         if altNames is None:
#             name_tag = meta.names[i]
#         else:
#             name_tag = altNames[i]
#         res.append((name_tag, out[i]))
#     res = sorted(res, key=lambda x: -x[1])
#     return res


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




# def generate_color(meta_path):
#     '''
#     Generate random colors for the number of classes mentioned in data file.
#     Arguments:
#     meta_path: Path to .data file.

#     Return:
#     color_array: RGB color codes for each class.
#     '''
#     random.seed(42)
#     with open(meta_path, 'r') as f:
#         content = f.readlines()
#     class_num = int(content[0].split("=")[1])
#     color_array = []
#     for x in range(0, class_num):
#         color_array.append((randint(0, 255), randint(0, 255), randint(0, 255)))
#     return color_array

# def c_array(ctype, values):
#     arr = (ctype*len(values))()
#     arr[:] = values
#     return arr


# class BOX(Structure):
#     _fields_ = [("x", c_float),
#                 ("y", c_float),
#                 ("w", c_float),
#                 ("h", c_float)]


# class DETECTION(Structure):
#     _fields_ = [("bbox", BOX),
#                 ("classes", c_int),
#                 ("prob", POINTER(c_float)),
#                 ("mask", POINTER(c_float)),
#                 ("objectness", c_float),
#                 ("sort_class", c_int),
#                 ("uc", POINTER(c_float)),
#                 ("points", c_int),
#                 ("embeddings", POINTER(c_float)),
#                 ("embedding_size", c_int),
#                 ("sim", c_float),
#                 ("track_id", c_int)]

# class IMAGE(Structure):
#     _fields_ = [("w", c_int),
#                 ("h", c_int),
#                 ("c", c_int),
#                 ("data", POINTER(c_float))]


# class METADATA(Structure):
#     _fields_ = [("classes", c_int),
#                 ("names", POINTER(c_char_p))]

# hasGPU = True
# if os.name == "nt":
#     cwd = os.path.dirname(__file__)
#     os.environ['PATH'] = cwd + ';' + os.environ['PATH']
#     winGPUdll = os.path.join(cwd, "yolo_cpp_dll.dll")
#     winNoGPUdll = os.path.join(cwd, "yolo_cpp_dll_nogpu.dll")
#     envKeys = list()
#     for k, v in os.environ.items():
#         envKeys.append(k)
#     try:
#         try:
#             tmp = os.environ["FORCE_CPU"].lower()
#             if tmp in ["1", "true", "yes", "on"]:
#                 raise ValueError("ForceCPU")
#             else:
#                 log.info("Flag value '"+tmp+"' not forcing CPU mode")
#         except KeyError:
#             # We never set the flag
#             if 'CUDA_VISIBLE_DEVICES' in envKeys:
#                 if int(os.environ['CUDA_VISIBLE_DEVICES']) < 0:
#                     raise ValueError("ForceCPU")
#             try:
#                 global DARKNET_FORCE_CPU
#                 if DARKNET_FORCE_CPU:
#                     raise ValueError("ForceCPU")
#             except NameError:
#                 pass
#             # log.info(os.environ.keys())
#             # log.warning("FORCE_CPU flag undefined, proceeding with GPU")
#         if not os.path.exists(winGPUdll):
#             raise ValueError("NoDLL")
#         lib = CDLL(winGPUdll, RTLD_GLOBAL)
#     except (KeyError, ValueError):
#         hasGPU = False
#         if os.path.exists(winNoGPUdll):
#             lib = CDLL(winNoGPUdll, RTLD_GLOBAL)
#             log.warning("Notice: CPU-only mode")
#         else:
#             # Try the other way, in case no_gpu was
#             # compile but not renamed
#             lib = CDLL(winGPUdll, RTLD_GLOBAL)
#             log.warning("Environment variables indicated a CPU run, but we didn't find `" +
#                         winNoGPUdll+"`. Trying a GPU run anyway.")
# else:
#     # lib = CDLL("../libdarknet/libdarknet.so", RTLD_GLOBAL)
#     lib = CDLL("./libdarknet/libdarknet.so", RTLD_GLOBAL)
# lib.network_width.argtypes = [c_void_p]
# lib.network_width.restype = c_int
# lib.network_height.argtypes = [c_void_p]
# lib.network_height.restype = c_int

# predict = lib.network_predict
# predict.argtypes = [c_void_p, POINTER(c_float)]
# predict.restype = POINTER(c_float)

# if hasGPU:
#     set_gpu = lib.cuda_set_device
#     set_gpu.argtypes = [c_int]

# make_image = lib.make_image
# make_image.argtypes = [c_int, c_int, c_int]
# make_image.restype = IMAGE

# get_network_boxes = lib.get_network_boxes
# get_network_boxes.argtypes = [c_void_p, c_int, c_int, c_float, c_float, POINTER(
#     c_int), c_int, POINTER(c_int), c_int]
# get_network_boxes.restype = POINTER(DETECTION)

# make_network_boxes = lib.make_network_boxes
# make_network_boxes.argtypes = [c_void_p]
# make_network_boxes.restype = POINTER(DETECTION)

# free_detections = lib.free_detections
# free_detections.argtypes = [POINTER(DETECTION), c_int]

# free_ptrs = lib.free_ptrs
# free_ptrs.argtypes = [POINTER(c_void_p), c_int]

# network_predict = lib.network_predict
# network_predict.argtypes = [c_void_p, POINTER(c_float)]

# reset_rnn = lib.reset_rnn
# reset_rnn.argtypes = [c_void_p]

# load_net = lib.load_network
# load_net.argtypes = [c_char_p, c_char_p, c_int]
# load_net.restype = c_void_p

# load_net_custom = lib.load_network_custom
# load_net_custom.argtypes = [c_char_p, c_char_p, c_int, c_int]
# load_net_custom.restype = c_void_p

# do_nms_obj = lib.do_nms_obj
# do_nms_obj.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# do_nms_sort = lib.do_nms_sort
# do_nms_sort.argtypes = [POINTER(DETECTION), c_int, c_int, c_float]

# free_image = lib.free_image
# free_image.argtypes = [IMAGE]

# letterbox_image = lib.letterbox_image
# letterbox_image.argtypes = [IMAGE, c_int, c_int]
# letterbox_image.restype = IMAGE

# load_meta = lib.get_metadata
# lib.get_metadata.argtypes = [c_char_p]
# lib.get_metadata.restype = METADATA

# load_image = lib.load_image_color
# load_image.argtypes = [c_char_p, c_int, c_int]
# load_image.restype = IMAGE

# rgbgr_image = lib.rgbgr_image
# rgbgr_image.argtypes = [IMAGE]

# predict_image = lib.network_predict_image
# predict_image.argtypes = [c_void_p, IMAGE]
# predict_image.restype = POINTER(c_float)


# ------------------------------------------------------- detection -------------------------------------------------
# ------------------------------------------------------- Matting Arguments -------------------------------------------------
parser = argparse.ArgumentParser(description='Inference images')
parser.add_argument('--model-type', type=str, required=False, choices=['mattingbase', 'mattingrefine'],
                    default='mattingrefine')
parser.add_argument('--model-backbone', type=str, required=False, choices=['resnet101', 'resnet50', 'mobilenetv2'],
                    default='resnet101')
parser.add_argument('--model-backbone-scale', type=float, default=0.25)
parser.add_argument('--model-checkpoint', type=str, required=False, default='./model_pth/pytorch_resnet101.pth')
parser.add_argument('--model-refine-mode', type=str, default='sampling', choices=['full', 'sampling', 'thresholding'])
parser.add_argument('--model-refine-sample-pixels', type=int, default=80_000)
parser.add_argument('--model-refine-threshold', type=float, default=0.9)
parser.add_argument('--model-refine-kernel-size', type=int, default=3)
parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], default='cuda')
parser.add_argument('--num-workers', type=int, default=0,
                    help='number of worker threads used in DataLoader. Note that Windows need to use single thread (0).')
parser.add_argument('--preprocess-alignment', action='store_true')
parser.add_argument('--output-dir', type=str, required=False, default='/home/user/shape_detection/circle/')
parser.add_argument('--output-types', type=str, required=False, nargs='+',
                    choices=['com', 'pha', 'fgr', 'err', 'ref', 'new'])
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

#original_img
save_path_columnar = "/home/user/shape_detection/columnar/orig/"
save_path_long = "/home/user/shape_detection/long/orig/"
save_path_circle = "/home/user/shape_detection/circle/orig/"
save_path_blade = "/home/user/shape_detection/blade/orig/"

#process
save_process_columnar = "/home/user/shape_detection/columnar/process/" 
save_process_circle = "/home/user/shape_detection/circle/process/"
save_process_long = "/home/user/shape_detection/long/process/"
save_process_blade = "/home/user/shape_detection/blade/process/"

#matting
save_mat_columnar = "/home/user/shape_detection/columnar/mat/"
save_mat_long = "/home/user/shape_detection/long/mat/"
save_mat_circle = "/home/user/shape_detection/circle/mat/"
save_mat_blade = "/home/user/shape_detection/blade/mat/"


dataset_root_path = r"/home/user/matting/imagedata"
img_floder = os.path.join(dataset_root_path,"img")
bgr_floder = os.path.join(dataset_root_path,"bgr")

#first_bgr
local_img_name=r'/home/user/shape_detection/bgr/1.jpg'
save_bgr = "/home/user/shape_detection/bgr/"

#second_bgr
bgrcircle_path = "/home/user/shape_detection/circle/bgr/"
bgrblade_path = "/home/user/shape_detection/blade/bgr/"
bgrlong_path = "/home/user/shape_detection/long/bgr/"
bgrcolumnar_path = "/home/user/shape_detection/columnar/bgr/"

#find center
center_circle = "/home/user/shape_detection/circle/center/"
center_long = "/home/user/shape_detection/long/center/"
center_blade = "/home/user/shape_detection/blade/center/"
center_columnar = "/home/user/shape_detection/columnar/center/"

resized_intrinsics = 431.74859619
left_to_right_distance_cm = 7.5
curr_time = datetime.datetime.now()


if __name__ == '__main__':
    #定義zed
    zed = sl.Camera()

    #init
    input_type = sl.InputType() # Set configuration parameters
    input_type.set_from_camera_id(zed_id)
    init = sl.InitParameters(input_t=input_type)  # 初始化
    init.camera_resolution = sl.RESOLUTION.HD1080 # 相机分辨率(默认-HD720)
    init.camera_fps = 10
    # init.coordinate_units = sl.UNIT.METER
    init.coordinate_units = sl.UNIT.MILLIMETER

    init.depth_mode = sl.DEPTH_MODE.ULTRA         # 深度模式  (默认-PERFORMANCE)
    init.depth_minimum_distance = 150
    init.depth_maximum_distance = 2000 
    # init.depth_minimum_distance = 0.15
    # init.depth_maximum_distance = 2 
    
    

    #open camera
    if not zed.is_opened():
        log.info("Opening ZED Camera...")
    status = zed.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        log.error(repr(status))
        exit()

    #set zed runtime value
    runtime_parameters =sl.RuntimeParameters()
    runtime_parameters.sensing_mode = sl.SENSING_MODE.STANDARD

    #set image size
    image_size = zed.get_camera_information().camera_resolution
    image_size.width = 1280
    image_size.height = 720

    #turn zed to numpy(for opencv)
    image_zed_left = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    image_zed_right = sl.Mat(image_size.width, image_size.height, sl.MAT_TYPE.U8_C4)
    depth_image_zed = sl.Mat(image_size.width,image_size.height, sl.MAT_TYPE.U8_C4)

    # global metaMain, netMain, altNames
    # netMain = None
    # metaMain = None
    # altNames = None
    # assert 0 < thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    # if not os.path.exists(config):
    #     raise ValueError("Invalid config path `" +
    #                      os.path.abspath(config)+"`")
    # if not os.path.exists(weights):
    #     raise ValueError("Invalid weight path `" +
    #                      os.path.abspath(weights)+"`")
    # if not os.path.exists(data):
    #     raise ValueError("Invalid data file path `" +
    #                      os.path.abspath(data)+"`")
    # if netMain is None:
    #     netMain = load_net_custom(config.encode("ascii"), weights.encode("ascii"), 0, 1)  # batch size = 1
    # if metaMain is None:
    #     metaMain = load_meta(data.encode("ascii"))
    # if altNames is None:
    #     # In thon 3, the metafile default access craps out on Windows (but not Linux)
    #     # Read the names file and create a list to feed to detect
    #     try:
    #         with open(data) as meta_fh:
    #             meta_contents = meta_fh.read()
    #             import re
    #             match = re.search("names *= *(.*)$", meta_contents,
    #                               re.IGNORECASE | re.MULTILINE)
    #             if match:
    #                 result = match.group(1)
    #             else:
    #                 result = None
    #             try:
    #                 if os.path.exists(result):
    #                     with open(result) as names_fh:
    #                         names_list = names_fh.read().strip().split("\n")
    #                         altNames = [x.strip() for x in names_list]
    #             except TypeError:
    #                 pass
    #     except Exception:
    #         pass

    # color_array = generate_color(data)

    # log.info("Running...")

    i = -10
    a = 0
    b = 0
    c = 0
    d = 0

    #yolo
    network, class_names, class_colors = darknet.load_network(config,data,weights,batch_size=1)
    
    key = ''
    while key != 113:   
        zed.grab() #開啟管道
        
        zed.retrieve_image(image_zed_left, sl.VIEW.LEFT, sl.MEM.CPU, image_size)
        zed.retrieve_image(image_zed_right, sl.VIEW.RIGHT, sl.MEM.CPU, image_size)
        zed.retrieve_measure(depth_image_zed, sl.MEASURE.DEPTH, sl.MEM.CPU, image_size)

        
        
        color_image = image_zed_left.get_data()
        # image_right = image_zed_left.get_data()
        depth_image = depth_image_zed.get_data()

        i += 1
        if i ==1:
            cv2.imwrite(save_bgr + '1.jpg',color_image)
        
        # detections = detect(netMain, metaMain, image_left, thresh)

        # for detection in detections:
        #         label = detection[0]
        #         confidence = detection[1]
        #         pstring = label+": "+str(np.rint(100 * confidence))+"%"
        #         log.info(pstring)
        #         bounds = detection[2]
        #         y_extent = int(bounds[3])
        #         x_extent = int(bounds[2])
        #         # Coordinates are around the center
        #         x_coord = int(bounds[0] - bounds[2]/2)
        #         y_coord = int(bounds[1] - bounds[3]/2)
        #         #boundingBox = [[x_coord, y_coord], [x_coord, y_coord + y_extent], [x_coord + x_extent, y_coord + y_extent], [x_coord + x_extent, y_coord]]
        #         thickness = 1
        #         x, y, z = get_object_depth(depth_image, bounds)
        #         distance = math.sqrt(x * x + y * y + z * z)
        #         distance = "{:.2f}".format(distance)
        #         # cv2.rectangle(image_left, (x_coord - thickness, y_coord - thickness),(x_coord + x_extent + thickness, y_coord + (18 + thickness*4)),
        #         #               color_array[detection[3]], -1)
        #         cv2.putText(image_left, label + " " +  (str(distance) + " m"),(x_coord + (thickness * 4), y_coord + (10 + thickness * 4)),
        #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        #         cv2.rectangle(image_left, (x_coord - thickness, y_coord - thickness),(x_coord + x_extent + thickness, y_coord + y_extent + thickness),
        #                       color_array[detection[3]], int(thickness*2))
        # cv2.imshow("rgbCam",color_image)
        cv2.imshow("depth",depth_image)
    
        # #shape detection
        width = color_image.shape[1]
        height = color_image.shape[0]

        t_prev = time.time()

        frame_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))

        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes()) 
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.print_detections(detections, show_coordinates)
        print(detections)
        
        darknet.free_image(darknet_image)

        # # draw bounding box
        image = darknet.draw_boxes(detections, frame_resized, class_colors)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        cv2.imshow("win_title", image)
        fps = int(1/(time.time()-t_prev))
#好想睡覺.....
        if  len(detections) != 0:
            if int(float(detections[0][1])) >= 90:
                if detections[0][0] == "long" and key == 114:
                    a += 1

                    long_center_x = detections[0][2][0]
                    long_center_y = detections[0][2][1]
                    long_width = detections[0][2][2]
                    long_height = detections[0][2][3]

                    left_up_x = int(round((long_center_x - (long_width/2.0)),3))
                    left_up_y = int(round((long_center_y - (long_height/2.0)),3))

                    right_down_x = int(round((long_center_x + long_width/2.0),3))
                    right_down_y = int(round((long_center_y + (long_height/2.0)),3))

                    orig_long = save_path_long+str(a) +'.jpg'

                    cv2.imwrite(orig_long,color_image)
                    shutil.copy(local_img_name, bgrlong_path +str(a) +'.jpg')
                    
                    matimg = handle(orig_long,bgrlong_path +str(a) +'.jpg')
                    process = post_processing(matimg[2])
                    process_long = save_process_long+"long_"+str(a)+'.jpg'
                    cv2.imwrite(process_long,process)

                    img = cv2.imread(process_long)

                    center = line_Segment(process_long,orig_long)

                    # 求取深度
                    z_value = depth_image_zed.get_value(center[1],center[2])
                    print(center[1],center[2])
                    print(z_value[1])

                    cv2.putText(center[0], "Object: " + str(detections[0][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Center: " + str(round(center[1],3)) +","+ str(round(center[2],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Angle: " + str(round(center[3],3)) , (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(center_long+"long_"+str(a)+'.jpg',center[0])
                    cv2.imshow("finish",center[0])
                elif detections[0][0] == "circle" and key == 114:
                    b += 1

                    orig_circle = save_path_circle+str(b) +'.jpg'
                    cv2.imwrite(orig_circle,color_image)

                    shutil.copy(local_img_name, bgrcircle_path + str(b) +'.jpg')
                    matimg = handle(orig_circle,bgrcircle_path + str(b) +'.jpg')
                    
                    process = post_processing(matimg[2])
                    process_circle = save_process_circle + "circle_" + str(b) + '.jpg'
                    cv2.imwrite(process_circle,process)

                    center = circle_transform(process_circle,orig_circle)

                    # 求取深度
                    z_value = depth_image_zed.get_value(center[1],center[2])
                    print(center[1],center[2])
                    print(z_value[1])

                    cv2.putText(center[0], "Object: " + str(detections[0][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Center: " + str(round(center[1],3)) +","+ str(round(center[2],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(center[0], "Angle: " + str(round(center[3],3)) , (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                    cv2.imwrite(center_circle+"circle_"+str(b)+'.jpg',center[0])
                    cv2.imshow("finish",center[0])

                elif detections[0][0] == "columnar" or detections[0][0] == "grip":
                    if key ==114:
                        c += 1
                        orig_columnar = save_path_columnar+str(c) +'.jpg'
                        cv2.imwrite(orig_columnar,color_image)

                        new_obj_name = str(c) +'.jpg'
                        shutil.copy(local_img_name, bgrcolumnar_path + new_obj_name)

                        matimg = handle(orig_columnar,bgrcolumnar_path + new_obj_name)                        

                        process = post_processing(matimg[2])
                        process_columnar = save_process_columnar + "columnar_" + str(c) + '.jpg'
                        cv2.imwrite(process_columnar,process)

                        if(len(detections) == 1):
                            center = line_Segment_cup(process_columnar,orig_columnar)
                            #depth
                            z_value = depth_image_zed.get_value(center[1],center[2])

                            cv2.putText(center[0], "Object: " + str(detections[0][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(center[0], "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(center[0], "Center: " + str(round(center[1],3)) +","+ str(round(center[2],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.putText(center[0], "Angle: " + str(round(center[3],3)) , (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                            cv2.imwrite(center_columnar+"columnar_"+str(c)+'.jpg',center[0])
                            cv2.imshow("finish",center[0])
                        #可以畢業就好><
                        else:
                            if detections[0][0] == "grip" and detections[1][0]== "columnar":
                                grip_center_x = detections[0][2][0]
                                grip_center_y = detections[0][2][1]
                                grip_width = detections[0][2][2]
                                grip_height = detections[0][2][3]

                                columnar_center_x = detections[1][2][0]
                                columnar_center_y = detections[1][2][1]
                                columnar_width = detections[1][2][2]
                                columnar_height = detections[1][2][3]
                                    
                                if(columnar_center_x > grip_center_x): #grip left
                                    if((columnar_center_x - grip_center_x) >= 70):
                                        left_up_x = int(grip_center_x - grip_width)
                                        left_up_y = int(grip_center_y - (grip_width/3.0))
                                        right_down_x = int(grip_center_x)
                                        right_down_y = int(left_up_y + (grip_width/3.0))

                                        cv2.rectangle(color_image,(left_up_x,left_up_y),(right_down_x,right_down_y),(255,0,255),2)
                                        center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y)
                                        real_angle = angle(left_up_x,left_up_y,left_up_x+grip_width,left_up_y)
                                            
                                    elif((columnar_center_x - grip_center_x) < 70 and (columnar_center_x - grip_center_x) >= 1 ):
                                        left_up_x = int(grip_center_x - (grip_height/2.0))
                                        left_up_y = int(grip_center_y - (grip_width/3.0))
                                        right_down_x = int(grip_center_x + (grip_height/2.0))
                                        right_down_y = int(grip_center_y + (grip_width/3.0))

                                        cv2.rectangle(color_image,(left_up_x,left_up_y),(right_down_x,right_down_y),(255,0,255),2)

                                        center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y)
                                        real_angle = angle(left_up_x,left_up_y,left_up_x+grip_width,left_up_y)
            
                                    cv2.circle(color_image,(int(center[0]),int(center[1])),2,(0,0,255),2)

                                    z_value = depth_image_zed.get_value(center[0],center[1])

                                    cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 40), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Center: " + str(round(center[0],3)) +","+ str(round(center[1],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: "+ str(real_angle), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_columnar+"columnar_grip_"+str(c)+'.jpg',color_image)  
                                    cv2.imshow("finish",color_image)

                                else: #grip right
                                    if((grip_center_x - columnar_center_x) >=70):
                                        left_up_x = int(grip_center_x)
                                        left_up_y = int(grip_center_y - (grip_width/3.0))
                                        right_down_x = int(grip_center_x + grip_width)
                                        right_down_y = int(left_up_y + (grip_width/3.0))

                                        cv2.rectangle(color_image,(left_up_x,left_up_y),(right_down_x,right_down_y),(255,0,255),2)  
                                        center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y) 
                                        real_angle = angle(left_up_x,left_up_y,left_up_x + grip_width,left_up_y)

                                    elif((grip_center_x - columnar_center_x) < 70 and (grip_center_x - columnar_center_x) >= 1):
                                        left_up_x = int(grip_center_x - (grip_height/2.0))
                                        left_up_y = int(grip_center_y - (grip_width/3.0))
                                        right_down_x = int(grip_center_x + (grip_height/2.0))
                                        right_down_y = int(grip_center_y + (grip_width/3.0))

                                        cv2.rectangle(color_image,(left_up_x,left_up_y),(right_down_x,right_down_y),(255,0,255),2)
                                        center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y)
                                        real_angle = angle(left_up_x,left_up_y,left_up_x+grip_width,left_up_y)
                                            
                                    cv2.circle(color_image,(int(center[0]),int(center[1])),2,(0,0,255),2)

                                    z_value = depth_image_zed.get_value(center[0],center[1])

                                    cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 40), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Center: " + str(round(center[0],3)) +","+ str(round(center[1],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: "+ str(real_angle), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_columnar+"columnar_grip_"+str(c)+'.jpg',color_image)  
                                    cv2.imshow("finish",color_image) 

                            elif detections[1][0] == "grip" and detections[0][0]== "columnar":
                                grip_center_x = detections[1][2][0]
                                grip_center_y = detections[1][2][1]
                                grip_width = detections[1][2][2]
                                grip_height = detections[1][2][3]

                                columnar_center_x = detections[0][2][0]
                                columnar_center_y = detections[0][2][1]
                                columnar_width = detections[0][2][2]
                                columnar_height = detections[0][2][3]

                                if(columnar_center_x > grip_center_x): #grip left
                                    left_up_x = int(grip_center_x - grip_width)
                                    left_up_y = int(grip_center_y - (grip_width/3.0))
                                    right_down_x = int(grip_center_x )
                                    right_down_y = int(left_up_y + (grip_width/3.0))

                                    cv2.rectangle(color_image,(left_up_x-10,left_up_y),(right_down_x+10,right_down_y),(255,0,255),2)
                                    center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y)
                                    cv2.circle(color_image,(int(center[0]),int(center[1])),2,(0,0,255),2)
                                    
                                    #depth
                                    z_value = depth_image_zed.get_value(center[0],center[1])

                                    cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 40), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Center: " + str(round(center[0],3)) +","+ str(round(center[1],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: "+ str(real_angle), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_columnar+"columnar_grip_"+str(c)+'.jpg',color_image)  
                                    cv2.imshow("finish",color_image) 

                                else: #grip eft_up_y = int(grip_center_y - (grip_height/3.0/2.0))

                                    right_down_x = int(grip_center_x + grip_width)
                                    right_down_y = int(left_up_y + (grip_height/3.0))

                                    cv2.rectangle(color_image,(left_up_x-10,left_up_y),(right_down_x+10,right_down_y),(255,0,255),2)  
                                    center = calculate_center(left_up_x,left_up_y,right_down_x,right_down_y)             

                                    cv2.circle(color_image,(int(center[0]),int(center[1])),2,(0,0,255),2)
                                    #depth
                                    z_value = depth_image_zed.get_value(center[0],center[1])

                                    cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[0][1]), (10, 40), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "Center: " + str(round(center[0],3)) +","+ str(round(center[1],3)), (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.putText(color_image, "angle: "+ str(real_angle), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    cv2.imwrite(center_columnar+"columnar_grip_"+str(c)+'.jpg',color_image)  
                                    cv2.imshow("finish",color_image) 
                                #人生好難阿 看不到眼前的希望  am1:47
                elif detections[0][0] == "blade" or detections[0][0] == "grasp": #detections[0][0]= blade,detections[1][0]= grasp
                    if key == 114:
                        d += 1
                        orig_blade = save_path_blade+str(d) +'.jpg'
                        cv2.imwrite(orig_blade,color_image)
                        new_obj_name = str(d) +'.jpg'
                        shutil.copy(local_img_name, bgrblade_path + new_obj_name)
                        matimg = handle(orig_blade,bgrblade_path + new_obj_name)
                        
                        process = post_processing(matimg[2])
                        process_blade = save_process_blade + "blade_" + str(d) + ".jpg"
                        cv2.imwrite(process_blade,process)  
                        if detections[1][0] == "grasp" and detections[0][0] == "blade":
                            grasp_center_x = round(detections[1][2][0])
                            grasp_center_y = round(detections[1][2][1])
                            grasp_width = round(detections[1][2][2])
                            grasp_height = round(detections[1][2][3])

                            blade_center_x = round(detections[0][2][0])
                            blade_center_y = round(detections[0][2][1])
                            blade_width = round(detections[0][2][2])
                            blade_height = round(detections[0][2][3])

                            left_up_x = round(grasp_center_x - (grasp_width/2.0))
                            left_up_y = round(grasp_center_y - (grasp_height/2.0))
                            right_down_x = round(grasp_center_x + grasp_width/2.0)
                            right_down_y = round(grasp_center_y + (blade_height/2.0))

                            blade_left_up_x = round(blade_center_x - (blade_width/2.0))
                            blade_left_up_y = round(blade_center_y - (blade_height/2.0))
                            blade_right_down_x = round(blade_center_x + blade_width/2.0)
                            blade_right_down_y = round(blade_center_y + (blade_height/2.0))

                            img = cv2.imread(process_blade)  
                            # img_org = cv2.imread(orig_blade) = color_image
                            cv2.rectangle(img,(left_up_x,left_up_y),(right_down_x,right_down_y),(50,205,50),2)

                            #just look rectangle
                            crop_img = img[left_up_y:left_up_y + (right_down_y - left_up_y), left_up_x:left_up_x + (right_down_x - left_up_x)]
                                
                            gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 70, 210)
                            # cv2.imshow("edges",edges)
                            contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
                            # print(contours)
                            if(len(contours) > 4): #剪刀   
                                #quadrant 1 , cut right up
                                if( blade_center_x > grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut right up" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 115 and(blade_center_x - grasp_center_x) >= 25): 
                                        cv2.putText(color_image, "cut right up 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        # cv2.line(img_org,(int(right_down_x),int(left_up_y)),(int(blade_left_up_x),int(blade_right_down_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (right_down_x + blade_left_up_x)/2.0 
                                        real_grasp_center_y = (left_up_y + blade_right_down_y)/2.0
                                            
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                            
                                        result = angle(right_down_x ,left_up_y,left_up_x ,right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        
                                    elif((blade_center_x - grasp_center_x) >= 115):
                                        cv2.putText(color_image, "cut right up 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        cv2.circle(color_image,(int(blade_center_x - blade_width/2.0),int(blade_center_y)),2,(255,0,0),2)
                                        cv2.rectangle(color_image, (int(blade_center_x - blade_width/2.0 - width/2.0), int(blade_center_y - height/2.0)), (int(blade_center_x - blade_width/2.0 + width/2.0), int(blade_center_y + height/2.0)),(255, 0, 255), 2)

                                        real_grasp_center_x = blade_center_x - blade_width/2.0
                                        real_grasp_center_y = blade_center_y

                                        result = angle(blade_left_up_x ,blade_left_up_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(2, 202, 119),2)
                                        
                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                    
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        
                                    elif((blade_center_x - grasp_center_x) < 25): 
                                        cv2.putText(color_image, "cut right up 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y + blade_height/2.0

                                        result = angle(real_grasp_center_x,real_grasp_center_y,blade_right_down_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                
                                # quadrant 2 , cut right down
                                elif(blade_center_x > grasp_center_x and blade_center_y > grasp_center_y): #右下角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut right down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 115 and(blade_center_x - grasp_center_x) >= 25):
                                        cv2.putText(color_image, "cut right down 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20 
                                        height = 40

                                        # cv2.line(img_org,(int(right_down_x),int(right_down_y)),(int(blade_left_up_x),int(blade_left_up_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (right_down_x + blade_left_up_x)/2.0 
                                        real_grasp_center_y = (right_down_y + blade_left_up_y)/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rec tangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                        result = angle(left_up_x ,left_up_y,right_down_x ,right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),
                                                (real_grasp_center_x - width/2,real_grasp_center_y + height/2.0),(real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        
                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) >= 115):
                                        cv2.putText(color_image, "cut right down 2222 1" , (10, 185), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40
                                
                                        real_grasp_center_x = blade_center_x - blade_width/2.0
                                        real_grasp_center_y = blade_center_y

                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        cv2.rectangle(color_image, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) < 25): 
                                        cv2.putText(color_image, "cut right down 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y - blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(blade_left_up_x,blade_left_up_y,real_grasp_center_x,real_grasp_center_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 3 , cut left up
                                elif(blade_center_x < grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut left up " , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((grasp_center_x - blade_center_x) < 115 and(grasp_center_x - blade_center_x)>=25):
                                        cv2.putText(color_image, "cut left up 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.line(img_org,(int(right_down_x),int(right_down_y)),(int(blade_left_up_x),int(blade_left_up_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (left_up_x + blade_right_down_x)/2.0 
                                        real_grasp_center_y = (left_up_y + blade_right_down_y)/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # # cv2.rectangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                        result = angle(blade_left_up_x ,blade_left_up_y,blade_right_down_x ,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),
                                                (real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),(real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        
                                    elif((grasp_center_x - blade_center_x) >= 115):
                                        cv2.putText(color_image, "cut left up 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        real_grasp_center_x = blade_center_x + blade_width/2.0
                                        real_grasp_center_y = blade_center_y

                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "cut left up 3333 1" , (10, 185), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y + blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(real_grasp_center_x,real_grasp_center_y,blade_right_down_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 4 , cut left down
                                elif(grasp_center_x > blade_center_x and blade_center_y > grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut left down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((grasp_center_x - blade_center_x) < 115 and(grasp_center_x - blade_center_x)>=25):
                                        cv2.putText(color_image, "cut left down 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        real_grasp_center_x = (blade_right_down_x + left_up_x)/2.0 
                                        real_grasp_center_y = (blade_left_up_y + right_down_y)/2.0

                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(blade_left_up_x,blade_right_down_y,blade_right_down_x,blade_left_up_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # print(real_grasp_center_x,real_grasp_center_y)     

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) >= 115):
                                        cv2.putText(color_image, "cut left down 2222 1" , (10, 185), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40
                                        
                                        real_grasp_center_x = blade_center_x + blade_width/2.0
                                        real_grasp_center_y = blade_center_y

                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "cut left down 3333 1" , (10, 185), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y - blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(real_grasp_center_x,real_grasp_center_y,blade_left_up_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
#########                       cv2.imwrite(center_blade + "blade_grip_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish",color_image)
                            else:  #鉗子
                                #quadrant 1,right up 
                                if( blade_center_x > grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right up" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 120 and(blade_center_x - grasp_center_x) > 25): 
                                        cv2.putText(color_image, "right up 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        # cv2.circle(img_org,(int(grasp_center_x),int(grasp_center_y + height/2)),2,(255,0,0),2)
                                        result = angle(left_up_x ,right_down_y,right_down_x ,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) >= 120):
                                        cv2.putText(color_image, "right up 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) < 25): 
                                        cv2.putText(color_image, "right up 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                       #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                #quadrant 2 , right down
                                elif(blade_center_x > grasp_center_x and blade_center_y > grasp_center_y): #右下角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) <= 120 and(blade_center_x - grasp_center_x)>25):
                                        cv2.putText(color_image, "right down 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        result = angle(left_up_x,left_up_y,right_down_x,right_down_y) + 90
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y + height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) >= 120): 
                                        cv2.putText(color_image, "right down 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) < 25):
                                        cv2.putText(color_image, "right down 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                       #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                #quadrant 3 , left up 
                                elif(blade_center_x < grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left up " , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x)>25):
                                        cv2.putText(color_image, "left up 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x+width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,right_down_x,right_down_y)
                                        box = [(grasp_center_x + width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y + height),(grasp_center_x + width/2.0 ,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((grasp_center_x - blade_center_x) >= 120): 
                                        cv2.putText(color_image, "left up 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,right_down_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((grasp_center_x - blade_center_x) < 25): 
                                        cv2.putText(color_image, "left up 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 4 , left down
                                elif(grasp_center_x > blade_center_x and blade_center_y > grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x)>25):
                                        cv2.putText(color_image, "left down 1111 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,right_down_y,right_down_x,left_up_y)
                                        box = [(grasp_center_x + width/2.0 ,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y + height),(grasp_center_x + width/2.0 ,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        # aa = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y + height/2))
                                        print(rota)

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        print(real_grasp_center_x,real_grasp_center_y)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
          
                                    elif((grasp_center_x - blade_center_x) >= 120): 
                                        cv2.putText(color_image, "down 2222 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y+height),(grasp_center_x + width/2.0,grasp_center_y + height)]
                                        rota = rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) < 25): 
                                        cv2.putText(color_image, "left down 3333 1" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    # cv2.imshow("img111",color_image)  

                                cv2.imwrite(center_blade + "blade_grip_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("img",color_image)

#===============================================================================================================================#
                        elif detections[0][0] == "grasp" and detections[1][0] == "blade":
                            grasp_center_x = detections[0][2][0]
                            grasp_center_y = detections[0][2][1]
                            grasp_width = detections[0][2][2]
                            grasp_height = detections[0][2][3]

                            blade_center_x = detections[1][2][0]
                            blade_center_y = detections[1][2][1]
                            blade_width = detections[1][2][2]
                            blade_height = detections[1][2][3]

                            left_up_x = int(grasp_center_x - (grasp_width/2.0))
                            left_up_y = int(grasp_center_y - (grasp_height/2.0))

                            right_down_x = int(grasp_center_x + grasp_width/2.0)
                            right_down_y = int(grasp_center_y + (grasp_height/2.0))

                            blade_left_up_x = round(blade_center_x - (blade_width/2.0))
                            blade_left_up_y = round(blade_center_y - (blade_height/2.0))
                            blade_right_down_x = round(blade_center_x + blade_width/2.0)
                            blade_right_down_y = round(blade_center_y + (blade_height/2.0))

                            img = cv2.imread(process_blade)
                            # img_org = cv2.imread(orig_blade)
                            cv2.rectangle(img,(left_up_x,left_up_y),(right_down_x,right_down_y),(50,205,50),2)
                            # cv2.imshow("img",img)

                            #just look rectangle
                            crop_img = img[left_up_y:left_up_y + (right_down_y - left_up_y), left_up_x:left_up_x + (right_down_x - left_up_x)]
                            
                            gray = cv2.cvtColor(crop_img,cv2.COLOR_BGR2GRAY)
                            edges = cv2.Canny(gray, 70, 210)
                            # cv2.imshow("edges",edges)
                            contours, hierarchy = cv2.findContours(edges, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

                            if(len(contours) > 4): #剪刀
                                #quadrant 1 , cut right up
                                if( blade_center_x > grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut right up" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 115 and(blade_center_x - grasp_center_x) >= 25): 
                                        cv2.putText(color_image, "cut right up 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        # cv2.line(img_org,(int(right_down_x),int(left_up_y)),(int(blade_left_up_x),int(blade_right_down_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (right_down_x + blade_left_up_x)/2.0 
                                        real_grasp_center_y = (left_up_y + blade_right_down_y)/2.0
                                        
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                        
                                        result = angle(right_down_x ,left_up_y,left_up_x ,right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        #depth
                                        z_value = depth_image[int(real_grasp_center_y),int(real_grasp_center_x)]
                                        depth_value = resized_intrinsics * left_to_right_distance_cm / z_value

                                        cv2.putText(img_org, "depth: " + str(round(depth_value,3)), (10, 35), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(img_org, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 95), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(img_org, "angle: " + str(round(result,3)), (10, 65), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) >= 115):
                                        cv2.putText(img_org, "cut right up 2222 2" , (10, 185), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40
                                        # cv2.line(img_org,(int(right_down_x),int(left_up_y)),(int(blade_left_up_x),int(blade_right_down_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        # real_grasp_center_x = (right_down_x + blade_left_up_x)/2.0 
                                        # real_grasp_center_y = (left_up_y + blade_right_down_y)/2.0
                                        
                                        cv2.circle(img_org,(int(blade_center_x - blade_width/2.0),int(blade_center_y)),2,(255,0,0),2)
                                        cv2.rectangle(img_org, (int(blade_center_x - blade_width/2.0 - width/2.0), int(blade_center_y - height/2.0)), (int(blade_center_x - blade_width/2.0 + width/2.0), int(blade_center_y + height/2.0)),(255, 0, 255), 2)

                                        real_grasp_center_x = blade_center_x - blade_width/2.0
                                        real_grasp_center_y = blade_center_y

                                        result = angle(blade_left_up_x ,blade_left_up_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                            
                                        cv2.line(img_org,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(img_org,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(img_org,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(img_org,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) < 25):
                                        cv2.putText(color_image, "cut right up 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y + blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(real_grasp_center_x,real_grasp_center_y,blade_right_down_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    #quadrant 2 , cut right down
                                elif(blade_center_x > grasp_center_x and blade_center_y > grasp_center_y): #右下角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut right down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        
                                    if((blade_center_x - grasp_center_x) < 115 and(blade_center_x - grasp_center_x) >= 25):
                                        cv2.putText(color_image, "cut right down 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20 
                                        height = 40

                                        # cv2.line(img_org,(int(right_down_x),int(right_down_y)),(int(blade_left_up_x),int(blade_left_up_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (right_down_x + blade_left_up_x)/2.0 
                                        real_grasp_center_y = (right_down_y + blade_left_up_y)/2.0
                                        
                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rec tangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                        result = angle(left_up_x ,left_up_y,right_down_x ,right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),
                                                (real_grasp_center_x - width/2,real_grasp_center_y + height/2.0),(real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) >= 115):
                                        cv2.putText(color_image, "cut right down 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        real_grasp_center_x = blade_center_x - blade_width/2.0
                                        real_grasp_center_y = blade_center_y
                    
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        cv2.rectangle(color_image, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) < 25):
                                        cv2.putText(color_image, "cut right down 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y - blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(blade_left_up_x,blade_left_up_y,real_grasp_center_x,real_grasp_center_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    #quadrant 3 , cut left up
                                elif(blade_center_x < grasp_center_x and blade_center_y < grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut left up " , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((grasp_center_x - blade_center_x) < 115 and(grasp_center_x - blade_center_x) >= 25):
                                        cv2.putText(color_image, "cut left up 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.line(img_org,(int(right_down_x),int(right_down_y)),(int(blade_left_up_x),int(blade_left_up_y)),(255, 0, 255),2,cv2.LINE_AA)
                                        real_grasp_center_x = (left_up_x + blade_right_down_x)/2.0 
                                        real_grasp_center_y = (left_up_y + blade_right_down_y)/2.0
                                        
                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # # cv2.rectangle(img_org, (int(real_grasp_center_x-width/2.0), int(real_grasp_center_y-height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)
                                        result = angle(blade_left_up_x ,blade_left_up_y,blade_right_down_x ,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),
                                                (real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),(real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((grasp_center_x - blade_center_x) >= 115):
                                        cv2.putText(color_image, "cut left up 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x + blade_width/2.0
                                        real_grasp_center_y = blade_center_y
                    
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    
                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "cut left up 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 40
                                        height = 20

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y + blade_height/2.0


                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(real_grasp_center_x,real_grasp_center_y,blade_right_down_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 4 , cut left down
                                elif(grasp_center_x > blade_center_x and blade_center_y > grasp_center_y): 
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "cut left down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((grasp_center_x - blade_center_x) < 115 and(grasp_center_x - blade_center_x)>=25):
                                        cv2.putText(color_image, "cut left down 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        real_grasp_center_x = (blade_right_down_x + left_up_x)/2.0 
                                        real_grasp_center_y = (blade_left_up_y + right_down_y)/2.0

                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(blade_left_up_x,blade_right_down_y,blade_right_down_x,blade_left_up_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # print(real_grasp_center_x,real_grasp_center_y)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((grasp_center_x - blade_center_x) >= 115):
                                        cv2.putText(color_image, "cut left down 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        real_grasp_center_x = blade_center_x + blade_width/2.0
                                        real_grasp_center_y = blade_center_y
                    
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(blade_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org, (int(real_grasp_center_x - width/2.0), int(real_grasp_center_y - height/2.0)), (int(real_grasp_center_x + width/2.0), int(real_grasp_center_y + height/2.0)),(255, 0, 255), 2)

                                        result = angle(real_grasp_center_x ,real_grasp_center_y,blade_center_x ,blade_center_y)
                                        box = [(real_grasp_center_x + width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                
                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "cut left down 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 40

                                        real_grasp_center_x = blade_center_x 
                                        real_grasp_center_y = blade_center_y - blade_height/2.0

                                        # cv2.circle(img_org,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        rresult = angle(real_grasp_center_x,real_grasp_center_y,blade_left_up_x,blade_right_down_y)
                                        box = [(real_grasp_center_x + width/2.0 ,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y - height/2.0),(real_grasp_center_x - width/2.0,real_grasp_center_y + height/2.0),
                                                (real_grasp_center_x + width/2.0 ,real_grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(real_grasp_center_x),int(real_grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    # cv2.imwrite(center_blade + "blade_grip_"+str(d)+'.jpg',color_image) 
                                    # cv2.imshow("img",color_image)
                                    # cv2.imshow("crop_img",crop_img)
                            else:  
                                #quadrant 1 , right up
                                if( blade_center_x > grasp_center_x and blade_center_y < grasp_center_y):
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right up" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 120 and(blade_center_x - grasp_center_x) >= 25): 
                                        cv2.putText(color_image, "right up 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        # cv2.circle(img_org,(int(grasp_center_x),int(grasp_center_y + height/2)),2,(255,0,0),2)
                                        result = angle(left_up_x ,right_down_y,right_down_x ,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
        
                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) >= 120):
                                        cv2.putText(color_image, "right up 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x +width/2.0), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) < 25):
                                        cv2.putText(color_image, "right up 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 2 , right down
                                elif(blade_center_x > grasp_center_x and blade_center_y > grasp_center_y): #右下角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "right down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    if((blade_center_x - grasp_center_x) < 120 and(blade_center_x - grasp_center_x) >= 25):
                                        cv2.putText(color_image, "right down 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        result = angle(left_up_x,left_up_y,right_down_x,right_down_y) + 90
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y + height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        # aa = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y + height/2))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((blade_center_x - grasp_center_x) >= 120):
                                        cv2.putText(color_image, "right down 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x +width/2), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)

                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                    elif((blade_center_x - grasp_center_x) < 25):
                                        cv2.putText(color_image, "right down 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    
                                #quadrant 3 , left up
                                elif(blade_center_x < grasp_center_x and blade_center_y < grasp_center_y): #左上角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left up" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x) >= 25): 
                                        cv2.putText(color_image, "left up 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 45

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2), int(grasp_center_y)), (int(grasp_center_x+width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,right_down_x,right_down_y)
                                        box = [(grasp_center_x + width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y + height),(grasp_center_x + width/2.0 ,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        # aa = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y + height/2))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        
                                    elif((grasp_center_x - blade_center_x) >= 120):
                                        cv2.putText(color_image, "left up 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 45
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(255, 0, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,right_down_y)
                                        box = [(grasp_center_x + width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y),(grasp_center_x - width/2,grasp_center_y+height),(grasp_center_x + width/2,grasp_center_y + height)]
                                        rota = rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "left up 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                #quadrant 4 , left down
                                elif(grasp_center_x > blade_center_x and blade_center_y > grasp_center_y): #左下角
                                    # cv2.putText(color_image, "distance: " + str(round(blade_center_x - grasp_center_x,3)) , (10, 125), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                    # cv2.putText(color_image, "left down" , (10, 155), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)                         
                                    if((grasp_center_x - blade_center_x) < 120 and(grasp_center_x - blade_center_x) >= 25):
                                        cv2.putText(color_image, "left down 1111 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,right_down_y,right_down_x,left_up_y)
                                        box = [(grasp_center_x + width/2.0 ,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y + height),(grasp_center_x + width/2.0 ,grasp_center_y + height)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]),int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA) 

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) >= 120):
                                        cv2.putText(color_image, "left down 2222 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 20
                                        height = 50
                                        
                                        # cv2.rectangle(img_org, (int(grasp_center_x-width/2.0), int(grasp_center_y)), (int(grasp_center_x + width/2.0), int(grasp_center_y + height)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x + width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y),(grasp_center_x - width/2.0,grasp_center_y+height),(grasp_center_x + width/2.0,grasp_center_y + height)]
                                        rota = rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))

                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                    elif((grasp_center_x - blade_center_x) < 25):
                                        cv2.putText(color_image, "left down 3333 2" , (10, 160), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        width = 50
                                        height = 20

                                        # cv2.rectangle(img_org,(int(grasp_center_x-width), int(grasp_center_y - height/2.0)), (int(grasp_center_x), int(grasp_center_y + height/2.0)),(0, 127, 255), 2)
                                        result = angle(left_up_x,left_up_y,left_up_x+grasp_width,left_up_y)
                                        box = [(grasp_center_x ,grasp_center_y - height/2.0),(grasp_center_x - width,grasp_center_y-height/2.0),(grasp_center_x - width,grasp_center_y+height/2.0),(grasp_center_x ,grasp_center_y + height/2.0)]
                                        rota = rota_rect(box,result,int(grasp_center_x),int(grasp_center_y))
                                        
                                        cv2.line(color_image,(int(rota[0][0]), int(rota[0][1])),(int(rota[1][0]),int(rota[1][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[1][0]),int(rota[1][1])),(int(rota[2][0]),int(rota[2][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[2][0]),int(rota[2][1])),(int(rota[3][0]),int(rota[3][1])),(255, 0, 255),2,cv2.LINE_AA)
                                        cv2.line(color_image,(int(rota[3][0]),int(rota[3][1])),(int(rota[0][0]),int(rota[0][1])),(255, 0, 255),2,cv2.LINE_AA)

                                        real_grasp_center_x = (rota[0][0] + rota[2][0])/2.0 
                                        real_grasp_center_y = (rota[0][1] + rota[2][1])/2.0
                                        cv2.circle(color_image,(int(real_grasp_center_x),int(real_grasp_center_y)),2,(255,0,0),2)
                                        
                                        #depth
                                        z_value = depth_image_zed.get_value(real_grasp_center_x,real_grasp_center_y)

                                        cv2.putText(color_image, "Object: " + str(detections[0][0]) +"," +str(detections[1][0]), (10, 30), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "Depth: " + str(round(z_value[1],3)), (10, 70), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "center: " + "("+ str(round(real_grasp_center_x,3)) +","+ str(round(real_grasp_center_y,3)) + ")", (10, 100), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)
                                        cv2.putText(color_image, "angle: " + str(round(result,3)), (10, 130), cv2.FONT_HERSHEY_DUPLEX,1, (0, 255, 255), 1, cv2.LINE_AA)

                                cv2.imwrite(center_blade + "blade_grip_"+str(d)+'.jpg',color_image) 
                                cv2.imshow("finish",color_image)                   
                                    # cv2.imshow("crop_img222",img_org)
                        else:
                            print("no")
        key = cv2.waitKey(5)                   
        if key == 27:
            break            
    cv2.destroyAllWindows()
    zed.close()













	




	

		

	

