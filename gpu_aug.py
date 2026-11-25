"""Batch augmentation that runs on the GPU.

The data-loader workers only decode, crop and flip each image (cheap) and hand
over uint8 tensors. The expensive steps (rotation, colour jitter, normalisation,
random erasing) run here on a whole batch at once. Every image still gets its
own random parameters, exactly like per-image CPU augmentation.

Works on any device; on a CPU-only machine the CPU pipeline in data.py is used
instead because it parallelises across loader workers.
"""
import math

import torch
import torch.nn.functional as F

from data import IMAGENET_MEAN, IMAGENET_STD

# luma weights used by torchvision for grayscale / saturation
_GRAY = torch.tensor([0.299, 0.587, 0.114])
# RGB <-> YIQ, used for a small hue rotation
_RGB2YIQ = torch.tensor([[0.299, 0.587, 0.114],
                         [0.596, -0.274, -0.322],
                         [0.211, -0.523, 0.312]])
_YIQ2RGB = torch.linalg.inv(_RGB2YIQ)


def _uniform(n, low, high, device):
    return torch.empty(n, device=device).uniform_(low, high)


class GPUAugment:
    """Call ``aug(batch_uint8, train=True)`` to get a normalised float batch."""

    def __init__(self, rotation=20.0, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02,
                 erase_p=0.2, erase_scale=(0.02, 0.33), erase_ratio=(0.3, 3.3)):
        self.rotation = rotation
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.erase_p = erase_p
        self.erase_scale = erase_scale
        self.erase_ratio = erase_ratio
        self._cache = {}

    def _const(self, name, tensor, device):
        key = (name, device)
        if key not in self._cache:
            self._cache[key] = tensor.to(device)
        return self._cache[key]

    def __call__(self, x, train=True):
        dev = x.device
        x = x.float().div_(255)
        if train:
            x = self._rotate(x)
            x = self._color(x)
        mean = self._const("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), dev)
        std = self._const("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), dev)
        x = (x - mean) / std
        if train and self.erase_p > 0:
            x = self._erase(x)
        return x

    def _rotate(self, x):
        b = x.shape[0]
        angle = _uniform(b, -self.rotation, self.rotation, x.device) * (math.pi / 180)
        cos, sin = torch.cos(angle), torch.sin(angle)
        zero = torch.zeros_like(cos)
        theta = torch.stack([torch.stack([cos, -sin, zero], 1),
                             torch.stack([sin, cos, zero], 1)], 1)
        grid = F.affine_grid(theta, list(x.shape), align_corners=False)
        # corners that rotate in from outside are black, like torchvision's default fill
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)

    def _color(self, x):
        b, dev = x.shape[0], x.device
        gray_w = self._const("gray", _GRAY.view(1, 3, 1, 1), dev)

        f = _uniform(b, 1 - self.brightness, 1 + self.brightness, dev).view(b, 1, 1, 1)
        x = (x * f).clamp_(0, 1)

        f = _uniform(b, 1 - self.contrast, 1 + self.contrast, dev).view(b, 1, 1, 1)
        m = (x * gray_w).sum(1, keepdim=True).mean((2, 3), keepdim=True)
        x = ((x - m) * f + m).clamp_(0, 1)

        f = _uniform(b, 1 - self.saturation, 1 + self.saturation, dev).view(b, 1, 1, 1)
        g = (x * gray_w).sum(1, keepdim=True)
        x = ((x - g) * f + g).clamp_(0, 1)

        if self.hue:
            # rotate the colour (I, Q) plane: a hue shift of h turns it by 2*pi*h
            a = _uniform(b, -self.hue, self.hue, dev) * (2 * math.pi)
            cos, sin = torch.cos(a), torch.sin(a)
            one, zero = torch.ones_like(cos), torch.zeros_like(cos)
            rot = torch.stack([torch.stack([one, zero, zero], 1),
                               torch.stack([zero, cos, -sin], 1),
                               torch.stack([zero, sin, cos], 1)], 1)          # (b, 3, 3)
            m = self._const("yiq2rgb", _YIQ2RGB, dev) @ rot @ self._const("rgb2yiq", _RGB2YIQ, dev)
            x = torch.einsum("bij,bjhw->bihw", m, x).clamp_(0, 1)
        return x

    def _erase(self, x):
        b, _, h, w = x.shape
        dev = x.device
        apply = torch.rand(b, device=dev) < self.erase_p
        if not apply.any():
            return x
        area = _uniform(b, *self.erase_scale, dev) * (h * w)
        log_r = _uniform(b, math.log(self.erase_ratio[0]), math.log(self.erase_ratio[1]), dev)
        ratio = torch.exp(log_r)
        eh = torch.sqrt(area * ratio).round().clamp(1, h)
        ew = torch.sqrt(area / ratio).round().clamp(1, w)
        top = (torch.rand(b, device=dev) * (h - eh + 1)).floor()
        left = (torch.rand(b, device=dev) * (w - ew + 1)).floor()
        ys = torch.arange(h, device=dev).view(1, h, 1)
        xs = torch.arange(w, device=dev).view(1, 1, w)
        box = ((ys >= top.view(b, 1, 1)) & (ys < (top + eh).view(b, 1, 1)) &
               (xs >= left.view(b, 1, 1)) & (xs < (left + ew).view(b, 1, 1)))
        mask = (box & apply.view(b, 1, 1)).unsqueeze(1)
        return x.masked_fill(mask, 0.0)
