import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_arch.RRDBNet_arch import RRDBNet


def rgb_to_yuv(img):
    """Tensor shape (B, 3, H, W) in range [0, 1]"""
    r, g, b = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    y =  0.29900 * r + 0.58700 * g + 0.11400 * b
    u = -0.14713 * r - 0.28886 * g + 0.43600 * b
    v =  0.61500 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)

def yuv_to_rgb(img):
    """Tensor shape (B, 3, H, W) - YUV"""
    y, u, v = img[:, 0:1, :, :], img[:, 1:2, :, :], img[:, 2:3, :, :]
    r = y + 1.13983 * v
    g = y - 0.39465 * u - 0.58060 * v
    b = y + 2.03211 * u
    return torch.cat([r, g, b], dim=1)


class YUV_Generator(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=64, nb=23, gc=32, scale=2):
        super(YUV_Generator, self).__init__()

        self.y_net = RRDBNet(in_nc=1, out_nc=1, nf=nf, nb=nb, gc=gc)
        self.scale = scale

    def forward(self, rgb_lr):
        yuv_lr = rgb_to_yuv(rgb_lr)
        y_lr = yuv_lr[:, 0:1, :, :]  # Канал яркости [B, 1, H, W]
        uv_lr = yuv_lr[:, 1:3, :, :] # Каналы цвета [B, 2, H, W]

        y_hr = self.y_net(y_lr)

        uv_hr = F.interpolate(uv_lr, size=(y_hr.size(2), y_hr.size(3)), mode='bilinear')
        yuv_hr = torch.cat([y_hr, uv_hr], dim=1)
        rgb_hr = yuv_to_rgb(yuv_hr)

        return rgb_hr