import torch.nn as nn
from torch.nn import functional as F
import torch
import math
from .tokenmixer.non_linear_layer import get_activation_fn
from typing import Optional

class AFNO2D_channelfirst(nn.Module):
    """
    hidden_size: channel dimension size
    num_blocks: how many blocks to use in the block diagonal weight matrices (higher => less complexity but less parameters)
    sparsity_threshold: lambda for softshrink
    hard_thresholding_fraction: how many frequencies you want to completely mask out (lower => hard_thresholding_fraction^2 less FLOPs)
    input shape [B N C]
    """

    def __init__(self, hidden_size, num_blocks=8, sparsity_threshold=0.01, hard_thresholding_fraction=1,
                 hidden_size_factor=1):
        super().__init__()
        assert hidden_size % num_blocks == 0, f"hidden_size {hidden_size} should be divisble by num_blocks {num_blocks}"

        self.hidden_size = hidden_size
        self.sparsity_threshold = 0.01
        self.num_blocks = num_blocks  # 8
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(
            self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))
        # self.norm_layer1 = get_normalization_layer(opts=opts, num_features=out_channels)
        self.act = self.build_act_layer()
        self.act2 = self.build_act_layer()

    @staticmethod
    def build_act_layer(act_type="relu", inplace=True, neg_slope=0.1) -> nn.Module:
        # act_type = getattr(opts, "model.activation.name", "relu")
        # neg_slope = getattr(opts, "model.activation.neg_slope", 0.1)
        # inplace = getattr(opts, "model.activation.inplace", False)

        act_layer = get_activation_fn(
            act_type=act_type,
            inplace=inplace,
            negative_slope=neg_slope,
            num_parameters=1,
        )
        return act_layer

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x, spatial_size=None):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, C, H, W = x.shape  # 8,32,128,128
        # x = self.fu(x)

        x = torch.fft.rfft2(x, dim=(2, 3), norm="ortho")  # 8,32,128,65
        origin_ffted = x  # 8,32,128,65 对B, C, H, W 进行傅立叶变换
        x = x.reshape(B, self.num_blocks, self.block_size, x.shape[2], x.shape[3])  # 8,8,4,128,65


        o1_real = self.act(
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[0]) - \
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[1]) + \
            self.b1[0, :, :, None, None]     # 8,8,4,128,65
        )

        o1_imag = self.act2(
            torch.einsum('bkihw,kio->bkohw', x.imag, self.w1[0]) + \
            torch.einsum('bkihw,kio->bkohw', x.real, self.w1[1]) + \
            self.b1[1, :, :, None, None]
        )

        o2_real = (
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[0]) - \
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[1]) + \
                self.b2[0, :, :, None, None]
        )

        o2_imag = (
                torch.einsum('bkihw,kio->bkohw', o1_imag, self.w2[0]) + \
                torch.einsum('bkihw,kio->bkohw', o1_real, self.w2[1]) + \
                self.b2[1, :, :, None, None]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, C, x.shape[3], x.shape[4])

        x = x * origin_ffted
        x = torch.fft.irfft2(x, s=(H, W), dim=(2, 3), norm="ortho")
        x = x.type(dtype)

        return x + bias

    # def profile_module(
    #         self, input: Tensor, *args, **kwargs
    #     ) -> Tuple[Tensor, float, float]:
    #     # TODO: to edit it
    #     b_sz, c, h, w = input.shape
    #     seq_len = h * w
    #
    #     # FFT iFFT
    #     p_ff, m_ff = 0, 5 * b_sz * seq_len * int(math.log(seq_len)) * c
    #     # others
    #     # params = macs = sum([p.numel() for p in self.parameters()])
    #     params = macs = self.hidden_size * self.hidden_size_factor * self.hidden_size * 2 * 2 // self.num_blocks
    #     # // 2 min n become half after fft
    #     macs = macs * b_sz * seq_len
    #
    #     # return input, params, macs
    #     return input, params, macs + m_ff