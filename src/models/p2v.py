# TODO: make norm3d module types changeable in temporal branch.
# P2V newbranch F1 SE sigmoid SANSAW
import torch
import torch.nn as nn
import torch.nn.functional as F
from ._attention import *
from ._blocks import Conv1x1, Conv3x3, MaxPool2x2
from .model_utils import BasicBlock, Bottleneck, segmenthead, DAPPM, PAPPM, PagFM, Bag, Light_Bag
from .SANSAW import SAN, SAW
from ._affblock import AFNO2D_channelfirst

class SimpleResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv2(x))

class SimpleResBlocksub1F1(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.fourierfuture = AFNO2D_channelfirst(hidden_size=32, num_blocks=8, sparsity_threshold=0.01,
                                                 hard_thresholding_fraction=1,
                                                 hidden_size_factor=1)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
        self.norm = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv1(x)
        Fo = self.fourierfuture(x)
        Fo = self.norm(Fo)
        return F.relu(x + Fo)

class SimpleResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 16, out_ch, bias=False),
        )
        self.conv3 = Conv1x1(out_ch*2, out_ch, norm=True)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)
        x = self.conv2(res)
        b, c, _, _ = x.size()
        x_avg = self.avg_pool(x).view(b, c)
        x_max = self.max_pool(x).view(b, c)
        attention = torch.cat([x_avg, x_max], dim=1).view(b, c*2, 1, 1)
        attention = self.sig(self.conv3(attention))
        # x_avg = self.fc(x_avg).view(b, c, 1, 1)
        # x_max = self.fc(x_max).view(b, c, 1, 1)
        # attention = self.sig(x_avg + x_max)
        x = x * attention.expand_as(x)
        return F.relu(res + x)

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)

    def forward(self, x):
        x = self.conv1(x)
        return F.relu(x + self.conv3(self.conv2(x)))

class ResBlockSE(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = Conv3x3(in_ch, out_ch, norm=True, act=True)
        self.conv2 = Conv3x3(out_ch, out_ch, norm=True, act=True)
        self.conv3 = Conv3x3(out_ch, out_ch, norm=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_ch, out_ch // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_ch // 16, out_ch, bias=False),
        )
        self.sig = nn.Sigmoid()
        self.conv4 = Conv1x1(out_ch * 2, out_ch, norm=True)

    def forward(self, x):
        res = self.conv1(x)
        x = self.conv3(self.conv2(res))
        b, c, _, _ = x.size()
        x_avg = self.avg_pool(x).view(b,c)
        x_max = self.max_pool(x).view(b,c)
        # x_avg = self.fc(x_avg).view(b, c, 1, 1)
        # x_max = self.fc(x_max).view(b, c, 1, 1)
        # attention = self.sig(x_avg + x_max)
        attention = torch.cat([x_avg, x_max], dim=1).view(b, c * 2, 1, 1)
        attention = self.sig(self.conv4(attention))
        x = x * attention.expand_as(x)
        return F.relu(res + x)

class DecBlock(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch):
        super().__init__()
        self.conv_fuse = SimpleResBlock(in_ch1 + in_ch2, out_ch)

    def forward(self, x1, x2):
        x2 = F.interpolate(x2, size=x1.shape[2:])
        x = torch.cat([x1, x2], dim=1)
        return self.conv_fuse(x)


class BasicConv3D(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size,
            bias='auto',
            bn=False, act=False,
            **kwargs
    ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append(nn.ConstantPad3d(kernel_size // 2, 0.0))
        seq.append(
            nn.Conv3d(
                in_ch, out_ch, kernel_size,
                padding=0,
                bias=(False if bn else True) if bias == 'auto' else bias,
                **kwargs
            )
        )
        if bn:
            seq.append(nn.BatchNorm3d(out_ch))
        if act:
            seq.append(nn.ReLU())
        self.seq = nn.Sequential(*seq)

    def forward(self, x):
        return self.seq(x)


class Conv3x3x3(BasicConv3D):
    def __init__(self, in_ch, out_ch, bias='auto', bn=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, 3, bias=bias, bn=bn, act=act, **kwargs)


class ResBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, itm_ch, stride=1, ds=None):
        super().__init__()
        self.conv1 = BasicConv3D(in_ch, itm_ch, 1, bn=True, act=True, stride=stride)
        self.conv2 = Conv3x3x3(itm_ch, itm_ch, bn=True, act=True)
        self.conv3 = BasicConv3D(itm_ch, out_ch, 1, bn=True, act=False)
        self.ds = ds

    def forward(self, x):
        res = x
        y = self.conv1(x)
        y = self.conv2(y)
        y = self.conv3(y)
        if self.ds is not None:
            res = self.ds(res)
        y = F.relu(y + res)
        return y


class PairEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(16, 32, 64), add_chs=(0, 0), attention_type_3='local', atten_k=7):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlock(2 * in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlock(enc_chs[0] + add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlockSE((enc_chs[1] + add_chs[1])//2, enc_chs[2]//2)
        self.pool3 = MaxPool2x2()

        self.attention_type_3 = attention_type_3
        self.middle_dim_3 = 64

        if attention_type_3 == 'global':
            self.fuse_attention_3 = MyAttentionGlobal(self.middle_dim_3, kScale=atten_k)
        elif attention_type_3 == 'local':
            self.fuse_attention_3 = MyAttention(self.middle_dim_3, kH=atten_k, kW=atten_k)

    def forward(self, x1, x2, add_feats=None):
        x = torch.cat([x1, x2], dim=1)
        feats = [x]

        x = self.conv1(x)
        x = self.pool1(x)
        feats.append(x)

        add_feat = F.interpolate(add_feats[0], size=x.shape[2:])
        x = torch.cat([x, add_feat], dim=1)
        x = self.conv2(x)
        x = self.pool2(x)
        feats.append(x)

        # 第三个block高低分辨率
        # 将spatial部分分为关键帧和非关键帧
        hr_S3_input_s = x[:, :32, :, :]  # 8,32,64,64
        lr_S3_input_s = x[:, 32:, :, :]  # 8,32,64,64
        lr_S3_input_s = F.interpolate(lr_S3_input_s, size=(32, 32), mode='bilinear', align_corners=True)  # 8,32,32,32
        # 将temporal部分分为关键帧和非关键帧
        add_feat = F.interpolate(add_feats[1], size=x.shape[2:])  # 8,512,64,64
        hr_S3_input_t = add_feat[:, :256, :, :]  # 8,256,64,64
        lr_S3_input_t = add_feats[1][:, 256:, :, :]  # 8,256,32,32
        # 高分辨率经过S-block2处理
        hr_S3_input = torch.cat([hr_S3_input_s, hr_S3_input_t], dim=1)  # 8,288,64,64
        hr_S3_output = self.conv3(hr_S3_input)  # 8,64,64,64
        # hr_S3_output_pool = self.pool3(hr_S3_output)  # 8,64,32,32
        hr_S3_output = self.pool3(hr_S3_output)  # 8,64,32,32
        # 低分辨率经过S-block2处理
        lr_S3_input = torch.cat([lr_S3_input_s, lr_S3_input_t], dim=1)  # 8,288,32,32
        lr_S3_output = self.conv3(lr_S3_input)
        # hr和lr特征融合 为图像高分辨率和低分辨率之间计算一个注意力权重
        lr_S1_feature_fuse1 = self.fuse_attention_3(hr_S3_output, lr_S3_output)  # 8,64,32,32
        output_tensor2 = torch.cat([hr_S3_output, lr_S1_feature_fuse1], dim=1)  # tensor 8,128,32,32
        feats.append(output_tensor2)

        return feats


class VideoEncoder(nn.Module):
    def __init__(self, in_ch, enc_chs=(64, 128)):
        super().__init__()
        if in_ch != 3:
            raise NotImplementedError

        self.n_layers = 2
        self.expansion = 4
        self.tem_scales = (1.0, 0.5)

        self.stem = nn.Sequential(
            nn.Conv3d(3, enc_chs[0], kernel_size=(3, 9, 9), stride=(1, 4, 4), padding=(1, 4, 4), bias=False),
            nn.BatchNorm3d(enc_chs[0]),
            nn.ReLU()
        )
        exps = self.expansion
        self.layer1 = nn.Sequential(
            ResBlock3D(
                enc_chs[0],
                enc_chs[0] * exps,
                enc_chs[0],
                ds=BasicConv3D(enc_chs[0], enc_chs[0] * exps, 1, bn=True)
            ),
            ResBlock3D(enc_chs[0] * exps, enc_chs[0] * exps, enc_chs[0])
        )
        self.layer2 = nn.Sequential(
            ResBlock3D(
                enc_chs[0] * exps,
                enc_chs[1] * exps,
                enc_chs[1],
                stride=(2, 2, 2),
                ds=BasicConv3D(enc_chs[0] * exps, enc_chs[1] * exps, 1, stride=(2, 2, 2), bn=True)
            ),
            ResBlock3D(enc_chs[1] * exps, enc_chs[1] * exps, enc_chs[1])
        )

    def forward(self, x):
        feats = [x]

        x = self.stem(x)
        for i in range(self.n_layers):
            layer = getattr(self, f'layer{i + 1}')
            x = layer(x)
            feats.append(x)

        return feats


class SimpleDecoder(nn.Module):
    def __init__(self, itm_ch, enc_chs, dec_chs):
        super().__init__()

        enc_chs = enc_chs[::-1]
        self.conv_bottom = Conv3x3(itm_ch, itm_ch, norm=True, act=True)
        self.blocks = nn.ModuleList([
            DecBlock(in_ch1, in_ch2, out_ch)
            for in_ch1, in_ch2, out_ch in zip(enc_chs, (itm_ch,) + dec_chs[:-1], dec_chs)
        ])
        self.conv_out = Conv1x1(dec_chs[-1], 1)

    def forward(self, x, feats):
        feats = feats[::-1]

        x = self.conv_bottom(x)
        a = []
        for feat, blk in zip(feats, self.blocks):
            x = blk(feat, x)
            a.append(x)
        b = a[0]
        y = self.conv_out(x)

        return y, b

bn_mom = 0.1
algc = False
class BoundaryEncoder(nn.Module):
    def __init__(self, in_ch, planes=32):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2*in_ch, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
            nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(planes, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )

        # self.conv1 = SimpleResBlock(2*in_ch, enc_chs[0])
        self.layer_d1 = self._make_single_layer(BasicBlock, planes, planes)
        self.diff1 = nn.Sequential(
            nn.Conv2d(planes * 8, planes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes, momentum=bn_mom),
        )
        # self.layer_d2 = self._make_single_layer(BasicBlock, planes, planes)
        self.layer_d2 = self._make_layer(Bottleneck, planes, planes*2, 1)
        self.diff2 = nn.Sequential(
            nn.Conv2d(planes * 16, planes*2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(planes*2, momentum=bn_mom),
        )
        # self.layer_d3 = self._make_layer(Bottleneck, planes, planes, 1)
        self.relu = nn.ReLU()
        # Prediction Head
    def forward(self, x1, x2, add_feats=None):
        x = torch.cat([x1,x2], dim=1) # 8,6,256,256

        x = self.conv1(x)  # 8,32,64,64
        x_d = self.layer_d1(x)  # 8,32,64,64
        feats = [x_d]
        # add videoencoder
        x_d = x_d + self.diff1(add_feats[0])  # 8,32,64,64
        x_d = self.layer_d2(self.relu(x_d))  # # 8,64,64,64
        feats.append(x_d)
        x_d = x_d + F.interpolate(
                        self.diff2(add_feats[1]),
                        size=[64, 64],
                        mode='bilinear', align_corners=algc)
        feats.append(x_d)
        return feats

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):  # 创建网络的一个层（或模块）。这个层通常由多个相同类型的块（block）组成
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:  # 根据 stride 和输入输出通道数的关系来确定是否需要添加一个downsample分支
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)

        return layer

class PairEncoderwithboundary(nn.Module):
    def __init__(self, in_ch, enc_chs=(16,32,64), add_chs=(0,0), attention_type_3='local', atten_k=7):
        super().__init__()

        self.n_layers = 3

        self.conv1 = SimpleResBlocksub1F1(2*in_ch, enc_chs[0])
        self.pool1 = MaxPool2x2()

        self.conv2 = SimpleResBlockSE(enc_chs[0]+add_chs[0], enc_chs[1])
        self.pool2 = MaxPool2x2()

        self.conv3 = ResBlockSE(64, enc_chs[2])
        self.pool3 = MaxPool2x2()

        self.middle_dim_2 = 32
        self.middle_dim_3 = 64
        self.attention_type_3 = attention_type_3
        self.bag = Bag(64, 64)
        self.downchannel_v = DAPPM(512, 128, 64)
        # self.downchannel_v_1 = PAPPM(256, 64, 32)
        self.diff1 = nn.Sequential(
            nn.Conv2d(256, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=bn_mom),
        )
        self.SAN_stage_1 = SAN(inplanes=32, selected_classes=[0,1])
        self.classifier_1 = nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True)
        self.SAW_stage_1 = SAW(dim=32, relax_denom=2.0, classifier=self.classifier_1)

        if attention_type_3 == 'global':
            self.fuse_attention_3 = MyAttentionGlobal(self.middle_dim_3, kScale=atten_k)
        elif attention_type_3 == 'local':
            self.fuse_attention_3 = MyAttention(self.middle_dim_3, kH=atten_k, kW=atten_k)

    def save_feature_map(self, grad):
        self.feature_map = grad

    def forward(self, x1, x2, add_feats_v=None, add_feats_b=None):
        x = torch.cat([x1,x2], dim=1)
        feats = [x] # 8,16,256,256
        x = self.conv1(x)
        x = self.pool1(x) # 8,32,128,128
        vis = x
        x1 = self.classifier_1(x.detach())   # 8,2,128,128
        # 分类结果
        x = self.SAN_stage_1(x, x1)  # 8,32,128,128
        saw_loss_lay1 = self.SAW_stage_1(x)
        feats.append(x)

        add_feat = F.interpolate(add_feats_v[0], size=x.shape[2:])
        x = torch.cat([x, add_feat], dim=1)
        x = self.conv2(x)
        x = self.pool2(x)  # 8,32,128,128
        feats.append(x)  # 8,64,64,64

        # 改变第三个block 为一高一低分辨率
        # 改变第三个block 为一高一低分辨率
        # 将spatial部分分为关键帧和非关键帧
        input_t = self.downchannel_v(add_feats_v[1]) # 8,64,32,32
        add_feat = F.interpolate(input_t, size=x.shape[2:])
        input_b = add_feats_b[1]  # 8,64,64,64

        # B引导S和V融合
        boundary_diffusion_1 = self.bag(x, add_feat, input_b)  # 8,32,64,64

        # 高分辨率经过S-block2处理
        hr_S3_output = self.conv3(boundary_diffusion_1)  # 8,64,64,64
        hr_S3_output = self.pool3(hr_S3_output)  # 8,128,32,32
        # x1 = self.classifier_1(hr_S3_output.detach())   # 8,2,32,32
        # # 分类结果
        # hr_S3_output = self.SAN_stage_1(hr_S3_output, x1)  # 8,128,32,32
        # saw_loss_lay1 = self.SAW_stage_1(hr_S3_output)
        feats.append(hr_S3_output)
        return feats, saw_loss_lay1, vis, boundary_diffusion_1

class segmenthead(nn.Module):
    def __init__(self, inplanes, interplanes, outplanes, scale_factor=None):
        super(segmenthead, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, momentum=bn_mom)
        self.conv1 = nn.Conv2d(inplanes, interplanes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(interplanes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(interplanes, outplanes, kernel_size=1, padding=0, bias=True)
        self.scale_factor = scale_factor
    def forward(self, x):
        x = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(x)))
        if self.scale_factor is not None:
            height = x.shape[-2] * self.scale_factor
            width = x.shape[-1] * self.scale_factor
            out = F.interpolate(out,
                                size=[height, width],
                                mode='bilinear', align_corners=algc)
        return out

class P2VNet(nn.Module):
    def __init__(self, in_ch, video_len=8, enc_chs_p=(32, 64, 128), enc_chs_v=(64, 128), dec_chs=(256, 128, 64, 32)):
        super().__init__()
        if video_len < 2:
            raise ValueError
        self.video_len = video_len
        self.encoder_v = VideoEncoder(in_ch, enc_chs=enc_chs_v)
        enc_chs_v = tuple(ch * self.encoder_v.expansion for ch in enc_chs_v)
        self.encoder_b = BoundaryEncoder(in_ch)
        self.encoder_p = PairEncoderwithboundary(in_ch, enc_chs=enc_chs_p, add_chs=enc_chs_v)
        self.conv_out_v = Conv1x1(enc_chs_v[-1], 1)
        self.convs_video = nn.ModuleList(
            [
                Conv1x1(2 * ch, ch, norm=True, act=True)
                for ch in enc_chs_v
            ]
        )
        self.decoder = SimpleDecoder(enc_chs_p[-1], (2 * in_ch,) + enc_chs_p, dec_chs)
        self.pred_b_head = segmenthead(64, 64, 1)

    def forward(self, t1, t2, return_aux=True):
        frames = self.pair_to_video(t1, t2)
        feats_v = self.encoder_v(frames.transpose(1, 2))
        feats_v.pop(0)

        for i, feat in enumerate(feats_v):
            feats_v[i] = self.convs_video[i](self.tem_aggr(feat))

        feats_b = self.encoder_b(t1, t2, feats_v)
        feats_p, lossaaa, vis, boundary_diffusion_1 = self.encoder_p(t1, t2, feats_v, feats_b)

        pred, vis1 = self.decoder(feats_p[-1], feats_p)

        if return_aux:
            pred_v = self.conv_out_v(feats_v[-1])
            pred_v = F.interpolate(pred_v, size=pred.shape[2:])
            pred_b = self.pred_b_head(feats_b[-1])
            pred_b = F.interpolate(pred_b, size=pred.shape[2:])
            return pred, pred_v, pred_b, lossaaa, vis1  # 64
            # return pred, pred_v, pred_b, lossaaa, feats_v[-1]  # 256 512
            # return pred, pred_v, pred_b, lossaaa, feats_b[-2]  # 64
        else:
            return pred

    def pair_to_video(self, im1, im2, rate_map=None):
        def _interpolate(im1, im2, rate_map, len):
            delta = 1.0 / (len - 1)
            delta_map = rate_map * delta
            steps = torch.arange(len, dtype=torch.float, device=delta_map.device).view(1, -1, 1, 1, 1)
            interped = im1.unsqueeze(1) + ((im2 - im1) * delta_map).unsqueeze(1) * steps
            return interped

        if rate_map is None:
            rate_map = torch.ones_like(im1[:, 0:1])
        frames = _interpolate(im1, im2, rate_map, self.video_len)
        return frames

    def tem_aggr(self, f):
        return torch.cat([torch.mean(f, dim=2), torch.max(f, dim=2)[0]], dim=1)