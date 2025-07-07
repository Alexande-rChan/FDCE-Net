import torch
import torch.nn as nn
import torch.nn.functional as F
from color_encoder import *

class fdce_net(nn.Module):

    def __init__(self, d_hist=64):
        super(fdce_net, self).__init__()
        self.fs_net = fs_net()
        self.dc_net = dc_net(d_hist)
        self.f_net = f_net()

    def forward(self, img_low):
        pre_enhancement = self.fs_net(img_low)
        color_hist, color_feature = self.dc_net(img_low)
        img_enhance = self.f_net(img_low, pre_enhancement, color_feature)

        return pre_enhancement, color_hist, img_enhance


class f_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(f_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[FRB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[FRB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[FRB(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])

        # Middle
        self.middle = nn.Sequential(*[FRB(base_channel * 8) for _ in range(depth[3])])

        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[FRB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[FRB(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[FRB(base_channel) for _ in range(depth[0])]),
        ])

        # conv
        self.conv_first = BasicConv(6, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)
        self.color_decoder = MultiScaleColorEncoder(
            in_channels=[32, 64, 128],
            num_queries=256,
            num_scales=3,
            dec_layers=9,
        )

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, img_low, pre_enhancement, color_shortcuts):
        x = torch.cat([img_low, pre_enhancement], 1)
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x = self.middle(x)
        fusion_feature = self.color_decoder(color_shortcuts, x)
        x = self.decoder(fusion_feature, shortcuts)
        x = self.conv_last(x)
        img_color = (torch.tanh(x) + 1) / 2
        return img_color


class dc_net(nn.Module):

    def __init__(self, d_hist, depth=[2, 2, 2]):
        super(dc_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[FRB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[FRB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[FRB(base_channel * 4) for _ in range(depth[2])]),
        ])

        self.conv_first = BasicConv(3, base_channel, 3, 1)

        # color hist
        self.conv_color = BasicConv(base_channel * 4, 256 * 3, 3, 1)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, d_hist)  ###### 64
        self.softmax = nn.Softmax(dim=2)

        self.d_hist = d_hist

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def color_forward(self, x):
        x = self.conv_color(x)
        x = self.pooling(x)
        x = torch.reshape(x, (-1, 3, 256))
        color_hist = self.softmax(self.fc(x))
        return color_hist

    def forward(self, x):

        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        color_hist = self.color_forward(x)

        return color_hist, shortcuts


class fs_net(nn.Module):

    def __init__(self, depth=[2, 2, 2, 2]):
        super(fs_net, self).__init__()

        base_channel = 32

        # encoder
        self.Encoder = nn.ModuleList([
            BasicConv(base_channel, base_channel, 3, 1),
            nn.Sequential(*[FRB(base_channel) for _ in range(depth[0])]),
            Down_scale(base_channel),
            BasicConv(base_channel * 2, base_channel * 2, 3, 1),
            nn.Sequential(*[FRB(base_channel * 2) for _ in range(depth[1])]),
            Down_scale(base_channel * 2),
            BasicConv(base_channel * 4, base_channel * 4, 3, 1),
            nn.Sequential(*[FRB(base_channel * 4) for _ in range(depth[2])]),
            Down_scale(base_channel * 4),
        ])

        # Middle
        self.middle = nn.Sequential(*[FRB(base_channel * 8) for _ in range(depth[3])])

        # decoder
        self.Decoder = nn.ModuleList([
            Up_scale(base_channel * 8),
            BasicConv(base_channel * 8, base_channel * 4, 3, 1),
            nn.Sequential(*[FRB(base_channel * 4) for _ in range(depth[2])]),
            Up_scale(base_channel * 4),
            BasicConv(base_channel * 4, base_channel * 2, 3, 1),
            nn.Sequential(*[FRB(base_channel * 2) for _ in range(depth[1])]),
            Up_scale(base_channel * 2),
            BasicConv(base_channel * 2, base_channel, 3, 1),
            nn.Sequential(*[FRB(base_channel) for _ in range(depth[0])]),

        ])

        # conv
        self.conv_first = BasicConv(3, base_channel, 3, 1)
        self.conv_last = nn.Conv2d(base_channel, 3, 3, 1, 1)

    def encoder(self, x):
        shortcuts = []
        for i in range(len(self.Encoder)):
            x = self.Encoder[i](x)
            if (i + 2) % 3 == 0:
                shortcuts.append(x)
        return x, shortcuts

    def decoder(self, x, shortcuts):
        for i in range(len(self.Decoder)):
            if (i + 2) % 3 == 0:
                index = len(shortcuts) - (i // 3 + 1)
                x = torch.cat([x, shortcuts[index]], 1)
            x = self.Decoder[i](x)
        return x

    def forward(self, x):
        x = self.conv_first(x)
        x, shortcuts = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x, shortcuts)
        x = self.conv_last(x)
        pre_enhancement = (torch.tanh(x) + 1) / 2
        return pre_enhancement


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=True, activation=True,
                 transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.InstanceNorm2d(out_channel))
        if activation:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class FreBlock(nn.Module):
    def __init__(self, nc):
        super(FreBlock, self).__init__()
        self.processmag = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))
        self.processpha = nn.Sequential(
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(nc, nc, 1, 1, 0))

    def forward(self, x):
        mag = torch.abs(x)
        pha = torch.angle(x)
        mag = self.processmag(mag)
        pha = self.processpha(pha)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)

        return x_out


class FRB(nn.Module):
    def __init__(self, channels):
        super(FRB, self).__init__()
        self.layer_1 = BasicConv(channels//2, channels//2, 3, 1,norm=False)
        self.layer_2 = BasicConv(channels//2, channels//2, 3, 1,norm=False)
        self.freblobk = FreBlock(channels//2)


    def forward(self, x):
        x1,x2 = torch.chunk(x,2,dim=1)
        _, _, H, W = x1.shape
        x_freq = torch.fft.rfft2(x1, norm='backward')
        x_freq = self.freblobk(x_freq)
        x_freq_spatial = torch.fft.irfft2(x_freq, s=(H, W), norm='backward')
        x_freq_spatial = x1+x_freq_spatial

        x_spatial = self.layer_1(x2)
        x_spatial = self.layer_2(x_spatial)
        x_spatial = x2 +x_spatial

        y = torch.cat([x_freq_spatial, x_spatial], dim=1)
        return y


class RB(nn.Module):
    def __init__(self, channels):
        super(RB, self).__init__()
        self.layer_1 = BasicConv(channels, channels, 3, 1)
        self.layer_2 = BasicConv(channels, channels, 3, 1)

    def forward(self, x):
        y = self.layer_1(x)
        y = self.layer_2(y)
        return y + x


class Down_scale(nn.Module):
    def __init__(self, in_channel):
        super(Down_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel * 2, 3, 2)

    def forward(self, x):
        return self.main(x)


class Up_scale(nn.Module):
    def __init__(self, in_channel):
        super(Up_scale, self).__init__()
        self.main = BasicConv(in_channel, in_channel // 2, kernel_size=4, activation=True, stride=2, transpose=True)

    def forward(self, x):
        return self.main(x)



