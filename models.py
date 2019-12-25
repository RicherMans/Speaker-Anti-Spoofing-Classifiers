import torch
import torch.nn as nn
import torch.nn.functional as F


class MFM(nn.Module):
    """
    Max Feature Map (MFM) Activation:

    Takes in a tensor of size 
    :math:`(N, C_{\text{in}}, H, W)` 
    and output 
    :math:`(N, C_{\text{out}}, H_{\text{out}}, W_{\text{out}})`
    by calculating:
    :math:`max((N, C_{\text{out}}1//2, H_{\text{out}}, W_{\text{out}}), :math:`(N, C_{\text{out}}2//2, H_{\text{out}}, W_{\text{out}})`)`


    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.filter = nn.Sequential(
            nn.Conv2d(in_channels,
                      2 * out_channels,
                      kernel_size=kernel_size,
                      padding=padding,
                      stride=stride))
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.filter(x)
        low, up = torch.chunk(x, 2, 1)
        return self.norm(torch.max(low, up))


class MFMGroup(nn.Module):
    """
    A Linear MFM followed by a kernel_size MFM
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 padding=1,
                 stride=1):
        super().__init__()
        self.group = nn.Sequential(
            MFM(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                padding=0,
                stride=1),
            MFM(in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride))

    def forward(self, x):
        return self.group(x)


class MovingAvgNorm(nn.Module):
    """
    Calculates multiple moving average estiamtes given a kernel_size
    Similar to kaldi's apply-cmvn-sliding 
    """
    def __init__(self, kernel_size=100, with_mean=True, with_std=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.with_mean = with_mean
        self.with_std = with_std
        self.eps = 1e-12

    def forward(self, x):
        assert x.ndim == 3, "Input needs to be tensor of shape B x T x D"
        n_batch, timedim, featdim = x.shape
        # Too small utterance, just normalize per time-step
        if timedim < self.kernel_size:
            return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) +
                                                    self.eps)
        else:
            sliding_window = F.pad(
                x.transpose(1, 2),
                (self.kernel_size // 2, self.kernel_size // 2 - 1),
                mode='reflect').unfold(-1, self.kernel_size,
                                       1).permute(0, 2, 1, 3).contiguous()
        m = sliding_window.mean(-1) if self.with_mean else torch.zeros_like(
            x)  # Mean estimate for each window
        v = sliding_window.std(-1) if self.with_std else torch.ones_like(
            x)  #Std estiamte for each window
        return (x - m) / (v + self.eps)


class LightCNN(nn.Module):
    def __init__(self, inputdim, outputdim, **kwargs):
        super().__init__()
        self._filtersizes = kwargs.get('filtersizes', [3, 3, 3, 3, 3])
        self._filter = [1] + kwargs.get('filter', [16, 24, 32, 16, 16])
        self._pooling = kwargs.get('pooling', [2, 2, 2, 2, 2])
        self._linear_dim = kwargs.get('lineardim', 128)
        self._cmn = kwargs.get(
            'cmn', True)  # Use or not use rollwing window standardization
        self.norm = MovingAvgNorm(150) if self._cmn else nn.Sequential()
        net = nn.ModuleList()
        for nl, (h0, h1, filtersize, poolingsize) in enumerate(
                zip(self._filter, self._filter[1:], self._filtersizes,
                    self._pooling)):
            if nl == 0:
                mfmtype = "MFM"
            else:
                mfmtype = "MFMGroup"
            net.append(globals()[mfmtype](
                in_channels=h0,
                out_channels=h1,
                kernel_size=filtersize,
                padding=int(filtersize) // 2,
                stride=1,
            ))
            net.append(nn.MaxPool2d(kernel_size=poolingsize, ceil_mode=True))
        self.network = nn.Sequential(*net)
        with torch.no_grad():
            feature_output = self.network(torch.randn(1, 1, 300,
                                                      inputdim)).shape
            feature_output = feature_output[1] * feature_output[3]
            # LeftOverFeatDim * ChannelDim

        self.timepool = nn.AdaptiveAvgPool2d((1, None))
        self.outputlayer = nn.Sequential(
            nn.Conv1d(feature_output, self._linear_dim * 2, kernel_size=1),
            nn.GLU(1), nn.Dropout(0.5),
            nn.Conv1d(self._linear_dim, outputdim, kernel_size=1, groups=1))

        self.network.apply(init_weights)
        self.outputlayer.apply(init_weights)

    def forward(self, x):
        x = self.norm(x)
        x = x.unsqueeze(1)
        x = self.network(x)  # pooled time dimension
        # Pool time dimension
        # Reshape to B x 1 x C x D, then pool (CxD) to one dimension and
        # Reshape to B x (CxD) x 1
        x = self.timepool(x).permute(0, 2, 1,
                                     3).contiguous().flatten(-2).permute(
                                         0, 2, 1).contiguous()
        return self.outputlayer(x).squeeze(-1)  # dim[-1] is one dimensional


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Conv1d)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
