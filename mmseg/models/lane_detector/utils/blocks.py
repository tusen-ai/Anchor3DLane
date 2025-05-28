import torch.nn as nn
import torch
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class DecodeLayer(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, bias=True):
        super(DecodeLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, mid_channel),
            nn.ReLU6(),
            nn.Linear(mid_channel, out_channel, bias=bias))
    def forward(self, x):
        return self.layer(x)
    
class SegHeadIns(nn.Module):
    def __init__(self, feat_channel, anchor_feat_channel, hidden_channel=64):
        super(SegHeadIns, self).__init__()
        self.anchor_project = DecodeLayer(anchor_feat_channel, 256, hidden_channel)
        self.feat_project = nn.Conv2d(feat_channel, hidden_channel, 3, padding=1)
    
    def forward(self, anchor_feat, img_feat):
        # [B, C, N, l] -> [B, N, C, l] -> [B, N, Cl]
        anchor_feat = anchor_feat.transpose(1, 2).flatten(2, 3)
        anchor_feat = self.anchor_project(anchor_feat)  # [B, N, C]
        img_feat = self.feat_project(img_feat)  # [B, C, H, W]
        b, _, h, w = img_feat.shape
        anchor_num = anchor_feat.shape[1]
        img_feat = img_feat.flatten(2, 3) # [B, C, HW]
        seg_label = torch.matmul(anchor_feat, img_feat)  # [B, N, HW]
        seg_label = seg_label.reshape(b, anchor_num, h, w)
        return seg_label
    
class SegHead(nn.Module):
    def __init__(self, feat_channel, hidden_channel=256):
        super(SegHead, self).__init__()
        self.seg_head = nn.Sequential(
            nn.Conv2d(feat_channel, hidden_channel, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.Conv2d(hidden_channel, hidden_channel, 3, stride=1, padding=1),
            nn.ReLU6(),
            nn.Conv2d(hidden_channel, 1, 1))
    
    def forward(self, img_feat):
        seg_label = self.seg_head(img_feat)
        return seg_label

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 kernel_size=None, padding=None, attn_groups=None, embed_shape=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
    

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x