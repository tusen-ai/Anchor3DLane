import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
import torchvision.models as models
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from itertools import chain
import geffnet
import pdb
import os

from ..builder import BACKBONES
from ..utils import ResLayer

def checkpoint_seq(
        functions,
        x,
        every=1,
        flatten=False,
        skip_last=False,
        preserve_rng_state=True
):
    r"""A helper function for checkpointing sequential models.
    Sequential models execute a list of modules/functions in order
    (sequentially). Therefore, we can divide such a sequence into segments
    and checkpoint each segment. All segments except run in :func:`torch.no_grad`
    manner, i.e., not storing the intermediate activations. The inputs of each
    checkpointed segment will be saved for re-running the segment in the backward pass.
    See :func:`~torch.utils.checkpoint.checkpoint` on how checkpointing works.
    .. warning::
        Checkpointing currently only supports :func:`torch.autograd.backward`
        and only if its `inputs` argument is not passed. :func:`torch.autograd.grad`
        is not supported.
    .. warning:
        At least one of the inputs needs to have :code:`requires_grad=True` if
        grads are needed for model inputs, otherwise the checkpointed part of the
        model won't have gradients.
    Args:
        functions: A :class:`torch.nn.Sequential` or the list of modules or functions to run sequentially.
        x: A Tensor that is input to :attr:`functions`
        every: checkpoint every-n functions (default: 1)
        flatten (bool): flatten nn.Sequential of nn.Sequentials
        skip_last (bool): skip checkpointing the last function in the sequence if True
        preserve_rng_state (bool, optional, default=True):  Omit stashing and restoring
            the RNG state during each checkpoint.
    Returns:
        Output of running :attr:`functions` sequentially on :attr:`*inputs`
    Example:
        >>> model = nn.Sequential(...)
        >>> input_var = checkpoint_seq(model, input_var, every=2)
    """
    def run_function(start, end, functions):
        def forward(_x):
            for j in range(start, end + 1):
                _x = functions[j](_x)
            return _x
        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = functions.children()
    if flatten:
        functions = chain.from_iterable(functions)
    if not isinstance(functions, (tuple, list)):
        functions = tuple(functions)

    num_checkpointed = len(functions)
    if skip_last:
        num_checkpointed -= 1
    end = -1
    for start in range(0, num_checkpointed, every):
        end = min(start + every - 1, num_checkpointed - 1)
        x = checkpoint(run_function(start, end, functions), x, preserve_rng_state=preserve_rng_state)
    if skip_last:
        return run_function(end + 1, len(functions) - 1, functions)(x)
    return x


def flatten_modules(named_modules, depth=1, prefix='', module_types='sequential'):
    prefix_is_tuple = isinstance(prefix, tuple)
    if isinstance(module_types, str):
        if module_types == 'container':
            module_types = (nn.Sequential, nn.ModuleList, nn.ModuleDict)
        else:
            module_types = (nn.Sequential,)
    for name, module in named_modules:
        if depth and isinstance(module, module_types):
            yield from flatten_modules(
                module.named_children(),
                depth - 1,
                prefix=(name,) if prefix_is_tuple else name,
                module_types=module_types,
            )
        else:
            if prefix_is_tuple:
                name = prefix + (name,)
                yield name, module
            else:
                if prefix:
                    name = '.'.join([prefix, name])
                yield name, module

@BACKBONES.register_module()
class EfficientNet(nn.Module):
    def __init__(self, arch, lv6=True, lv5=True, lv4=True, lv3=True, pretrained=True, \
    stride=1, lv5_partial=False, with_cp=False):
        super(EfficientNet, self).__init__()
        self.pretrain_path = {'b5': 'pretrained/tf_efficientnet_b5_ns-6f26d0cf.pth',
                    'b4': 'pretrained/tf_efficientnet_b4_ns-d6313a46.pth', 
                    'b3': 'pretrained/tf_efficientnet_b3_ns-9d44bf68.pth', 
                    'b2': 'pretrained/tf_efficientnet_b2_ns-00306e48.pth', 
                    'b1': 'pretrained/tf_efficientnet_b1_ns-99dd0c41.pth'}
        self.arch = arch
        self.with_cp = with_cp
        if arch == "b1":
            if stride == 1:
                self.encoder = geffnet.tf_efficientnet_b1_ns_s8(pretrained=False)
            else:
                self.encoder = geffnet.tf_efficientnet_b1_ns(pretrained=False)
            self.dimList = [16, 24, 40, 112, 1280] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 40, 112, 320] #5th feature is extracted after blocks[6]
        elif arch == "b2":
            if stride == 1:
                self.encoder = geffnet.tf_efficientnet_b2_ns_s8(pretrained=False)
            else:
                self.encoder = geffnet.tf_efficientnet_b2_ns(pretrained=False)
            self.dimList = [16, 24, 48, 120, 1408] #5th feature is extracted after conv_head or bn2
            #self.dimList = [16, 24, 48, 120, 352] #5th feature is extracted after blocks[6]
        elif arch == "b3":
            if stride == 1:
                self.encoder = geffnet.tf_efficientnet_b3_ns_s8(pretrained=False)
            else:
                self.encoder = geffnet.tf_efficientnet_b3_ns(pretrained=False)
            self.dimList = [24, 32, 48, 136, 1536] #5th feature is extracted after conv_head or bn2
        elif arch == "b4":
            if stride == 1:
                self.encoder = geffnet.tf_efficientnet_b4_ns_s8(pretrained=False)
            else:
                self.encoder = geffnet.tf_efficientnet_b4_ns(pretrained=False)
            self.dimList = [24, 32, 56, 160, 1792] #5th feature is extracted after conv_head or bn2
        elif arch == "b5":
            if stride == 1:
                self.encoder = geffnet.tf_efficientnet_b5_ns_s8(pretrained=False)
            else:
                self.encoder = geffnet.tf_efficientnet_b5_ns(pretrained=False)
            self.dimList = [24, 40, 64, 176, 2048] #5th feature is extracted after conv_head or bn2
        else:
            raise Exception("Not implemented arch type:", arch)
        del self.encoder.global_pool
        del self.encoder.classifier
        del self.encoder.conv_head
        del self.encoder.bn2
        del self.encoder.act2
        self.block_idx = [3, 4, 5, 7, 11] #5th feature is extracted after bn2
        if lv6 is False:
            del self.encoder.blocks[6]
            self.block_idx = self.block_idx[:4]
            self.dimList = self.dimList[:4]
        if lv5 is False:
            del self.encoder.blocks[5]
            self.block_idx = self.block_idx[:3]
            self.dimList = self.dimList[:3]
        if lv5_partial is True:
            del self.encoder.blocks[5][-1]
            del self.encoder.blocks[5][-1]
            del self.encoder.blocks[5][-1]
            del self.encoder.blocks[5][-1]
            self.block_idx = self.block_idx[:3]
            self.dimList = self.dimList[:3]
        if lv4 is False:
            del self.encoder.blocks[4]
            self.block_idx = self.block_idx[:2]
            self.dimList = self.dimList[:2]
        if lv3 is False:
            del self.encoder.blocks[3]
            self.block_idx = self.block_idx[:1]
            self.dimList = self.dimList[:1]

        if pretrained:
            self.load_checkpoint()
        # self.fixList = ['blocks.0.0','bn']

        # for name, parameters in self.encoder.named_parameters():
        #     if name == 'conv_stem.weight':
        #         parameters.requires_grad = False
        #     if any(x in name for x in self.fixList):
        #         parameters.requires_grad = False

    def load_checkpoint(self):
        checkpoint_path = self.pretrain_path[self.arch]
        if checkpoint_path and os.path.isfile(checkpoint_path):
            print("=> Loading checkpoint '{}'".format(checkpoint_path))
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    if k.startswith('module'):
                        name = k[7:]  # remove `module.`
                    else:
                        name = k
                    new_state_dict[name] = v
                self.encoder.load_state_dict(new_state_dict, strict=False)
            else:
                self.encoder.load_state_dict(checkpoint, strict=False)
            print("=> Loaded checkpoint '{}'".format(checkpoint_path))
        else:
            print("=> Error: No checkpoint found at '{}'".format(checkpoint_path))
            raise FileNotFoundError()


    def forward(self, x):
        out_featList = []
        x = self.encoder.conv_stem(x)
        x = self.encoder.bn1(x)
        x = self.encoder.act1(x)
        for i in range(len(self.encoder.blocks)):
            if self.with_cp and x.requires_grad:
                x = checkpoint(self.encoder.blocks[i], x)
            else:
                x = self.encoder.blocks[i](x)
            out_featList.append(x)
        return out_featList

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()
                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

# ckpt = 'pretrained/tf_efficientnet_b5_ns-6f26d0cf.pth'
# m = deepFeatureExtractor_EfficientNet('EfficientNet-B5', ckpt=ckpt, lv3=True, lv4=True, lv5=True, lv6=True)
# print(m)
# i = torch.randn(2, 3, 480, 360)
# o = m.features(i)
# # o = m.forward(i)
# pdb.set_trace()
# print(o[0].shape)