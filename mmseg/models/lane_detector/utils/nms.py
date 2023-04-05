# --------------------------------------------------------
# Source code for Anchor3DLane
# Copyright (c) 2023 TuSimple
# @Time    : 2023/04/05
# @Author  : Shaofei Huang
# nowherespyfly@gmail.com
# --------------------------------------------------------

import torch

@torch.no_grad()
def nms_3d(proposals, scores, vises, thresh, anchor_len=10):
    # proposals: [N, 35], scores: [N]
    order = scores.argsort(descending=True)
    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        x1 = proposals[i][5:5+anchor_len]  # [l]
        z1 = proposals[i][5+anchor_len:5+anchor_len*2]   # [l]
        vis1 = vises[i]  # [l]
        x2s = proposals[order[1:]][:, 5:5+anchor_len]  # [n, l]
        z2s = proposals[order[1:]][:, 5+anchor_len:5+anchor_len*2]   # [n, l]
        vis2s = vises[order[1:]]  # [n, l]
        matched = vis1 * vis2s  # [n, l]
        lengths = matched.sum(dim=1)   # [n]
        dis = ((x1 - x2s) ** 2 + (z1 - z2s) ** 2) ** 0.5  # [n, l]
        dis = (dis * matched + 1e-6).sum(dim=1) / (lengths + 1e-6)  # [n], incase no matched points
        inds = torch.where(dis > thresh)[0]  # [n']
        order = order[inds + 1]   # [n']

    return torch.tensor(keep)

if __name__ == '__main__':
    anchors = []
    x_base = [i + 0. for i in range(10)]
    z_base = [i / 10. for i in range(10)]
    vis = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    anchor1 = [0, 0.99, 0, 0, 0] + x_base + z_base + vis
    anchor1 = torch.tensor(anchor1)
    anchor2 = anchor1.clone()
    anchor2[1] = 0.9
    anchor2[5:15] = anchor2[5:15] + torch.randn(10) - 0.5
    anchor2[33:35] = 0
    anchor3 = anchor1.clone()
    anchor3[1] = 0.6
    anchor3[15:25] = anchor3[15:25] + torch.randn(10) / 10 
    anchor3[25:28] = 0
    anchor4 = anchor1.clone()
    anchor4[1] = 0.5
    anchor4[5:15] = anchor4[5:15] + 10
    anchor4[15:25] = anchor4[15:25] + 0.1
    anchor4[25:34] = 0
    anchor5 = anchor4.clone()
    anchor5[1] = 0.8
    anchor5[25:30] = 1
    anchors = torch.stack([anchor2, anchor1, anchor3, anchor4, anchor5], dim=0)  # [2, 35]
    scores = anchors[:, 1]
    print(anchors)
    keep = nms_3d(anchors, scores, 1.)
    import pdb
    pdb.set_trace()
