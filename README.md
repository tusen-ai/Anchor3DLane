# Anchor3DLane
This repo is the official PyTorch implementation for paper:

[Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection](https://arxiv.org/abs/2301.02371). Accepted by CVPR 2023.

Code is scheduled to be released before April 15th, 2023.

![pipeline](images/pipeline.png)
In this paper, we deﬁne 3D lane anchors in the 3D space and propose a BEV-free method named Anchor3DLane to predict 3D lanes directly from FV representations. 3D lane anchors are projected to the FV features to extract their features which contain both good structural and context information to make accurate predictions. We further extend Anchor3DLane to the multi-frame setting to incorporate temporal information for performance improvement.

## Visualization
We represent the visualization results of Anchor3DLane on ApolloSim, OpenLane and ONCE-3DLane datasets.

* Visualization results on ApolloSim dataset.
![apollo](images/vis_apollo.png)

* Visualization results on OpenLane dataset.
![openlane](images/vis_openlane.png)

* Visualization results on OpenLane dataset.
![once](images/vis_once.png)


# Citation
If you find this repo useful for your research, please cite
```
@article{anchor3dlane,
  author    = {Huang, Shaofei and Shen, Zhenwei and Huang, Zehao and Ding, Zi-han and Dai, Jiao and Han, Jizhong and Wang, Naiyan and Liu, Si},
  title     = {Anchor3DLane: Learning to Regress 3D Anchors for Monocular 3D Lane Detection},
  journal   = {arXiv preprint arXiv:2301.02371},
  year      = {2023}
}
```
# Contact
For questions about our paper or code, please contact [Shaofei Huang](mailto:nowherespyfly@gmail.com).