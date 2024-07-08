## BEVUDA: Multi-geometric Space Alignments for Domain Adaptive BEV 3D Object Detection
![Python 3.9](https://img.shields.io/badge/Python-3.9-red)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/pdf/2211.17126)

<img src="bevuda.png"/>

## Quick Start
### Installation
**Step 0.** Install [pytorch](https://pytorch.org/)(v1.9.0).

**Step 1.** Install [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)(v1.0.0rc4).

**Step 2.** Install requirements.
```shell
pip install -r requirements.txt
```
**Step 3.** Install BEVDepth(gpu required).
```shell
python setup.py develop
```

### Data preparation
**Step 0.** Download nuScenes official dataset.

**Step 1.** Symlink the dataset root to `./data/`.
```
ln -s [nuscenes root] ./data/
```
The directory will be as follows.
```
BEVDepth
├── data
│   ├── nuScenes
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
```
**Step 2.** Prepare infos.
```
python scripts/gen_info.py
```

### Tutorials
**Train.**
```
python [EXP_PATH] --amp_backend native -b 8 --gpus 8
```
**Eval.**
```
python [EXP_PATH] --ckpt_path [CKPT_PATH] -e -b 8 --gpus 8
```

## Citation 
Please cite our work if you find it useful.
```bibtex
@article{liu2022multi,
  title={Multi-latent Space Alignments for Unsupervised Domain Adaptation in Multi-view 3D Object Detection},
  author={Liu, Jiaming and Zhang, Rongyu and Chi, Xiaowei and Li, Xiaoqi and Lu, Ming and Guo, Yandong and Zhang, Shanghang},
  journal={arXiv preprint arXiv:2211.17126},
  year={2022}
}
```
