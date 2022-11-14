# mmdet-ISCL
This is the official implementation of TGRS paper "Instance Switching-Based Contrastive Learning for Fine-Grained Airplane Detection".[IEEE Xplore](https://ieeexplore.ieee.org/document/9933796)

## Introduction
ISCL is a augmentation method that can be added to most two-stage detectors to boost their performance on fine-grained airplane detection in remote sensing images.

**Abstract**: Detecting airplanes from high-resolution remote sensing images has a variety of applications. The characteristics of clear details, rich spatial, and texture information of objects in high-resolution remote sensing images make it possible to identify different types of airplanes from backgrounds. However, airplanes usually exhibit slight interclass discrepancy and unbalanced class distribution, which pose significant challenges to the fine-grained detection of airplanes. In this article, we propose the ISCL, an instance switching-based contrastive learning method for fine-grained airplane detection. Specifically, we introduce a contrastive learning-based module (CLM) to widen the interclass distance while narrowing the intraclass distance by optimizing feature space distribution with the InfoNCE+ loss, which is built on a serial head in a cascaded way. Then, we design a refined instance switching (ReIS) module to alleviate the class imbalance problem. To take full advantage of the CLM and ReIS, we further introduce an optimization strategy, which is an organic combination of the two modules to widen the distances of different airplane categories that are easily confused. In addition, we contribute a fine-grained attribute-assisted dataset, dubbed GF-RarePlanes dataset (GRD), to help the detectors better learn the subtle differences between the airplanes. Extensive experiments on two datasets (i.e., GF and FAIR1M) demonstrate that our proposed method can significantly improve the accuracy of fine-grained airplane detection under both horizontal bounding box (HBB) and oriented bounding box (OBB) scenarios. Dataset and codes will be available at [https://lanxin1011.github.io/ISCL/](https://lanxin1011.github.io/ISCL/).


![demo image](figures/iscl_framework.png)

## Installation and Get Started

Required enviroments: (note that this implementation is based on MMDetection v2.18)
* Linux
* Python 3.6+
* Pytorch 1.5+
* CUDA 9.2+
* [MMCV](https://mmcv.readthedocs.io/en/latest/#installation)


Install:

Note that this repository is based on the [MMDetection](https://github.com/open-mmlab/mmdetection). Assume that your environment has satisfied the above requirements, please follow the following steps for installation.

```shell script
git clone https://github.com/Lanxin1011/mmdet-ISCL.git
pip install -r requirements/build.txt
python setup.py develop
```

## Main Results


## Visualization


## Citation
If you find this work helpful,please consider citing:
```bibtex
@article{zeng2022instance,
  title={Instance Switching-based Contrastive Learning for Fine-grained Airplane Detection},
  author={Zeng, Lanxin and Guo, Haowen and Yang, Wen and Yu, Huai and Yu, Lei and Zhang, Peng and Zou, Tongyuan},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2022},
  publisher={IEEE}
}
```
