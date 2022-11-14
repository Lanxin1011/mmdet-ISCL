## AITOOL
aitool is a Python library for academic research in remote sensing field.

It will provide the following functionalities.

- Basic parse and dump functions for datasets of aerial images
- Evaluation tools
- Visualization tools

### Requirements

- Python 3.6+
- Pytorch 1.1+
- CUDA 9.0+
- [mmcv](https://github.com/open-mmlab/mmcv)
- pycocotools (pip install lvis@git+https://github.com/open-mmlab/cocoapi.git#subdirectory=lvis)

### Installation
```
git clone https://github.com/jwwangchn/aitool.git
cd aitool
python setup.py develop
```