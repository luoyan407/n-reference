# n-Reference Transfer Learning for Saliency Prediction
n-Reference Transfer Learning enables efficient transfer of knowledge learned from the existing large-scale saliency datasets to a target domain with limited labeled examples. The code is tested under Ubuntu 1804 LTS with Python 3.7.

## Requirements
It requires [PyTorch 1.2.0+](https://pytorch.org/get-started/previous-versions/).
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

## Data
The saliency prediction task requires blurred fixation maps and/or binary fixation maps. Images and corresponding binary fixation maps (i.e., ground truth) can be downloaded via the following links.
- [SALICON](http://salicon.net/download/)
- [MIT1003](http://people.csail.mit.edu/tjudd/WherePeopleLook/index.html)
- [Webpage saliency dataset (FiWI)](https://www-users.cs.umn.edu/~qzhao/webpage_saliency.html)
- [CAT2000](http://saliency.mit.edu/results_cat2000.html)

Note that SALICON and FiWI only provide binary fixation maps. FiWI provides a Matlab script to generate blurred fixation maps. You can either refer to the script to do the preprocessing yourself or download the preprocessed blurred fixation maps via this [link](https://drive.google.com/drive/folders/1d3lANSKYwjmvpiAkmOx_e76jaTcc6saS?usp=sharing).

## Core Modules
The reference process is implemented in [models.py](src/models.py), named **Referencer**. TR-Ref is implemented in [train_nref.py](train_nref.py). TR, FT, and FT|Ref with n samples are implemented in [train_nshot.py](train_nshot.py). Conventional TR is implemented in [train.py](train.py).

## Training
To train a model in various schemes (i.e., TR, TR-Ref, FT, and FT|Ref), run 
```bash
./train_script.sh
```
Note that variable **split_file** indicates the path to the file of the sample splits (i.e., subsets of training/validation/reference samples). If the variable is empty, it will generate ramdon subsets. 
The splits used in this work can be found in this shared [folder](https://drive.google.com/drive/folders/19FXP9wgDxtrJ20zgUwGbZ0rCkAmGj4lb?usp=sharing).

## Citation
If you think our work is interesting or useful, please cite our paper
```BibTex
@InProceedings{Luo_2020_ECCV,
author = {Luo, Yan and Wong, Yongkang and Kankanhalli, Mohan S. and Zhao, Qi},
title = {n-Reference Transfer Learning for Saliency Prediction},
booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
month = {September},
year = {2020}
} 
```