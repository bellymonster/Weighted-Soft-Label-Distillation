# Rethinking soft labels for knowledge distillation: a bias-variance tradeoff perspective
Accepted by ICLR 2021

This is the offical PyTorch implementation of paper "Rethinking soft labels for knowledge distillation: a bias-variance tradeoff perspective".

## Requirements
+ Python >= 3.6
+ PyTorch >= 1.0.1 

## ImageNet Training

The code is used for training Imagenet. Our pre-trained teacher models are Pytorch official models. By default, we pack the ImageNet data as the lmdb file for faster IO. The lmdb files can be made as follows.

1. Generate the list of the image data.
python dataset/mk_img_list.py --image_path 'the path of your image data' --output_path 'the path to output the list file'

2. Use the image list obtained above to make the lmdb file.
python dataset/img2lmdb.py --image_path 'the path of your image data' --list_path 'the path of your image list' --output_path 'the path to output the lmdb file' --split 'split folder (train/val)'

+ train_with_distillation.py: train the model with our distillation method
+ imagenet_train_cfg.py: all dataset and hyperparameter settings
+ knowledge_distiller.py: our weighted soft label distillation loss