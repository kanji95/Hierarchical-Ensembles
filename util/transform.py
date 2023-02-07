# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL

from torchvision import datasets, transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

TIERED_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TIERED_IMAGENET_STD = [0.229, 0.224, 0.225]

INATURALIST_MEAN = [0.454, 0.474, 0.367]
INATURALIST_STD = [0.237, 0.230, 0.249]

mean_ilsvrc12 = [0.485, 0.456, 0.406]
std_ilsvrc12 = [0.229, 0.224, 0.225]
mean_inat19 = [0.454, 0.474, 0.367]
std_inat19 = [0.237, 0.230, 0.249]

normalize_tfs_ilsvrc12 = transforms.Normalize(mean=mean_ilsvrc12, std=std_ilsvrc12)
normalize_tfs_inat19 = transforms.Normalize(mean=mean_inat19, std=std_inat19)
normalize_tfs_dict = {
    "tiered-imagenet-84": normalize_tfs_ilsvrc12,
    "tiered-imagenet-224": normalize_tfs_ilsvrc12,
    "ilsvrc12": normalize_tfs_ilsvrc12,
    "inaturalist19-84": normalize_tfs_inat19,
    "inaturalist19-224": normalize_tfs_inat19,
}


def train_transforms(img_resolution, dataset, augment=True, normalize=True):
    if augment and normalize:
        return transforms.Compose(
            [
                # extract random crops and resize to img_resolution
                transforms.RandomResizedCrop(img_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize_tfs_dict[dataset],
            ]
        )
    elif not augment and normalize:
        return transforms.Compose([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    elif augment and not normalize:
        return transforms.Compose([transforms.RandomResizedCrop(img_resolution), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    else:
        return transforms.Compose([transforms.ToTensor()])


def val_transforms(dataset, normalize=True, resize=None, crop=None):
    trsfs = []
    
    if resize:
        trsfs.append(transforms.Resize((resize, resize)))

    if crop:
        trsfs.append(transforms.CenterCrop(crop))

    if normalize:
        trsfs.extend([transforms.ToTensor(), normalize_tfs_dict[dataset]])
    else:
        trsfs.append([*transforms.ToTensor()])

    return transforms.Compose(trsfs)