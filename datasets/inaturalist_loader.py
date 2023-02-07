""" A PyTorch dataset for loading iNaturalist data.
    
    TODO: missing credits.
"""
import os
import json
import numpy as np
from glob import glob
from collections import defaultdict

import torch
import torch.utils.data as data
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torchvision.datasets.folder import default_loader

TAXONOMY_MAP = {
    "species": 0,
    "genus": 1,
    "family": 2,
    "order": 3,
    "class": 4,
    "phylum": 5,
    "kingdom": 6,
}

TAXONOMY_NUM = {
    "species": 1010,
    "genus": 72,
    "family": 57,
    "order": 34,
    "class": 9,
    "phylum": 4,
    "kingdom": 3,
}

ANN_FILE = {
    "train": "train2019.json",
    "val": "val2019.json",
    "test": "test2019.json",
}

def load_taxonomy(ann_data, classes):
    # loads the taxonomy data and converts to ints
    tax_levels = [
        "id",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
    ]
    taxonomy = {}

    if "categories" in ann_data.keys():
        num_classes = len(ann_data["categories"])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data["categories"]]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0] * len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    return taxonomy, classes_taxonomic

class iNaturalist(data.Dataset):
    def __init__(self, root, mode="train", transform=None, taxonomy="species"):
      
        self._mode = mode
        self.taxonomy_name = taxonomy
        
        split = "train" if mode == "train" else "val"
        self.inaturalist_dir = os.path.join(root, split)
        
        split_path = os.path.join("./datasets/data_splits/inaturalist19/", split)
        class_files = glob(split_path + "/*")
        
        self.classes = sorted([os.path.splitext(os.path.basename(cls_file))[0] for cls_file in class_files])
        self.num_classes = TAXONOMY_NUM[self.taxonomy_name]
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        # make pathlib.Paths
        ann_file = ANN_FILE[mode]
        try:
            self._root = root
            self.annotations_path = root / ann_file
        except TypeError:
            self._root = root = Path(root)
            self._ann_file = ann_file = root / ann_file

        # load annotations
        print(f"iNaturalist: loading annotations from: {ann_file}.")
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # # set up the filenames and annotations
        self._img_paths = [root / aa["file_name"] for aa in ann_data["images"]]

        # if we dont have class labels set them to '0'
        if "annotations" in ann_data.keys():
            self._classes = [a["category_id"] for a in ann_data["annotations"]]
        else:
            self._classes = [0] * len(self._img_paths)

        self._taxonomy, self._classes_taxonomic = load_taxonomy(
            ann_data, self._classes
        )
        self._taxonomy['species'] = self._taxonomy['id']
        
        self.parent_to_child = defaultdict(list)
        # self.family_to_genus = defaultdict(list)
        # self.family_to_species = defaultdict(list)
            
        for species_key in self._taxonomy['family']:
            # family_key = self._taxonomy['family'][species_key]
            genus_key = self._taxonomy['genus'][species_key]
            self.parent_to_child[genus_key].append(species_key)
            # self.family_to_genus[family_key].append(genus_key)
            # self.family_to_species[family_key].append(species_key)
        
        self.classes = sorted(list(set(self._classes)))
        self.classes = [f'nat{x:04}' for x in self.classes]
        
        self.samples = []
        for class_file in class_files:
            class_name = os.path.splitext(os.path.basename(class_file))[0]
            with open(class_file, 'r') as f:
                image_names = f.readlines()
            image_names = [image_name.strip() for image_name in image_names]
            
            species_id = self.class_to_idx[class_name]
            tax_id = self._taxonomy[self.taxonomy_name][species_id]
            
            class_samples = [(os.path.join(self.inaturalist_dir, class_name, image_name), tax_id, class_name) for image_name in image_names]
            self.samples.extend(class_samples)
        
        # image loading, preprocessing and augmentations
        self.loader = default_loader
        
        self.transform = transform

        # print out some stats
        print(f"iNaturalist: found {len(self.samples)} images.")
        print(f"iNaturalist: found {self.num_classes} classes.")
    
    def __getitem__(self, index):
        
        img_path, tax_id, class_name = self.samples[index]
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)
        
        return {
            "img":img, 
            "target": tax_id,
            "class_name": class_name 
        }

    
    def __str__(self):
        details = f"len={len(self)}, mode={self._mode}, root={self._root}"
        return f"iNaturalistDataset({details})"

    def __len__(self):
        return len(self.samples)