import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import torchvision
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

"""
changed..
"""

class custom_dataset(VisionDataset):
    """
    A dataset that returns a split of the CIFAR10 dataset.
    input: dataset_name, train, transform, target_transform, download, split_number, split_id, iid
    """

    def __init__(
        self,
        root:str,
        train=True,
        transform=None,
        target_transform=None,
        download=False,
        split_number=10,
        split_id=0,
        iid=True,   ##이걸 false로 바꾸면 non-iid
        dataset_name="cifar100",
    ):
        super(custom_dataset, self).__init__(root=root, transform=transform, target_transform=target_transform)
        self.dataset_name = dataset_name
        self.train = train
        self.root =root
        self.transform = transform
        self.target_transform = target_transform
        self.download = download
        self.split_number = split_number
        self.split_id = split_id
        self.iid = iid
        self.data = []
        self.targets = []
        self.data, self.targets = self.get_data()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

    def get_data(self):
        if self.dataset_name == "cifar10":
            dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=self.download)
        elif self.dataset_name == "cifar100":
            dataset = torchvision.datasets.CIFAR100(root=self.root, train=self.train, download=self.download)
        elif self.dataset_name == "svhn":
            dataset = torchvision.datasets.svhn(root=self.root, train=self.train, download=self.download)
        else:
            raise ValueError("Dataset not found")

        if self.iid:
            data = dataset.data
            targets = dataset.targets
            data_len = len(data)
            data_len_per_split = data_len // self.split_number
            data = data[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
            targets = targets[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
        

        
        else:
            # non-iid split
            Adata = [[] for _ in range(10)]
            Atargets = [[] for _ in range(10)]
            data=[]
            targets=[]
            data_len = len(dataset.data)
            class_num = 10
            if self.dataset_name == "cifar100":
                class_num = 100
            data_len_per_class = data_len // class_num
            
            # for i in range(class_num):
            #     data.append(dataset.data[i * data_len_per_class : (i + 1) * data_len_per_class])
            #     targets.append(dataset.targets[i * data_len_per_class : (i + 1) * data_len_per_class])
            
            for n in range(data_len):
                Adata[dataset.targets[n]].append(dataset.data[n])
                Atargets[dataset.targets[n]].append(dataset.targets[n])

            # numdata = max( 10 // self.split_number,1)
            # for d in Adata[self.split_id * numdata : (self.split_id + 1) * numdata]:
            #     data += d
            # for t in Atargets[self.split_id * numdata : (self.split_id + 1) * numdata]:
            #     targets += t
            
            split_id = self.split_id    
                                        
            if split_id == 0:
                for i in range(10):
                    if (i==0) or (i==1):
                        data += Adata[i][0:4200]
                        targets += Atargets[i][0:4200]
                    else:
                        data += Adata[i][4200:]
                        targets += Atargets[i][4200:]
                    # data += Adata[0][25000:50000]
                    # targets += Atargets[0][25000:50000]
                    # data += Adata[1][0:50000]
                    # targets += Atargets[1][0:50000]
                    # data += Adata[2][0:25000]
                    # targets += Atargets[2][0:25000]                    
                    

            if split_id == 1:
                for i in range(10):
                    if (i==2) or (i==3):
                        data += Adata[i][0:4200]
                        targets += Atargets[i][0:4200]
                    else:
                        data += Adata[i][4200:]
                        targets += Atargets[i][4200:]
                    # data += Adata[2][25000:50000]
                    # targets += Atargets[2][25000:50000]
                    # data += Adata[3][0:50000]
                    # targets += Atargets[3][0:50000]
                    # data += Adata[4][0:25000]
                    # targets += Atargets[4][0:25000]
                
            if split_id == 2:
                for i in range(10):
                    if (i==4) or (i==5):
                        data += Adata[i][0:4200]
                        targets += Atargets[i][0:4200]
                    else:
                        data += Adata[i][4200:]
                        targets += Atargets[i][4200:]
                    # data += Adata[4][25000:50000]
                    # targets += Atargets[4][25000:50000]
                    # data += Adata[5][0:50000]
                    # targets += Atargets[5][0:50000]
                    # data += Adata[6][0:25000]
                    # targets += Atargets[6][0:25000]
                    
            if split_id == 3:
                for i in range(10):
                    if (i==6) or (i==7):
                        data += Adata[i][0:4200]
                        targets += Atargets[i][0:4200]
                    else:
                        data += Adata[i][4200:]
                        targets += Atargets[i][4200:]
                    # data += Adata[6][25000:50000]
                    # targets += Atargets[6][25000:50000]
                    # data += Adata[7][0:50000]
                    # targets += Atargets[7][0:50000]
                    # data += Adata[8][0:25000]
                    # targets += Atargets[8][0:25000]
                    
            if split_id == 4:
                for i in range(10):
                    if (i==8) or (i==9):
                        data += Adata[i][0:4200]
                        targets += Atargets[i][0:4200]
                    else:
                        data += Adata[i][4200:]
                        targets += Atargets[i][4200:]
                    # data += Adata[8][25000:50000]
                    # targets += Atargets[8][25000:50000]
                    # data += Adata[9][0:50000]
                    # targets += Atargets[9][0:50000]
                    # data += Adata[0][0:25000]
                    # targets += Atargets[0][0:25000]
                    
            # if split_id == 5:
            #     for i in [10, 5]:
            #         data += Adata[i][3500:5000]
            #         targets += Atargets[i][3500:5000]
                                                    
        return data, targets
    
