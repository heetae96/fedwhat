import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

import random
import torch
import torchvision
from torchvision.datasets.utils import check_integrity, download_and_extract_archive
from torchvision.datasets.vision import VisionDataset

"""
changed..
"""
false=False

np.random.seed(12000)

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
        iid=False,   ##이걸 False로 바꾸면 non-iid
        dataset_name="cifar10",
        unbalanced_sgm=0,
        rule_arg=0.3,
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
        self.unbalanced_sgm = unbalanced_sgm
        self.rule_arg = rule_arg
        self.data, self.targets = self.get_data()
        print('Split num : [%d / %d]' %(split_id, split_number))

    def __getitem__(self, index):
        

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        
        if self.dataset_name=='emnist':
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img.numpy(), mode="L")

        else:
            img, target = self.data[index], self.targets[index]
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
        elif self.dataset_name == "emnist":
            dataset = torchvision.datasets.EMNIST(root=self.root, split='balanced', train=self.train, download=self.download)
        else:
            raise ValueError("Dataset not found")

        # if self.iid:
        #     data = dataset.data
        #     targets = dataset.targets
        #     data_len = len(data)
        #     data_len_per_split = data_len // self.split_number
        #     data = data[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
        #     targets = targets[self.split_id * data_len_per_split : (self.split_id + 1) * data_len_per_split]
        split_id = self.split_id
        split_number = self.split_number 
        if self.iid:
            data=[]
            targets=[]
            data_len = len(dataset.data)
            class_num = 10
            Adata = [[] for _ in range(class_num)]
            Atargets = [[] for _ in range(class_num)]
            
            if self.dataset_name == "cifar10":
                class_num = 10
            data_len_per_class = data_len // class_num           

            for n in range(data_len):
                Adata[dataset.targets[n]].append(dataset.data[n])
                Atargets[dataset.targets[n]].append(dataset.targets[n])
              
            #split_id = self.split_id
            #split_number = self.split_number 
               
            data_per_class_per_worker = data_len // (class_num * split_number)
            start_idx = split_id * data_per_class_per_worker
            end_idx = (split_id + 1) * data_per_class_per_worker
            
            for i in range(class_num):
                data.extend(Adata[i][start_idx:end_idx])
                targets.extend(Atargets[i][start_idx:end_idx])
            self.data=data
            self.targets=targets
        

          
                       
        #non-iid    
        else:
            if os.path.isfile('%s/dat/diri%d_%d/%d' %(self.root,self.rule_arg,self.split_number,self.split_id)):
                fp=open('%s/dat/diri%d_%d/%d' %(self.root,self.rule_arg,self.split_number,self.split_id), 'rb')
                dat=pickle.load(fp)
                clnt_xi, clnt_yi=dat
                self.data = list(clnt_xi); self.targets = clnt_yi.T.tolist()[0]
                
            else:
                dataset = torchvision.datasets.CIFAR10(root=self.root, train=self.train, download=self.download, transform=self.transform)
                
                #trn_load = torch.utils.data.DataLoader(dataset, batch_size=50000, shuffle=True, num_workers=2)
                self.channels = 3; self.width = 32; self.height = 32; self.n_cls = 10;

                # trn_itr = trn_load.__iter__(); 
                # trn_x, trn_y = trn_itr.__next__()
                # trn_x = trn_x.numpy(); trn_y = trn_y.numpy().reshape(-1,1)
                
                
                trn_x =dataset.data; trn_y =dataset.targets
                
                # rand_perm = np.random.permutation(len(trn_y))
                # trn_x = trn_x[rand_perm]
                # trn_y = trn_y[rand_perm]
                
                self.trn_x = trn_x
                self.trn_y = trn_y
                
                #unbalaced mount    
                n_data_per_clnt = int((len(trn_y)) / split_number)

                # if self.unbalanced_sgm != 0:
                #     # Draw from lognormal distribution
                #     clnt_data_list = (np.random.lognormal(mean=np.log(n_data_per_clnt), sigma=self.unbalanced_sgm, size=self.n_client))
                #     clnt_data_list = (clnt_data_list/np.sum(clnt_data_list)*len(trn_y)).astype(int)
                #     diff = np.sum(clnt_data_list) - len(trn_y)

                #     # Add/Subtract the excess number starting from first client
                #     if diff!= 0:
                #         for clnt_i in range(split_number):
                #             if clnt_data_list[clnt_i] > diff:
                #                 clnt_data_list[clnt_i] -= diff
                #                 break
                            
                # else:
                #print(trn_y)
                clnt_data_list = (np.ones(split_number) * n_data_per_clnt).astype(int)
                cls_priors   = np.random.dirichlet(alpha=[self.rule_arg]*self.n_cls,size=split_number)    
                
                prior_cumsum = np.cumsum(cls_priors, axis=1)
                
                idx_list = [np.array([], dtype=int) for _ in range(self.n_cls)]

                for i, dat in enumerate(trn_y):
                    idx_list[dat] = np.append(idx_list[dat], i)
                    
                
                #idx_list = [trn_y[k]==i for k in range(len(trn_y)) for i in range(self.n_cls)]
                #print(idx_list)
                # exit()
                cls_amount = [len(idx_list[i]) for i in range(self.n_cls)]
            
                clnt_x = [ np.zeros((clnt_data_list[clnt__], self.height, self.width, self.channels)).astype(np.uint8) for clnt__ in range(split_number) ]
                clnt_y = [ np.zeros((clnt_data_list[clnt__], 1)).astype(np.int64) for clnt__ in range(split_number) ]
                #print(cls_amount)
                while(np.sum(clnt_data_list)!=0):
                    curr_clnt = np.random.randint(split_number)
                    # If current node is full resample a client
                    #print('Remaining Data: %d' %np.sum(clnt_data_list))
                    if clnt_data_list[curr_clnt] <= 0:
                        continue
                    clnt_data_list[curr_clnt] -= 1
                    curr_prior = prior_cumsum[curr_clnt]
                    while True:
                        cls_label = np.argmax(np.random.uniform() <= curr_prior)
                        # Redraw class label if trn_y is out of that class
                        if cls_amount[cls_label] <= 0:
                            continue
                        cls_amount[cls_label] -= 1
                        clnt_x[curr_clnt][clnt_data_list[curr_clnt]] = trn_x[idx_list[cls_label][cls_amount[cls_label]]]
                        clnt_y[curr_clnt][clnt_data_list[curr_clnt]] = trn_y[idx_list[cls_label][cls_amount[cls_label]]]

                        break

                clnt_x = np.asarray(clnt_x)
                clnt_y = np.asarray(clnt_y)

                cls_means = np.zeros((split_number, self.n_cls))
                for clnt in range(split_number):
                    for cls in range(self.n_cls):
                        cls_means[clnt,cls] = np.mean(clnt_y[clnt]==cls)
                prior_real_diff = np.abs(cls_means-cls_priors)
                print('--- Max deviation from prior: %.4f' %np.max(prior_real_diff))
                print('--- Min deviation from prior: %.4f' %np.min(prior_real_diff))
            
                self.clnt_x = clnt_x; self.clnt_y = clnt_y

            
                print('Class frequencies:')
                count = 0
                for clnt in range(split_number):
                    print("Client %3d: " %clnt + 
                        ', '.join(["%.3f" %np.mean(self.clnt_y[clnt]==cls) for cls in range(self.n_cls)]) + 
                        ', Amount:%d' %self.clnt_y[clnt].shape[0])
                    count += self.clnt_y[clnt].shape[0]
            
            
                print('Total Amount:%d' %count)
                print('--------')
                
                self.data = list(clnt_x[split_id]); self.targets = clnt_y[split_id].T.tolist()[0]
                i=0
                os.makedirs('%s/dat/diri%d_%d' %(self.root,self.rule_arg,self.split_number))
                for clnt_xi, clnt_yi in zip(clnt_x, clnt_y):
                    fp=open('%s/dat/diri%d_%d/%d' %(self.root,self.rule_arg,self.split_number,i), 'wb')
                    pickle.dump((clnt_xi, clnt_yi), fp)
                    fp.close()
                    i+=1

        return self.data, self.targets

import torchvision.transforms as transforms   
if __name__ =="__main__":
    transform = transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])])
    cd = custom_dataset(root='data/', dataset_name="cifar10", transform=transform, rule_arg=0.6, iid=false, split_id=0, split_number=5)

    print(type(cd.data), type(cd.data[0]), type(cd.data[0][0]), type(cd.data[0][0][0]), type(cd.data[0][0][0][0]))
    print(np.array(cd.data).shape)
    print(type(cd.targets), type(cd.targets[0]))
    print(np.array(cd.targets).shape)

