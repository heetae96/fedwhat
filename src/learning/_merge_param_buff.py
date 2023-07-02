from collections import OrderedDict
import torch

def merge_param(P : list(OrderedDict([(str, torch.Tensor),]),), device='cpu'):
    
    # for now, ordered Dict is default
    par = P[0]
        
    if isinstance(par, OrderedDict):
        for i in range(1,len(P)):
            for key in par.keys():
                par[key]=torch.add((par[key]).to(torch.device(device)),(P[i][key]).to(torch.device(device)))
        for key in par.keys():
            par[key] = par[key]/len(P)
            
    elif isinstance(par, list):
        with torch.no_grad():
            for i in range(1,len(P)):
                for j in range(len(par)):
                    par[j].add_(P[i][j].to(torch.device(device)))
            for j in range(len(par)):
                par[j] = par[j]/len(P)
    else:
        raise TypeError("parameter type is not supported")
    return par