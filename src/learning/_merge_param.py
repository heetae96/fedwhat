from collections import OrderedDict
import torch

def merge_param(P : list((int,int,OrderedDict([(str, torch.Tensor),])),), device):

    par = P[0][2]
    
    for p in P:
        print(p[1],end=' ')
    print()
    
    if isinstance(par, OrderedDict):
        for i in range(1,len(P)):
            for key in par.keys():
                par[key]=torch.add(par[key],(P[i][2][key]).to(torch.device(device)))
        for key in par.keys():
            par[key] = par[key]/len(P)
            
    elif isinstance(par, list):
        with torch.no_grad():
            for i in range(1,len(P)):
                for j in range(len(par)):
                    par[j].add_(P[i][2][j].to(torch.device(device)))
            for j in range(len(par)):
                par[j] = par[j]/len(P)
    else:
        raise TypeError("parameter type is not supported")
    return par