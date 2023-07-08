from collections import OrderedDict
import torch
from abc import abstractmethod
import copy

def merge_param(P : list((int,int,OrderedDict([(str, torch.Tensor),])),), device):

    par = P[0][2]
    '''
    for p in P:
        print(p[1],end=' ')
    print()


    if isinstance(par, OrderedDict):
        """
        case when use model.statedict.
        """
        for i in range(1,len(P)):
            for key in par.keys():
                par[key]=torch.add(par[key],(P[i][2][key]).to(torch.device(device)))
        for key in par.keys():
            par[key] = par[key]/len(P)

    elif isinstance(par, list):
        """
        case when use model.param etc.
        """ 
        with torch.no_grad():
            for i in range(1,len(P)):
                for j in range(len(par)):
                    par[j].add_(P[i][2][j].to(torch.device(device)))
            for j in range(len(par)):
                par[j] = par[j]/len(P)
    else:
        raise TypeError("parameter type is not supported")
    '''
    
    optim = FedYogi()
    aggregated = optim.do(par=par,P=P,dev=device)
    
    return aggregated

class FedAvg_modif:
    '''
    rate = 1/len(p) : 현재
    '''
    def __init__(self):
        """Initialize FedAvg instance."""
        self.agg_weights = None
        self.aggregate_fn = self._aggregate_pytorch
        
    def do(self,
           par,
           P,
           *,
           dev,
           total: int = 0,
           version: int = 0,
           **kwargs):

        assert (par is not None)

        # if len(P) == 0 or total == 0:
        #     return None
        
        """
        rate계산위치가 에매해짐
        """
        rate = 1/len(P)
        # reset global weights before aggregation
        self.agg_weights = copy.deepcopy(par)
        for key in par.keys():
            self.agg_weights[key] = par[key]/len(P)
        
        for k in range(1,len(P)):
            # after popping, the item is removed from the cache
            # hence, explicit cache cleanup is not needed

            ## rate = tres.count / total
            self.aggregate_fn(P[k][2], rate, dev)

        return self.agg_weights

    def _aggregate_pytorch(self, par, rate, dev):
        
        for k in par.keys():
            tmp= par[k] *rate
            tmp = tmp.to(torch.device(dev))
            self.agg_weights[k] += tmp

class FedOPT(FedAvg_modif):

    """FedOPT class."""

    def __init__(self, beta_1, beta_2, eta, tau):
        """Initialize FedOPT instance."""
        super().__init__()
        self.current_weights = None
        self.d_t = None
        self.m_t = None
        self.v_t = None
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.eta = eta
        self.tau = tau
        self.adapt_fn = self._adapt_pytorch
        
    def do(self,
           par,
           P,
           *,
           dev,
           total: int = 0,
           version: int = 0,
           **kwargs):
        """Do aggregates models of trainers.

        Parameters
        ----------
        base_weights: weights to be used as base
        cache: a container that includes a list of weights for aggregation
        total: a number of data samples used to train weights in cache
        version: a version number of base weights

        Returns
        -------
        aggregated model: type is either list (tensorflow) or dict (pytorch)
        """

        self.agg_weights = super().do(par=par,P=P,dev=dev)
        if self.agg_weights is None:
            return self.current_weights

        if self.current_weights is None:
            self.current_weights = self.agg_weights
        else:
            self.adapt_fn(self.agg_weights, self.current_weights)

        return self.current_weights

    @abstractmethod
    def _delta_v_pytorch(self):
        return

    def _adapt_pytorch(self, average, current):
        import torch


        self.d_t = {k: average[k] - current[k] for k in average.keys()}

        if self.m_t is None:
            self.m_t = {
                k: torch.zeros_like(self.d_t[k])
                for k in self.d_t.keys()
            }
        self.m_t = {
            k: self.beta_1 * self.m_t[k] + (1 - self.beta_1) * self.d_t[k]
            for k in self.m_t.keys()
        }

        if self.v_t is None:
            self.v_t = {
                k: torch.zeros_like(self.d_t[k])
                for k in self.d_t.keys()
            }
        self._delta_v_pytorch()

        self.current_weights = OrderedDict({
            k: self.current_weights[k] + self.eta * self.m_t[k] /
            (torch.sqrt(self.v_t[k]) + self.tau)
            for k in self.current_weights.keys()
        })

class FedAdaGrad(FedOPT):
    """FedAdaGrad class."""

    def __init__(self, beta_1=0.9, beta_2=0.99, eta=1e-2, tau=1e-3):
        """Initialize Fedadagrad instance."""
        super().__init__(beta_1, beta_2, eta, tau)

    def _delta_v_pytorch(self):
        self.v_t = {k: self.v_t[k] + self.d_t[k]**2 for k in self.v_t.keys()}
        return

class FedAdam(FedOPT):
    """FedAdam class."""

    def __init__(self, beta_1=0.9, beta_2=0.99, eta=1e-2, tau=1e-3):
        """Initialize FedAdam instance."""
        super().__init__(beta_1, beta_2, eta, tau)

    def _delta_v_pytorch(self):
        self.v_t = {k: self.beta_2 * self.v_t[k] + (1 - self.beta_2) * self.d_t[k]**2 for k in self.v_t.keys()}
        return
    
class FedYogi(FedOPT):
    """FedYogi class."""

    def __init__(self, beta_1=0.9, beta_2=0.99, eta=1e-2, tau=1e-3):
        """Initialize FedYogi instance."""
        super().__init__(beta_1, beta_2, eta, tau)

    def _delta_v_pytorch(self):
        import torch
        self.v_t = {k: self.v_t[k] - (1 - self.beta_2) * self.d_t[k]**2 * torch.sign(self.v_t[k] - self.d_t[k]**2) for k in self.v_t.keys()}
        return
