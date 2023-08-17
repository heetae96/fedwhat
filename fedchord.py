"""
This is implementation of Federated learning with P2P communication with DHT-chord protocol.

- run (must run root node first)
    root node  : python p2p_tcp.py 8000(this node's port)
    other node : python p2p_tcp.py 7000(this node's port) 8000(root node port)
"""
from src.fednode import fednode
from src.utils import handle_args_f
N=9
if __name__ == '__main__':
    this_addr, help_addr, logger, container, device, case_n, iid = handle_args_f()
    # if case_n in [0,1,2,3,4,5,6,7]:
    #     case = [8,case_n]
    if case_n in range(N):
        case = [N,case_n]
    else:
        raise ValueError("case_n must be 0, 1, 2, or 3")
        
    #node = P2PNode(logger, (this_addr), (help_addr))
    node = fednode(logger, (this_addr), (help_addr), container, device, case, iid)
    
    
"""
python fedchord.py -p 12000 -g 0 -t 1

python fedchord.py -p 17271 -P 12000 -g 1 -t 2


"""