try:
    from . import node
except:
    import node

import time

'''Train CIFAR10 with PyTorch.'''

import socket, time,pickle, json
import gzip

from .learning import *
from .learning import learning as ln

import io

__all__ = ['fednode']

class fednode(node.P2PNode):
    def __init__(self, logger, addr, host_addr=None, container=False, device=None, data_config:list=[1,0], iid=True):
        self.st = time.time()
        self.device = device
        self.data_config = data_config
        self.iid = iid
        self.learning= self.learning1(self)
        
        
        super().__init__(logger, addr, host_addr, container, data_config[1])
            
    def _handle(self, data, conn):
        """
        handle data from other nodes
        """
        if data[0] == 'push_param':
            "the case of push param"
            self._handle_push_param(data,conn)
            #pass
        elif data[0] == 'weight_req':
            "the case of pull param"
            self._handle_pull_param(conn)
            #pass
        super()._handle(data, conn)
    
    def mainjob(self):
        print("start learning")
        self.learning.run(self)
            
    def _handle_push_param(self, para, conn: socket.socket):
        """
        push param to other node
        """
        #print("----- recv push_req")
        size = para[1]
        rounds = para[2]
        id = para[3]
        #print("----- send ok")
        conn.send(b'ok')
        #print(size, rounds, id)
        recv_t = time.time()            
        data=[]
        while size>0:
            temp =time.time()
            s = conn.recv(size)
            #print("----- recv_t {}".format(len(s)))
            if not s: break
            data.append(s)
            size -= len(s)
            #print("               {:.4f}".format(time.time()-temp))
        data = b''.join(data)
        
        conn.close()
        print('recv parameters time:{:.4f}'.format(time.time()-recv_t))
        unzip_t = time.time()

        data = pickle.loads(data)
        # data = torch.load(io.BytesIO(data))
        #print(type(data))
        print('deserialize time:{:.4f}'.format(time.time()-unzip_t))
        check=False
        for i,par in enumerate(self.learning.parQueue):
            if(par[0]==id):
                self.learning.parQueue[i] = (id, rounds, data)
                check=True
        if check!=True:
            self.learning.parQueue.append((id, rounds, data))

    def _handle_pull_param(self ,conn: socket.socket):
        """
        pull param from other node
        """
        #print("recv_weight_req from",conn.getsockname())
        seri_t = time.time()
        #par = self.learning.net.state_dict()
        #print(self.id)
        #self.learning.parQueue[0]=(self.id, self.learning.rounds, par)
        #print(par)
        while(self.learning.parQueue[0][2]==None):
            time.sleep(0.1)
        par = pickle.dumps(self.learning.parQueue[0][2])
        #print size of par for set the buffer size
        #print('size of par:{}B'.format(len(par)))
        par = gzip.compress(par)
        #print('size of par:{}B'.format(len(par)))
        
        #print(par)
        #par = msgpack.packb(par.numpy())
        #print('serialize the model',end=' ')
        #print('time:{:.2f}'.format(time.time()-seri_t))
        send_t = time.time()
        #print("send_meta_data")
        conn.send(pickle.dumps(('push_param', len(par), self.learning.rounds, self.id)))

        #sock.send(pickle.dumps((len(par))))
        #conn.settimeout(10)
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        if conn.recv(1024).decode('utf-8')=='ok':
            #print('recv ok')
            conn.send(par)
        conn.close()
        #print('send parameters time:{:.2f}'.format(time.time()-send_t))
      
    class learning1(ln):
            
        def push_param(self, rounds: int=0):
            seri_t = time.time()
            
            par = pickle.dumps(self.parQueue[0][2])
            # par = io.BytesIO()
            # torch.save(self.parQueue[0][2], par, pickle_protocol=5)
            
            
            print('serialize the model',end=' ')
            print('time:{:.4f}'.format(time.time()-seri_t))
            send_t = time.time()
            for finger in set(self.finger_table):
                if(finger[1]==self.id):
                    continue
                else:
                    send_t = time.time()
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(finger[0])
                    #print("---- send meta{}".format(len(par)))
                    
                    #get size of par. par is io.BytesIO()
                    sock.send(pickle.dumps(('push_param', len(par), rounds, self.id)))
                    # sock.send(pickle.dumps(('push_param', len(par.getvalue()), rounds, self.id)))

                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    if sock.recv(1024).decode('utf-8')=='ok':
                        #print("---- recv ok, send par")
                        send_t = time.time()
                        sock.send(par)
                        # sock.send(par.getvalue())
                    sock.close()
                    #print('send parameters time:{:.4f}({})'.format(time.time()-send_t, finger[0]))

        def pull_param(self):

            #finger 에 잌ㅅ는 애들..
            for finger in set(self.finger_table):

                if(finger[1]==self.id):
                    continue
                else:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(finger[0])
                    sock.send(pickle.dumps(('weight_req',)))
                    print(finger,"send_weight_req")
                    sock.setblocking(True)
                    para = sock.recv(1024)
                    para = pickle.loads(para)
                    size = para[1]
                    rounds = para[2]
                    id = para[3]
                    sock.send(b'ok')
                    print('sent ok')
                    print(size, rounds, id)
                    
                    data=[]
                    while size>0:
                        s = sock.recv(size)
                        if not s: break
                        data.append(s)
                        size -= len(s)
                    data = b''.join(data)
                    print(len(data))
                    
                    sock.close()
                    print('received data')
                    data = gzip.decompress(data)
                    print(len(data))
                    data = pickle.loads(data)
                    check=False
                    for i,par in enumerate(self.parQueue):
                        if(par[0]==id):
                            self.parQueue[i] = (id, rounds, data)
                            check=True
                    if check!=True:
                        print('------------',id)
                        self.parQueue.append((id, rounds, data))
