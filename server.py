import socket, time
import threading
import pickle
import selectors
from src.learning._merge_param_buff import *
import random

# TODO client heartbeat 받기..?
#K = 1
K = 3
T = 5 #the total number of clients

class Server:
    """
    central server in federated learning
    
    func: 
        - Daemon listening
        - Accept client's join
        - Save all client's address(ip,port)
        - Request learning Some or All client
        - Wait all chosen client's weight (exact fedavg)
            - Wait K of chosem client's weight (fedbuff (semi-async))
            - Wait client's weight for seconds (async)
        - Aggregate weights
        - Send aggregated weight to all client
    
    thread:
        listening

    """
    def __init__(self, addr:tuple):
        # self socket
        self.addr = addr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(addr)
        self.socket.setblocking(False)
        self.socket.listen(5)
        
        # state of self > maybe deprecate later
        self.alive = True
        self.start =0
        
        # set listening daemon
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        print('start') 
        self.listen_t.start()
        
        # ready for manage client
        self.c_list = {0:[], 1:[]}
        
        #self.K = 0
        self.weights = dict()
        self.rounds=0
        
        # default job(for now inf loop)
        self.mainjob()

    
    def mainjob(self):
        while self.alive:
            time.sleep(5)
    
    def run(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
            while self.alive:
                for (key,mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        #print(data)
        #data = pickle.loads(data)
        #self.logger.debug("[recv data : {}]".format(data))
        if data[0] == 'join':
            self._join(data)
        
        elif data[0] == 'exit':
            self._exit(data)
            
        elif data[0] == 'func0':
            self._func0(data, conn)
        
        elif data[0] == 'func1':
            self._func1(data, conn)

        elif data[0] == 'test_chat':
            self._test_chat(data, conn)
            
        elif data[0] == 'push_param':
            self._handle_push_param(data, conn)
        else:
            a=0
            
        if conn:
            conn.close
    
    def _func0(self, data, conn):
        return 0
    
    def _func1(self, data, conn):
        
        return 0
    
    def _handle_push_param(self, data, conn: socket.socket):
        #print(self.c_list, data[2])
        self.c_list[1].remove(data[2])
        self.c_list[0].append(data[2])
        
        size = data[1]
        conn.send(b'ok')
        
        #receive param                   
        data=[]
        while size>0:
            s = conn.recv(size)
            if not s: break
            data.append(s)
            size -= len(s)
        data = b''.join(data)
        
        conn.close()

        data = pickle.loads(data)
        self.weights[conn.getpeername] = data
        #print(self.c_list)
        if len(self.weights) >= K:
            par = list(self.weights.values())
            self.weights = dict()
            print("received %d weights, now round %d" %(len(par),self.rounds))
            
            self.rounds+=1
            par = merge_param(par)
            self.weights = dict()
            self.update_client(par)
            


    def update_client(self, param):
        par = pickle.dumps(param)
        
        clients = self.c_list[0]
        random.shuffle(clients)
        
        for c in clients:
            print(c)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(c)
            sock.send(pickle.dumps(('_weight', len(par))))

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            if sock.recv(1024).decode('utf-8')=='ok':
                sock.send(par)
                self.c_list[0].remove(c)
                self.c_list[1].append(c)
            sock.close()
    
    def _join(self, data):
        """
        join and start
        
        two posible method 
            [v] Start All : server knows the total number(constant) of client.
            [ ] Start minimum : server doesn't know the total number of client, 
                so server starts the minimum number(constant) of client.
        """
        
        new_client = data[1]
        self.c_list[1].append(new_client)
        print('new client {} joined!\nnow {} clients in list!'.format(new_client, len(self.c_list[1])))
        

        if self.start==0 and (len(self.c_list[1])) >= T:
            for c in self.c_list[1]:
                #print(c)
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(c)
                sock.send(pickle.dumps(('_start_learning',0)))
                print('learning start {}'.format(c))
                sock.close()
            self.start=1
    
    def _exit(self, data):
        client = data[1]
        self.c_list[1].remove(client)
        print('client {} exit!\nnow {} clients in list!'.format(client, len(self.c_list)))
    
    def _test_chat(self, data, conn:socket.socket):
        print(" > ", conn.getpeername(), " says ", data[1])
    
    def accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = sock.accept()
        sel.register(conn, selectors.EVENT_READ, self.read_handler)

    def read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        message = "---- wait for recv[any other] from {}".format(conn.getpeername())
        #self.logger.debug(message)  
        data = conn.recv(1024)
        time.sleep(0.5)
        #self._handle(data, conn)
        data = pickle.loads(data)
        threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        sel.unregister(conn)
        

import subprocess
import argparse
        
def get_global_ip():
    """
    get global ip address
    """
    return subprocess.check_output("wget http://ipecho.net/plain -O - -q ; echo", shell=True).decode().strip()

def get_self_ip():
    return socket.gethostbyname(socket.gethostname())

if __name__ == '__main__':
    this_ip = get_self_ip()
    parser = argparse.ArgumentParser()
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=12000)
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    args = parser.parse_args()
    
    this_addr = (args.addr, args.port)

    server = Server(this_addr)
