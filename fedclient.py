import socket, time
import threading
import pickle
import selectors
import subprocess
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data

from src.models import *

    
from src.learning._custom_dataloader import *

E = 5

import os

class fedclient:
    
    def __init__(self, addr:tuple, hostaddr:tuple, test, data_config, device, iid):
        
        self.alive = True
        self.state = "waiting"
        
        self.mode = test
        # daemon listening
        self.addr = addr
        self.host_addr = hostaddr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(addr)
        self.socket.setblocking(False)
        self.socket.listen(5)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.listen_t.start()
        
        self.data_config = data_config
        self.device = device
        self.iid = iid
        
        # import learning
        self.learning= self.Learning(self)
        
        # notify self
        self.join()
        
        #interactive mode for test maybe default setting or change 
        #this may become a default or may be replaced by a sending a heartbeat.
        if self.mode:
            dat = input("input the test str\n")
            while dat!='exit':
                self.send(dat)
                dat = input("input the test str\n")
            
            self.exit()
        else:
            while(self.alive):
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

    def join(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('join', self.addr)))
        sock.close()
    
    def exit(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('exit', self.addr)))
        sock.close()
            
    def send(self, data):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.host_addr)
        sock.send(pickle.dumps(('test_chat', data)))
        sock.close()
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        #print(data)
        #data = pickle.loads(data)
        #self.logger.debug("[recv data : {}]".format(data))
        if data[0] == '_func0':
            self._func0(data, conn)
        
        elif data[0] == '_start_learning':
            print("run")
            self.learning.run(self, None)
        
        elif data[0] == '_weight':
            self._update_weight(data, conn)
        
        elif data[0] == '_state':
            self._send_state(conn)
        else:
            a=0
            
        if conn:
            conn.close
    
    def _send_state(self, conn):
        conn.send(pickle.dumps(self.state))
    
    def _update_weight(self, data, conn):
        size = data[1]
        #conn.send(pickle.dumps(self.state))
        conn.send(b'ok')
        
     
        if self.state == 'waiting':
            self.state = "running"
            data=[]
            while size>0:
                s = conn.recv(size)
                #print("----- recv_t {}".format(len(s)))
                if not s: break
                data.append(s)
                size -= len(s)
                #print("               {:.4f}".format(time.time()-temp))
            data = b''.join(data)
            
            conn.close()
            
            data = pickle.loads(data)        
            # for param, _new in zip(net.parameters(), new_param):
            #     param.data = _new
            self.learning.run(self,data)
        else:
            conn.close()  
    
    def _func0(self, data, conn):
        return 0

    ####################### for handling connection ######################
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
                
    class Learning:
        
        def __init__(self, cli):

            self.st = time.time()
            self.prev_t=time.time()
            self.alive = True
            self.rounds =0
            self.parQueue = [(id, 0,None),] #will be size same as number of workers
            
            self.dataconfig = cli.data_config
            
            self.device = cli.device if torch.cuda.is_available() else 'cpu'
            
            #torch.cuda.set_per_process_memory_fraction(0.5, device=self.device)
            self.host_addr = cli.host_addr
            self.addr = cli.addr
            self.iid = cli.iid

            best_acc = 0  # best test accuracy
            self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

            # Data
            #print('==> Preparing data..')
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ])

            trainset = custom_dataset(
                root='./data', train=True, download=True, transform=transform_train, 
                split_number=self.dataconfig[0], split_id=self.dataconfig[1], iid=self.iid)
            self.trainloader = data.DataLoader(
                trainset, batch_size=128, shuffle=True, num_workers=2)
            
            testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform_test)
            self.testloader = data.DataLoader(
                testset, batch_size=100, shuffle=False, num_workers=2)

            self.net = ResNet18()
            self.net = self.net.to(self.device)
            if self.device!='cpu':
                cudnn.benchmark = True
            self.criterion = nn.CrossEntropyLoss()
            
            self.optimizer = optim.Adam(self.net.parameters())

        # Training
        def train(self, epoch):
            net, trainloader, device, optimizer, criterion = self.net, self.trainloader, self.device, self.optimizer, self.criterion
            
            print('\nEpoch: %d Round: %d' % (epoch, self.rounds))
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):

                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                #             % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

            print("train loss : %0.4f  train acc : %0.2f" %(train_loss/(len(trainloader)+1),100.*correct/total))


        def test(self):
            net, testloader, device, criterion = self.net, self.testloader, self.device, self.criterion
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()

                    if (total==10000):
                        print("test  loss : %0.4f  test acc : %0.2f" %(test_loss/(batch_idx+1),100.*correct/total))


        def run(self, cli, new_param=None):
            self.st = time.time()
            cli.state = "running"
            net = self.net
            train, test = self.train, self.test
            
            # check first learn
            if new_param!=None:
                self.net.load_state_dict(new_param)
                self.net = self.net.to(torch.device(self.device))
                test()
                
            for epoch in range(E):
                train(epoch)
                #test(epoch)
                #scheduler.step()
                print("epoch time : {:.4f} ({:.4f})".format(time.time()-self.st, time.time()-self.prev_t))
                
            self.rounds = self.rounds + 1
            self.par = self.net.state_dict()
            
            cli.state = "waiting"
            send = threading.Thread(target=self.push_param)
            send.start()
                
        def push_param(self):
        
                par = pickle.dumps(self.par)

                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(self.host_addr)
                sock.send(pickle.dumps(('push_param', len(par), self.addr)))

                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                if sock.recv(1024).decode('utf-8')=='ok':

                    sock.send(par)
                sock.close()
                
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
    parser.add_argument("--port","-p", help="this peer's port number", type=int, default=random.randint(12001, 40000))
    parser.add_argument("--addr","-a", help="this peer's ip address", type=str, default=this_ip)
    parser.add_argument("--host_port","-P", help="help peer's port number", type=int, default=12000)
    parser.add_argument("--host_addr","-A", help="help peer's ip address", type=str, default='220.67.133.165')
    parser.add_argument('--test', '-t', help="option to test", action="store_true", default=False)
    
    parser.add_argument("--data","-d", help="use (N)'th block in data", type=int, default=0)
    
    parser.add_argument("--gpu","-g", help="gpu_num", type=int, default=0)
    
    parser.add_argument("--iid","-i", help="non-iid data", action="store_false", default=True)   
    
    args = parser.parse_args()
    
    if args.gpu == -1:
        device = 'cpu'
    else:
        device = 'cuda:' + str(args.gpu)
    
    if args.data in [0,1,2,3,4,5]:
        case = [6,args.data]
    else:
        raise ValueError("case_n must be less than 6")
    
    this_addr = (args.addr, args.port)
    host_addr = (args.host_addr, args.host_port)

    client = fedclient(this_addr, host_addr, args.test, case, device, args.iid)
