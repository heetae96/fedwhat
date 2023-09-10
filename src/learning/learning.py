import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import threading
try:
    from models import *
except:
    from ..models import *
    
from ._custom_dataloader import *
from ._merge_param import *

class learning:
    
    def __init__(self, fednode):
        self.st = fednode.st
        self.prev_t=time.time()
        self.alive = True
        self.rounds =0
        
        self.dataconfig = fednode.data_config
        self.device = fednode.device if torch.cuda.is_available() else 'cpu'
        self.iid = fednode.iid
        
        self.start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        # Data

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5,), (0.5,)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            # transforms.Normalize((0.5,), (0.5,)),
        ])

        trainset = custom_dataset(
            root='./data', train=True, download=True, transform=transform_train, 
            split_number=self.dataconfig[0], split_id=self.dataconfig[1], iid=self.iid, dataset_name="cifar10" )
        self.trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=64, shuffle=True, num_workers=2)

        # testset = torchvision.datasets.EMNIST(
        #     root='./data', split='balanced', train=False, download=False, transform=transform_test)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test)
        self.testloader = data.DataLoader(
            testset, batch_size=80, shuffle=False, num_workers=2)

        #self.classes = ('plane', 'car', 'bird', 'cat', 'deer',
        #        'dog', 'frog', 'horse', 'ship', 'truck')

        # Model
        # must check num_classes!!!
        
        self.net = ResNet18()
        self.net = self.net.to(self.device)
        if self.device!='cpu':
            cudnn.benchmark = True
        self.criterion = nn.CrossEntropyLoss()
        
        #self.optimizer = optim.Adam(self.net.parameters())
        #self.optimizer = optim.SGD(self.net.parameters())
        self.optimizer = optim.SGD(self.net.parameters(),lr=0.01,momentum=0.9, weight_decay=5e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500)
        self.fedopt = FedAvg_modif()
        
    # Training
    def train(self, epoch):
        net, trainloader, device, optimizer, criterion = self.net, self.trainloader, self.device, self.optimizer, self.criterion
        
        print('\nEpoch: %d' % epoch)
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

        print("train loss : %0.4f  train acc : %0.2f" %(train_loss/(batch_idx+1),100.*correct/total))


    def test(self, epoch=0):
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


    def run(self, fednode):
        self.addr = fednode.addr
        
        self.id = fednode.id
        self.finger_table = fednode.finger_table
        
        print(self.id, self.addr)
        net, start_epoch = self.net, self.start_epoch
        train, test = self.train, self.test
        
        for rounds in range(200):
            self.rounds = rounds
            if rounds!=0:
                """
                wait until self.par == rounds
                """
                
                while True:
                    if len(self.parQueue)>1:
                        break
                    time.sleep(1)
                up_t = time.time()
                
                # self.net.load_state_dict(merge_param(self.parQueue, self.device))
                with torch.no_grad():
                    #new_param = merge_param(self.parQueue, self.device)
                    new_param = self.fedopt.do(par=self.parQueue[0][2], P=self.parQueue,dev=self.device)
                
                # for param, _new in zip(net.parameters(), new_param):
                #     param.data = _new
                self.net.load_state_dict(new_param)
                
                self.net = self.net.to(torch.device(self.device))
                #print("parameters loaded time:{:.4f}".format(time.time()-up_t))
                self.parQueue = [(self.id, 0,None),]
                
                test(epoch)
            else:
                self.parQueue = [(self.id, 0,None),]
            
            for epoch in range(start_epoch, start_epoch+5):
                train(epoch)
                #test(epoch)
                #scheduler.step()
                print("epoch time : {:.4f} ({:.4f})".format(time.time()-self.st, time.time()-self.prev_t))
                self.prev_t = time.time()
                self.scheduler.step()

            start_epoch+=5
            par = self.net.state_dict()
            
            #save model
            #torch.save(net.state_dict(), PATH)
            
            #par = [param for param in self.net.parameters()]
            self.parQueue[0]=(self.id, self.rounds, par)
            self.finger_table = fednode.finger_table
            send = threading.Thread(target=self.push_param, args=(self.rounds,))
            send.start()