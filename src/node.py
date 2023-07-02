import socket, time
import threading
import hashlib
import pickle
import selectors

try:
    from . import utils
except:
    import utils
NUM_OF_BITS = 16

__all__ = ['P2PNode']

#TODO : error handling, define solo remain node's behavior

contain = utils.contain

class P2PNode:
    """
    node of Chord DHT
    node's id is generated by hash function
    """
    
    def __init__(self, logger, addr, host_addr=None, container=False, data_no=0):
        
        self.addr = addr
        self.host_addr = host_addr
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(addr)
        self.socket.setblocking(False)
        self.socket.listen(5)
        self.id = self.__hash(self.addr)
        #self.id = int(((1<<16)/5)*int(data_no))         

        #for container..
        if container:
            self.addr = (utils.get_global_ip(), self.addr[1])
        
        self.finger_table = list(("",0) for i in range(NUM_OF_BITS))
        self.__init_finger_table()
        self.predecessor_addr = None
        self.predecessor_id = -1
        self.successor_addr = self.addr
        self.successor_id = self.id
        self.socketlist=[self.socket]

        self.alive = True
        
        self.logger = logger
        """
        loop for stablize after few seconds, to update finger table, successor and predecessor after join or unjoin of other nodes
        """
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.socket, selectors.EVENT_READ, self.__accept_handler)
        self.listen_t = threading.Thread(target=self.run, daemon=True, name="run")
        self.daemon_t = threading.Thread(target=self.daemon, daemon=True, name="daemon")
        
        self.listen_t.start()
        self.join()
        self.daemon_t.start()
        
        #while self.alive:
        self.mainjob()
    
    ######################## methods called by init #######################
    
    def mainjob(self):
        while self.alive:
            time.sleep(5)
        #pass
            
    def __init_finger_table(self):
        """
        initialize finger table
        """
        for i in range(NUM_OF_BITS):
            self.finger_table[i] = (self.addr, self.id)    
    
    def __hash(self, addr):
        """
        hash function, that generated id of node by address, using sha1
        """
        return int(hashlib.sha1(str(addr).encode()).hexdigest(), 16) % (2**NUM_OF_BITS)
    
    ##################### daemon thread called by init ####################
    
    def daemon(self):
        while self.alive:
            self.__stabilize()
            self.__fix_fingers()
            time.sleep(3)

    def __stabilize(self):
        """
        stabilize the network
        """
        if(self.successor_addr != self.addr):
            message = "start stabilizing"
            #self.logger.debug(message)  
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.connect(self.successor_addr)
                sock.send(pickle.dumps(('get_predecessor', self.addr, self.id)))
                message = "---- wait for recv[get_predecessor] from {}".format(self.successor_addr)
                #self.logger.debug(message)
                data = self.recv_data(sock)
                if data==None:
                    raise MyError
                message = "received data from successor\n{}".format(data)
                #self.logger.debug(message) 
                if data[1] != -1 and contain(data[1], self.id, self.successor_id):
                    self.successor_addr = data[0]
                    self.successor_id = data[1]
                    message = "updated successor to\n{}".format(self.successor_addr)
                    #self.logger.debug(message) 
                self._notify()
                message = "notified me to successor"
                #self.logger.debug(message) 
            except MyError:
                #self.logger.warning("error to stabilize(maybe deadlock)")
                pass
            except:
                """
                connect to next address in finger table
                """
                message = "Node {} left".format(self.successor_addr)
                #self.logger.debug(message)
                for i,addr in enumerate(self.finger_table):
                    if (addr[0] != self.successor_addr):
                        self.successor_addr = addr[0]
                        self.successor_id = addr[1]
                        for j in range(i):
                            self.finger_table[j] = self.finger_table[i]
                        break
                    else:
                        self.finger_table[i]= (self.predecessor_addr,self.predecessor_id)
                #if(self._notify_leave()==False):
                    #message = "error to notifying leave"
                    #self.logger.warning(message) 
                    #self.print_finger_table()   
                # message = "ended stabilizing"
                # self.logger.debug(message) 
                #time.sleep(5)
            message = "\nself id : %d || succ id : %d || pred id : %d" % (self.id, self.successor_id, self.predecessor_id)
            self.logger.info(message)
                   
    def __fix_fingers(self):
        """
        fix fingers
        """
        if self.successor_addr != self.addr:
            message = "=> start fixing fingers"
            self.logger.info(message)  
            #self.print_finger_table("debug")
            start=time.time()
            self._update_finger_table()
            self.print_finger_table()
            #time.sleep(5)
            end=time.time()
            message = "=> ended fixing fingers in {} seconds\n".format(end-start)
            self.logger.info(message) 
 
    ################### listening thread called by init ###################
    
    def run(self):
        """
        thread for listening
        """
        while self.alive:
            self.selector = selectors.DefaultSelector()
            self.selector.register(self.socket, selectors.EVENT_READ, self.__accept_handler)
            while self.alive:
                for (key,mask) in self.selector.select():
                    key: selectors.SelectorKey
                    srv_sock, callback = key.fileobj, key.data
                    callback(srv_sock, self.selector)

    def __accept_handler(self, sock: socket.socket, sel: selectors.BaseSelector):
        """
        accept connection from other nodes
        """
        conn: socket.socket
        conn, addr = sock.accept()
        sel.register(conn, selectors.EVENT_READ, self.__read_handler)

    def __read_handler(self, conn: socket.socket, sel: selectors.BaseSelector):
        """
        read data from other nodes
        """
        message = "---- wait for recv[any other] from {}".format(conn.getpeername())
        #self.logger.debug(message)  
        data = conn.recv(1024)
        time.sleep(0.5)
        #self._handle(data, conn)
        data = pickle.loads(data)
        self.logger.debug("[recv data : {}]".format(data))
        #print(data)
        threading.Thread(target=self._handle, args=((data,conn)), daemon=True).start()
        sel.unregister(conn)

    ################## core method thread called by init ##################

    def join(self):
        """
        join to the network
        """
        if self.host_addr:
            self._find_successor(self.host_addr)
            self._notify()
        else:
            self.successor_addr = self.addr
            self.predecessor_addr = self.addr
            self.successor_id = self.id
            self.predecessor_id = self.id
    
    def _find_successor(self, addr):
        """
        find successor of id, data sent by pickle
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(addr)
        sock.send(pickle.dumps(('find_successor', self.addr, self.id)))
        message = "---- wait for recv[find_successor] from {}".format(addr)
        #self.logger.debug(message)
        try:
            data=self.recv_data(sock)
            if data==None:
                raise MyError
        except MyError:
            #self.logger.warning("error to find_successor(maybe deadlock)")
            pass
        """
        get successor address and id with mutex
        """
        self.successor_addr = data[0]
        self.successor_id = data[1]
        sock.close()
  
    def _notify(self):
        """
        notify successor about self
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(self.successor_addr)
        sock.send(pickle.dumps(('notify', self.addr, self.id)))
        sock.close()

    def _update_finger_table(self):
        """
        update finger table
        """
        self.finger_table[0] = (self.successor_addr, self.successor_id)
        for i in range(1, NUM_OF_BITS):
            id = (self.id + 2 ** i) % 2 ** NUM_OF_BITS
            addr = self._find_successor_by_id(id, None)
            if(addr!=None):
                if self.finger_table[i] != addr:
                    self.logger.info("update finger table : %d(%d)" % (id,i))
                    self.finger_table[i] = addr

    def _find_successor_by_id(self, id, conn: socket.socket):
        """
        find successor of id
        """
        if contain(id, self.id, self.successor_id):
            if conn:
                conn.send(pickle.dumps((self.successor_addr, self.successor_id)))
            else:
                return (self.successor_addr, self.successor_id)
        else: 
            if conn:
                self._find_closest_preceding_finger(id, conn)
            else:
                return self._find_closest_preceding_finger(id, None)
    
    def _find_closest_preceding_finger(self, id:int, conn: socket.socket):
        """
        find closest preceding finger
        """
        check=False
        for i in range(NUM_OF_BITS - 1, -1, -1):
            
            if contain(self.finger_table[i][1], self.id, id):
                check=True
                try:
                    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    sock.connect(self.finger_table[i][0])
                    self.logger.debug("--> sent find_successor_by_id({}) to {}".format(id,self.finger_table[i][0]))
                    sock.send(pickle.dumps(('find_successor_by_id', self.addr, self.id, id)))
                    message = "-|- wait for recv[find_successor_by_id({})] from {}".format(id,self.finger_table[i][0])
                    self.logger.debug(message)
                    data=self.recv_data(sock)
                    message = "<-- recv data [{}] {}".format(data, threading.current_thread().getName())
                    if threading.current_thread().getName() == self.daemon_t.getName():
                        self.logger.debug(message, extra={'color': 'red'})
                    if data==None:
                        raise MyError
                except MyError:
                    self.logger.warning("error to find_closest_preceding_finger(maybe deadlock)")
                except:
                    data = self.finger_table[(i+1)%NUM_OF_BITS]
                    
                if conn:
                    conn.send(pickle.dumps(data))
                else:
                    return data
                break
        if check==False:
            if (self.id==id):
                return((self.addr, self.id))
            return ((self.successor_addr, self.successor_id))
            #self.logger.warning("this is terrible situation")

    def _notify_leave(self):
        """
        notify successor that node's predecessor has left
        """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(self.successor_addr)
            sock.send(pickle.dumps(('notify_leave', self.addr, self.id)))
            sock.close()
            return True
        except:
            return False
    
    ######################## for send/recv data ##########################
    
    def send_data(self,addr,data):
        """
        send data to other nodes
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(addr)
        sock.send(pickle.dumps(data))
        sock.close()
    
    def recv_data(self,sock):
        """
        receive data from other nodes
        """
        #sock.settimeout(5)
        
        # for i in range (10): 
        #     try:
        #         data = pickle.loads(sock.recv(1024))
        #     except TimeoutError:
        #         data = None
        #     if data!=None: break
        
        data = pickle.loads(sock.recv(1024))
        sock.close()
        return data

    ######################## for handle a request ########################
    
    def _handle(self, data, conn: socket.socket):
        """
        handle data from other nodes
        """
        #data = pickle.loads(data)
        #self.logger.debug("[recv data : {}]".format(data))
        if data[0] == 'find_successor':
            self._handle_find_successor(data, conn)
        elif data[0] == 'notify':
            self._handle_notify(data)
        elif data[0] == 'notify_leave':
            self._handle_notify_leave(data)
        elif data[0] == 'get_predecessor':
            self._handle_get_predecessor(data, conn)
        elif data[0] == 'find_successor_by_id':
            self._handle_find_successor_by_id(data, conn)
            
        # #These two requests are for testing
        
        # elif data[0] == 'hop_count':
        #     self._handle_hop_count(data, conn)
        # elif data[0] == 'find_id':
        #     data=self._find_closest_preceding_finger(data[1], None)
        #     conn.send(pickle.dumps(data))
        
        if conn:    #if connect remain, close here.
            conn.close
       
    def _handle_find_successor(self, data, conn):
        """
        handle find_successor request
        """
        if self.id == self.successor_id:
            conn.send(pickle.dumps((self.addr, self.id)))
            self.successor_addr = data[1]
            self.successor_id = data[2]
            self.finger_table[0] = (self.successor_addr, self.successor_id)
        elif contain(data[2], self.id, self.successor_id):
            conn.send(pickle.dumps((self.successor_addr, self.successor_id)))
            self.successor_addr = data[1]
            self.successor_id = data[2]
            self.finger_table[0] = (self.successor_addr, self.successor_id)
        else:
            self._find_successor_by_id(data[2], conn)
     
    def _handle_notify(self, data):
        """
        handle notify request
        """
        if self.predecessor_id == -1 or contain(data[2], self.predecessor_id, self.id) or self.predecessor_id == self.id:
            self.predecessor_addr = data[1]
            self.predecessor_id = data[2]
 
    def _handle_notify_leave(self, data):
        """
        handle notify_leave request
        """
        self.predecessor_addr = data[1]
        self.predecessor_id = data[2]
    
    def _handle_get_predecessor(self, data, conn):
        """
        handle get_predecessor request
        """
        conn.send(pickle.dumps((self.predecessor_addr, self.predecessor_id)))

    def _handle_find_successor_by_id(self, data, conn: socket.socket):
        """
        handle find_successor_by_id request
        """
        if(data[3]==self.id):
            conn.send(pickle.dumps((self.addr, self.id)))
        else:
            self._find_successor_by_id(data[3], conn)


    # def _handle_hop_count(self, data, conn):
    #     """
    #     handle hop_count request
    #     """
    #     #time.sleep(5)
    #     if(data[1] < 3):
    #         for finger in set(self.finger_table):
    #             if(finger[0] == self.addr):
    #                 continue
    #             sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    #             sock.connect(finger[0])
    #             sock.send(pickle.dumps(('hop_count', data[1]+1,self.addr[1])))
    #             sock.close()
    #             message ="hop_count: from: "+str(data[2])+"::"+str(data[1])+"-> to: "+str(finger[0][1])+"::"+str(data[1]+1)
    #             self.logger.warning(message)
    #             print(message)
    # ############################# for logging ############################
    def print_finger_table(self, level="info"):
        """
        print finger table with ideal id
        """
        message="\n<finger table>"
        
        for i,elem in enumerate(self.finger_table):
            id = (self.id + 2 ** i) % 2 ** NUM_OF_BITS
            message+="\n%2d | %s:%d (%2d)" % (id, elem[0][0], elem[0][1], elem[1])
        message+="\n"
        if level == "info":
            self.logger.info(message)
        elif level == "debug":
            self.logger.debug(message)

class MyError(Exception):
    pass

if __name__ == '__main__':
    this_addr, help_addr, logger, container= utils.handle_args()

        
    node = P2PNode(logger, (this_addr), (help_addr))