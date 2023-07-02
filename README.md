# **Federated learning with P2P network(DHT-chord) implemented on python**

## **Usage**

### generate chord ring and start learning
dataset : cifar100(or 10) \
model : resnet18 \
\
basic example:
```bash
#root(start) node
python fedchord.py -g <gid> -t <data_split_id> -p <root node port(12000 default)> (-i (use non_iid case(not completed yet)))

#peer node
python fedchord.py -g <gid> -t <data_split_id> -p <self port> -P <root node port> -A <root node address> (-i (use non_iid case(not completed yet)))
```