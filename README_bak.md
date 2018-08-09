# Deeplab-mxnet
Deeplab-mxnet Re-Implement
## Tips
1. When your use other network for segmentation, please add newtwork class name into `symbols.__init__.py`

## Installation
1. `conda create --name mxnet python=2.7`
2. `pip install -r requirements.txt`
3. `cp /usr/local/lib/python2.7/dist-packages/cv* anaconda3/envs/mxnet/lib/python2.7/`
4. `sh ./init.sh`



## Usage:
1. visdom

python -m visdom.server(or proxychains python -m visdom.server)
## FAQ
1.
Program hang if your system opencv is 2.x and your opencv-python is 3.x
you can  `cp /usr/local/lib/python2.7/dist-packages/cv* .mxnet_0.12/local/lib/python2.7/`

2. visdom hang 
sudo apt-get install proxychains

Make a config file at ~/.proxychains/proxychains.conf with content:
```
strict_chain
proxy_dns 
remote_dns_subnet 224
tcp_read_time_out 15000
tcp_connect_time_out 8000
localnet 127.0.0.0/255.0.0.0
quiet_mode

[ProxyList]
socks5  127.0.0.1 1080
```

Then run command with proxychains. Examples:

`proxychains curl https://www.twitter.com/`

sslocal -s 198.181.33.53 -p 443 -k "shanghaitechcvml" -l 1080 -t 600 -m aes-256-cfb

## Uninstall mxnet (Build from source)
``
`/anaconda3/envs/mxnet/lib/python2.7/site-packages/easy-install.pth`

## Docker build
`docker build -t mxnet_plus/python:gpu -f ./Dockerfile.python.gpu .`
## Enter an container
`nvidia-docker run -it -v /home/syzhang:/home/syzhang -v /home/PublicDataset:/home/PublicDataset -v /home/PublicModel:/home/PublicModel  mxnet_plus/python:gpu bash`

python experiments/deeplab/deeplab_train_test.py --cfg experiments/deeplab/cfgs/deeplab_resnet_v1_101_cityscapes_segmentation_base.yaml

pypi

https://mirrors.geekpie.club/pypi/