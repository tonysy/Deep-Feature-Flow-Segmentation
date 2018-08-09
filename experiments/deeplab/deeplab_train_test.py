# --------------------------------------------------------
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2017 ShanghaiTech PLUS Group
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# Written by Songyang Zhang 
# E-main: sy.zhangbuaa#gmail.com
# --------------------------------------------------------

import os
import sys
os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
this_dir = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(this_dir, '..', '..', 'deeplab'))

import train
import test

if __name__ == "__main__":
    train.main()
    # test.main()




