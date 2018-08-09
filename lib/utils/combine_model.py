# --------------------------------------------------------
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Copyright (c) 2017 ShanghaiTech PLUS Group
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Zheng Zhang
# Written by Songyang Zhang 
# E-main: sy.zhangbuaa#gmail.com
# --------------------------------------------------------

from load_model import load_checkpoint
from save_model import save_checkpoint


def combine_model(prefix1, epoch1, prefix2, epoch2, prefix_out, epoch_out):
    args1, auxs1 = load_checkpoint(prefix1, epoch1)
    args2, auxs2 = load_checkpoint(prefix2, epoch2)
    arg_names = args1.keys() + args2.keys()
    aux_names = auxs1.keys() + auxs2.keys()
    args = dict()
    for arg in arg_names:
        if arg in args1:
            args[arg] = args1[arg]
        if arg in args2:
            args[arg] = args2[arg]
    auxs = dict()
    for aux in aux_names:
        if aux in auxs1:
            auxs[aux] = auxs1[aux]
        if aux in auxs2:
            auxs[aux] = auxs2[aux]
    save_checkpoint(prefix_out, epoch_out, args, auxs)
