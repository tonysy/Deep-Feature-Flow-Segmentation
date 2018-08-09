# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Modified by Zheng Zhang
# --------------------------------------------------------

import time
import logging
import mxnet as mx
# from lib.logger.visdomlogger import VisdomPlotLogger
class Speedometer(object):
    def __init__(self, batch_size, frequent=50):
        self.batch_size = batch_size
        self.frequent = frequent
        self.init = False
        self.tic = 0
        self.last_count = 0
        self.loss_idx_abs = 0 # remember total index from the first epoch
        # self.train_loss_logger = VisdomPlotLogger('line', env='deeplab_duc_dff',opts={'title': 'Train FCNLoss'})
    def __call__(self, param):
        """Callback to Show speed."""
        count = param.nbatch
        if self.last_count > count:
            self.init = False
        self.last_count = count

        if self.init:
            if count % self.frequent == 0:
                speed = self.frequent * self.batch_size / (time.time() - self.tic)
                s = ''
                if param.eval_metric is not None:
                    name, value = param.eval_metric.get()
                    s = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec\tTrain-" % (param.epoch, count, speed)
                    for n, v in zip(name, value):
                        s += "%s=%f,\t" % (n, v)
                        if n == 'FCNLogLoss':
                            self.loss_idx_abs += count
                            FCNLogLoss = v
                    # self.train_loss_logger.log(self.loss_idx_abs,FCNLogLoss)
                else:
                    s = "Iter[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (param.epoch, count, speed)
                    
                logging.info(s)
                print(s)
                self.tic = time.time()
        else:
            self.init = True
            self.tic = time.time()
