import mxnet as mx 
from deeplab import _init_paths
# from deeplab.symbols.resnet_v1_101_deeplab_dcn_duc import resnet_v1_101_deeplab_dcn_duc
# from deeplab.symbols.duc_hdc_symbol.network_duc_hdc import get_symbol_duc_hdc
from deeplab.symbols.densenet_bc_deeplab_base import densenet_bc_deeplab_base

def plot_network(symbol, input_data_shape):
    t = mx.viz.plot_network(symbol, shape={'data' : input_data_shape})
    t.render()
    

if __name__ == '__main__':
    # For resnet-dcn 
    # resnet_dcn = resnet_v1_101_deeplab_dcn_duc()
    # symbol = resnet_dcn.get_train_duc_symbol(19)
    # input_data_shape = (1, 3, 1024, 2048)
    # plot_network(symbol, input_data_shape)

    # symbol_duc = get_symbol_duc_hdc(19, 16)
    # plot_network(symbol_duc, input_data_shape)
    
    # For DenseNet
    # depth = 121

    # if depth == 121:
    #     units = [6, 12, 24, 16]
    # elif depth == 169:
    #     units = [6, 12, 32, 32]
    # elif depth == 201:
    #     units = [6, 12, 48, 32]
    # elif depth == 161:
    #     units = [6, 12, 36, 24]
    # else:
    #     raise ValueError("no experiments done on detph {}, you can do it youself".format(depth))
    
    # reduction = 0.5

    # symbol_densenet = DenseNet(units=units, num_stage=4, growth_rate=48 if depth==161 else 32, 
    #                     num_class=1000, data_type='imagenet', reduction=reduction, drop_out=0, bottle_neck=True,
    #                     bn_mom=0.9, workspace=512)
    densenets = densenet_bc_deeplab_base()
    input_data_shape = (1, 3, 1024, 2048)
    symbol_densenet = densenets.get_train_symbol(19)
    plot_network(symbol_densenet, input_data_shape)
