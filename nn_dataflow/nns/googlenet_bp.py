from nn_dataflow.core import Network
from nn_dataflow.core.layer import InputLayer, ConvLayer, FCLayer, PoolingLayer, \
    ConvBackActLayer, ConvBackWeightLayer, FCBackActLayer, FCBackWeightLayer, PoolingBackLayer, EltwiseLayer

'''
GoogLeNet

ILSVRC 2014
'''

NN = Network('GoogLeNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
NN.add('pool1', PoolingLayer(64, 56, 3, strd=2))
# Norm layer is ignored.

NN.add('conv2_3x3_reduce', ConvLayer(64, 64, 56, 1))
NN.add('conv2_3x3', ConvLayer(64, 192, 56, 3))
# Norm layer is ignored.
NN.add('pool2', PoolingLayer(192, 28, 3, strd=2))


def add_inception(network, incp_id, sfmap, nfmaps_in, nfmaps_1, nfmaps_3r,
                  nfmaps_3, nfmaps_5r, nfmaps_5, nfmaps_pool, prevs):
    ''' Add an inception module to the network. '''
    pfx = 'inception_{}_'.format(incp_id)
    # 1x1 branch.
    network.add(pfx + '1x1', ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1),
                prevs=prevs)
    # 3x3 branch.
    network.add(pfx + '3x3_reduce', ConvLayer(nfmaps_in, nfmaps_3r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '3x3', ConvLayer(nfmaps_3r, nfmaps_3, sfmap, 3))
    # 5x5 branch.
    network.add(pfx + '5x5_reduce', ConvLayer(nfmaps_in, nfmaps_5r, sfmap, 1),
                prevs=prevs)
    network.add(pfx + '5x5', ConvLayer(nfmaps_5r, nfmaps_5, sfmap, 5))
    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1),
                prevs=prevs)
    # Merge branches.
    return (pfx + '1x1', pfx + '3x3', pfx + '5x5', pfx + 'pool_proj')


def add_back_inception(network, incp_id, sfmap, nfmaps_out, nfmaps_1, nfmaps_3r, nfmaps_3,
                       nfmaps_5r, nfmaps_5, nfmaps_pool, nfmaps_in, prevs):
    pfx = 'inception_{}_back'.format(incp_id)
    pfx_w = 'inception_{}_back_w_'.format(incp_id)

    nfmaps_1_input_name = 'inception_{}_back_1x1_input'.format(incp_id)
    nfmaps_3_input_name = 'inception_{}_back_3x3_input'.format(incp_id)
    nfmaps_5_input_name = 'inception_{}_back_5x5_input'.format(incp_id)
    nfmaps_pool_input_name = 'inception_{}_back_pool_input'.format(incp_id)

    network.add(nfmaps_1_input_name, ConvLayer(nfmaps_in, nfmaps_1, sfmap, 1), prevs=prevs)
    network.add(nfmaps_3_input_name, ConvLayer(nfmaps_in, nfmaps_3, sfmap, 1), prevs=prevs)
    network.add(nfmaps_5_input_name, ConvLayer(nfmaps_in, nfmaps_5, sfmap, 1), prevs=prevs)
    network.add(nfmaps_pool_input_name, ConvLayer(nfmaps_in, nfmaps_pool, sfmap, 1), prevs=prevs)

    # 1x1 branch.
    network.add(pfx + '1x1', ConvBackActLayer(nfmaps_1, nfmaps_out, sfmap, 1), prevs=nfmaps_1_input_name)
    network.add(pfx_w + '1x1', ConvBackWeightLayer(nfmaps_1, nfmaps_out, sfmap, 1), prevs=nfmaps_1_input_name)

    # 3x3 branch.
    network.add(pfx + '3x3', ConvBackActLayer(nfmaps_3, nfmaps_3r, sfmap, 3), prevs=nfmaps_3_input_name)
    network.add(pfx_w + '3x3', ConvBackWeightLayer(nfmaps_3, nfmaps_3r, sfmap, 3), prevs=nfmaps_3_input_name)
    network.add(pfx + '3x3_reduce', ConvBackActLayer(nfmaps_3r, nfmaps_out, sfmap, 1),
        prevs=(pfx + '3x3',))
    network.add(pfx_w + '3x3_reduce', ConvBackWeightLayer(nfmaps_3r, nfmaps_out, sfmap, 1),
        prevs=(pfx + '3x3',))

    # 5x5 branch.
    network.add(pfx + '5x5', ConvBackActLayer(nfmaps_5, nfmaps_5r, sfmap, 5), prevs=nfmaps_5_input_name)
    network.add(pfx_w + '5x5', ConvBackWeightLayer(nfmaps_5, nfmaps_5r, sfmap, 5), prevs=nfmaps_5_input_name)
    network.add(pfx + '5x5_reduce', ConvBackActLayer(nfmaps_5r, nfmaps_out, sfmap, 1),
        prevs=(pfx + '5x5',))
    network.add(pfx_w + '5x5_reduce', ConvBackWeightLayer(nfmaps_5r, nfmaps_out, sfmap, 1),
        prevs=(pfx + '5x5',))

    # Pooling branch.
    network.add(pfx + 'pool_proj', ConvBackActLayer(nfmaps_pool, nfmaps_out, sfmap, 1),
        prevs=nfmaps_pool_input_name)
    network.add(pfx_w + 'pool_proj', ConvBackWeightLayer(nfmaps_pool, nfmaps_out, sfmap, 1),
        prevs=nfmaps_pool_input_name)

    aggr_name = 'inception_{}_aggr'.format(incp_id)
    network.add(aggr_name, EltwiseLayer(nfmaps_out, sfmap, 4), prevs=(pfx + '1x1', pfx + '3x3_reduce', pfx + '5x5_reduce', pfx + 'pool_proj'))

    return aggr_name


_PREVS = ('pool2',)

# Inception 3.
_PREVS = add_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '3b', 28, 256, 128, 128, 192, 32, 96, 64,
                       prevs=_PREVS)

NN.add('pool3', PoolingLayer(480, 14, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool3',)

# Inception 4.
_PREVS = add_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4d', 14, 512, 112, 144, 288, 32, 64, 64,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '4e', 14, 528, 256, 160, 320, 32, 128, 128,
                       prevs=_PREVS)

NN.add('pool4', PoolingLayer(832, 7, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool4',)

# Inception 5.
_PREVS = add_inception(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128,
                       prevs=_PREVS)
_PREVS = add_inception(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128,
                       prevs=_PREVS)

NN.add('pool5', PoolingLayer(1024, 1, 7), prevs=_PREVS)

NN.add('fc', FCLayer(1024, 1000))


NN.add('fc_back', FCBackActLayer(1000, 1024))
NN.add('fc_back_w', FCBackWeightLayer(1000, 1024), prevs=('fc',))

NN.add('pool5_back', PoolingBackLayer(1024, 7, 7))

_PREVS = add_back_inception(NN, '5b', 7, 832, 384, 192, 384, 48, 128, 128, 1024, prevs=('pool5_back',))
_PREVS = add_back_inception(NN, '5a', 7, 832, 256, 160, 320, 32, 128, 128, 832,
                       prevs=_PREVS)

NN.add('pool4_back', PoolingBackLayer(832, 14, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool4_back',)

_PREVS = add_back_inception(NN, '4e', 14, 528, 256, 160, 320, 32, 128, 128, 832,
                       prevs=_PREVS)
_PREVS = add_back_inception(NN, '4d', 14, 512, 112, 144, 288, 32, 64, 64, 528,
                       prevs=_PREVS)
_PREVS = add_back_inception(NN, '4c', 14, 512, 128, 128, 256, 24, 64, 64, 512,
                       prevs=_PREVS)
_PREVS = add_back_inception(NN, '4b', 14, 512, 160, 112, 224, 24, 64, 64, 512,
                       prevs=_PREVS)
_PREVS = add_back_inception(NN, '4a', 14, 480, 192, 96, 208, 16, 48, 64, 512,
                       prevs=_PREVS)

NN.add('pool3_back', PoolingBackLayer(480, 28, 3, strd=2), prevs=_PREVS)
_PREVS = ('pool3_back',)

_PREVS = add_back_inception(NN, '3b', 28, 256, 128, 128, 192, 32, 96, 64, 480,
                       prevs=_PREVS)
_PREVS = add_back_inception(NN, '3a', 28, 192, 64, 96, 128, 16, 32, 32, 256,
                       prevs=_PREVS)

NN.add('pool2_back', PoolingBackLayer(192, 56, 3, strd=2))
NN.add('conv2_back_3x3', ConvBackActLayer(192, 64, 56, 3))
NN.add('conv2_back_w_3x3', ConvBackWeightLayer(192, 64, 56, 3), prevs=('pool2_back',))
NN.add('conv2_back_3x3_reduce', ConvBackActLayer(64, 64, 56, 1), prevs=('conv2_back_3x3',))
NN.add('conv2_back_w_3x3_reduce', ConvBackWeightLayer(64, 64, 56, 1), prevs=('conv2_back_3x3',))
NN.add('pool1_back', PoolingBackLayer(64, 112, 3, strd=2), prevs=('conv2_back_3x3_reduce',))
NN.add('conv1_back', ConvBackActLayer(64, 3, 224, 7, 2))
NN.add('conv1_back_w', ConvBackWeightLayer(64, 3, 224, 7, 2), prevs=('pool1_back'))
