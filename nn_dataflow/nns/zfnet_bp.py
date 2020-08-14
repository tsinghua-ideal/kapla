from nn_dataflow.core import Network
from nn_dataflow.core.layer import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConvBackActLayer, \
    ConvBackWeightLayer, FCBackActLayer, FCBackWeightLayer, PoolingBackLayer

NN = Network('ZFNetBP')
NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 96, 110, 7, 2))
NN.add('pool1', PoolingLayer(96, 55, 3, strd=2))
# Norm layer is ignored.
NN.add('conv2', ConvLayer(96, 256, 26, 5, 2))
NN.add('pool2', PoolingLayer(256, 13, 3, strd=2))
# Norm layer is ignored.
NN.add('conv3', ConvLayer(256, 512, 13, 3))
NN.add('conv4', ConvLayer(512, 1024, 13, 3))
NN.add('conv5', ConvLayer(1024, 512, 13, 3))
NN.add('pool3', PoolingLayer(512, 6, 3, strd=2))
NN.add('fc1', FCLayer(512, 4096, 6))
NN.add('fc2', FCLayer(4096, 4096))
NN.add('fc3', FCLayer(4096, 1000))

NN.add('fc3_back', FCBackActLayer(1000, 4096))
NN.add('fc2_back', FCBackActLayer(4096, 4096))
NN.add('fc1_back', FCBackActLayer(4096, 512, 6))
NN.add('pool3_back', PoolingBackLayer(512, 13, 3, strd=2))
NN.add('conv5_back', ConvBackActLayer(512, 1024, 13, 3))
NN.add('conv4_back', ConvBackActLayer(1024, 512, 13, 3))
NN.add('conv3_back', ConvBackActLayer(512, 256, 13, 3))
NN.add('pool2_back', PoolingBackLayer(256, 26, 3, strd=2))
NN.add('conv2_back', ConvBackActLayer(256, 96, 55, 5, 2))
NN.add('pool1_back', PoolingBackLayer(96, 110, 3, strd=2))
NN.add('conv1_back', ConvBackActLayer(96, 3, 224, 7, 2))

NN.add('fc3_back_w', FCBackActLayer(1000, 4096), prevs=('fc3'))
NN.add('fc2_back_w', FCBackActLayer(4096, 4096), prevs=('fc3_back'))
NN.add('fc1_back_w', FCBackActLayer(4096, 512, 6), prevs=('fc2_back'))
NN.add('pool3_back_w', PoolingBackLayer(512, 13, 3, strd=2), prevs=('fc1_back'))
NN.add('conv5_back_w', ConvBackActLayer(512, 1024, 13, 3), prevs=('pool3_back'))
NN.add('conv4_back_w', ConvBackActLayer(1024, 512, 13, 3), prevs=('conv5_back'))
NN.add('conv3_back_w', ConvBackActLayer(512, 256, 13, 3), prevs=('conv4_back'))
NN.add('pool2_back_w', PoolingBackLayer(256, 26, 3, strd=2), prevs=('conv3_back'))
NN.add('conv2_back_w', ConvBackActLayer(256, 96, 55, 5, 2), prevs=('pool2_back'))
NN.add('pool1_back_w', PoolingBackLayer(96, 110, 3, strd=2), prevs=('conv2_back'))
NN.add('conv1_back_w', ConvBackActLayer(96, 3, 224, 7, 2), prevs=('pool1_back'))