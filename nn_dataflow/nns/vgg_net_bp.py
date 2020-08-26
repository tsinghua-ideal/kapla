from nn_dataflow.core import Network
from nn_dataflow.core.layer import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConvBackActLayer, \
    ConvBackWeightLayer, FCBackActLayer, FCBackWeightLayer, PoolingBackLayer

'''
VGGNet-16 Back-prop Version.

Simonyan and Zisserman, 2014
'''

NN = Network('VGG')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 224, 3))
NN.add('conv2', ConvLayer(64, 64, 224, 3))
NN.add('pool1', PoolingLayer(64, 112, 2))

NN.add('conv3', ConvLayer(64, 128, 112, 3))
NN.add('conv4', ConvLayer(128, 128, 112, 3))
NN.add('pool2', PoolingLayer(128, 56, 2))

NN.add('conv5', ConvLayer(128, 256, 56, 3))
NN.add('conv6', ConvLayer(256, 256, 56, 3))
NN.add('conv7', ConvLayer(256, 256, 56, 3))
NN.add('pool3', PoolingLayer(256, 28, 2))

NN.add('conv8', ConvLayer(256, 512, 28, 3))
NN.add('conv9', ConvLayer(512, 512, 28, 3))
NN.add('conv10', ConvLayer(512, 512, 28, 3))
NN.add('pool4', PoolingLayer(512, 14, 2))

NN.add('conv11', ConvLayer(512, 512, 14, 3))
NN.add('conv12', ConvLayer(512, 512, 14, 3))
NN.add('conv13', ConvLayer(512, 512, 14, 3))
NN.add('pool5', PoolingLayer(512, 7, 2))

NN.add('fc1', FCLayer(512, 4096, 7))
NN.add('fc2', FCLayer(4096, 4096))
NN.add('fc3', FCLayer(4096, 1000))

NN.add('fc3_back', FCBackActLayer(1000, 4096))
NN.add('fc2_back', FCBackActLayer(4096, 4096))
NN.add('fc1_back', FCBackActLayer(4096, 512, 7))
NN.add('pool5_back', PoolingBackLayer(512, 14, 2))
NN.add('conv13_back', ConvBackActLayer(512, 512, 14, 3))
NN.add('conv12_back', ConvBackActLayer(512, 512, 14, 3))
NN.add('conv11_back', ConvBackActLayer(512, 512, 14, 3))
NN.add('pool4_back', PoolingBackLayer(512, 28, 2))
NN.add('conv10_back', ConvBackActLayer(512, 512, 28, 3))
NN.add('conv9_back', ConvBackActLayer(512, 512, 28, 3))
NN.add('conv8_back', ConvBackActLayer(512, 256, 28, 3))
NN.add('pool3_back', PoolingBackLayer(256, 56, 2))
NN.add('conv7_back', ConvBackActLayer(256, 256, 56, 3))
NN.add('conv6_back', ConvBackActLayer(256, 256, 56, 3))
NN.add('conv5_back', ConvBackActLayer(256, 128, 56, 3))
NN.add('pool2_back', PoolingBackLayer(128, 112, 2))
NN.add('conv4_back', ConvBackActLayer(128, 128, 112, 3))
NN.add('conv3_back', ConvBackActLayer(128, 64, 112, 3))
NN.add('pool1_back', PoolingBackLayer(64, 224, 2))
NN.add('conv2_back', ConvBackActLayer(64, 64, 224, 3))
NN.add('conv1_back', ConvBackActLayer(64, 3, 224, 3))

NN.add('fc3_back_w', FCBackActLayer(1000, 4096), prevs=('fc3'))
NN.add('fc2_back_w', FCBackActLayer(4096, 4096), prevs=('fc3_back'))
NN.add('fc1_back_w', FCBackActLayer(4096, 512, 7), prevs=('fc2_back'))
NN.add('pool5_back_w', PoolingBackLayer(512, 14, 2), prevs=('fc1_back'))
NN.add('conv13_back_w', ConvBackActLayer(512, 512, 14, 3), prevs=('pool5_back'))
NN.add('conv12_back_w', ConvBackActLayer(512, 512, 14, 3), prevs=('conv13_back'))
NN.add('conv11_back_w', ConvBackActLayer(512, 512, 14, 3), prevs=('conv12_back'))
NN.add('pool4_back_w', PoolingBackLayer(512, 28, 2), prevs=('conv11_back'))
NN.add('conv10_back_w', ConvBackActLayer(512, 512, 28, 3), prevs=('pool4_back'))
NN.add('conv9_back_w', ConvBackActLayer(512, 512, 28, 3), prevs=('conv10_back'))
NN.add('conv8_back_w', ConvBackActLayer(512, 256, 28, 3), prevs=('conv9_back'))
NN.add('pool3_back_w', PoolingBackLayer(256, 56, 2), prevs=('conv8_back'))
NN.add('conv7_back_w', ConvBackActLayer(256, 256, 56, 3), prevs=('pool3_back'))
NN.add('conv6_back_w', ConvBackActLayer(256, 256, 56, 3), prevs=('conv7_back'))
NN.add('conv5_back_w', ConvBackActLayer(256, 128, 56, 3), prevs=('conv6_back'))
NN.add('pool2_back_w', PoolingBackLayer(128, 112, 2), prevs=('conv5_back'))
NN.add('conv4_back_w', ConvBackActLayer(128, 128, 112, 3), prevs=('pool2_back'))
NN.add('conv3_back_w', ConvBackActLayer(128, 64, 112, 3), prevs=('conv4_back'))
NN.add('pool1_back_w', PoolingBackLayer(64, 224, 2), prevs=('conv3_back'))
NN.add('conv2_back_w', ConvBackActLayer(64, 64, 224, 3), prevs=('pool1_back'))
NN.add('conv1_back_w', ConvBackActLayer(64, 3, 224, 3), prevs=('conv2_back'))
