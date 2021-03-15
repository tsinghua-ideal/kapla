from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer, DepthwiseConvolutionLayer

'''
MobileNet
Howard et.al., 2017
'''

NN = Network('MobileNet')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv0', ConvLayer(3, 32, 112, 3, 2))
NN.add('dep_conv1', DepthwiseConvolutionLayer(32, 112, 3, 1))
NN.add('point_conv1', ConvLayer(32, 64, 112, 1, 1))
NN.add('dep_conv2', DepthwiseConvolutionLayer(64, 56, 3, 2))
NN.add('point_conv2', ConvLayer(64, 128, 56, 1, 1))
NN.add('dep_conv3', DepthwiseConvolutionLayer(128, 56, 3, 1))
NN.add('point_conv3', ConvLayer(128, 128, 56, 1, 1))
NN.add('dep_conv4', DepthwiseConvolutionLayer(128, 28, 3, 2))
NN.add('point_conv4', ConvLayer(128, 256, 28, 1, 1))
NN.add('dep_conv5', DepthwiseConvolutionLayer(256, 28, 3, 1))
NN.add('point_conv5', ConvLayer(256, 256, 28, 1, 1))
NN.add('dep_conv6', DepthwiseConvolutionLayer(256, 14, 3, 2))
NN.add('point_conv6', ConvLayer(256, 512, 14, 1, 1))

for i in range(5):
    NN.add(f'dep_conv{i+7}', DepthwiseConvolutionLayer(512, 14, 3, 1))
    NN.add(f'point_conv{i+7}', ConvLayer(512, 512, 14, 1, 1))

NN.add('dep_conv12', DepthwiseConvolutionLayer(512, 7, 3, 2))
NN.add('point_conv12', ConvLayer(512, 1024, 7, 1, 1))
NN.add('dep_conv13', DepthwiseConvolutionLayer(1024, 7, 3, 1))
NN.add('point_conv13', ConvLayer(1024, 1024, 7, 1, 1))
NN.add('pooling13', PoolingLayer(1024, 1, 7))
NN.add('fc', FCLayer(1024, 1000))
