from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer, \
        DepthwiseConvolutionLayer, ConvBackActLayer, ConvBackWeightLayer, FCBackActLayer, \
        FCBackWeightLayer, PoolingBackLayer, DepthwiseConvolutionBackActLayer, \
        DepthwiseConvolutionBackWeightLayer

'''
MobileNet Back-prop Version.
Howard et.al., 2017
'''

NN = Network('MobileNetBP')

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

NN.add('fc_back', FCBackActLayer(1000, 1024))
NN.add('pooling13_back', PoolingBackLayer(1024, 7, 7))
NN.add('point_conv13_back', ConvBackActLayer(1024, 1024, 7, 1, 1))
NN.add('dep_conv13_back', DepthwiseConvolutionBackActLayer(1024, 7, 3, 1))
NN.add('point_conv12_back', ConvBackActLayer(1024, 512, 7, 1, 1))
NN.add('dep_conv12_back', DepthwiseConvolutionBackActLayer(512, 14, 3, 2))

for i in range(5):
    NN.add(f'point_conv{11-i}_back', ConvBackActLayer(512, 512, 14, 1, 1))
    NN.add(f'dep_conv{11-i}_back', DepthwiseConvolutionBackActLayer(512, 14, 3, 1))

NN.add('point_conv6_back', ConvBackActLayer(512, 256, 14, 1, 1))
NN.add('dep_conv6_back', DepthwiseConvolutionBackActLayer(256, 28, 3, 2))
NN.add('point_conv5_back', ConvBackActLayer(256, 256, 28 ,1, 1))
NN.add('dep_conv5_back', DepthwiseConvolutionBackActLayer(256, 28, 3, 1))
NN.add('point_conv4_back', ConvBackActLayer(256, 128, 28, 1, 1))
NN.add('dep_conv4_back', DepthwiseConvolutionBackActLayer(128, 56, 3, 2))
NN.add('point_conv3_back', ConvBackActLayer(128, 128, 56, 1, 1))
NN.add('dep_conv3_back', DepthwiseConvolutionBackActLayer(128, 56, 3, 1))
NN.add('point_conv2_back', ConvBackActLayer(128, 64, 56, 1, 1))
NN.add('dep_conv2_back', DepthwiseConvolutionBackActLayer(64, 112, 3, 2))
NN.add('point_conv1_back', ConvBackActLayer(64, 32, 112, 1, 1))
NN.add('dep_conv1_back', DepthwiseConvolutionBackActLayer(32, 112, 3, 1))
NN.add('conv0_back', ConvBackActLayer(32, 3, 224, 3, 2))

NN.add('fc_back_w', FCBackWeightLayer(1000, 1024), prevs=('fc'))
NN.add('pooling13_back_w', PoolingBackLayer(1024, 7, 7), prevs=('fc_back'))
NN.add('point_conv13_back_w', ConvBackWeightLayer(1024, 1024, 7, 1, 1), prevs=('pooling13_back'))
NN.add('dep_conv13_back_w', DepthwiseConvolutionBackWeightLayer(1024, 7, 3, 1),
       prevs=('point_conv13_back'))
NN.add('point_conv12_back_w', ConvBackWeightLayer(1024, 512, 7, 1, 1), prevs=('dep_conv13_back'))
NN.add('dep_conv12_back_w', DepthwiseConvolutionBackWeightLayer(512, 14, 3, 2),
       prevs=('point_conv12_back'))

for i in range(5):
    NN.add(f'point_conv{11-i}_back_w', ConvBackWeightLayer(512, 512, 14, 1, 1),
           prevs=(f'dep_conv{12-i}_back'))
    NN.add(f'dep_conv{11-i}_back_w', DepthwiseConvolutionBackWeightLayer(512, 14, 3, 1),
           prevs=(f'point_conv{11-i}_back'))

NN.add('point_conv6_back_w', ConvBackWeightLayer(512, 256, 14, 1, 1), prevs=('dep_conv7_back'))
NN.add('dep_conv6_back_w', DepthwiseConvolutionBackWeightLayer(256, 28, 3, 2),
       prevs=('point_conv6_back'))
NN.add('point_conv5_back_w', ConvBackWeightLayer(256, 256, 28 ,1, 1), prevs=('dep_conv6_back'))
NN.add('dep_conv5_back_w', DepthwiseConvolutionBackWeightLayer(256, 28, 3, 1),
       prevs=('point_conv5_back'))
NN.add('point_conv4_back_w', ConvBackWeightLayer(256, 128, 28, 1, 1), prevs=('dep_conv5_back'))
NN.add('dep_conv4_back_w', DepthwiseConvolutionBackWeightLayer(128, 56, 3, 2),
       prevs=('point_conv4_back'))
NN.add('point_conv3_back_w', ConvBackWeightLayer(128, 128, 56, 1, 1), prevs=('dep_conv4_back'))
NN.add('dep_conv3_back_w', DepthwiseConvolutionBackWeightLayer(128, 56, 3, 1),
       prevs=('point_conv3_back'))
NN.add('point_conv2_back_w', ConvBackWeightLayer(128, 64, 56, 1, 1), prevs=('dep_conv3_back'))
NN.add('dep_conv2_back_w', DepthwiseConvolutionBackWeightLayer(64, 112, 3, 2),
       prevs=('point_conv2_back'))
NN.add('point_conv1_back_w', ConvBackWeightLayer(64, 32, 112, 1, 1), prevs=('dep_conv2_back'))
NN.add('dep_conv1_back_w', DepthwiseConvolutionBackWeightLayer(32, 112, 3, 1),
       prevs=('point_conv1_back'))
NN.add('conv0_back_w', ConvBackWeightLayer(32, 3, 224, 3, 2), prevs=('dep_conv1_back'))
