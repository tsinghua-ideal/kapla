""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from nn_dataflow.core import Network
from nn_dataflow.core.layer import InputLayer, ConvLayer, FCLayer, \
        PoolingLayer, EltwiseLayer, ConvBackActLayer, ConvBackWeightLayer, FCBackActLayer, \
        FCBackWeightLayer, PoolingBackLayer, EltwiseBackLayer

'''
ResNet-50

He, Zhang, Ren, and Sun, 2015
'''

NN = Network('ResNetBP')

NN.set_input_layer(InputLayer(3, 224))

NN.add('conv1', ConvLayer(3, 64, 112, 7, 2))
NN.add('pool1', PoolingLayer(64, 56, 3, 2))

RES_PREV = 'pool1'

for i in range(3):
    NN.add('conv2_{}_a'.format(i), ConvLayer(64 if i == 0 else 256, 64, 56, 1))
    NN.add('conv2_{}_b'.format(i), ConvLayer(64, 64, 56, 3))
    NN.add('conv2_{}_c'.format(i), ConvLayer(64, 256, 56, 1))

    # With residual shortcut.
    if i == 0:
        NN.add('conv2_br', ConvLayer(64, 256, 56, 1), prevs=(RES_PREV,))
        RES_PREV = 'conv2_br'
    NN.add('conv2_{}_res'.format(i), EltwiseLayer(256, 56, 2),
           prevs=(RES_PREV, 'conv2_{}_c'.format(i)))
    RES_PREV = 'conv2_{}_res'.format(i)

for i in range(4):
    NN.add('conv3_{}_a'.format(i),
           ConvLayer(256, 128, 28, 1, 2) if i == 0
           else ConvLayer(512, 128, 28, 1))
    NN.add('conv3_{}_b'.format(i), ConvLayer(128, 128, 28, 3))
    NN.add('conv3_{}_c'.format(i), ConvLayer(128, 512, 28, 1))

    # With residual shortcut.
    if i == 0:
        NN.add('conv3_br', ConvLayer(256, 512, 28, 1, 2), prevs=(RES_PREV,))
        RES_PREV = 'conv3_br'
    NN.add('conv3_{}_res'.format(i), EltwiseLayer(512, 28, 2),
           prevs=(RES_PREV, 'conv3_{}_c'.format(i)))
    RES_PREV = 'conv3_{}_res'.format(i)

for i in range(6):
    NN.add('conv4_{}_a'.format(i),
           ConvLayer(512, 256, 14, 1, 2) if i == 0
           else ConvLayer(1024, 256, 14, 1))
    NN.add('conv4_{}_b'.format(i), ConvLayer(256, 256, 14, 3))
    NN.add('conv4_{}_c'.format(i), ConvLayer(256, 1024, 14, 1))

    # With residual shortcut.
    if i == 0:
        NN.add('conv4_br', ConvLayer(512, 1024, 14, 1, 2), prevs=(RES_PREV,))
        RES_PREV = 'conv4_br'
    NN.add('conv4_{}_res'.format(i), EltwiseLayer(1024, 14, 2),
           prevs=(RES_PREV, 'conv4_{}_c'.format(i)))
    RES_PREV = 'conv4_{}_res'.format(i)

for i in range(3):
    NN.add('conv5_{}_a'.format(i),
           ConvLayer(1024, 512, 7, 1, 2) if i == 0
           else ConvLayer(2048, 512, 7, 1))
    NN.add('conv5_{}_b'.format(i), ConvLayer(512, 512, 7, 3))
    NN.add('conv5_{}_c'.format(i), ConvLayer(512, 2048, 7, 1))

    # With residual shortcut.
    if i == 0:
        NN.add('conv5_br', ConvLayer(1024, 2048, 7, 1, 2), prevs=(RES_PREV,))
        RES_PREV = 'conv5_br'
    NN.add('conv5_{}_res'.format(i), EltwiseLayer(2048, 7, 2),
           prevs=(RES_PREV, 'conv5_{}_c'.format(i)))
    RES_PREV = 'conv5_{}_res'.format(i)

NN.add('pool5', PoolingLayer(2048, 1, 7))

NN.add('fc', FCLayer(2048, 1000))


NN.add('fc_back', FCBackActLayer(1000, 2048))
NN.add('fc_back_w', FCBackWeightLayer(1000, 2048), prevs=('fc',))

NN.add('pool5_back', PoolingBackLayer(2048, 7, 7), prevs=('fc_back',))

RES_PREV = 'pool5_back'

for i in reversed(range(3)):
    NN.add('conv5_{}_res_back_c'.format(i), EltwiseLayer(2048, 7, 1), prevs=(RES_PREV,))
    NN.add('conv5_{}_res_back_br'.format(i), EltwiseLayer(2048, 7, 1), prevs=(RES_PREV,))

    NN.add('conv5_{}_c_back'.format(i), ConvBackActLayer(2048, 512, 7, 1),
           prevs=('conv5_{}_res_back_c'.format(i),))
    NN.add('conv5_{}_c_back_w'.format(i), ConvBackWeightLayer(2048, 512, 7, 1),
           prevs=('conv5_{}_res_back_c'.format(i),))
    NN.add('conv5_{}_b_back'.format(i), ConvBackActLayer(512, 512, 7, 3),
           prevs=('conv5_{}_c_back'.format(i),))
    NN.add('conv5_{}_b_back_w'.format(i), ConvBackWeightLayer(512, 512, 7, 3),
           prevs=('conv5_{}_c_back'.format(i),))
    NN.add('conv5_{}_a_back'.format(i), ConvBackActLayer(512, 1024, 14, 1, 2) if i == 0
           else ConvBackActLayer(512, 2048, 7, 1), prevs=('conv5_{}_b_back'.format(i),))
    NN.add('conv5_{}_a_back_w'.format(i), ConvBackWeightLayer(512, 1024, 14, 1, 2) if i == 0
           else ConvBackWeightLayer(512, 2048, 7, 1), prevs=('conv5_{}_b_back'.format(i),))

    RES_PREV = 'conv5_{}_a_back'.format(i)

for i in reversed(range(6)):
    NN.add('conv4_{}_res_back_c'.format(i), EltwiseLayer(1024, 14, 1), prevs=(RES_PREV,))
    NN.add('conv4_{}_res_back_br'.format(i), EltwiseLayer(1024, 14, 1), prevs=(RES_PREV,))

    NN.add('conv4_{}_c_back'.format(i), ConvBackActLayer(1024, 256, 14, 1),
           prevs=('conv4_{}_res_back_c'.format(i),))
    NN.add('conv4_{}_c_back_w'.format(i), ConvBackWeightLayer(1024, 256, 14, 1),
           prevs=('conv4_{}_res_back_c'.format(i),))
    NN.add('conv4_{}_b_back'.format(i), ConvBackActLayer(256, 256, 14, 3),
           prevs=('conv4_{}_c_back'.format(i),))
    NN.add('conv4_{}_b_back_w'.format(i), ConvBackWeightLayer(256, 256, 14, 3),
           prevs=('conv4_{}_c_back'.format(i),))
    NN.add('conv4_{}_a_back'.format(i), ConvBackActLayer(256, 512, 28, 1, 2) if i == 0
           else ConvBackActLayer(256, 1024, 14, 1), prevs=('conv4_{}_b_back'.format(i),))
    NN.add('conv4_{}_a_back_w'.format(i), ConvBackWeightLayer(256, 512, 28, 1, 2) if i == 0
           else ConvBackWeightLayer(256, 1024, 14, 1), prevs=('conv4_{}_b_back'.format(i),))

    RES_PREV = 'conv4_{}_a_back'.format(i)

for i in reversed(range(4)):
    NN.add('conv3_{}_res_back_c'.format(i), EltwiseLayer(512, 28, 1), prevs=(RES_PREV,))
    NN.add('conv3_{}_res_back_br'.format(i), EltwiseLayer(512, 28, 1), prevs=(RES_PREV,))

    NN.add('conv3_{}_c_back'.format(i), ConvBackActLayer(512, 128, 28, 1),
           prevs=('conv3_{}_res_back_c'.format(i),))
    NN.add('conv3_{}_c_back_w'.format(i), ConvBackWeightLayer(512, 128, 28, 1),
           prevs=('conv3_{}_res_back_c'.format(i),))
    NN.add('conv3_{}_b_back'.format(i), ConvBackActLayer(128, 128, 28, 3),
           prevs=('conv3_{}_c_back'.format(i),))
    NN.add('conv3_{}_b_back_w'.format(i), ConvBackWeightLayer(128, 128, 28, 3),
           prevs=('conv3_{}_c_back'.format(i),))
    NN.add('conv3_{}_a_back'.format(i), ConvBackActLayer(128, 256, 56, 1, 2) if i == 0
           else ConvBackActLayer(128, 512, 28, 1), prevs=('conv3_{}_b_back'.format(i),))
    NN.add('conv3_{}_a_back_w'.format(i), ConvBackWeightLayer(128, 256, 56, 1, 2) if i == 0
           else ConvBackWeightLayer(128, 512, 28, 1), prevs=('conv3_{}_b_back'.format(i),))

    RES_PREV = 'conv3_{}_a_back'.format(i)

for i in reversed(range(3)):
    NN.add('conv2_{}_res_back_c'.format(i), EltwiseLayer(256, 56, 1), prevs=(RES_PREV,))
    NN.add('conv2_{}_res_back_br'.format(i), EltwiseLayer(256, 56, 1), prevs=(RES_PREV,))

    NN.add('conv2_{}_c_back'.format(i), ConvBackActLayer(256, 64, 56, 1),
           prevs=('conv2_{}_res_back_c'.format(i),))
    NN.add('conv2_{}_c_back_w'.format(i), ConvBackWeightLayer(256, 64, 56, 1),
           prevs=('conv2_{}_res_back_c'.format(i),))
    NN.add('conv2_{}_b_back'.format(i), ConvBackActLayer(64, 64, 56, 3),
           prevs=('conv2_{}_c_back'.format(i),))
    NN.add('conv2_{}_b_back_w'.format(i), ConvBackWeightLayer(64, 64, 56, 3),
           prevs=('conv2_{}_c_back'.format(i),))
    NN.add('conv2_{}_a_back'.format(i), ConvBackActLayer(64, 64, 56, 1) if i == 0
           else ConvBackActLayer(64, 256, 56, 1), prevs=('conv2_{}_b_back'.format(i),))
    NN.add('conv2_{}_a_back_w'.format(i), ConvBackWeightLayer(64, 64, 56, 1) if i == 0
           else ConvBackWeightLayer(64, 256, 56, 1), prevs=('conv2_{}_b_back'.format(i),))

    RES_PREV = 'conv2_{}_a_back'.format(i)

NN.add('pool1_back', PoolingBackLayer(64, 112, 3, 2), prevs=(RES_PREV,))
NN.add('conv1_back', ConvBackActLayer(64, 3, 224, 7, 2), prevs=('pool1_back'))
NN.add('conv1_back_w', ConvBackWeightLayer(64, 3, 224, 7, 2), prevs=('pool1_back'))
