from nn_dataflow.core import Network
from nn_dataflow.core.layer import InputLayer, FCLayer, FCBackActLayer, FCBackWeightLayer

'''
MLP-L

PRIME, 2016
'''

NN = Network('MLP-L')

NN.set_input_layer(InputLayer(784, 1))

NN.add('fc1', FCLayer(784, 1500))
NN.add('fc2', FCLayer(1500, 1000))
NN.add('fc3', FCLayer(1000, 500))
NN.add('fc4', FCLayer(500, 10))

NN.add('fc4_back', FCBackActLayer(10, 500))
NN.add('fc3_back', FCBackActLayer(500, 1000))
NN.add('fc2_back', FCBackActLayer(1000, 1500))
NN.add('fc1_back', FCBackActLayer(1500, 784))

NN.add('fc4_back_w', FCBackWeightLayer(10, 500), prevs=('fc4',))
NN.add('fc3_back_w', FCBackWeightLayer(500, 1000), prevs=('fc4_back',))
NN.add('fc2_back_w', FCBackWeightLayer(1000, 1500), prevs=('fc3_back',))
NN.add('fc1_back_w', FCBackWeightLayer(1500, 784), prevs=('fc2_back',))