from nn_dataflow.core import Network
from nn_dataflow.core import InputLayer, EltwiseLayer

from nn_dataflow.nns import add_lstm_cell, add_back_lstm_cell

'''
LSTM from GNMT in back-prop.

Sutskever, Vinyals, Le, Google, NIPS 2014
'''

NN = Network('GNMTBP')

NN.set_input_layer(InputLayer(1000, 1))

NL = 4

# Word embedding is a simple lookup.
# Exclude or ignore embedding processing.
WE = NN.INPUT_LAYER_KEY

# layered LSTM.
X = WE
for l in range(NL):
    cell = 'cell_l{}'.format(l)
    C, H = add_lstm_cell(NN, cell, 1000, X)
    X = H

# log(p), softmax.
NN.add('Wd', EltwiseLayer(1000, 1, 1), prevs=(X,))

NN.add('Wd_back', EltwiseLayer(1000, 1, 1), prevs=('Wd'))

H = 'Wd_back'
for l in reversed(range(NL)):
    cell = 'cell_l{}'.format(l)
    H, _ = add_back_lstm_cell(NN, cell, 1000, H)
