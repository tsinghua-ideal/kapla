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

def import_network(name):
    '''
    Import an example network.
    '''
    import importlib

    if name not in all_networks():
        raise ImportError('nns: NN {} has not been defined!'.format(name))
    netmod = importlib.import_module('.' + name, 'nn_dataflow.nns')
    network = netmod.NN
    return network


def all_networks():
    '''
    Get all defined networks.
    '''
    import os

    nns_dir = os.path.dirname(os.path.abspath(__file__))
    nns = [f[:-len('.py')] for f in os.listdir(nns_dir)
           if f.endswith('.py') and not f.startswith('__')]
    return list(sorted(nns))


def add_lstm_cell(network, name, size, xin, cin=None, hin=None):
    '''
    Add a LSTM cell named `name` to the `network`, with the dimension `size`.
    `xin`, `cin`, `hin` are the layers' names whose outputs are x_t, C_{t-1},
    h_{t-1}, respectively. Return the layers' names whose outputs are C_t, h_t.
    '''
    from nn_dataflow.core import Network
    from nn_dataflow.core import InputLayer, FCLayer, EltwiseLayer

    if not isinstance(network, Network):
        raise TypeError('add_lstm_cell: network must be a Network instance.')

    if cin is None:
        cin = '{}_cinit'.format(name)
        network.add_ext(cin, InputLayer(size, 1))
    if hin is None:
        hin = '{}_hinit'.format(name)
        network.add_ext(hin, InputLayer(size, 1))

    if (cin not in network) or (hin not in network) or (xin not in network):
        raise ValueError('add_lstm_cell: cin {}, hin {}, xin {} must all be '
                         'in the network.'.format(cin, hin, xin))

    def gate_name(gate):
        ''' Name of a gate. '''
        return '{}_{}gate'.format(name, gate)

    # Candidate.
    cand_name = '{}_cand'.format(name)
    prevs = (hin, xin) if hin else (xin,)
    network.add(cand_name, FCLayer(len(prevs) * size, size), prevs=prevs)

    # Three gates.
    prevs = (hin, xin) if hin else (xin,)
    for g in ['i', 'f', 'o']:
        network.add(gate_name(g), FCLayer(len(prevs) * size, size), prevs=prevs)

    # C_t.
    cout_name = '{}_cout'.format(name)
    cout_f_name = cout_name + '_f'
    prevs = (cin, gate_name('f')) if cin else (gate_name('f'),)
    network.add(cout_f_name, EltwiseLayer(size, 1, len(prevs)), prevs=prevs)
    cout_i_name = cout_name + '_i'
    prevs = (cand_name, gate_name('i'))
    network.add(cout_i_name, EltwiseLayer(size, 1, 2), prevs=prevs)
    prevs = (cout_i_name, cout_f_name)
    network.add(cout_name, EltwiseLayer(size, 1, 2), prevs=prevs)

    # h_t.
    hout_name = '{}_hout'.format(name)
    prevs = (cout_name, gate_name('o'))
    network.add(hout_name, EltwiseLayer(size, 1, 2), prevs=prevs)

    return cout_name, hout_name


def add_back_lstm_cell(network, name, size, hout_name, cout_name=None, has_hin=False):
    from nn_dataflow.core import Network
    from nn_dataflow.core.layer import EltwiseLayer, FCBackActLayer, FCBackWeightLayer, EltwiseBackLayer
    # hout.
    hout_back_o_gate_name = '{}_hout_back_o_gate'.format(name)
    hout_back_cout_name = '{}_hout_back_cout'.format(name)
    prevs = (hout_name,)
    network.add(hout_back_o_gate_name, EltwiseBackLayer(size, 1, 1), prevs=prevs)
    network.add(hout_back_cout_name, EltwiseBackLayer(size, 1, 1), prevs=prevs)

    # Cout.
    cout_aggr_name = '{}_cout_aggr'.format(name)
    cout_back_coutf_name = '{}_cout_back_coutf'.format(name)
    cout_back_couti_name = '{}_cout_back_couti'.format(name)
    prevs_hout = (hout_back_cout_name, cout_name) if cout_name else (hout_back_cout_name,)
    cout_aggr_num = len(prevs_hout)
    network.add(cout_aggr_name, EltwiseLayer(size, 1, cout_aggr_num), prevs=prevs_hout)
    network.add(cout_back_coutf_name, EltwiseBackLayer(size, 1, 1), prevs=(cout_aggr_name,))
    network.add(cout_back_couti_name, EltwiseBackLayer(size, 1, 1), prevs=(cout_aggr_name,))

    # Coutf. No need to back-prop to cin.
    coutf_back_f_gate_name = '{}_coutf_back_f_gate'.format(name)
    prevs = (cout_back_coutf_name,)
    network.add(coutf_back_f_gate_name, EltwiseBackLayer(size, 1, 1), prevs=prevs)

    # Couti.
    couti_back_i_gate_name = '{}_couti_back_i_gate'.format(name)
    couti_back_cand_name = '{}_couti_back_cand'.format(name)
    prevs = (cout_back_couti_name,)
    network.add(couti_back_i_gate_name, EltwiseBackLayer(size, 1, 1), prevs=prevs)
    network.add(couti_back_cand_name, EltwiseBackLayer(size, 1, 1), prevs=prevs)

    weight_upd_size = 2 * size if has_hin else size

    # Ogate.
    ogate_back_xin_name = '{}_ogate_back_xin'.format(name)
    ogate_back_hin_name = '{}_ogate_back_hin'.format(name)
    ogate_back_w_upd_name = '{}_ogate_back_w_upd'.format(name)
    prevs = (hout_back_o_gate_name,)
    network.add(ogate_back_xin_name, FCBackActLayer(size, size), prevs=prevs)
    if has_hin:
        network.add(ogate_back_hin_name, FCBackActLayer(size, size), prevs=prevs)
    network.add(ogate_back_w_upd_name, FCBackWeightLayer(size, weight_upd_size), prevs=prevs)

    # Fgate.
    fgate_back_xin_name = '{}_fgate_back_xin'.format(name)
    fgate_back_hin_name = '{}_fgate_back_hin'.format(name)
    fgate_back_w_upd_name = '{}_fgate_back_w_upd'.format(name)
    prevs = (coutf_back_f_gate_name,)
    network.add(fgate_back_xin_name, FCBackActLayer(size, size), prevs=prevs)
    if has_hin:
        network.add(fgate_back_hin_name, FCBackActLayer(size, size), prevs=prevs)
    network.add(fgate_back_w_upd_name, FCBackWeightLayer(size, weight_upd_size), prevs=prevs)

    # Igate.
    igate_back_xin_name = '{}_igate_back_xin'.format(name)
    igate_back_hin_name = '{}_igate_back_hin'.format(name)
    igate_back_w_upd_name = '{}_igate_back_w_upd'.format(name)
    prevs = (couti_back_i_gate_name,)
    network.add(igate_back_xin_name, FCBackActLayer(size, size), prevs=prevs)
    if has_hin:
        network.add(igate_back_hin_name, FCBackActLayer(size, size), prevs=prevs)
    network.add(igate_back_w_upd_name, FCBackWeightLayer(size, weight_upd_size), prevs=prevs)

    # Cand.
    cand_back_xin_name = '{}_cand_back_xin'.format(name)
    cand_back_hin_name = '{}_cand_back_hin'.format(name)
    cand_back_w_upd_name = '{}_cand_back_w_upd'.format(name)
    prevs = (couti_back_cand_name,)
    network.add(cand_back_xin_name, FCBackActLayer(size, size), prevs=prevs)
    if has_hin:
        network.add(cand_back_hin_name, FCBackActLayer(size, size), prevs=prevs)
    network.add(cand_back_w_upd_name, FCBackWeightLayer(size, weight_upd_size), prevs=prevs)

    # Xin aggr.
    xin_aggr_name = '{}_xin_aggr'.format(name)
    prevs = (ogate_back_xin_name, fgate_back_xin_name, igate_back_xin_name, cand_back_xin_name)
    network.add(xin_aggr_name, EltwiseLayer(size, 1, 4), prevs=prevs)

    # Hin aggr.
    if has_hin:
        hin_aggr_name = '{}_hin_aggr'.format(name)
        prevs = (ogate_back_hin_name, fgate_back_hin_name, igate_back_hin_name, cand_back_hin_name)
        network.add(hin_aggr_name, EltwiseLayer(size, 1, 4), prevs=prevs)
    else:
        hin_aggr_name = None

    return xin_aggr_name, hin_aggr_name
