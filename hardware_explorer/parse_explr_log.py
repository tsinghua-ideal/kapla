import os
import re
import numpy as np
import pickle
import datetime
import itertools, copy, functools, sys, pprint, math, time, argparse

explr_log_dir = 'hardware_explorer'
pickle_dir = 'design_explr_pickle'

def parse_log(log_path, pickle_path):
    with open(log_path, 'r') as f:
        nn = next(f)[:-1]
        print(f'---{nn}')
        results = []
        for line in f:
            if '.json' in line:
                configs = line.split('/')[-1].split('.')[0].split('_')[:-1]
                configs = [int(config[1:]) for config in configs]
                print(configs)
            elif 'nndf_total_cost' in line:
                cost = float(line[1:-2].split(' ')[1])
                print(cost)
                results.append(configs + [cost,])
        results = np.array(results)
        pickle_path = os.path.join(pickle_path, nn)
        print(pickle_path)
        with open(pickle_path, 'wb+') as p:
            pickle.dump(results, p)


if __name__ == '__main__':
    # for filename in os.listdir(explr_log_dir):
    #     if '.log' not in filename:
    #         continue
    #     log_path = os.path.abspath(os.path.join(explr_log_dir, filename))
    #     parse_log(log_path)

    # filename = 'energy_explr_alex_net_april_06_15_07.log'
    # filename = 'energy_explr_mlp_l_april_01_19_37.log'
    # filename = 'energy_explr_lstm_gnmt_april_06_15_07.log'
    # filename = 'bp_energy_explr_mlp_l_bp_april_07_21_49.log'
    # filename = 'bp_energy_explr_lstm_gnmt_bp_april_07_21_49.log'
    filename = 'bp_energy_explr_alex_net_bp_april_07_21_49.log'
    log_path = os.path.abspath(os.path.join(explr_log_dir, filename))
    pickle_path = os.path.abspath(os.path.join(explr_log_dir, pickle_dir))
    parse_log(log_path, pickle_path)
