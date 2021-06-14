import os
import itertools
import json


NODE_NUMS = [8, 16, 32]
PE_NUMS = [8, 16, 32]
BUFFER_SIZES = [16384, 32768, 65536]
REGF_SIZES = [64]

BUFFER_COST = {
    16384: 2.29698e-12,
    32768: 2.8325450000000004e-12,
    65536: 3.8839000000000004e-12
}

# STORAGE_IDLE_COST = {
#     64: 5.85485e-06
#     16384: 0.000523695
# }

base_hw_fp = 'nn_dataflow/hardwares/multi_node.json'
config_dir = 'hardware_explorer/hw_explr/'


if __name__ == '__main__':
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)

    with open(base_hw_fp) as f:
        hw = json.load(f)

    for (node_num, buffer_size, pe_num, regf_size) in itertools.product(NODE_NUMS, BUFFER_SIZES, PE_NUMS, REGF_SIZES):
        hw['GBUF']['size'] = buffer_size
        hw['GBUF']['array'] = [node_num, node_num]
        hw['GBUF']['access_cost'] = BUFFER_COST[buffer_size]
        hw['REGF']['size'] = regf_size
        hw['REGF']['array'] = [pe_num, pe_num]

        config_fn = f'n{node_num}_b{buffer_size}_p{pe_num}_r{regf_size}.json'
        fpath = os.path.join(config_dir, config_fn)
        with open(fpath, mode='w+') as f:
            json.dump(hw, f)
