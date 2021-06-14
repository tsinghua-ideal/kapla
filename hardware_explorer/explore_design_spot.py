import os
import datetime
import itertools, copy, functools, sys, pprint, math, time, argparse

from multiprocessing import Pool
from collections import defaultdict, OrderedDict

import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
import nn_dataflow.core.parallel_enum as nndf_pe

from nn_dataflow import util
from nn_dataflow.core import InterLayerPipeline, PhyDim2, PartitionScheme, FmapRange, \
    FmapPosition, DataLayout, partition, BufShrScheme, NestedLoopDesc, LoopBlockingScheme, \
    SchedulingResult, NNDataflowScheme, ConvLayer, LocalRegionLayer, DepthwiseConvolutionLayer, \
    DepthwiseConvolutionBackActLayer, DepthwiseConvolutionBackWeightLayer, NodeRegion
from nn_dataflow.nns import import_network

from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.array_mapping_templates.tensor_dim_map import SearchMethodEnum as sme

from nn_dataflow.solver.fast_explorer import gen_segment_set, segment_occp_is_valid

from nn_dataflow.searcher.kapla_searcher import KaplaSearcher

from nn_dataflow.array_mapping_templates.row_stationary import RowStationary
from nn_dataflow.array_mapping_templates.systolic import Systolic
from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, SystolicTensorDimMap
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
    shape_init_data_block, SimpleCstr, BL, nn_rearrange, layer2workload, is_valid, get_min_factor, \
    ident_layer_type, get_conv_strds
from nn_dataflow.solver.kapla_solver import KaplaSolver


def solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, back_prop=False):
    hw = parse_json(hw_fp)
    resource, unit_cost = parse_hardware(hw)

    opts = parse_json(opt_fp)
    options = parse_options(opts)
    if back_prop and (options.partition_interlayer or options.hw_gbuf_save_writeback):
        print('run_back_prop(): back_prop should disable interlayer pipelining')
        sys.exit(1)

    solver = KaplaSolver(network, array_mapping, batch_size, resource, unit_cost, options)

    tbeg = time.time()
    df_top = solver.solve_dataflow()
    tend = time.time()
    telapsed = tend - tbeg

    pp = pprint.PrettyPrinter(indent=2)
    if df_top is None:
        pp.pprint("*** No valid schedule found for {}".format(network.net_name))
        return float('inf'), df_top

    df, cost, total_time, nndf, total_cost = df_top

    print("*---------------*")
    print("Network: {}".format(network.net_name))
    print("batch_size: {}".format(batch_size))
    print("{}".format("EYERISS" if array_mapping == ame.ROW_STATIONARY else "SYSTOLIC"))
    print("nprocess: {}".format(options.nprocesses))

    print("*-* Kapla:")
    for key, value in df.items():
        pp.pprint(key)
        pp.pprint(value)
    print("*-* NNDF:")
    for key, value in nndf.items():
        pp.pprint("{}:".format(key))
        pp.pprint(value)
    print("*-* Stats:")
    pp.pprint("elapsed: {}".format(telapsed))
    print("Kapla:")
    pp.pprint(cost)
    pp.pprint("total_cost: {}".format(total_cost))
    pp.pprint("total_time: {}".format(total_time))
    print("NNDF:")
    total_op_cost = nndf.total_ops * unit_cost.mac_op
    total_access_cost = sum(a * c for a, c in zip(nndf.total_accesses, unit_cost.mem_hier))
    total_noc_cost = nndf.total_noc_hops * unit_cost.noc_hop
    total_static_cost = nndf.total_time * unit_cost.idl_unit

    total_access = [0 for _ in range(me.NUM)]
    total_rmt_gbuf_acc = 0
    total_node_hop_cost = 0
    total_mem_hop_cost = 0
    for sr in nndf.values():
        for m in range(me.NUM):
            total_access[m] += sum(sr.scheme["access"][m])
        total_rmt_gbuf_acc += sum(sr.scheme["remote_gbuf_access"])
        total_node_hop_cost += sr.scheme["cost_node_nhops"]
        total_mem_hop_cost += sr.scheme["cost_mem_nhops"]
    access_cost = tuple(c * a for c, a in zip(unit_cost.mem_hier, total_access))
    remote_gbuf_access_cost = total_rmt_gbuf_acc * unit_cost.mem_hier_at(me.GBUF)
    pp.pprint("total_dram_cost: {}".format(access_cost[me.DRAM]))
    pp.pprint("total_sram_cost: {}".format(access_cost[me.GBUF]))
    pp.pprint("total_itcn_cost: {}".format(access_cost[me.ITCN]))
    pp.pprint("total_regf_cost: {}".format(access_cost[me.REGF]))
    pp.pprint("total_rmt_gbuf_cost: {}".format(remote_gbuf_access_cost))
    pp.pprint("total_node_hop_cost: {}".format(total_node_hop_cost))
    pp.pprint("total_mem_hop_cost: {}".format(total_mem_hop_cost))
    pp.pprint("total_noc_cost: {}".format(total_noc_cost))
    pp.pprint("total_op_cost: {}".format(total_op_cost))
    pp.pprint("total_static_cost: {}".format(total_static_cost))
    pp.pprint("nndf_total_cost: {}".format(nndf.total_cost))
    pp.pprint("nndf_total_time: {}".format(nndf.total_time))

    return nndf.total_cost, nndf


def argparser():
    ap = argparse.ArgumentParser()

    ap.add_argument('net', help='network name, should be a .py file under "nns".')
    ap.add_argument('batch', type=int, help='batch_size')
    ap.add_argument('array_mapping', type=str, help='array-mapping')
    ap.add_argument('--back-prop', action='store_true', help='Run in back_propagation setting.')
    return ap


def main():
    args = argparser().parse_args()
    print(args.net)
    network = import_network(args.net)
    batch_size = args.batch
    if args.array_mapping == 'eyeriss':
        array_mapping = ame.ROW_STATIONARY
    elif args.array_mapping == 'systolic':
        array_mapping = ame.SYSTOLIC
    else:
        raise TypeError("Unsupported array mapping {}".format(args.array_mapping))

    min_cost = float('inf')
    min_nndf = None
    min_config_name = None
    cost_map = {}

    hw_config_dir = 'hardware_explorer/hw_explr/'

    for filename in os.listdir(hw_config_dir):
        if 'n32_b65536_p32_r64' not in filename:
            continue
        hw_fp = os.path.abspath(os.path.join(hw_config_dir, filename))
        print(hw_fp)
        if array_mapping == ame.ROW_STATIONARY:
            if args.back_prop:
                opt_fp = "nn_dataflow/options/option_training.json"
                c, nndf = solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, args.back_prop)
            else:
                opt_fp = "nn_dataflow/options/option_inference.json"
                c, nndf = solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)
        elif array_mapping == ame.SYSTOLIC:
            opt_fp = "nn_dataflow/options/option_inference.json"
            nndf = solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)

        if c < min_cost:
            min_cost = c
            min_nndf = nndf
            min_config_name = filename
        
        cost_map[filename] = c

    dt = datetime.datetime.now().strftime('%m_%d_%H_%M')
    bp_str = 'bp_' if args.back_prop else ''
    mapping_str = 'systolic_' if args.array_mapping == 'systolic' else ''

    with open(f'hardware_explorer/{bp_str}{mapping_str}{args.net}_{dt}.json', 'w+') as f:
        print('---config file name---', file=f)
        print(min_config_name, file=f)
        print('---cost---', file=f)
        print(str(min_cost), file=f)
        print('---nndf---')
        print(min_nndf, file=f)
        print('---cost map---')
        print(cost_map, file=f)


if __name__ == '__main__':
    sys.exit(main())

