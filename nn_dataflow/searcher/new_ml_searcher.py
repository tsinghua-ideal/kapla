import sys
import pprint
import math
import time
import argparse
from collections import defaultdict, OrderedDict

from nn_dataflow import util
import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
from nn_dataflow.core import InterLayerPipeline, PhyDim2, FmapRange, FmapPosition, DataLayout, \
    partition, NestedLoopDesc, LoopBlockingScheme, SchedulingResult, NNDataflowScheme
from nn_dataflow.core.layer import ConvLayer, LocalRegionLayer
from nn_dataflow.nns import import_network

from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, SystolicTensorDimMap
from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
    SegDfCache, SimpleCstr, BL, nn_rearrange
from nn_dataflow.ml_related.ml_tuner import XGBTuner

class MLSearcher():
    def __init__(self, network, array_mapping, batch_size, resource, unit_cost, options, ntops=1):
        self.network = network
        self.array_mapping = array_mapping
        self.batch_size = batch_size
        self.resource = resource
        self.unit_cost = unit_cost
        self.options = options
        self.ntops = ntops

        self.ilp = InterLayerPipeline(network, batch_size, resource)
        self.segments = self.gen_segments()
        self.ordered_layer_list = self.ilp.ordered_layer_list()

        if array_mapping == ame.ROW_STATIONARY:
            self.tdm = RSTensorDimMap()
        elif array_mapping == ame.SYSTOLIC:
            self.tdm = SystolicTensorDimMap()
        else:
            raise ValueError("No corresponding dim map: {}".format(array_mapping))

        self.cost_model = KaplaCostModel(self.tdm)
        self.seg_df_cache = SegDfCache(network)
        self.xgbtuner = XGBTuner(array_mapping, batch_size, unit_cost, options, plan_size=100,
                                 num_threads=options.nprocesses, log_interval=50)

    def search_dataflow(self):
        df_tops = defaultdict(lambda: None)
        nndf_tops = {}
        for input_layout, ext_layout_dict in self._gen_input_layout():
            nndf = NNDataflowScheme(self.network, input_layout, ext_layout_dict)
            nndf_tops[None] = nndf
            break

        layer_counter = 0
        seg_no_counter = 0
        for layer_name in self.ordered_layer_list:
            print('{}: {}'.format(layer_counter, layer_name))
            # if layer_counter != 3:
            #     layer_counter += 1
            #     continue
            seg_counter = 0
            nndf_list = []
            for seg in self.segments[layer_name]:
                print('- {}: {}'.format(seg_counter, seg.seg))
                # if seg_counter != 3:
                #     seg_counter += 1
                #     continue
                allocation = seg.allocation()
                seg_dfs = list()

                # Get previous nndf.
                curr_layer_idx = self.ordered_layer_list.index(seg[0][0])
                if curr_layer_idx == 0:
                    prev_df = None
                    prev_nndf = nndf_tops[None]
                else:
                    prev_df = df_tops.get(self.ordered_layer_list[curr_layer_idx-1], None)
                    prev_nndf = nndf_tops.get(self.ordered_layer_list[curr_layer_idx-1], None)

                if prev_nndf is None:
                    continue

                # Forwarding data regions. Map a spatial index to the forwarding region.
                fwd_data_region_dict = {}




def search_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, back_prop=False):
    hw = parse_json(hw_fp)
    resource, unit_cost = parse_hardware(hw)

    opts = parse_json(opt_fp)
    options = parse_options(opts)
    if back_prop and (options.partition_interlayer or options.hw_gbuf_save_writeback):
        print('run_back_prop(): back_prop should disable interlayer pipelining')
        sys.exit(1)

    searcher = MLSearcher(network, array_mapping, batch_size, resource, unit_cost, options)

    tbeg = time.time()
    df_top = searcher.search_dataflow()
    tend = time.time()
    telapsed = tend - tbeg

    pp = pprint.PrettyPrinter(indent=2)
    if df_top is None:
        pp.pprint("*** No valid schedule found for {}".format(network.net_name))
        sys.exit(0)

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
    total_access_cost = sum(a * c for a, c
                            in zip(nndf.total_accesses, unit_cost.mem_hier))
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

    if array_mapping == ame.ROW_STATIONARY:
        if args.back_prop:
            hw_fp = "nn_dataflow/hardwares/multi_node.json"
            opt_fp = "nn_dataflow/options/option_training.json"
            search_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, args.back_prop)
        else:
            hw_fp = "nn_dataflow/hardwares/multi_node.json"
            opt_fp = "nn_dataflow/options/option_inference.json"
            search_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)
    elif array_mapping == ame.SYSTOLIC:
        hw_fp = "nn_dataflow/hardwares/single_node.json"
        opt_fp = "nn_dataflow/options/option_inference.json"
        search_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)


if __name__ == "__main__":
    sys.exit(main())
