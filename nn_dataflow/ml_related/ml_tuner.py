import numpy as np
import fastcache
import itertools
import copy
import functools
import sys
import pprint
import math
import time
import argparse
from nn_dataflow.ml_related.ml_cost_model import XGBoostCostModel
from nn_dataflow.ml_related.ml_optimizer import SimulatedAnnealingOptimizer
from nn_dataflow.ml_related.ml_design_space import DesignSpace

from multiprocessing import Pool
from collections import defaultdict, OrderedDict, namedtuple

from nn_dataflow import util
import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
import nn_dataflow.core.parallel_enum as nndf_pe
from nn_dataflow.core import InterLayerPipeline, PhyDim2, PartitionScheme, FmapRange, \
        FmapPosition, DataLayout, partition, BufShrScheme, NestedLoopDesc, LoopBlockingScheme, \
        SchedulingResult, NNDataflowScheme, Resource
from nn_dataflow.core.layer import ConvLayer, LocalRegionLayer, ConvBackActLayer, ConvBackWeightLayer, LocalRegionBackLayer
from nn_dataflow.nns import import_network
from nn_dataflow.core.node_region import NodeRegion

from nn_dataflow.solver.fast_explorer import gen_segment_set, segment_occp_is_valid
from nn_dataflow.array_mapping_templates.row_stationary import RowStationary
from nn_dataflow.array_mapping_templates.systolic import Systolic
from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, SystolicTensorDimMap
from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ParallelEnum as pe
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
     shape_init_data_block, SegDfCache, SimpleCstr, BL, nn_rearrange, layer2workload, is_valid, \
     get_min_factor, part_workload, construct_stack, layer_rearrange, ident_layer_type, get_conv_strds


class ParallelEnum():
    OUTP = 0
    OFMP = 1
    BATP = 2
    INPP = 3
    IFMP = 4
    NUM = 5

pe = ParallelEnum()

ARRAY_MAPPING_LIST = ["os", "ws", "rs", "dla"]
PARA_DIM_LIST = ["N", "C", "K", "Xo", "Yo"]
NN_DIM_LIST = ["N", "C", "K", "R", "S", "Xo", "Yo", "Xi", "Yi"]
DATA_LIST = [["C", "K", "R", "S"], ["N", "C", "Xi", "Yi"], ["N", "K", "Xo", "Yo"]]
CSTR_LIST = ['topbat', 'topifm', 'topofm']


class SimpleCstr(namedtuple('SimpleCstr', CSTR_LIST)):
    '''
    Simplified constraint specification.
    '''
    def __new__(cls, *args, **kwargs):
        ntp = super(SimpleCstr, cls).__new__(cls, *args, **kwargs)
        return ntp


class SimpleBufshr():
    '''
    Simplified buffer sharing.
    '''
    def __init__(self, layer_type, pdims, porders):
        self.dims = [PhyDim2(1, 1) for _ in range(de.NUM)]
        if layer_type in (0, 1):
            self.dims[de.FIL] = pdims[pe.OFMP] * pdims[pe.BATP]
        elif layer_type in (2, 3):
            self.dims[de.FIL] = pdims[pe.IFMP] * pdims[pe.BATP]

        if layer_type in (0, 2):
            self.dims[de.IFM] = pdims[pe.OUTP]
            self.dims[de.OFM] = pdims[pe.INPP]

    def size(self, dce):
        return self.dims[dce].size()


class SegDfCache():
    '''
    Cache segment dataflow and its cost.
    '''
    def __init__(self, network):
        self.seg_df_cache = dict()
        self.network = network

    def hash_seg_desc(self, segment, allocation, constraint):
        seg_layers = []
        for sp in segment:
            sp_list = []
            for layer_name in sp:
                sp_list.append(self.network[layer_name])
            seg_layers.append(tuple(sp_list))
        seg_layers = tuple(seg_layers)
        hash_key = hash((seg_layers, allocation, constraint))

        return hash_key

    def insert_to_cache(self, segment, allocation, constraint, sched_result):
        hash_key = self.hash_seg_desc(segment, allocation, constraint)
        self.seg_df_cache[hash_key] = sched_result

    def get_cached_result(self, segment, allocation, constraint):
        hash_key = self.hash_seg_desc(segment, allocation, constraint)
        return self.seg_df_cache.get(hash_key, None)


class XGBTuner(object):
    """Tuner that uses xgboost as cost model"""
    def __init__(self, array_mapping, batch_size, unit_cost, options, plan_size=100, num_threads=1, log_interval=50):
        self.cost_model = XGBoostCostModel(n_threads=num_threads,
                                           log_interval=log_interval)

        self.plan_size = plan_size
        self.options = options
        self.array_mapping = array_mapping
        self.unit_cost = unit_cost
        self.options = options
        self.batch_size = batch_size

        if array_mapping == ame.ROW_STATIONARY:
            self.tdm = RSTensorDimMap()
        elif array_mapping == ame.SYSTOLIC:
            self.tdm = SystolicTensorDimMap()
        else:
            raise ValueError("No corresponding dim map: {}".format(array_mapping))

    def init_design_space(self, layer, resource, cur_cstr):
        self.design_space = self._construct_design_space(layer, resource, cur_cstr)
        self.optimizer = SimulatedAnnealingOptimizer(design_space=self.design_space)

        self.layer = layer
        self.resource = resource
        self.cur_cstr = cur_cstr
        self.layer_type = ident_layer_type(layer)
        self.conv_strds = get_conv_strds(self.layer_type, layer)

        # trial plan
        self.trials = []
        self.trial_pt = 0
        self.visited = set()

        # observeed samples.
        self.xs = []
        self.ys = []
        self.flops_max = 0.0
        self.train_ct = 0

        self.best_config = None
        self.best_cost = float('inf')

        self.reset()

    def next_batch(self, batch_size):
        # sys.stderr.write("----next_batch\n")
        ret = []
        indexes = []

        counter = 0
        start_time = time.time()
        while counter < batch_size:
            if len(self.visited) >= len(self.design_space):
                break

            while self.trial_pt < len(self.trials):
                index = self.trials[self.trial_pt]
                if index not in self.visited:
                    break
                self.trial_pt += 1

            if self.trial_pt >= len(self.trials) - int(0.05 * self.plan_size):
                # if the trial list is empty or
                # the tuner is doing the last 5% trials (e-greedy), choose randomly
                index = np.random.randint(0, len(self.design_space))

            indexes.append(index)
            self.visited.add(index)

            counter += 1

        ret = self.design_space.indexes2configs(indexes)
        # sys.stderr.write("next_batch elapsed: {}\n".format(time.time() - start_time))
        return ret

    def update(self, inputs, results):
        for inp, res in zip(inputs, results):
            feature = self.design_space.config2feature(inp)
            self.xs.append(feature)
            self.ys.append(res)

        # if we have enough new training samples.
        if len(self.xs) >= self.plan_size * (self.train_ct + 1):
            # print("XS: {}".format(self.xs))
            # print("YS: {}".format(self.ys))
            self.cost_model.fit(tuple(self.xs), tuple(self.ys), self.plan_size)
            maximums = self.optimizer.find_maximums(
                    self.cost_model, self.plan_size, self.visited)

            self.trials = maximums
            self.trial_pt = 0
            self.train_ct += 1

    def has_next(self):
        return len(self.visited) < len(self.design_space)

    def tune(self, ifmap_layout, n_trial, n_parallel, early_stopping=None):
        sys.stderr.write("----Start tune.\n")
        early_stopping = early_stopping or 1e9
        self.n_trial = n_trial
        self.early_stopping = early_stopping

        i = 0
        while i < n_trial:
            if not self.has_next():
                break

            configs = self.next_batch(min(n_parallel, n_trial - i))
            results = self.measure_batch(configs, ifmap_layout)
            costs = []

            # keep best config
            for k, (config, result) in enumerate(zip(configs, results)):
                if result[0] is None:
                    total_cost = 100
                else:
                    cost_dict = result[1]
                    total_cost = sum(cost_dict.values())
                costs.append(total_cost)
                if total_cost < self.best_cost:
                    self.best_config = config
                    self.best_cost = total_cost

                print("Trial No: {} cur_cost: {} best_cost: {}".format(
                      i+k+1, total_cost, self.best_cost))
            i += len(results)
            self.update(configs, costs)

    @fastcache.clru_cache(maxsize=1024)
    def search(self, layer, batch_size, array_mapping, unit_cost, cur_cstr, resource, sp_idx, tm_idx,
               ifmap_layout, options, n_trial, n_parallel):
        self.reset()
        print("Start init design space")
        self.init_design_space(layer, resource, cur_cstr)
        if len(self.design_space) == 0:
            return None

        print("---Start new search\n")
        print("design_space length: {}, n_trial: {}\n".format(len(self.design_space), n_trial))

        n_trial = min(n_trial, len(self.design_space))
        self.tune(ifmap_layout, n_trial, n_parallel)
        if self.best_config == None:
            layer_sched_tops = None
        else:
            layer_sched_tops = self.measure_batch([self.best_config,], ifmap_layout)

        return layer_sched_tops[0]

    def reset(self):
        """reset the status of tuner"""
        self.best_config = None
        self.best_cost = float('inf')

    # def _construct_design_space(self, layer, batch_size, resource, cur_cstr, array_mapping,
    #                             unit_cost, options):
    #     layer_type = ident_layer_type(layer)
    #     conv_strds = get_conv_strds(layer_type, layer)

    #     partition_list = list(generate_partition(
    #             layer, batch_size,
    #             resource.proc_region.dim, self.options,
    #             guaranteed=True))

    #     array_mapping_list = []
    #     loop_blocking_array = []
    #     lbs_part_len = []
    #     lbs_nld_len = []

    #     for pdims, porders in partition_list:
    #         gbuf_workload = part_workload(pdims, layer, batch_size)
    #         if options.hw_gbuf_sharing:
    #             buf_sharing = SimpleBufshr(layer_type, pdims, porders)
    #         else:
    #             buf_sharing = None
    #         mapping = construct_array_mapping(layer_type, gbuf_workload, array_mapping, resource, conv_strds)
    #         counter = 0
    #         for loopcnt, unit_size, logic_region, regf_stack, origin_regf_update, unit_ops, regf_repls in mapping.gen_array_mapping():
    #             lbs_list = []
    #             for knobs_tuple, bl_ts, real_cstr in generate_loop_blocking(loopcnt, cur_cstr):
    #                 regf_tensor_dims = derive_regf_tensor_dim(layer_type, knobs_tuple, bl_ts[BL.REGF+1], unit_size[BL.REGF])
    #                 gbuf_bl_tp = [bl_a * bl_b for bl_a, bl_b in zip(bl_ts[BL.REGF+1], bl_ts[BL.REGF])]
    #                 gbuf_tensor_dims = derive_gbuf_tensor_dim(layer_type, knobs_tuple, gbuf_bl_tp, unit_size[BL.GBUF])

    #                 regf_tensor_dims = format_tensor_dim(layer_type, regf_tensor_dims, conv_strds)
    #                 gbuf_tensor_dims = format_tensor_dim(layer_type, gbuf_tensor_dims, conv_strds)
    #                 if not (is_valid(layer_type, regf_tensor_dims, resource.size_regf) and
    #                         is_valid(layer_type, gbuf_tensor_dims, resource.size_gbuf, buf_sharing)):
    #                     continue
    #                 for bl_ords in itertools.product(*[itertools.permutations(range(len(knobs_tuple))) for _ in range(BL.NUM)]):
    #                     outermost = len(knobs_tuple) - 1
    #                     if "N" in knobs_tuple:
    #                         if bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("N"):
    #                             continue
    #                         outermost -= 1
    #                     if "C" in knobs_tuple:
    #                         if (cur_cstr.topifm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("C"):
    #                             continue
    #                     if "K" in knobs_tuple:
    #                         if (cur_cstr.topofm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("K"):
    #                             continue

    #                     lbs_list.append((knobs_tuple, bl_ts, real_cstr, bl_ords))
    #             lbs_nld_len.append(len(lbs_list))
    #             loop_blocking_array.extend(lbs_list)
    #             array_mapping_list.append((loopcnt, unit_size, logic_region, regf_stack, origin_regf_update, unit_ops, regf_repls))
    #             counter += len(lbs_list)
    #         lbs_part_len.append(counter)

    #     return DesignSpace(partition_list, array_mapping_list, loop_blocking_array, lbs_part_len,
    #                        lbs_nld_len, resource, cur_cstr, unit_cost, options)

    def _construct_design_space(self, layer, resource, constraint):
        layer_type = ident_layer_type(layer)
        conv_strds = get_conv_strds(layer_type, layer)

        partition_list = list(partition.gen_partition(layer, self.batch_size, resource.proc_region.dim,
            self.options, guaranteed=True))

        array_mapping_list = []
        loop_blocking_array = []
        lbs_part_len = []
        lbs_nld_len = []

        for part in partition_list:
            parted_workload = part_workload(self.array_mapping, part, layer, self.batch_size)
            if self.options.hw_gbuf_sharing:
                buf_sharing = BufShrScheme(resource.proc_region, part, layer.data_loops())
            else:
                buf_sharing = None
            mapping = construct_array_mapping(layer_type, parted_workload, self.array_mapping, resource,
                                              conv_strds)
            counter = 0
            for loopcnt, unit_size, logic_region, regf_stack, origin_regf_update, unit_ops, regf_repls in mapping.gen_array_mapping():
                lbs_list = []
                for knobs_tuple, bl_ts, real_cstr in generate_loop_blocking(self.array_mapping, layer_type, loopcnt, constraint):
                    regf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, bl_ts[BL.REGF+1], unit_size[BL.REGF])
                    gbuf_bl_tp = [bl_a * bl_b for bl_a, bl_b in zip(bl_ts[BL.REGF+1], bl_ts[BL.REGF])]
                    gbuf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, gbuf_bl_tp, unit_size[BL.GBUF])

                    regf_tensor_dims = self.tdm.format_tensor_dim(layer_type, regf_tensor_dims, conv_strds)
                    gbuf_tensor_dims = self.tdm.format_tensor_dim(layer_type, gbuf_tensor_dims, conv_strds)
                    shr_node_num = tuple(buf_sharing.size(dce) for dce in range(de.NUM))
                    if not (is_valid(self.tdm, layer_type, regf_tensor_dims, resource.size_regf) and
                            is_valid(self.tdm, layer_type, gbuf_tensor_dims, resource.size_gbuf, shr_node_num)):
                        continue
                    # for bl_ords in itertools.product(*[itertools.permutations(range(len(knobs_tuple))) for _ in range(BL.NUM)]):
                    #     outermost = len(knobs_tuple) - 1
                    #     if "N" in knobs_tuple:
                    #         if constraint.topbat > 1 and (constraint.topifm > 1 or constraint.topofm > 1) and \
                    #             bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("N"):
                    #             continue
                    #         outermost -= 1
                    #     if "C" in knobs_tuple:
                    #         if (constraint.topifm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("C"):
                    #             continue
                    #     if "K" in knobs_tuple:
                    #         if (constraint.topofm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("K"):
                    #             continue
                        # lbs_list.append((knobs_tuple, bl_ts, real_cstr, bl_ords))
                    lbs_list.append((knobs_tuple, bl_ts, real_cstr))
                lbs_nld_len.append(len(lbs_list))
                loop_blocking_array.extend(lbs_list)
                mapping_fold = mapping.fold
                mapping_repls = mapping.repls.copy()
                array_mapping_list.append((loopcnt, unit_size, logic_region, mapping_fold, mapping_repls, regf_stack, origin_regf_update, unit_ops, regf_repls))
                counter += len(lbs_list)
            lbs_part_len.append(counter)

        return DesignSpace(partition_list, array_mapping_list, loop_blocking_array, lbs_part_len,
                           lbs_nld_len, resource, constraint, self.unit_cost, self.options)

    def measure_batch(self, configs, ifmap_layout):
        print("----measure_batch\n")
        results = []
        collection = []

        def retrieve_result():
            ''' Retrieve results from multiprocessing.Pool. '''
            for r in collection:
                yield r.get(timeout=3600)

        def retrieve_result_st():
            ''' Retrieve results from single-process processing. '''
            for r in collection:
                yield r

        if self.options.nprocesses > 1:
            pool = Pool(processes=self.options.nprocesses)
            apply_func = pool.apply_async
            retrieve_func = retrieve_result()
        else:
            pool = None
            apply_func = util.apply
            retrieve_func = retrieve_result_st()

        for config in configs:
            part, array_mapping, bl_ts_ord, resource, constraint, unit_cost, options = config
            gbuf_workload = part_workload(self.array_mapping, part, self.layer, self.batch_size)
            if self.options.hw_gbuf_sharing:
                buf_sharing = BufShrScheme(resource.proc_region, part, self.layer.data_loops())
            else:
                buf_sharing = None

            # Filter nodes. All memory nodes can store filters. Deduplicate.
            filter_nodes = frozenset(resource.dram_region.iter_node())
            # Ofmap layout.
            ofmap_range = FmapRange(
                FmapPosition(b=0, n=0, h=0, w=0),
                FmapPosition(b=self.batch_size, n=self.layer.nofm,
                             h=self.layer.hofm, w=self.layer.wofm))
            ofmap_data_region = resource.dst_data_region
            ofmap_layout = DataLayout(
                frngs=(ofmap_range,),
                regions=(ofmap_data_region,),
                parts=(part.projection(ofmap_data_region, appl2frng=True),))
            # Partition NoC hop cost.
            unit_nhops = partition.unit_nhops_to_proc_region(
                self.layer, self.batch_size, resource.proc_region, part,
                filter_nodes, ifmap_layout, ofmap_layout, self.options)

            loopcnt, unit_size, logic_region, mapping_fold, mapping_repls, regf_stack, origin_regf_update, unit_ops, regf_repls = array_mapping
            knobs_tuple, bl_ts, real_cstr = bl_ts_ord

            if self.options.hw_gbuf_sharing:
                buf_sharing = BufShrScheme(resource.proc_region, part, self.layer.data_loops())
            else:
                buf_sharing = None
            r = apply_func(_get_layer_df_preprocess, (self.array_mapping, self.layer_type, self.conv_strds, frozenset(gbuf_workload.items()), part, buf_sharing, frozenset(loopcnt.items()), tuple(frozenset(us.items()) for us in unit_size),
                logic_region, mapping_fold, frozenset(mapping_repls.items()), tuple(regf_stack), tuple(origin_regf_update), unit_ops, frozenset(regf_repls.items()), tuple(knobs_tuple),
                bl_ts, real_cstr, constraint, resource, unit_nhops, self.unit_cost, self.options))
            collection.append(r)

        for (layer_df, cost_dict, layer_time, real_cstr, layer_vars) in retrieve_func:
            results.append((layer_df, cost_dict, layer_time, real_cstr, layer_vars))

        if pool is not None:
            pool.close()
            pool.join()

        return results


def _get_layer_df_preprocess(array_mapping, layer_type, conv_strds, gbuf_workload, part, buf_sharing,
                                 loopcnt, unit_size, logic_region, mapping_fold, froz_mapping_repls, regf_stack, origin_regf_update,
                                 unit_ops, regf_repls, knobs_tuple, bl_ts, real_cstr,
                                 constraint, resource, unit_nhops, unit_cost, options):

    if array_mapping == ame.ROW_STATIONARY:
        tdm = RSTensorDimMap()
    elif array_mapping == ame.SYSTOLIC:
        tdm = SystolicTensorDimMap()
    else:
        raise ValueError("No corresponding dim map: {}".format(array_mapping))

    cost_model = KaplaCostModel(tdm)
    cost_model.set_cur_layer_type(layer_type)

    gbuf_workload = dict(gbuf_workload)
    loopcnt = dict(loopcnt)
    unit_size = [dict(us) for us in unit_size]
    regf_repls = dict(regf_repls)
    mapping_repls = dict(froz_mapping_repls)

    min_layer_df = None
    min_total_cost = 100
    min_cost_dict = dict()
    min_layer_time = float("inf")
    min_layer_vars = None
    min_real_cstr = None

    src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
    dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)

    gbuf_stack = construct_stack(array_mapping, layer_type, part, gbuf_workload)
    gbuf_stack_num = util.prod((util.prod(pdim) for pdim in part.pdims))
    regf_stack_num = logic_region.h * logic_region.w * util.prod(regf_repls.values())

    regf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, bl_ts[BL.REGF+1], unit_size[BL.REGF])
    gbuf_bl_tp = [bl_a * bl_b for bl_a, bl_b in zip(bl_ts[BL.REGF+1], bl_ts[BL.REGF])]
    gbuf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, gbuf_bl_tp, unit_size[BL.GBUF])

    regf_tensor_dims = tdm.format_tensor_dim(layer_type, regf_tensor_dims, conv_strds)
    gbuf_tensor_dims = tdm.format_tensor_dim(layer_type, gbuf_tensor_dims, conv_strds)

    regf_workload = copy.deepcopy(gbuf_tensor_dims)
    for dim, r in regf_repls.items():
        regf_workload[dim] = util.idivc(regf_workload[dim], r)

    if array_mapping == ame.SYSTOLIC:
        for stc in regf_stack:
            stack_repl = stc[-1]
            for item in stc[:-1]:
                if item in knobs_tuple:
                    regf_workload[item] = util.idivc(regf_workload[item], stack_repl)

    shr_node_num = tuple(buf_sharing.size(dce) for dce in range(de.NUM))
    if not (is_valid(tdm, layer_type, regf_tensor_dims, resource.size_regf) and
            is_valid(tdm, layer_type, gbuf_tensor_dims, resource.size_gbuf, shr_node_num)):
        return min_layer_df, min_cost_dict, min_layer_time, min_real_cstr, min_layer_vars
    # opt_out_bufshr = False
    # if is_valid(tdm, layer_type, gbuf_tensor_dims, resource.size_gbuf):
    #     opt_out_bufshr = True

    opt_out_bufshr = [False for _ in range(de.NUM)]
    _shr_node = [1 for _ in range(de.NUM)]
    for didx, node_num in enumerate(shr_node_num):
        _shr_node[didx] = node_num

    for d in range(de.NUM):
        origin_shr_node = _shr_node[d]
        _shr_node[d] = 1
        if is_valid(tdm, layer_type, gbuf_tensor_dims, resource.size_gbuf, _shr_node):
            opt_out_bufshr[d] = True
        else:
            _shr_node[d] = origin_shr_node
    opt_out_bufshr = tuple(opt_out_bufshr)

    regf_updates = derive_update(tdm, layer_type, knobs_tuple, bl_ts[BL.REGF],
                                    regf_tensor_dims, conv_strds, origin_regf_update)
    gbuf_updates = derive_update(tdm, layer_type, knobs_tuple, bl_ts[BL.GBUF],
                                    gbuf_tensor_dims, conv_strds)

    accesses_result = [[0 for _ in range(de.NUM)] for _ in range(me.NUM)]

    g_init_datas, g_froz_init_datas = shape_init_data_block(tdm, gbuf_tensor_dims)
    g_upd_dims, g_iter_times = cost_model.analyze_dim_iters(gbuf_tensor_dims, gbuf_updates, gbuf_workload)
    g_logical_dim, g_buf_sharings, _, _ = cost_model.analyze_stacks(g_froz_init_datas, gbuf_stack,
                                                            resource.proc_region.dim,
                                                            None,
                                                            BL.GBUF)
    gbuf_unit_accesses = cost_model.analyze_relevant_accesses(g_init_datas, g_upd_dims,
                                                    g_iter_times)

    r_init_datas, r_froz_init_datas = shape_init_data_block(tdm, regf_tensor_dims)
    r_upd_dims, r_iter_times = cost_model.analyze_dim_iters(regf_tensor_dims, regf_updates, regf_workload)
    r_froz_updates = tuple(regf_updates)
    r_logical_dim, _, r_remote_sharings, r_temporal_sharings = \
        cost_model.analyze_stacks(r_froz_init_datas, regf_stack, resource.dim_array, r_froz_updates, BL.REGF)
    regf_unit_accesses = cost_model.analyze_relevant_accesses(r_init_datas, r_upd_dims,
                                                    r_iter_times)
    regf_upper_iters = cost_model.upper_fetch(r_upd_dims, r_iter_times, g_upd_dims, g_iter_times)

    # Regfile accesses
    regf_accesses = [0 for _ in range(de.NUM)]
    flat_iter_times = r_iter_times + g_iter_times
    if layer_type in (0, 2):
        regf_accesses[de.FIL] = unit_ops * util.prod(bl_ts[BL.REGF+1]) * \
                                util.prod(flat_iter_times) * gbuf_stack_num * \
                                regf_stack_num

    regf_accesses[de.IFM] = unit_ops * util.prod(bl_ts[BL.REGF+1]) * \
                            util.prod(flat_iter_times) * regf_stack_num * \
                            gbuf_stack_num
    regf_accesses[de.OFM] = unit_ops * util.prod(bl_ts[BL.REGF+1]) * \
                            util.prod(flat_iter_times) * regf_stack_num * \
                            gbuf_stack_num * 2

    accesses_result[me.REGF] = regf_accesses

    nndf_ops = unit_ops * util.prod(bl_ts[BL.REGF+1]) * util.prod(flat_iter_times) * \
        gbuf_stack_num * regf_stack_num

    proc_time = unit_ops * util.prod(bl_ts[BL.REGF+1]) * util.prod(flat_iter_times)

    for bl_ords in itertools.product(*[itertools.permutations(range(len(knobs_tuple))) for _ in range(BL.NUM)]):
        # print("knobs_tuple", knobs_tuple)
        # print(bl_ords)
        layer_df = None
        total_cost = 100
        cost_dict = dict()
        layer_time = float("inf")
        layer_vars = None

        g_rd_iters = cost_model.redundant_iter(g_upd_dims, g_iter_times, bl_ords[BL.GBUF])
        r_rd_iters = cost_model.redundant_iter(r_upd_dims, r_iter_times, bl_ords[BL.REGF])

        if resource.no_time_mux:
            trivial_iter = True
            for dim in tdm.get_dtype_rlvt_dims(layer_type, de.FIL):
                for upd_idx, dps in enumerate(g_upd_dims):
                    if any(dim in dp for dp in dps) and (g_iter_times[upd_idx] != 1):
                        trivial_iter = False

            if layer_type == lte.CONV_BACK_W:
                trivial_iter = False

            if trivial_iter:
                g_rd_iters = list(g_rd_iters)
                g_rd_iters[de.FIL] = 0
                g_rd_iters = tuple(g_rd_iters)

        bufshr_rdt_iters = cost_model.bufshr_redundant_iter(g_upd_dims, g_iter_times, r_upd_dims,
                r_iter_times, bl_ords[BL.REGF], tuple(g_rd_iters), opt_out_bufshr)

        # Inter-layer constraints
        # batch, input, output update order
        outermost = len(knobs_tuple) - 1
        if "N" in knobs_tuple:
            if constraint.topbat > 1 and (constraint.topifm > 1 or constraint.topofm > 1) and \
                bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("N"):
                continue
            outermost -= 1
        if "C" in knobs_tuple:
            if (constraint.topifm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("C"):
                continue
        if "K" in knobs_tuple:
            if (constraint.topofm > 1) and bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("K"):
                continue
        # if data regions are not DRAM, can only access once, no spilling.
        if (not src_is_dram) and (g_rd_iters[de.IFM] > 1):
            continue
        if (not dst_is_dram) and (g_rd_iters[de.OFM] > 1):
            continue

        dram_accesses, fwd_hops, buf_shr_hops = \
            cost_model.analyze_gbuf_level_access(gbuf_unit_accesses, g_rd_iters, g_logical_dim,
                                        g_buf_sharings, bufshr_rdt_iters, options)
        gbuf_accesses, itcn_accesses = \
            cost_model.analyze_regf_level_access(regf_unit_accesses, r_rd_iters, r_logical_dim,
                                        r_remote_sharings, r_temporal_sharings,
                                        regf_upper_iters, gbuf_stack_num)

        # inter-layer data sharing.
        remote_gbuf_access = [0 for _ in range(de.NUM)]
        if not src_is_dram:
            remote_gbuf_access[de.IFM] += dram_accesses[de.IFM]
            dram_accesses[de.IFM] = 0
        if not dst_is_dram:
            remote_gbuf_access[de.OFM] += dram_accesses[de.OFM]
            dram_accesses[de.OFM] = 0

        accesses_result[me.DRAM] = dram_accesses
        accesses_result[me.GBUF] = gbuf_accesses
        accesses_result[me.ITCN] = itcn_accesses
        node_hops = [fwd_hop + bufshr_hop for fwd_hop, bufshr_hop in zip(fwd_hops, buf_shr_hops)]
        mem_hops = [unh * f for unh, f in zip(unit_nhops, g_rd_iters)]

        # calculate the cost
        cost_dict["dram_cost"] = sum(accesses_result[me.DRAM]) * unit_cost.mem_hier_at(me.DRAM)
        cost_dict["sram_cost"] = sum(accesses_result[me.GBUF]) * unit_cost.mem_hier_at(me.GBUF)
        cost_dict["itcn_cost"] = sum(accesses_result[me.ITCN]) * unit_cost.mem_hier_at(me.ITCN)
        cost_dict["regf_cost"] = sum(accesses_result[me.REGF]) * unit_cost.mem_hier_at(me.REGF)

        cost_dict["remote_sram_cost"] = unit_cost.mem_hier_at(me.GBUF) * sum(remote_gbuf_access)
        cost_dict["noc_hop_cost"] = sum(node_hops) * unit_cost.noc_hop
        cost_dict["mem_hop_cost"] = sum(mem_hops) * unit_cost.noc_hop
        cost_dict["op_cost"] = nndf_ops * unit_cost.mac_op
        total_cost = sum(cost_dict.values())

        sorted_regf_updates = sort_update(regf_updates, bl_ords[BL.REGF], origin_regf_update)
        sorted_gbuf_updates = sort_update(gbuf_updates, bl_ords[BL.GBUF])

        # calculate the time
        dram_time = int(math.ceil(sum(accesses_result[me.DRAM]) / resource.dram_bandwidth))
        bus_time = util.idivc(int(math.ceil(1. * max(accesses_result[me.GBUF])
                                / gbuf_stack_num)), resource.array_bus_width)
        layer_time = (proc_time, dram_time, bus_time)

        layer_vars = (layer_type, logic_region, mapping_fold, mapping_repls, part, conv_strds,
                      loopcnt, unit_size, knobs_tuple, bl_ts, bl_ords, unit_ops, resource,
                      buf_sharing, unit_nhops, g_logical_dim, g_buf_sharings, gbuf_unit_accesses,
                      g_rd_iters, g_upd_dims, g_iter_times, bl_ords[BL.GBUF], accesses_result)
        layer_df = layer_rearrange(tdm, gbuf_tensor_dims, gbuf_stack, sorted_gbuf_updates,
                                   regf_tensor_dims, regf_stack, sorted_regf_updates, buf_sharing)

        if total_cost < min_total_cost:
            min_layer_df = layer_df
            min_cost_dict = cost_dict
            min_layer_time = layer_time
            min_real_cstr = real_cstr
            min_layer_vars = layer_vars

    return (min_layer_df, min_cost_dict, min_layer_time, min_real_cstr, min_layer_vars)



def generate_loop_blocking(array_mapping, layer_type, loopcnt, constraint):
    knobs = OrderedDict()
    for dim, count in loopcnt.items():
        if count > 1:
            knobs[dim] = util.factorize(count, BL.NUM + 1)

    if array_mapping == ame.ROW_STATIONARY:
        if layer_type == lte.CONV:
            dim_list = ["N", "K", "C", "Yo"]
        elif layer_type == lte.LOCAL:
            dim_list = ["N", "K", "Yo"]
        elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
            dim_list = ["N", "K", "C", "Yi"]
        elif layer_type == lte.LOCAL_BACK_H:
            dim_list = ["N", "C", "Yi"]
    elif array_mapping == ame.SYSTOLIC:
        if layer_type == lte.CONV:
            dim_list = ["N", "K", "C", "XY"]
        elif layer_type == lte.LOCAL:
            dim_list = ["N", "K", "XY"]

    for dim in dim_list:
        if dim not in knobs:
            knobs[dim] = util.factorize(1, BL.NUM + 1)

    knobs_tuple = tuple(knobs.keys())

    for bl_ts in itertools.product(*knobs.values()):
        bl_ts = tuple(zip(*bl_ts))
        # Construct the real constraint.
        if "N" in knobs_tuple:
            topbat = bl_ts[0][knobs_tuple.index("N")]
        else:
            topbat = 1
        if "C" in knobs_tuple:
            topifm = bl_ts[0][knobs_tuple.index("C")]
        else:
            topifm = 1
        if "K" in knobs_tuple:
            topofm = bl_ts[0][knobs_tuple.index("K")]
        else:
            topofm = 1
        real_cstr = SimpleCstr(topbat, topifm, topofm)

        # for bl_ords in itertools.product(*[itertools.permutations(range(len(knobs))) for _ in range(BL.NUM)]):
        if constraint.topbat and ("N" in knobs_tuple) and (constraint.topbat != topbat):
            continue
        if constraint.topifm and ("C" in knobs_tuple) and (constraint.topifm != topifm):
            continue
        if constraint.topofm and ("K" in knobs_tuple) and (constraint.topofm != topofm):
            continue


        yield knobs_tuple, bl_ts, real_cstr

def construct_array_mapping(layer_type, workload, array_mapping, resource, conv_strds):
    if array_mapping == ame.ROW_STATIONARY:
        mapping = RowStationary(layer_type, workload, resource, conv_strds)
    elif array_mapping == ame.SYSTOLIC:
        mapping = Systolic(layer_type, workload, resource, conv_strds)
    else:
        raise ValueError("Not yet implement {}".format(array_mapping))

    return mapping


def derive_tensor_dim(layer_type, knobs_tuple, bl_t, unit_size):
    repl_size = copy.copy(unit_size)
    if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
        for kidx, knob in enumerate(knobs_tuple):
            repl_size.setdefault(knob, 1)
            repl_size[knob] *= bl_t[kidx]
    elif layer_type == lte.LOCAL:
        for kidx, knob in enumerate(knobs_tuple):
            if knob == "K":
                repl_size.setdefault("K", 1)
                repl_size.setdefault("C", 1)
                repl_size["K"] *= bl_t[kidx]
                repl_size["C"] *= bl_t[kidx]
            else:
                repl_size.setdefault(knob, 1)
                repl_size[knob] *= bl_t[kidx]
    elif layer_type == lte.LOCAL_BACK_H:
        for kidx, knob in enumerate(knobs_tuple):
            if knob == "C":
                repl_size.setdefault("K", 1)
                repl_size.setdefault("C", 1)
                repl_size["K"] *= bl_t[kidx]
                repl_size["C"] *= bl_t[kidx]
            else:
                repl_size.setdefault(knob, 1)
                repl_size[knob] *= bl_t[kidx]

    return repl_size


def derive_update(tdm, layer_type, knobs_tuple, bl_t, stacked_dims, conv_strds, origin_updates=()):
    results = []
    existed_upd_dims = set()
    for upd in origin_updates:
        for item in upd:
            if item in tdm.dim_set:
                existed_upd_dims.add(item)

    if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
        for dim, t in zip(knobs_tuple, bl_t):
            if dim in existed_upd_dims:
                continue
            results.append((dim, stacked_dims[dim]))
    elif layer_type == lte.LOCAL:
        for dim, t in zip(knobs_tuple, bl_t):
            if dim in existed_upd_dims:
                continue
            if dim == "C":
                continue
            elif dim == "K":
                results.append(("C", stacked_dims[dim] * conv_strds[2], "K", stacked_dims[dim]))
            else:
                results.append((dim, stacked_dims[dim]))
    elif layer_type == lte.LOCAL_BACK_H:
        for dim, t in zip(knobs_tuple, bl_t):
            if dim in existed_upd_dims:
                continue
            if dim == "K":
                continue
            elif dim == "C":
                results.append(("C", stacked_dims[dim], "K", stacked_dims[dim] * conv_strds[2]))
            else:
                results.append((dim, stacked_dims[dim]))

    results = tuple(results)
    return origin_updates + results


def sort_update(unordered_updates, bl_ord, origin_updates=()):
    return origin_updates + tuple(x for x, _ in sorted(zip(unordered_updates[-len(bl_ord):], bl_ord), key=lambda item: item[1]))
