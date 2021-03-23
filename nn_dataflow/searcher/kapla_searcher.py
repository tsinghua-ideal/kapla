import itertools
import copy
import functools
import sys
import pprint
import math
import time
import argparse

from multiprocessing import Pool
from collections import defaultdict, OrderedDict

import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me

from nn_dataflow import util
from nn_dataflow.core import InterLayerPipeline, PhyDim2, FmapRange, FmapPosition, DataLayout, \
    partition, BufShrScheme, NestedLoopDesc, LoopBlockingScheme, SchedulingResult, \
    NNDataflowScheme, NodeRegion
from nn_dataflow.core.layer import ConvLayer, LocalRegionLayer, DepthwiseConvolutionLayer
from nn_dataflow.nns import import_network

from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.array_mapping_templates.tensor_dim_map import SearchMethodEnum as sme

from nn_dataflow.array_mapping_templates.row_stationary import RowStationary
from nn_dataflow.array_mapping_templates.systolic import Systolic
from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, SystolicTensorDimMap
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
    shape_init_data_block, SimpleCstr, BL, nn_rearrange, SegDfCache, is_valid, part_workload, \
    ident_layer_type, get_conv_strds, construct_stack, layer_rearrange
from nn_dataflow.ml_related.ml_tuner import XGBTuner


class KaplaSearcher:
    """
    Search for an optimal dataflow represented by the Kapla directives for a given
    accelerator and a specified array mapping.
    """
    def __init__(self, search_method, network, array_mapping, batch_size, resource, unit_cost, options, ntops=1):
        self.search_method = search_method
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
        if search_method == sme.ML_SEARCHER:
            self.xgbtuner = XGBTuner(array_mapping, batch_size, unit_cost, options, plan_size=100,
                                     num_threads=options.nprocesses, log_interval=50)

        if array_mapping == ame.ROW_STATIONARY:
            self.tdm = RSTensorDimMap()
        elif array_mapping == ame.SYSTOLIC:
            self.tdm = SystolicTensorDimMap()
        else:
            raise ValueError("No corresponding dim map: {}".format(array_mapping))

        self.cost_model = KaplaCostModel(self.tdm)
        self.seg_df_cache = SegDfCache(network)

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
            print("{}: {}".format(layer_counter, layer_name))
            # if layer_counter != 3:
            #     layer_counter += 1
            #     continue
            seg_counter = 0
            nndf_list = []
            for seg in self.segments[layer_name]:
                print("{}: {}".format(seg_counter, seg.seg))
                # if seg_counter != 0:
                #     seg_counter += 1
                #     continue
                allocation = seg.allocation()
                seg_dfs = list()

                # Get previous nndf.
                cur_layer_idx = self.ordered_layer_list.index(seg[0][0])
                if cur_layer_idx == 0:
                    prev_df = None
                    prev_nndf = nndf_tops[None]
                else:
                    prev_df = df_tops.get(self.ordered_layer_list[cur_layer_idx - 1], None)
                    prev_nndf = nndf_tops.get(self.ordered_layer_list[cur_layer_idx - 1], None)
                if prev_nndf is None:
                    continue

                # Iterate through all valid constraints.
                constraint_counter = 0

                constraint_prob = 0.04
                constraint_handle = self.gen_constraint(seg)
                if self.search_method == sme.RANDOM_SEARCHER:
                    constraint_handle = util.random_collect(constraint_handle, constraint_prob)

                for constraint, _ in constraint_handle:
                    print("-- constraint: {}".format(constraint))
                    # if constraint_counter != 2:
                    #     constraint_counter += 1
                    #     continue
                    constraint_counter += 1
                    cur_nndf = prev_nndf.copy()
                    seg_df, cost_dict, seg_time, cur_nndf, total_cost = \
                        self.search_segment_df(seg, allocation, constraint, cur_nndf)
                    if len(seg_df) == 0:
                        continue
                    seg_dfs.append((seg_df, cost_dict, seg_time, cur_nndf, total_cost))

                # Select best seg df.
                if len(seg_dfs) == 0:
                    continue
                if self.options.opt_goal == 'e':
                    top_seg_df = sorted(seg_dfs, key=lambda x: x[-1])[0]
                elif self.options.opt_goal == 'd':
                    top_seg_df = sorted(seg_dfs, key=lambda x: x[-2].total_time)[0]
                elif self.options.opt_goal == 'ed':
                    top_seg_df = sorted(seg_dfs, key=lambda x: x[-1]*x[-2].total_time)[0]
                else:
                    raise ValueError('Kapla searcher: Invalid opt goal {}'.format(self.options.opt_goal))
                nndf_result = nn_rearrange(top_seg_df, prev_df)
                nndf_list.append(nndf_result)
                seg_no_counter += 1
                seg_counter += 1

            if len(nndf_list) == 0:
                continue

            if self.options.opt_goal == 'e':
                df_tops[layer_name] = sorted(nndf_list, key=lambda x: x[-1])[0]
            elif self.options.opt_goal == 'd':
                df_tops[layer_name] = sorted(nndf_list, key=lambda x: x[-3])[0]
            elif self.options.opt_goal == 'ed':
                df_tops[layer_name] = sorted(nndf_list, key=lambda x: x[-1]*x[-3])[0]
            else:
                raise ValueError('Kapla searcher: Invalid opt goal {}'.format(self.options.opt_goal))

            nndf_tops[layer_name] = df_tops[layer_name][3]
            layer_counter += 1

        return df_tops[self.ordered_layer_list[-1]]

    def search_segment_df(self, segment, allocation, constraint, cur_nndf):
        seg_df = defaultdict(lambda: dict())
        seg_times = list()
        cstr_collections = dict()
        seg_costs = dict()
        total_cost = 0

        # Forwarding data regions. Map a spatial index to the forwarding region.
        fwd_data_region_dict = {}
        for sh_list in segment.ifm_fwd_dict.values():
            # A list of spatial indices that share the same ifmaps.
            r = allocation[sh_list[0].sp_idx][sh_list[0].tm_idx].proc_region
            for idx in sh_list[1:]:
                fwd_data_region_dict[idx] = r
        for fwd_src, fwd_dst_list in segment.ofm_fwd_dict.items():
            # Ofmaps forwarded to neighbors.
            r = allocation[fwd_src.sp_idx][fwd_src.tm_idx].proc_region
            for idx in fwd_dst_list:
                fwd_data_region_dict[idx] = r

        seg_idx = cur_nndf.last_seg_idx + 1

        for sp_idx, (ltpl, rtpl, ctpl) in enumerate(zip(segment, allocation, constraint)):
            seg_times.append([])
            for tm_idx, (layer_name, resource, cstr) in enumerate(zip(ltpl, rtpl, ctpl)):
                # Update the constraint. Currently only support ofm update.
                topbat = cstr.topbat
                topifm = cstr.topifm
                topofm = cstr.topofm
                for prev_layer, _ in cstr.update_dict.items():
                    topofm = cstr_collections[prev_layer].topofm
                cur_cstr = SimpleCstr(topbat, topifm, topofm)

                ifmap_layout = cur_nndf.fmap_layout(self.network.prevs(layer_name))
                fwd_data_region = fwd_data_region_dict.get((sp_idx, tm_idx))
                if fwd_data_region is not None:
                    # Remap source data regions to the forwarding region.
                    ifmap_layout = DataLayout(frngs=ifmap_layout.frngs,
                                              regions=(fwd_data_region,) * len(ifmap_layout.frngs),
                                              parts=tuple(p.projection(fwd_data_region,
                                                                       appl2frng=True)
                                                          for p in ifmap_layout.parts))
                # Search the layer's dataflow.
                if self.search_method in {sme.KAPLA_SEARCHER, sme.RANDOM_SEARCHER}:
                    df, real_cstr, cost_dict, layer_time, sched_vars, accesses_result, noc_hops = \
                        self.search_layer_df(layer_name, cur_cstr, resource, ifmap_layout)
                elif self.search_method == sme.ML_SEARCHER:
                    result = self.xgbtuner.search(self.network[layer_name], self.batch_size,
                                             self.array_mapping, self.unit_cost, cur_cstr, resource,
                                             sp_idx, tm_idx, ifmap_layout, self.options,
                                             n_trial=1024, n_parallel=128)
                    if result is None:
                        df, cost_dict, layer_time, real_cstr, sched_vars = (None for _ in range(5))
                    else:
                        df, cost_dict, layer_time, real_cstr, sched_vars = result
                else:
                    raise TypeError('Invalid search method: {}.'.format(self.search_method))

                if df is None:
                    return dict(), None, None, None, None
                print("layer: {}".format(layer_name))
                print("df: {}".format(df))
                print("cost: {}".format(cost_dict))
                print("total cost: {}".format(sum(cost_dict.values())))
                print("time: {}".format(layer_time))
                print("sched vars:")
                for idx, var in enumerate(sched_vars):
                    print(var)
                print("")
                # sched_vars = sched_vars[:-1]
                sched_vars = sched_vars[:-9]
                sched_result = self.derive_sched_result(layer_name, seg_idx, sp_idx, tm_idx,
                                                        ifmap_layout, sched_vars)
                print("sched result: {}".format(sched_result))
                cur_nndf[layer_name] = sched_result
                seg_df[layer_name]["dataflow"] = df
                seg_df[layer_name]["sched_seq"] = [0, sp_idx, tm_idx]
                for key, value in cost_dict.items():
                    seg_costs[key] = seg_costs.setdefault(key, 0) + value
                cstr_collections[layer_name] = real_cstr
                seg_times[sp_idx].append(layer_time)
                total_cost += sum(cost_dict.values())

        seg_time = self.cost_model.seg_time_estimation(self.network, segment, seg_times, cstr_collections)

        return seg_df, seg_costs, seg_time, cur_nndf, total_cost

    @functools.lru_cache(maxsize=1024)
    def search_layer_df(self, layer_name, cstr, resource, ifmap_layout):
        opt_value = float('inf')
        min_cost = float("inf")
        min_cost_dict = dict()
        min_layer_time = float("inf")
        min_cstr = None
        min_noc_hops = None
        min_accesses_result = None
        min_layer_df = None
        min_layer_vars = None

        layer = self.network[layer_name]
        layer_type = ident_layer_type(layer)
        conv_strds = get_conv_strds(layer_type, layer)

        results = []

        def retrieve_result():
            ''' Retrieve results from multiprocessing.Pool. '''
            for r in results:
                yield r.get(timeout=3600)

        def retrieve_result_st():
            ''' Retrieve results from single-process processing. '''
            for r in results:
                yield r

        if self.options.nprocesses > 1:
            pool = Pool(processes=self.options.nprocesses)
            apply_func = pool.apply_async
            retrieve_func = retrieve_result
        else:
            pool = None
            apply_func = util.apply
            retrieve_func = retrieve_result_st

        partition_prob = 0.04
        partition_handle = partition.gen_partition(layer, self.batch_size, resource.proc_region.dim,
                                                   self.options, guaranteed=True)
        if self.search_method == sme.RANDOM_SEARCHER:
            partition_handle = util.random_collect(partition_handle, partition_prob)

        for part in partition_handle:
            parted_workload = part_workload(self.array_mapping, part, layer, self.batch_size)
            if self.options.hw_gbuf_sharing:
                buf_sharing = BufShrScheme(resource.proc_region, part, layer.data_loops())
            else:
                buf_sharing = None

            # Filter nodes. All memory nodes can store filters. Deduplicate.
            filter_nodes = frozenset(resource.dram_region.iter_node())
            # Ofmap layout.
            ofmap_range = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                                    FmapPosition(b=self.batch_size, n=layer.nofm, h=layer.hofm,
                                                 w=layer.wofm))
            ofmap_data_region = resource.dst_data_region
            ofmap_layout = DataLayout(frngs=(ofmap_range,), regions=(ofmap_data_region,),
                                      parts=(part.projection(ofmap_data_region, appl2frng=True),))
            # Partition NoC hop cost.
            unit_nhops = \
                partition.unit_nhops_to_proc_region(layer, self.batch_size, resource.proc_region,
                                                    part, filter_nodes, ifmap_layout, ofmap_layout,
                                                    self.options)

            r = apply_func(_search_layer_df_perprocess,
                           (self.search_method, layer_type, conv_strds,
                            frozenset(parted_workload.items()), part,
                            buf_sharing, resource, cstr, self.array_mapping, unit_nhops,
                            self.unit_cost, self.options))
            results.append(r)

        for (layer_df, cost_dict, layer_time, real_cstr, noc_hops, accesses_result,
             layer_vars) in retrieve_func():
            if layer_df is None:
                continue
            if self.options.opt_goal == 'e' and (sum(cost_dict.values()) < opt_value):
                min_cost = sum(cost_dict.values())
                opt_value = min_cost
                min_layer_vars = layer_vars
                min_cost_dict = cost_dict
                min_layer_time = layer_time
                min_cstr = real_cstr
                min_noc_hops = noc_hops
                min_accesses_result = accesses_result
                min_layer_df = layer_df
            elif self.options.opt_goal == 'd' and (sum(layer_time) < opt_value):
                opt_value = sum(layer_time)
                min_cost = sum(cost_dict.values())
                opt_value = min_cost
                min_layer_vars = layer_vars
                min_cost_dict = cost_dict
                min_layer_time = layer_time
                min_cstr = real_cstr
                min_noc_hops = noc_hops
                min_accesses_result = accesses_result
                min_layer_df = layer_df
            elif self.options.opt_goal == 'ed' and ((sum(cost_dict.values()) * sum(layer_time)) < opt_value):
                opt_value = sum(cost_dict.values()) * sum(layer_time)
                min_cost = sum(cost_dict.values())
                opt_value = min_cost
                min_layer_vars = layer_vars
                min_cost_dict = cost_dict
                min_layer_time = layer_time
                min_cstr = real_cstr
                min_noc_hops = noc_hops
                min_accesses_result = accesses_result
                min_layer_df = layer_df

        if pool is not None:
            pool.close()
            pool.join()

        return min_layer_df, min_cstr, min_cost_dict, min_layer_time, min_layer_vars, \
                min_accesses_result, min_noc_hops

    def gen_segments(self):
        segments = defaultdict(list)
        for seg in self.ilp.gen_segment(self.options):
            if seg not in segments[seg[-1][-1]]:
                segments[seg[-1][-1]].append(seg)

        return segments

    def gen_constraint(self, segment):
        for constraint, hints in segment.gen_constraint():
            yield constraint, hints

    def derive_sched_result(self, layer_name, seg_idx, sp_idx, tm_idx, ifmap_layout, sched_vars):
        (layer_type, logic_region, mapping_fold, mapping_repls, part, conv_strds, loopcnt,
         unit_size, knobs_tuple, bl_ts, bl_ords, unit_ops, resource, bufshr) = sched_vars

        layer = self.network[layer_name]

        proc_region = resource.proc_region

        # Ofmap layout.
        ofmap_range = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                                FmapPosition(b=self.batch_size, n=layer.nofm, h=layer.hofm,
                                             w=layer.wofm))
        ofmap_data_region = resource.dst_data_region
        ofmap_layout = DataLayout(frngs=(ofmap_range,), regions=(ofmap_data_region,),
                                  parts=(part.projection(ofmap_data_region, appl2frng=True),))

        filter_nodes = frozenset(resource.dram_region.iter_node())
        unit_nhops = partition.unit_nhops_to_proc_region(layer, self.batch_size, proc_region,
                                                         part, filter_nodes, ifmap_layout,
                                                         ofmap_layout, self.options)

        lbs = self.derive_nndf_lbs(layer_type, layer, logic_region, mapping_fold, mapping_repls,
                                   part, conv_strds, loopcnt, unit_size, knobs_tuple, bl_ts,
                                   bl_ords, resource, bufshr)

        sched_seq = (seg_idx, sp_idx, tm_idx)

        sched_result = self._construct_sched_result(lbs, part, ofmap_layout, sched_seq, unit_nhops)

        return sched_result

    def derive_nndf_lbs(self, layer_type, layer, logic_region, mapping_fold, mapping_repls, part,
                        conv_strds, loopcnt, unit_size, knobs_tuple, bl_ts, bl_ords, resource,
                        bufshr):
        for bl in range(BL.NUM):
            unit_size[bl] = self.tdm.format_tensor_dim(layer_type, unit_size[bl], conv_strds)

        if self.array_mapping == ame.ROW_STATIONARY:
            part_layer = part_workload(self.array_mapping, part, layer, self.batch_size)
            if layer_type == lte.CONV:
                acclayer = ConvLayer(1, 1, (util.idivc(part_layer["Yo"], mapping_fold.w),
                                            part_layer["Xo"]),
                                     (part_layer["S"], part_layer["R"]),
                                     strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yo"], mapping_fold.w))
            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(1, (util.idivc(part_layer["Yo"], mapping_fold.w),
                                                part_layer["Xo"]), conv_strds[2],
                                            (part_layer["S"], part_layer["R"]),
                                            ntrd=conv_strds[2], strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yo"], mapping_fold.w))
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                acclayer = ConvLayer(1, 1, (util.idivc(part_layer["Yi"], mapping_fold.w),
                                            part_layer["Xi"]),
                                     (part_layer["S"], part_layer["R"]),
                                     strd=(conv_strds[1], conv_strds[0]), rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yi"], mapping_fold.w))
            elif layer_type == lte.LOCAL_BACK_H:
                acclayer = LocalRegionLayer(1, (util.idivc(part_layer["Yi"], mapping_fold.w),
                                                part_layer["Xi"]), conv_strds[2],
                                            (part_layer["S"], part_layer["R"]),
                                            strd=(conv_strds[1], conv_strds[0]), ntrd=conv_strds[2],
                                            rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yi"], mapping_fold.w))
            elif layer_type == lte.DW_CONV:
                acclayer = DepthwiseConvolutionLayer(1,
                    (util.idivc(part_layer["Yo"], mapping_fold.w), part_layer["Xo"]),
                    (part_layer["S"], part_layer["R"]),
                    (conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yo"], mapping_fold.w))
            elif layer_type in (lte.DW_CONV_H, lte.DW_CONV_W):
                acclayer = DepthwiseConvolutionLayer(1,
                    (util.idivc(part_layer["Yi"], mapping_fold.w), part_layer["Xi"]),
                    (part_layer["S"], part_layer["R"]), strd=(conv_strds[1], conv_strds[0]),
                    rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yi"], mapping_fold.w))
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))
            flpesets_per_unitpass = mapping_fold.h
            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"] * \
                                               mapping_repls["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"] * \
                                               mapping_repls["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL_BACK_H:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.DW_CONV:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["K"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type in (lte.DW_CONV_H, lte.DW_CONV_W):
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM] * flpesets_per_unitpass
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM] * flpesets_per_unitpass

            unit_access[me.ITCN][de.IFM] = acclayer.wifm * dim_flpeset.size() * \
                                           flpesets_per_unitpass * util.prod(mapping_repls.values())
            unit_access[me.ITCN][de.OFM] = acclayer.wofm * dim_flpeset.size() * \
                                           flpesets_per_unitpass * util.prod(mapping_repls.values())

            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W, lte.DW_CONV,
                              lte.DW_CONV_H, lte.DW_CONV_W):
                unit_access[me.ITCN][de.FIL] = acclayer.wfil * dim_flpeset.size() * \
                                               flpesets_per_unitpass * \
                                               util.prod(mapping_repls.values())
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_access[me.ITCN][de.FIL] = 0
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))

            unit_access[me.REGF] = \
                [acclayer.total_ops() * util.prod(mapping_repls.values())] * de.NUM
            if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_access[me.REGF][de.FIL] = 0

            sz_gbuf = self.tdm.get_tensor_size(layer_type, unit_size[BL.GBUF])
            sz_gbuf[de.IFM] /= mapping_fold.w
            sz_gbuf[de.OFM] /= mapping_fold.w

            if layer_type in (lte.CONV, lte.LOCAL, lte.DW_CONV):
                sz_gbuf[de.IFM] /= amp_acc_ifm
            else:
                sz_gbuf[de.OFM] /= amp_acc_ifm

            sz_regf = [0] * de.NUM
            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W, lte.DW_CONV,
                              lte.DW_CONV_H, lte.DW_CONV_W):
                sz_regf[de.FIL] = acclayer.wfil
                sz_regf[de.IFM] = acclayer.wfil
                sz_regf[de.OFM] = 1
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                sz_regf[de.FIL] = 0
                sz_regf[de.IFM] = acclayer.wreg * acclayer.nreg
                sz_regf[de.OFM] = 1

            ops_lpe = acclayer.total_ops() * util.prod(mapping_repls.values())
            loopcnt = [loopcnt["C"], loopcnt["K"], loopcnt["N"] * mapping_fold.w]
            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W, lte.DW_CONV,
                              lte.DW_CONV_H, lte.DW_CONV_W):
                unit_time = acclayer.wfil * acclayer.hfil
                regf_reusable[de.IFM] = (acclayer.wfil == acclayer.wifm)
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_time = acclayer.nreg * acclayer.wreg * acclayer.hreg
                regf_reusable[de.IFM] = (acclayer.wreg == acclayer.wifm)
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))
            regf_reusable[de.OFM] = (acclayer.wofm == 1)
            regf_reusable[de.FIL] = (mapping_fold.h == 1)

            if layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
                sz_regf[de.IFM], sz_regf[de.OFM] = sz_regf[de.OFM], sz_regf[de.IFM]
                regf_reusable[de.IFM], regf_reusable[de.OFM] = \
                    regf_reusable[de.OFM], regf_reusable[de.IFM]
                for i in range(me.NUM):
                    unit_access[i][de.IFM], unit_access[i][de.OFM] = \
                        unit_access[i][de.OFM], unit_access[i][de.IFM]



            comp_bl_ts = [[1 for _ in range(3)] for _ in range(BL.NUM + 1)]
            comp_bl_ords = [[1 for _ in range(3)] for _ in range(BL.NUM)]

            unset_ords = set()
            for d_idx, dim in enumerate(["C", "K", "N"]):
                if dim in knobs_tuple:
                    idx = knobs_tuple.index(dim)
                    for bl in range(BL.NUM + 1):
                        comp_bl_ts[bl][d_idx] = bl_ts[bl][idx]
                    comp_bl_ords[BL.GBUF][d_idx] = bl_ords[BL.GBUF][idx]
                    comp_bl_ords[BL.REGF][d_idx] = bl_ords[BL.REGF][idx]
                else:
                    unset_ords.add(d_idx)

            comp_bl_ts[BL.GBUF + 1][le.BAT] *= mapping_fold.w

            counter = len(bl_ords[BL.GBUF])
            for d_idx in unset_ords:
                for bl in range(BL.NUM):
                    comp_bl_ords[bl][d_idx] = counter
                counter += 1

            for bl in range(BL.NUM):
                for order, didx in enumerate(sorted(range(len(comp_bl_ords[bl])),
                        key=lambda k: comp_bl_ords[bl][k])):
                    comp_bl_ords[bl][didx] = order

            # For back-prop localregion layer, since the data_loop mapping is using the conventional
            # forward version local-region layer, we need to reverse the i and o dimension.
            if layer_type in (lte.LOCAL_BACK_H, lte.DW_CONV_H, lte.DW_CONV_W):
                for ble in range(BL.NUM+1):
                    comp_bl_ts[ble][le.IFM], comp_bl_ts[ble][le.OFM] = \
                        comp_bl_ts[ble][le.OFM], comp_bl_ts[ble][le.IFM]
                for ble in range(me.NUM):
                    unit_access[ble][de.IFM], unit_access[ble][de.OFM] = \
                        unit_access[ble][de.OFM], unit_access[ble][de.IFM]
                for ble in range(BL.NUM):
                    comp_bl_ords[ble][le.IFM], comp_bl_ords[ble][le.OFM] = \
                        comp_bl_ords[ble][le.OFM], comp_bl_ords[ble][le.IFM]
                loopcnt[le.IFM], loopcnt[le.OFM] = loopcnt[le.OFM], loopcnt[le.IFM]

            loopcnt = tuple(loopcnt)

            for i in range(BL.NUM):
                comp_bl_ords[i] = tuple(comp_bl_ords[i])
            comp_bl_ords = tuple(comp_bl_ords)
            for i in range(BL.NUM + 1):
                comp_bl_ts[i] = tuple(comp_bl_ts[i])
            comp_bl_ts = tuple(comp_bl_ts)

            sz_gbuf = tuple(sz_gbuf)
            sz_regf = tuple(sz_regf)
            for i in range(me.NUM):
                unit_access[i] = tuple(unit_access[i])
            unit_access = tuple(unit_access)
            regf_reusable = tuple(regf_reusable)

            nld = NestedLoopDesc(loopcnt=loopcnt, unit_access=unit_access, usize_gbuf=sz_gbuf,
                                 usize_regf=sz_regf, unit_ops=ops_lpe, unit_time=unit_time,
                                 data_loops=acclayer.data_loops(), regf_reusable=regf_reusable,
                                 rw_data=layer.rw_data)

            real_resource = resource._replace(size_gbuf=resource.size_gbuf / 0.99,
                                              size_regf=resource.size_regf / 0.99)

            print('###DEBUG###')
            print('comp_bl_ts: {}'.format(comp_bl_ts))
            print('comp_bl_ords: {}'.format(comp_bl_ords))
            lbs = LoopBlockingScheme(nld, comp_bl_ts, comp_bl_ords, real_resource, bufshr,
                                     self.options)
            if not lbs.is_valid():
                print("LBS INVALID!")
                print("loopcnt", loopcnt)
                print("unit_access", unit_access)
                print("sz_gbuf", sz_gbuf)
                print("sz_regf", sz_regf)
                print("fold", mapping_fold)
                print("repl", mapping_repls)
                print("logic_region", logic_region)
                print("unit_size", unit_size)
                print("bl_ts", bl_ts)
                print("comp_bl_ts", comp_bl_ts)
                print("bl_ords", bl_ords)
                print("comp_bl_ords", comp_bl_ords)
                print("bufshr", tuple(bufshr.size(dce) for dce in range(de.NUM)))

        elif self.array_mapping == ame.SYSTOLIC:
            part_layer, p_batch_size, p_occ = bufshr.part.part_layer(layer, self.batch_size)
            fold_h, fold_w = mapping_fold.h, mapping_fold.w
            full_xy = layer.hofm * layer.wofm
            fold_xy = util.idivc(full_xy, mapping_fold.h)
            fold_x = util.closest_factor(fold_xy, factor=math.sqrt(fold_xy))[0]
            fold_y = fold_xy / fold_x
            if layer_type == lte.CONV:
                acclayer = ConvLayer(1, 1. * part_layer.nofm / fold_w, (fold_x, fold_y),
                                     (part_layer.hfil, part_layer.wfil),
                                     strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / part_layer.hifm / \
                              part_layer.wifm
                unit_time = layer.wfil * layer.hfil
            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(1. * layer.nofm / fold_w, (fold_x, fold_y),
                                            layer.nreg, (layer.hreg, layer.wreg),
                                            strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / part_layer.hifm / \
                              part_layer.wifm
                unit_time = layer.nreg * layer.wreg * layer.hreg
            elif layer_type == lte.DW_CONV:
                acclayer = DepthwiseConvolutionLayer(1. * layer.nofm / fold_w, (fold_x, fold_y),
                                                     (layer.hfil, layer.wfil),
                                                     strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / part_layer.hifm / \
                              part_layer.wifm
                unit_time = layer.wfil * layer.hfil
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))

            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["N"]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * \
                                               mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["N"] / \
                                               amp_acc_ifm
            elif layer_type == lte.DW_CONV:
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size()
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["N"] / \
                                               amp_acc_ifm
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM]
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM]

            unit_access[me.ITCN][de.IFM] = acclayer.total_ifmap_size() * \
                                           util.prod(mapping_repls.values())
            unit_access[me.ITCN][de.OFM] = acclayer.total_ofmap_size() * \
                                           util.prod(mapping_repls.values())
            if layer_type in (lte.CONV, lte.DW_CONV):
                unit_access[me.ITCN][de.FIL] = acclayer.total_filter_size() * \
                                               util.prod(mapping_repls.values())
            elif layer_type == lte.LOCAL:
                unit_access[me.ITCN][de.FIL] = 0
            else:
                raise TypeError("Unsupported layer type {}".format(layer_type))


            unit_access[me.REGF] = \
                [acclayer.total_ops() * util.prod(mapping_repls.values())] * de.NUM

            sz_gbuf = [0] * de.NUM
            sz_gbuf[de.IFM] = acclayer.total_ifmap_size() * util.prod(mapping_repls.values())
            sz_gbuf[de.OFM] = acclayer.total_ofmap_size() * util.prod(mapping_repls.values())
            if layer_type in (lte.CONV, lte.DW_CONV):
                sz_gbuf[de.FIL] = acclayer.total_filter_size() * util.prod(mapping_repls.values())
            else:
                sz_gbuf[de.FIL] = 0

            sz_regf = [0] * de.NUM
            if layer_type in (lte.CONV, lte.DW_CONV):
                sz_regf[de.FIL] = 1
                sz_regf[de.IFM] = 1
                sz_regf[de.OFM] = 1
            elif layer_type == lte.LOCAL:
                sz_regf[de.FIL] = 0
                sz_regf[de.IFM] = acclayer.nreg
                sz_regf[de.OFM] = 1

            ops_lpe = acclayer.total_ops() * util.prod(mapping_repls.values())
            loopcnt = (loopcnt["C"], loopcnt["K"], loopcnt["N"] * mapping_fold.h)
            regf_reusable[de.IFM] = False
            regf_reusable[de.FIL] = False
            regf_reusable[de.OFM] = True

            comp_bl_ts = [[1 for _ in range(3)] for _ in range(BL.NUM + 1)]
            comp_bl_ords = [[1 for _ in range(3)] for _ in range(BL.NUM)]

            set_flags = [False] * le.NUM
            for lidx, dims in enumerate([["C"], ["K"], ["N", "XY"]]):
                for dim in dims:
                    if dim in knobs_tuple:
                        set_flags[lidx] = True
                        idx = knobs_tuple.index(dim)
                        for bl in range(BL.NUM + 1):
                            comp_bl_ts[bl][lidx] *= bl_ts[bl][idx]
                        comp_bl_ords[BL.GBUF][lidx] = bl_ords[BL.GBUF][idx]
                        comp_bl_ords[BL.REGF][lidx] = bl_ords[BL.REGF][idx]

            counter = len(bl_ords[BL.GBUF])
            for lidx, set_flag in enumerate(set_flags):
                if not set_flag:
                    for bl in range(BL.NUM):
                        comp_bl_ords[bl][lidx] = counter
                    counter += 1

            for bl in range(BL.NUM):
                for order, didx in enumerate(sorted(range(len(comp_bl_ords[bl])),
                                                    key=lambda k: comp_bl_ords[bl][k])):
                    comp_bl_ords[bl][didx] = order

            for i in range(BL.NUM):
                comp_bl_ords[i] = tuple(comp_bl_ords[i])
            comp_bl_ords = tuple(comp_bl_ords)
            for i in range(BL.NUM + 1):
                comp_bl_ts[i] = tuple(comp_bl_ts[i])
            comp_bl_ts = tuple(comp_bl_ts)

            sz_gbuf = tuple(sz_gbuf)
            sz_regf = tuple(sz_regf)
            for i in range(me.NUM):
                unit_access[i] = tuple(unit_access[i])
            unit_access = tuple(unit_access)
            regf_reusable = tuple(regf_reusable)

            nld = NestedLoopDesc(loopcnt=loopcnt, unit_access=unit_access, usize_gbuf=sz_gbuf,
                                 usize_regf=sz_regf, unit_ops=ops_lpe, unit_time=unit_time,
                                 data_loops=acclayer.data_loops(), regf_reusable=regf_reusable,
                                 rw_data=layer.rw_data)

            real_resource = resource._replace(size_gbuf=resource.size_gbuf / 0.99,
                                              size_regf=resource.size_regf / 0.99)

            lbs = LoopBlockingScheme(nld, comp_bl_ts, comp_bl_ords, real_resource, bufshr,
                                     self.options)
            if not lbs.is_valid():
                print("LBS INVALID!")
                print("acclayer.nofm", acclayer.nofm)
                print('acclayer.hofm', acclayer.hofm)
                print('acclayer.wofm', acclayer.wofm)
                print('acclyaer.hfil', acclayer.hfil)
                print('acclayer.wfil', acclayer.wfil)
                print("mapping_fold", mapping_fold)
                print("sz_gbuf", sz_gbuf)
                print("sz_regf", sz_regf)
                print("bl_ts", bl_ts)
                print("comp_bl_ts", comp_bl_ts)
                print("bl_ords", bl_ords)
                print("comp_bl_ords", comp_bl_ords)
                print("bufshr", tuple(bufshr.size(dce) for dce in range(de.NUM)))
        else:
            raise TypeError("Unsupported array mapping type {}".format(self.array_mapping))

        return lbs

    def _construct_sched_result(self, lbs, part, ofmap_layout, sched_seq, unit_nhops):
        scheme = OrderedDict()

        # Cost components.
        cost_access = lbs.get_access_cost(self.unit_cost)

        # Inter-node data forwarding/rotation hops.
        node_nhops = lbs.get_noc_access()
        # Memory access hops.
        print("Memory access hops:")
        print(unit_nhops)
        print(lbs.get_top_level_fetch())
        mem_nhops = [unh * f for unh, f in zip(unit_nhops, lbs.get_top_level_fetch())]
        # Total hops = inter-node hops + memory hops.
        total_nhops = [nnh + mnh for nnh, mnh in zip(node_nhops, mem_nhops)]
        cost_noc = self.unit_cost.noc_hop * sum(total_nhops)
        cost_node_nhops = self.unit_cost.noc_hop * sum(node_nhops)
        cost_mem_nhops = self.unit_cost.noc_hop * sum(mem_nhops)

        cost_op = self.unit_cost.mac_op * lbs.ops

        cost_static = self.unit_cost.idl_unit * lbs.time

        # Calculate the categorical access.
        access = lbs.get_access()
        cost_dram = sum(access[me.DRAM]) * self.unit_cost.mem_hier_at(me.DRAM)
        cost_sram = sum(access[me.GBUF]) * self.unit_cost.mem_hier_at(me.GBUF)
        cost_itcn = sum(access[me.ITCN]) * self.unit_cost.mem_hier_at(me.ITCN)
        cost_regf = sum(access[me.REGF]) * self.unit_cost.mem_hier_at(me.REGF)

        assert not math.isnan(cost_op + cost_access + cost_noc + cost_static)

        # Overall stats.
        scheme['cost'] = cost_op + cost_access + cost_noc + cost_static
        scheme['time'] = lbs.time
        scheme['ops'] = lbs.ops
        scheme['num_nodes'] = lbs.num_nodes
        scheme['is_dram'] = (lbs.src_is_dram, lbs.dst_is_dram)
        scheme['cost_op'] = cost_op
        scheme['cost_access'] = cost_access
        scheme['cost_dram'] = cost_dram
        scheme['cost_sram'] = cost_sram
        scheme['cost_itcn'] = cost_itcn
        scheme['cost_regf'] = cost_regf
        scheme['cost_node_nhops'] = cost_node_nhops
        scheme['cost_mem_nhops'] = cost_mem_nhops
        scheme['cost_noc'] = cost_noc
        scheme['cost_static'] = cost_static
        scheme['proc_time'] = lbs.proc_time
        scheme['bus_time'] = lbs.bus_time
        scheme['dram_time'] = lbs.dram_time
        scheme['access'] = lbs.get_access()
        scheme['unit_access'] = lbs.nld.unit_access
        scheme['remote_gbuf_access'] = lbs.remote_gbuf_access
        scheme['total_nhops'] = total_nhops
        scheme['fetch'] = lbs.fetch

        # Loop blocking.
        lp_ts = list(zip(*lbs.bl_ts))
        scheme['ti'] = tuple(lp_ts[le.IFM])
        scheme['to'] = tuple(lp_ts[le.OFM])
        scheme['tb'] = tuple(lp_ts[le.BAT])
        scheme['tvals'] = lbs.bl_ts
        scheme['orders'] = lbs.bl_ords
        scheme['size'] = [[lbs.data_size(bl, dce) for dce in range(de.NUM)]
                          for bl in range(lbs.BL.NUM)]
        scheme['unit_size'] = lbs.unit_size
        scheme['unit_cnt'] = lbs.unit_cnt
        scheme['accfwd_reduction'] = lbs.accfwd_reduction
        scheme['bufshr_grp_size'] = lbs.bufshr_grp_size
        scheme['bufshr_subgrp_size'] = lbs.bufshr_subgrp_size
        scheme['bufshr_bs_t'] = lbs.bufshr_bs_t
        scheme['bufshr_bs_ord'] = lbs.bufshr_bs_ord
        scheme['bufshr_rot_fetch'] = lbs.bufshr_rot_fetch
        scheme['bufshr_rot_round_cnt'] = lbs.bufshr_rot_round_cnt
        scheme['bufshr_rot_unit_cnt'] = lbs.bufshr_rot_unit_cnt
        scheme['bufshr_wide_fetch'] = lbs.bufshr_wide_fetch
        scheme['bufshr_wide_fetch_width'] = lbs.bufshr_wide_fetch_width

        # Partitioning.
        scheme['part'] = part
        scheme['mem_nhops'] = mem_nhops
        scheme['node_nhops'] = node_nhops
        scheme['unit_nhops'] = unit_nhops

        return SchedulingResult(scheme=scheme, ofmap_layout=ofmap_layout, sched_seq=sched_seq)

    def _gen_input_layout(self):
        input_layer = self.network.input_layer()
        input_frng = FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                               FmapPosition(b=self.batch_size,
                                            n=input_layer.nofm,
                                            h=input_layer.hofm,
                                            w=input_layer.wofm))

        ext_layer_names = self.network.ext_layers()
        ext_layers = [self.network[l] for l in ext_layer_names]
        ext_frngs = [FmapRange(FmapPosition(b=0, n=0, h=0, w=0),
                               FmapPosition(b=self.batch_size, n=ext_layer.nofm, h=ext_layer.hofm,
                                            w=ext_layer.wofm))
                     for ext_layer in ext_layers]

        # Input and external layers share the same region.

        input_region = ext_region = self.resource.src_data_region

        for part in partition.gen_partition(input_layer, self.batch_size, input_region.dim,
                                            self.options, guaranteed=True):
            input_layout = DataLayout(frngs=(input_frng,), regions=(input_region,),
                                      parts=(part.projection(input_region, appl2frng=True),))

            ext_layout_dict = dict(zip(ext_layer_names,
                                       [DataLayout(frngs=(ext_frng,), regions=(ext_region,),
                                                   parts=(part.projection(ext_region,
                                                                          appl2frng=True),))
                                        for ext_frng in ext_frngs])) if ext_layers else None

            yield input_layout, ext_layout_dict

def _search_layer_df_perprocess(search_method, layer_type, conv_strds, froz_parted_workload, part, buf_sharing,
                                resource, constraint, array_mapping, unit_nhops, unit_cost, options):
    opt_value = float('inf')
    min_layer_df = None
    min_layer_vars = None
    min_cost_dict = None
    min_total_cost = float('inf')
    min_layer_time = float('inf')
    min_cstr = None
    min_noc_hops = None
    min_accesses_result = None

    if array_mapping == ame.ROW_STATIONARY:
        tdm = RSTensorDimMap()
    elif array_mapping == ame.SYSTOLIC:
        tdm = SystolicTensorDimMap()
    else:
        raise ValueError("No corresponding dim map: {}".format(array_mapping))

    cost_model = KaplaCostModel(tdm)
    cost_model.set_cur_layer_type(layer_type)

    gbuf_workload = dict(froz_parted_workload)
    gbuf_stack = construct_stack(array_mapping, layer_type, part, gbuf_workload)
    gbuf_stack_num = util.prod((util.prod(pdim) for pdim in part.pdims))

    src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
    dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)

    mapping = construct_array_mapping(layer_type, gbuf_workload, array_mapping, resource, conv_strds)
    for loopcnt, unit_size, logic_region, regf_stack, origin_regf_update, unit_ops, regf_repls in \
            mapping.gen_array_mapping():
        regf_stack_num = logic_region[0] * logic_region[1] * util.prod(regf_repls.values())

        bl_ts_prob = 0.1
        lb_handle = generate_loop_blocking(loopcnt, constraint)
        if search_method == sme.RANDOM_SEARCHER:
            lb_handle = util.random_collect(lb_handle, bl_ts_prob)
        for knobs_tuple, bl_ts, real_cstr in lb_handle:
            regf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, bl_ts[BL.REGF + 1],
                                                 unit_size[BL.REGF])
            gbuf_bl_tp = [bl_a * bl_b for bl_a, bl_b in zip(bl_ts[BL.REGF + 1], bl_ts[BL.REGF])]
            gbuf_tensor_dims = derive_tensor_dim(layer_type, knobs_tuple, gbuf_bl_tp,
                                                 unit_size[BL.GBUF])

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
                continue

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
            g_upd_dims, g_iter_times = cost_model.analyze_dim_iters(gbuf_tensor_dims, gbuf_updates,
                                                                    gbuf_workload)
            g_logical_dim, g_buf_sharings, _, _ = \
                cost_model.analyze_stacks(g_froz_init_datas, gbuf_stack, resource.proc_region.dim,
                                          None, BL.GBUF)
            gbuf_unit_accesses = cost_model.analyze_relevant_accesses(g_init_datas, g_upd_dims,
                                                                      g_iter_times)

            r_init_datas, r_froz_init_datas = shape_init_data_block(tdm, regf_tensor_dims)
            r_upd_dims, r_iter_times = cost_model.analyze_dim_iters(regf_tensor_dims, regf_updates,
                                                                    regf_workload)
            r_froz_updates = tuple(regf_updates)
            r_logical_dim, _, r_remote_sharings, r_temporal_sharings = \
                cost_model.analyze_stacks(r_froz_init_datas, regf_stack, resource.dim_array,
                                          r_froz_updates, BL.REGF)
            regf_unit_accesses = cost_model.analyze_relevant_accesses(r_init_datas, r_upd_dims,
                                                                      r_iter_times)
            regf_upper_iters = cost_model.upper_fetch(r_upd_dims, r_iter_times, g_upd_dims,
                                                      g_iter_times)

            # Regfile accesses
            regf_accesses = [0 for _ in range(de.NUM)]
            flat_iter_times = r_iter_times + g_iter_times
            if layer_type in (lte.CONV, lte.DW_CONV, lte.CONV_BACK_H, lte.CONV_BACK_W,
                              lte.DW_CONV_H, lte.DW_CONV_W):
                regf_accesses[de.FIL] = unit_ops * util.prod(bl_ts[BL.REGF + 1]) * \
                                        util.prod(flat_iter_times) * gbuf_stack_num * \
                                        regf_stack_num

            regf_accesses[de.IFM] = unit_ops * util.prod(bl_ts[BL.REGF + 1]) * \
                                    util.prod(flat_iter_times) * regf_stack_num * \
                                    gbuf_stack_num
            regf_accesses[de.OFM] = unit_ops * util.prod(bl_ts[BL.REGF + 1]) * \
                                    util.prod(flat_iter_times) * regf_stack_num * \
                                    gbuf_stack_num * 2

            accesses_result[me.REGF] = regf_accesses

            nndf_ops = unit_ops * util.prod(bl_ts[BL.REGF + 1]) * util.prod(flat_iter_times) * \
                       gbuf_stack_num * regf_stack_num

            proc_time = unit_ops * util.prod(bl_ts[BL.REGF + 1]) * util.prod(flat_iter_times)

            bl_ord_counter = 0
            for bl_ords in itertools.product(*[itertools.permutations(range(len(knobs_tuple)))
                                               for _ in range(BL.NUM)]):
                g_rd_iters = cost_model.redundant_iter(g_upd_dims, g_iter_times, bl_ords[BL.GBUF])
                r_rd_iters = cost_model.redundant_iter(r_upd_dims, r_iter_times, bl_ords[BL.REGF])

                trivial_iter = False
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

                bufshr_rdt_iters = \
                    cost_model.bufshr_redundant_iter(g_upd_dims, g_iter_times, r_upd_dims,
                                                     r_iter_times, bl_ords[BL.REGF],
                                                     tuple(g_rd_iters), opt_out_bufshr)

                # Inter-layer constraints
                # batch, input, output update order
                outermost = len(knobs_tuple) - 1
                if "N" in knobs_tuple:
                    if constraint.topbat > 1 and \
                            (constraint.topifm > 1 or constraint.topofm > 1) and \
                            bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("N"):
                        continue
                    outermost -= 1
                if "C" in knobs_tuple:
                    if (constraint.topifm > 1) and \
                            bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("C"):
                        continue
                if "K" in knobs_tuple:
                    if (constraint.topofm > 1) and \
                            bl_ords[BL.GBUF].index(outermost) != knobs_tuple.index("K"):
                        continue
                # if data regions are not DRAM, can only access once, no spilling.
                if (not src_is_dram) and (g_rd_iters[de.IFM] > 1):
                    continue
                if (not dst_is_dram) and (g_rd_iters[de.OFM] > 1):
                    continue

                dram_accesses, fwd_hops, buf_shr_hops = \
                    cost_model.analyze_gbuf_level_access(gbuf_unit_accesses, g_rd_iters,
                                                         g_logical_dim, g_buf_sharings,
                                                         bufshr_rdt_iters, options)
                gbuf_accesses, itcn_accesses = \
                    cost_model.analyze_regf_level_access(regf_unit_accesses, r_rd_iters,
                                                         r_logical_dim, r_remote_sharings,
                                                         r_temporal_sharings, regf_upper_iters,
                                                         gbuf_stack_num)

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
                node_hops = [fwd_hop + bufshr_hop for fwd_hop, bufshr_hop in
                             zip(fwd_hops, buf_shr_hops)]
                mem_hops = [unh * f for unh, f in zip(unit_nhops, g_rd_iters)]
                noc_hops = [nnh + mnh for nnh, mnh in zip(node_hops, mem_hops)]

                # calculate the cost
                cost_dict = dict()
                cost_dict["dram_cost"] = sum(accesses_result[me.DRAM]) * \
                                         unit_cost.mem_hier_at(me.DRAM)
                cost_dict["sram_cost"] = sum(accesses_result[me.GBUF]) * \
                                         unit_cost.mem_hier_at(me.GBUF)
                cost_dict["itcn_cost"] = sum(accesses_result[me.ITCN]) * \
                                         unit_cost.mem_hier_at(me.ITCN)
                cost_dict["regf_cost"] = sum(accesses_result[me.REGF]) * \
                                         unit_cost.mem_hier_at(me.REGF)

                cost_dict["remote_sram_cost"] = unit_cost.mem_hier_at(me.GBUF) * \
                                                sum(remote_gbuf_access)
                cost_dict["node_hop_cost"] = sum(node_hops) * unit_cost.noc_hop
                cost_dict["mem_hop_cost"] = sum(mem_hops) * unit_cost.noc_hop
                cost_dict["op_cost"] = nndf_ops * unit_cost.mac_op
                total_cost = sum(cost_dict.values())

                # calculate the time
                dram_time = int(math.ceil(sum(accesses_result[me.DRAM]) / resource.dram_bandwidth))
                bus_time = util.idivc(int(math.ceil(1. * max(accesses_result[me.GBUF])
                                                    / gbuf_stack_num)), resource.array_bus_width)
                layer_time = (proc_time, dram_time, bus_time)

                # print("")
                # pp = pprint.PrettyPrinter(2)
                # pp.pprint(part)
                # pp.pprint(gbuf_workload)
                # pp.pprint(regf_workload)
                # pp.pprint(regf_repls)
                # pp.pprint(loopcnt)
                # pp.pprint(knobs_tuple)
                # pp.pprint(bl_ts)
                # pp.pprint(bl_ords)
                # pp.pprint(unit_nhops)
                # print("")
                # pp.pprint(g_upd_dims)
                # pp.pprint("fil rlvt dims: {}".format(tdm.get_dtype_rlvt_dims(layer_type, de.FIL)))
                # pp.pprint(g_iter_times)
                # pp.pprint(trivial_iter)
                # pp.pprint(g_rd_iters)
                # pp.pprint(gbuf_unit_accesses)
                # pp.pprint(gbuf_tensor_dims)
                # pp.pprint(gbuf_stack)
                # pp.pprint(gbuf_updates)
                # pp.pprint(g_buf_sharings)
                # print("")
                # pp.pprint(r_upd_dims)
                # pp.pprint(r_iter_times)
                # pp.pprint(r_rd_iters)
                # pp.pprint(regf_unit_accesses)
                # pp.pprint(regf_tensor_dims)
                # pp.pprint(regf_stack)
                # pp.pprint(regf_updates)
                # pp.pprint(r_remote_sharings)
                # pp.pprint(r_temporal_sharings)
                # print("")
                # pp.pprint(accesses_result)
                # pp.pprint(fwd_hops)
                # pp.pprint(bufshr_rdt_iters)
                # pp.pprint(buf_shr_hops)
                # pp.pprint(node_hops)
                # pp.pprint(cost_dict)
                # pp.pprint(total_cost)

                sorted_regf_updates = sort_update(regf_updates, bl_ords[BL.REGF],
                                                  origin_regf_update)
                sorted_gbuf_updates = sort_update(gbuf_updates, bl_ords[BL.GBUF])
                if options.opt_goal == 'e' and (total_cost < opt_value):
                    opt_value = total_cost
                    logic_region = mapping.logic_region
                    mapping_fold = mapping.fold
                    mapping_repls = mapping.repls.copy()
                    min_layer_vars = (layer_type, logic_region, mapping_fold, mapping_repls, part,
                                      conv_strds, loopcnt, unit_size, knobs_tuple, bl_ts, bl_ords,
                                      unit_ops, resource, buf_sharing, unit_nhops, g_logical_dim, g_buf_sharings, gbuf_unit_accesses, g_rd_iters, g_upd_dims, g_iter_times, bl_ords[BL.GBUF], accesses_result)
                    min_layer_df = layer_rearrange(tdm, gbuf_tensor_dims, gbuf_stack,
                                                   sorted_gbuf_updates, regf_tensor_dims,
                                                   regf_stack, sorted_regf_updates, buf_sharing)
                    min_total_cost = total_cost
                    min_layer_time = layer_time
                    min_cost_dict = cost_dict
                    min_cstr = real_cstr
                    min_noc_hops = noc_hops
                    min_accesses_result = accesses_result
                elif options.opt_goal == 'd' and (sum(layer_time) < opt_value):
                    opt_value = sum(layer_time)
                    logic_region = mapping.logic_region
                    mapping_fold = mapping.fold
                    mapping_repls = mapping.repls.copy()
                    min_layer_vars = (layer_type, logic_region, mapping_fold, mapping_repls, part,
                                      conv_strds, loopcnt, unit_size, knobs_tuple, bl_ts, bl_ords,
                                      unit_ops, resource, buf_sharing, unit_nhops, g_logical_dim, g_buf_sharings, gbuf_unit_accesses, g_rd_iters, g_upd_dims, g_iter_times, bl_ords[BL.GBUF], accesses_result)
                    min_layer_df = layer_rearrange(tdm, gbuf_tensor_dims, gbuf_stack,
                                                   sorted_gbuf_updates, regf_tensor_dims,
                                                   regf_stack, sorted_regf_updates, buf_sharing)
                    min_total_cost = total_cost
                    min_layer_time = layer_time
                    min_cost_dict = cost_dict
                    min_cstr = real_cstr
                    min_noc_hops = noc_hops
                    min_accesses_result = accesses_result
                elif options.opt_goal == 'ed' and (total_cost * sum(layer_time) < opt_value):
                    opt_value = total_cost * sum(layer_time)
                    logic_region = mapping.logic_region
                    mapping_fold = mapping.fold
                    mapping_repls = mapping.repls.copy()
                    min_layer_vars = (layer_type, logic_region, mapping_fold, mapping_repls, part,
                                      conv_strds, loopcnt, unit_size, knobs_tuple, bl_ts, bl_ords,
                                      unit_ops, resource, buf_sharing, unit_nhops, g_logical_dim, g_buf_sharings, gbuf_unit_accesses, g_rd_iters, g_upd_dims, g_iter_times, bl_ords[BL.GBUF], accesses_result)
                    min_layer_df = layer_rearrange(tdm, gbuf_tensor_dims, gbuf_stack,
                                                   sorted_gbuf_updates, regf_tensor_dims,
                                                   regf_stack, sorted_regf_updates, buf_sharing)
                    min_total_cost = total_cost
                    min_layer_time = layer_time
                    min_cost_dict = cost_dict
                    min_cstr = real_cstr
                    min_noc_hops = noc_hops
                    min_accesses_result = accesses_result
                bl_ord_counter += 1

    return min_layer_df, min_cost_dict, min_layer_time, min_cstr, min_noc_hops, \
            min_accesses_result, min_layer_vars


def generate_loop_blocking(loopcnt, constraint):
    knobs = OrderedDict()
    for dim, count in loopcnt.items():
        if count > 1:
            knobs[dim] = util.factorize(count, BL.NUM + 1)

    if len(knobs) == 0:
        knobs["N"] = util.factorize(1, BL.NUM + 1)
        knobs["K"] = util.factorize(1, BL.NUM + 1)
        knobs["C"] = util.factorize(1, BL.NUM + 1)

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
    elif layer_type in (lte.LOCAL, lte.DW_CONV):
        for kidx, knob in enumerate(knobs_tuple):
            if knob == "K":
                repl_size.setdefault("K", 1)
                repl_size.setdefault("C", 1)
                repl_size["K"] *= bl_t[kidx]
                repl_size["C"] *= bl_t[kidx]
            else:
                repl_size.setdefault(knob, 1)
                repl_size[knob] *= bl_t[kidx]
    elif layer_type in (lte.LOCAL_BACK_H, lte.DW_CONV_H, lte.DW_CONV_W):
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
    elif layer_type in (lte.LOCAL, lte.DW_CONV):
        for dim, t in zip(knobs_tuple, bl_t):
            if dim in existed_upd_dims:
                continue
            if dim == "C":
                continue
            elif dim == "K":
                results.append(("C", stacked_dims[dim] * conv_strds[2], "K", stacked_dims[dim]))
            else:
                results.append((dim, stacked_dims[dim]))
    elif layer_type in (lte.LOCAL_BACK_H, lte.DW_CONV_H, lte.DW_CONV_W):
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
    return origin_updates + tuple(x for x, _ in sorted(
        zip(unordered_updates[-len(bl_ord):], bl_ord), key=lambda item: item[1]))


def search_dataflow(search_method, network, batch_size, array_mapping, hw_fp, opt_fp, back_prop=False):
    hw = parse_json(hw_fp)
    resource, unit_cost = parse_hardware(hw)

    opts = parse_json(opt_fp)
    options = parse_options(opts)
    if back_prop and (options.partition_interlayer or options.hw_gbuf_save_writeback):
        print('run_back_prop(): back_prop should disable interlayer pipelining')
        sys.exit(1)

    searcher = KaplaSearcher(search_method, network, array_mapping, batch_size, resource, unit_cost, options)

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
    ap.add_argument('--method', default='searcher', choices=['searcher', 'ml', 'random'],
                    help='Method to find the optimal dataflow.')
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

    if args.method == 'searcher':
        search_method = sme.KAPLA_SEARCHER
    elif args.method == 'ml':
        search_method = sme.ML_SEARCHER
    elif args.method == 'random':
        search_method = sme.RANDOM_SEARCHER
    else:
        raise ValueError('Invalid search method: {}'.format(args.method))

    if array_mapping == ame.ROW_STATIONARY:
        if args.back_prop:
            hw_fp = "nn_dataflow/hardwares/multi_node.json"
            opt_fp = "nn_dataflow/options/option_training.json"
            search_dataflow(search_method, network, batch_size, array_mapping, hw_fp, opt_fp, args.back_prop)
        else:
            hw_fp = "nn_dataflow/hardwares/multi_node.json"
            opt_fp = "nn_dataflow/options/option_inference.json"
            search_dataflow(search_method, network, batch_size, array_mapping, hw_fp, opt_fp)
    elif array_mapping == ame.SYSTOLIC:
        hw_fp = "nn_dataflow/hardwares/single_node.json"
        opt_fp = "nn_dataflow/options/option_inference.json"
        search_dataflow(search_method, network, batch_size, array_mapping, hw_fp, opt_fp)


if __name__ == "__main__":
    sys.exit(main())
