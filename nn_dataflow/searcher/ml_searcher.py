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
from nn_dataflow.ml_related.ml_tuner import XGBTuner


class KaplaSearcher():
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
            print("{}: {}".format(layer_counter, layer_name))
            # if layer_counter != 3:
            #     layer_counter += 1
            #     continue
            seg_counter = 0
            nndf_list = []
            for seg in self.segments[layer_name]:
                print("- {}: {}".format(seg_counter, seg.seg))
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
                for sh_list in seg.ifm_fwd_dict.values():
                    # A list of spatial indices that share the same ifmaps.
                    r = allocation[sh_list[0].sp_idx][sh_list[0].tm_idx].proc_region
                    for idx in sh_list[1:]:
                        fwd_data_region_dict[idx] = r
                for fwd_src, fwd_dst_list in seg.ofm_fwd_dict.items():
                    # Ofmaps forwarded to neighbors.
                    r = allocation[fwd_src.sp_idx][fwd_src.tm_idx].proc_region
                    for idx in fwd_dst_list:
                        fwd_data_region_dict[idx] = r

                for constraint, _ in self.gen_constraint(seg):
                    print("-- constraint: {}".format(constraint))
                    cur_nndf = prev_nndf.copy()
                    seg_df, cost_dict, seg_time, cur_nndf, total_cost = self.search_segment_df(seg, allocation, constraint, cur_nndf)
                    if len(seg_df) == 0:
                        print("Constraint iter: No valid seg nndf.")
                        continue
                    print("** cur_nndf:")
                    print(cur_nndf)
                    seg_dfs.append((seg_df, cost_dict, seg_time, cur_nndf, total_cost))

                # Select best seg df.
                if len(seg_dfs) == 0:
                    print("Segment iter: No valid seg nndf.")
                    continue
                top_seg_df = sorted(seg_dfs, key=lambda x: x[-1])[0]
                print("***Best seg_dfs: {}".format(top_seg_df))

                nndf_result = nn_rearrange(seg_no_counter, top_seg_df, prev_df)
                nndf_list.append(nndf_result)
                seg_no_counter += 1
                seg_counter += 1
            if len(nndf_list) == 0:
                print("Last layer iter: No valid nndf.")
                continue
            df_tops[layer_name] = sorted(nndf_list, key=lambda x: x[-1])[0]
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
                layer = self.network[layer_name]

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
                    ifmap_layout = DataLayout(
                        frngs=ifmap_layout.frngs,
                        regions=(fwd_data_region,) * len(ifmap_layout.frngs),
                        parts=tuple(p.projection(fwd_data_region, appl2frng=True)
                                    for p in ifmap_layout.parts))

                search_result = self.xgbtuner.search(layer, self.batch_size, self.array_mapping, self.unit_cost,
                    cur_cstr, resource, sp_idx, tm_idx, ifmap_layout, self.options, n_trial=1024, n_parallel=128)
                if search_result is None:
                    return dict(), None, None, None, None
                df, cost_dict, layer_time, real_cstr, sched_vars = search_result
                if df is None:
                    return dict(), None, None, None, None
                print("layer: {}".format(layer_name))
                print("df: {}".format(df))
                print("cost: {}".format(cost_dict))
                print("total cost: {}".format(sum(cost_dict.values())))
                print("time: {}".format(layer_time))
                print("sched_vars:")
                for idx, var in enumerate(sched_vars):
                    print(var)
                print("---")
                sched_vars = sched_vars[:-1]
                sched_result = self.derive_sched_result(layer_name, seg_idx, sp_idx, tm_idx,
                    resource, ifmap_layout, sched_vars)
                print("sched_result", sched_result)
                cur_nndf[layer_name] = sched_result

                seg_df[layer_name]["dataflow"] = df
                seg_df[layer_name]["sched_seq"] = [0, sp_idx, tm_idx]
                for key, value in cost_dict.items():
                    seg_costs[key] = seg_costs.setdefault(key, 0) + value
                cstr_collections[layer_name] = real_cstr
                seg_times[sp_idx].append(layer_time)
                total_cost += sum(cost_dict.values())

        seg_time = self.cost_model.seg_time_estimation(self.network, segment, seg_times, cstr_collections)
        print("* cur_nndf")
        print(cur_nndf)
        return seg_df, seg_costs, seg_time, cur_nndf, total_cost


    def gen_segments(self):
        segments = defaultdict(list)
        for seg in self.ilp.gen_segment(self.options):
            if seg not in segments[seg[-1][-1]]:
                segments[seg[-1][-1]].append(seg)

        return segments

    def gen_constraint(self, segment):
        for constraint, hints in segment.gen_constraint():
            yield constraint, hints


    def derive_sched_result(self, layer_name, seg_idx, sp_idx, tm_idx, resource, ifmap_layout, sched_vars):
        (layer_type, logic_region, mapping_fold, mapping_repls, part, conv_strds, loopcnt, unit_size, knobs_tuple,
         bl_ts, bl_ords, unit_ops, resource, bufshr) = sched_vars

        layer = self.network[layer_name]

        proc_region = resource.proc_region

        # Ofmap layout.
        ofmap_range = FmapRange(
            FmapPosition(b=0, n=0, h=0, w=0),
            FmapPosition(b=self.batch_size, n=layer.nofm,
                            h=layer.hofm, w=layer.wofm))
        ofmap_data_region = resource.dst_data_region
        ofmap_layout = DataLayout(
            frngs=(ofmap_range,),
            regions=(ofmap_data_region,),
            parts=(part.projection(ofmap_data_region, appl2frng=True),))

        filter_nodes = frozenset(resource.dram_region.iter_node())
        unit_nhops = partition.unit_nhops_to_proc_region(layer, self.batch_size, proc_region,
            part, filter_nodes, ifmap_layout, ofmap_layout, self.options)

        lbs = self.derive_nndf_lbs(layer_type, layer, logic_region, mapping_fold, mapping_repls, part, conv_strds, loopcnt, unit_size, knobs_tuple,
                                   bl_ts, bl_ords, unit_ops, resource, bufshr)

        sched_seq = (seg_idx, sp_idx, tm_idx)

        sched_result = self._construct_sched_result(lbs, part, ofmap_layout, sched_seq, unit_nhops)

        return sched_result

    def derive_nndf_lbs(self, layer_type, layer, logic_region, mapping_fold, mapping_repls, part, conv_strds, loopcnt, unit_size, knobs_tuple,
                                   bl_ts, bl_ords, unit_ops, resource, bufshr):
        for bl in range(BL.NUM):
            unit_size[bl] = self.tdm.format_tensor_dim(layer_type, unit_size[bl], conv_strds)

        if self.array_mapping == ame.ROW_STATIONARY:
            part_layer = part_workload(self.array_mapping, part, layer, self.batch_size)
            if layer_type == lte.CONV:
                acclayer = ConvLayer(
                    1, 1,
                    (util.idivc(part_layer["Yo"], mapping_fold.w), part_layer["Xo"]),
                    (part_layer["S"], part_layer["R"]),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                      w=util.idivc(part_layer["Yo"], mapping_fold.w))
            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(
                    1,
                    (util.idivc(part_layer["Yo"], mapping_fold.w), part_layer["Xo"]),
                    conv_strds[2], (part_layer["S"], part_layer["R"]),
                    ntrd=conv_strds[2],
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                        w=util.idivc(part_layer["Yo"], mapping_fold.w))
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                acclayer = ConvLayer(
                    1, 1,
                    (util.idivc(part_layer["Yi"], mapping_fold.w), part_layer["Xi"]),
                    (part_layer["S"], part_layer["R"]),
                    strd=(conv_strds[1], conv_strds[0]), rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                        w=util.idivc(part_layer["Yi"], mapping_fold.w))
            elif layer_type == lte.LOCAL_BACK_H:
                acclayer = LocalRegionLayer(
                    1,
                    (util.idivc(part_layer["Yi"], mapping_fold.w), part_layer["Xi"]),
                    conv_strds[2], (part_layer["S"], part_layer["R"]),
                    ntrd=conv_strds[2],
                    strd=(conv_strds[1], conv_strds[0]), rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * mapping_fold.w / part_layer["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_layer["S"], mapping_fold.h),
                                        w=util.idivc(part_layer["Yi"], mapping_fold.w))
            # print(acclayer)
            flpesets_per_unitpass = mapping_fold.h
            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["K"] * mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"] * mapping_repls["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * mapping_repls["N"] / amp_acc_ifm
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["C"] * mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"] * mapping_repls["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["K"] * mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["K"] * mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["K"] * mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL_BACK_H:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["C"] * mapping_repls["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * mapping_repls["N"] / amp_acc_ifm

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM] * flpesets_per_unitpass
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM] * flpesets_per_unitpass

            unit_access[me.ITCN][de.IFM] = acclayer.wifm * dim_flpeset.size() \
                    * flpesets_per_unitpass * util.prod(mapping_repls.values())
            unit_access[me.ITCN][de.OFM] = acclayer.wofm * dim_flpeset.size() \
                    * flpesets_per_unitpass * util.prod(mapping_repls.values())

            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_access[me.ITCN][de.FIL] = acclayer.wfil * dim_flpeset.size() \
                        * flpesets_per_unitpass * util.prod(mapping_repls.values())
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_access[me.ITCN][de.FIL] = 0

            unit_access[me.REGF] = [acclayer.total_ops() * util.prod(mapping_repls.values())] * de.NUM
            if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_access[me.REGF][de.FIL] = 0

            sz_gbuf = self.tdm.get_tensor_size(layer_type, unit_size[BL.GBUF])
            sz_gbuf[de.IFM] /= mapping_fold.w
            sz_gbuf[de.OFM] /= mapping_fold.w

            if layer_type in (lte.CONV, lte.LOCAL):
                sz_gbuf[de.IFM] /= amp_acc_ifm
            else:
                sz_gbuf[de.OFM] /= amp_acc_ifm

            sz_regf = [0] * de.NUM
            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
                sz_regf[de.FIL] = acclayer.wfil
                sz_regf[de.IFM] = acclayer.wfil
                sz_regf[de.OFM] = 1
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                sz_regf[de.FIL] = 0
                sz_regf[de.IFM] = acclayer.wreg * acclayer.nreg
                sz_regf[de.OFM] = 1

            ops_lpe = acclayer.total_ops() * util.prod(mapping_repls.values())
            loopcnt = [loopcnt["C"], loopcnt["K"], loopcnt["N"] * mapping_fold.w]
            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_time = acclayer.wfil * acclayer.hfil
                regf_reusable[de.IFM] = (acclayer.wfil == acclayer.wifm)
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_time = acclayer.nreg * acclayer.wreg * acclayer.hreg
                regf_reusable[de.IFM] = (acclayer.wreg == acclayer.wifm)
            regf_reusable[de.OFM] = (acclayer.wofm == 1)
            regf_reusable[de.FIL] = (mapping_fold.h == 1)

            if layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
                sz_regf[de.IFM], sz_regf[de.OFM] = sz_regf[de.OFM], sz_regf[de.IFM]
                regf_reusable[de.IFM], regf_reusable[de.OFM] = regf_reusable[de.OFM], regf_reusable[de.IFM]
                for i in range(me.NUM):
                    unit_access[i][de.IFM], unit_access[i][de.OFM] = unit_access[i][de.OFM], unit_access[i][de.IFM]



            comp_bl_ts = [[1 for _ in range(3)] for _ in range(BL.NUM+1)]
            comp_bl_ords = [[1 for _ in range(3)] for _ in range(BL.NUM)]

            unset_ords = set()
            for d_idx, dim in enumerate(["C", "K", "N"]):
                if dim in knobs_tuple:
                    idx = knobs_tuple.index(dim)
                    for bl in range(BL.NUM+1):
                        comp_bl_ts[bl][d_idx] = bl_ts[bl][idx]
                    comp_bl_ords[BL.GBUF][d_idx] = bl_ords[BL.GBUF][idx]
                    comp_bl_ords[BL.REGF][d_idx] = bl_ords[BL.REGF][idx]
                else:
                    unset_ords.add(d_idx)

            comp_bl_ts[BL.GBUF+1][le.BAT] *= mapping_fold.w

            counter = len(bl_ords[BL.GBUF])
            for d_idx in unset_ords:
                for bl in range(BL.NUM):
                    comp_bl_ords[bl][d_idx] = counter
                counter += 1

            for bl in range(BL.NUM):
                for order, didx in enumerate(sorted(range(len(comp_bl_ords[bl])), key=lambda k: comp_bl_ords[bl][k])):
                    comp_bl_ords[bl][didx] = order

            # For back-prop localregion layer, since the data_loop mapping is using the conventional
            # forward version local-region layer, we need to reverse the i and o dimension.
            if layer_type == lte.LOCAL_BACK_H:
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
            for i in range(BL.NUM+1):
                comp_bl_ts[i] = tuple(comp_bl_ts[i])
            comp_bl_ts = tuple(comp_bl_ts)

            sz_gbuf = tuple(sz_gbuf)
            sz_regf = tuple(sz_regf)
            for i in range(me.NUM):
                unit_access[i] = tuple(unit_access[i])
            unit_access = tuple(unit_access)
            regf_reusable = tuple(regf_reusable)

            print("nld making")
            print(knobs_tuple)
            print(bl_ts)
            print(bl_ords)
            print(comp_bl_ts)
            print(comp_bl_ords)

            nld = NestedLoopDesc(loopcnt=loopcnt, unit_access=unit_access,
                                usize_gbuf=sz_gbuf, usize_regf=sz_regf,
                                unit_ops=ops_lpe, unit_time=unit_time, data_loops=acclayer.data_loops(),
                                regf_reusable=regf_reusable, rw_data=layer.rw_data)

            real_resource = resource._replace(size_gbuf=resource.size_gbuf/0.99,
                                              size_regf=resource.size_regf/0.99)


            lbs = LoopBlockingScheme(nld, comp_bl_ts, comp_bl_ords, real_resource, bufshr, self.options)
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
            # fold_hofm = util.closest_factor(mapping_fold.h, factor=mapping_fold.h/2)[0]
            # fold_wofm = mapping_fold.h / fold_hofm
            full_xy = layer.hofm * layer.wofm
            fold_xy = util.idivc(full_xy, mapping_fold.h)
            fold_x = util.closest_factor(fold_xy, factor=math.sqrt(fold_xy))[0]
            fold_y = fold_xy / fold_x
            if layer_type == lte.CONV:
                acclayer = ConvLayer(
                    1, 1. * part_layer.nofm / fold_w,
                    (fold_x, fold_y),
                    (part_layer.hfil, part_layer.wfil),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / \
                    part_layer.hifm / part_layer.wifm
                dim_flpeset = PhyDim2(h=logic_region.h, w=logic_region.w)
                unit_time = layer.wfil * layer.hfil
            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(
                    1. * layer.nofm / fold_w,
                    (fold_x, fold_y),
                    layer.nreg, (layer.hreg, layer.wreg),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / \
                    part_layer.hifm / part_layer.wifm
                dim_flpeset = PhyDim2(h=logic_region.h, w=logic_region.w)
                unit_time = layer.nreg * layer.wreg * layer.hreg

            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * mapping_repls["N"]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * mapping_repls["C"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["C"] * mapping_repls["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * mapping_repls["N"] / amp_acc_ifm

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM]
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM]

            unit_access[me.ITCN][de.IFM] = acclayer.total_ifmap_size() * util.prod(mapping_repls.values())
            unit_access[me.ITCN][de.OFM] = acclayer.total_ofmap_size() * util.prod(mapping_repls.values())
            if layer_type == lte.CONV:
                unit_access[me.ITCN][de.FIL] = acclayer.total_filter_size() * util.prod(mapping_repls.values())
            elif layer_type == lte.LOCAL:
                unit_access[me.ITCN][de.FIL] = 0

            unit_access[me.REGF] = [acclayer.total_ops() * util.prod(mapping_repls.values())] * de.NUM

            sz_gbuf = [0] * de.NUM
            sz_gbuf[de.IFM] = acclayer.total_ifmap_size() * util.prod(mapping_repls.values())
            sz_gbuf[de.OFM] = acclayer.total_ofmap_size() * util.prod(mapping_repls.values())
            if layer_type == lte.CONV:
                sz_gbuf[de.FIL] = acclayer.total_ifmap_size() * util.prod(mapping_repls.values())
            else:
                sz_gbuf[de.FIL] = 0

            sz_regf = [0] * de.NUM
            if layer_type == lte.CONV:
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

            comp_bl_ts = [[1 for _ in range(3)] for _ in range(BL.NUM+1)]
            comp_bl_ords = [[1 for _ in range(3)] for _ in range(BL.NUM)]

            unset_ords = set()
            set_flags = [False] * le.NUM
            for lidx, dims in enumerate([["C"], ["K"], ["N", "XY"]]):
                for dim in dims:
                    if dim in knobs_tuple:
                        set_flags[lidx] = True
                        idx = knobs_tuple.index(dim)
                        for bl in range(BL.NUM+1):
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
                for order, didx in enumerate(sorted(range(len(comp_bl_ords[bl])), key=lambda k: comp_bl_ords[bl][k])):
                    comp_bl_ords[bl][didx] = order

            for i in range(BL.NUM):
                comp_bl_ords[i] = tuple(comp_bl_ords[i])
            comp_bl_ords = tuple(comp_bl_ords)
            for i in range(BL.NUM+1):
                comp_bl_ts[i] = tuple(comp_bl_ts[i])
            comp_bl_ts = tuple(comp_bl_ts)

            sz_gbuf = tuple(sz_gbuf)
            sz_regf = tuple(sz_regf)
            for i in range(me.NUM):
                unit_access[i] = tuple(unit_access[i])
            unit_access = tuple(unit_access)
            regf_reusable = tuple(regf_reusable)

            print(bl_ords)
            print(comp_bl_ords)

            nld = NestedLoopDesc(loopcnt=loopcnt, unit_access=unit_access, usize_gbuf=sz_gbuf,
                                 usize_regf=sz_regf, unit_ops=ops_lpe, unit_time=unit_time,
                                 data_loops=acclayer.data_loops(), regf_reusable=regf_reusable,
                                 rw_data=layer.rw_data)

            real_resource = resource._replace(size_gbuf=resource.size_gbuf/0.99,
                                              size_regf=resource.size_regf/0.99)

            lbs = LoopBlockingScheme(nld, comp_bl_ts, comp_bl_ords, real_resource, bufshr, self.options)
            if not lbs.is_valid():
                print("LBS INVALID!")
                print("sz_gbuf", sz_gbuf)
                print("sz_regf", sz_regf)
                print("bl_ts", bl_ts)
                print("comp_bl_ts", comp_bl_ts)
                print("bl_ords", bl_ords)
                print("comp_bl_ords", comp_bl_ords)
                print("bufshr", tuple(bufshr.size(dce) for dce in range(de.NUM)))

        return lbs

    def _construct_sched_result(self, lbs, part, ofmap_layout, sched_seq, unit_nhops):
        scheme = OrderedDict()

        # Cost components.
        cost_access = lbs.get_access_cost(self.unit_cost)

        # Inter-node data forwarding/rotation hops.
        node_nhops = lbs.get_noc_access()
        # Memory access hops.
        mem_nhops = [unh * f for unh, f
                     in zip(unit_nhops, lbs.get_top_level_fetch())]
        # Total hops = inter-node hops + memory hops.
        # total_nhops = [nnh for nnh in node_nhops]
        total_nhops = [nnh + mnh for nnh, mnh in zip(node_nhops, mem_nhops)]
        cost_noc = self.unit_cost.noc_hop * sum(total_nhops)
        cost_node_nhops = self.unit_cost.noc_hop * sum(node_nhops)
        cost_mem_nhops = self.unit_cost.noc_hop * sum(mem_nhops)

        cost_op = self.unit_cost.mac_op * lbs.ops

        cost_static = self.unit_cost.idl_unit * lbs.time
        # cost_static = 0

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

        return SchedulingResult(scheme=scheme, ofmap_layout=ofmap_layout,
                                sched_seq=sched_seq)

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
                               FmapPosition(b=self.batch_size,
                                            n=ext_layer.nofm,
                                            h=ext_layer.hofm,
                                            w=ext_layer.wofm))
                     for ext_layer in ext_layers]

        # Input and external layers share the same region.

        input_region = ext_region = self.resource.src_data_region

        for part in partition.gen_partition(input_layer, self.batch_size,
                                            input_region.dim, self.options,
                                            guaranteed=True):
            input_layout = DataLayout(
                frngs=(input_frng,),
                regions=(input_region,),
                parts=(part.projection(input_region, appl2frng=True),))

            ext_layout_dict = dict(zip(
                ext_layer_names,
                [DataLayout(
                    frngs=(ext_frng,),
                    regions=(ext_region,),
                    parts=(part.projection(ext_region, appl2frng=True),))
                 for ext_frng in ext_frngs])) if ext_layers else None

            yield input_layout, ext_layout_dict



def search_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, back_prop=False):

    hw = parse_json(hw_fp)
    resource, unit_cost = parse_hardware(hw)

    opts = parse_json(opt_fp)
    options = parse_options(opts)
    if back_prop and (options.partition_interlayer or options.hw_gbuf_save_writeback):
        print('run_back_prop(): back_prop should disable interlayer pipelining')
        sys.exit(1)

    searcher = KaplaSearcher(network, array_mapping, batch_size, resource, unit_cost, options)

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
    # total_static_cost = 0

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
