import itertools
import copy
import functools
import sys
import pprint
import math
import time

from multiprocessing import Pool
from collections import defaultdict

from nn_dataflow import util
import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
from nn_dataflow.core import InterLayerPipeline
from nn_dataflow.core.layer import ConvLayer, LocalRegionLayer, ConvBackLayer, LocalRegionBackLayer
from nn_dataflow.nns import import_network
from nn_dataflow.core.node_region import NodeRegion

from nn_dataflow.solver.fast_explorer import gen_segment_set, segment_occp_is_valid
from nn_dataflow.array_mapping_templates.row_stationary import RowStationary
from nn_dataflow.array_mapping_templates.systolic import Systolic
from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, ident_layer_type, \
                                                               get_conv_strds, SystolicTensorDimMap
from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ParallelEnum as pe
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
     shape_init_data_block, SegDfCache, SimpleCstr, BL, nn_rearrange, layer2workload, is_valid, \
     get_min_factor



class KaplaSolver():
    def __init__(self, network, array_mapping, batch_size, resource, unit_cost, options, ntops=1):
        self.network = network
        self.array_mapping = array_mapping
        self.batch_size = batch_size
        self.resource = resource
        self.unit_cost = unit_cost
        self.options = options
        self.ntops = ntops

        self.ilp = InterLayerPipeline(network, batch_size, resource)
        self.priored_segments = self.solve_priortize_segment()
        self.ordered_layer_list = self.ilp.ordered_layer_list()

        if array_mapping == ame.ROW_STATIONARY:
            self.tdm = RSTensorDimMap()
        elif array_mapping == ame.SYSTOLIC:
            self.tdm = SystolicTensorDimMap()
        else:
            raise ValueError("No corresponding dim map: {}".format(array_mapping))

        self.cost_model = KaplaCostModel(network, self.tdm)

    def solve_dataflow(self):
        df_tops = defaultdict(lambda: list())
        layer_counter = 0
        seg_no_counter = 0
        for layer_name in self.ordered_layer_list:
            print("{}: {}".format(layer_counter, layer_name))
            seg_counter = 0
            for seg in self.priored_segments[layer_name]:
                print("- {}: {}".format(seg_counter, seg.seg))
                allocation = seg.allocation()
                seg_dfs = list()
                for constraint, _ in self.solve_constraint(seg):
                    print("-- constraint: {}".format(constraint))
                    seg_df, cost_dict, seg_time, total_cost = self.solve_segment_df(seg, allocation, constraint)
                    if len(seg_df) == 0:
                        print("No valid seg nndf.")
                        continue
                    seg_dfs.append((seg_df, cost_dict, seg_time, total_cost))

                # Select best seg df.
                seg_dfs = sorted(seg_dfs, key=lambda x: x[-1])[:self.ntops]
                print("***Best seg_dfs: {}".format(seg_dfs))

                # Append to nndf.
                curr_layer_idx = self.ordered_layer_list.index(seg[0][0])
                if curr_layer_idx == 0:
                    prev_nndfs = []
                else:
                    prev_nndfs = df_tops[self.ordered_layer_list[curr_layer_idx-1]]
                nndf_result = nn_rearrange(seg_no_counter, seg_dfs, prev_nndfs)
                nndf_result = sorted(nndf_result, key=lambda x: x[-1])[:self.ntops]
                df_tops[layer_name].extend(nndf_result)
                seg_no_counter += 1
                seg_counter += 1
            df_tops[layer_name] = sorted(df_tops[layer_name], key=lambda x: x[-1])[:self.ntops]
            layer_counter += 1

        return df_tops[self.ordered_layer_list[-1]][:self.ntops]

    def solve_segment_df(self, segment, allocation, constraint):
        seg_df = defaultdict(lambda: dict())
        seg_times = list()
        cstr_collections = dict()
        seg_costs = defaultdict(lambda: dict())
        total_cost = 0
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

                df, real_cstr, cost_dict, layer_time = self.solve_layer_df(layer, cur_cstr, resource, \
                    sp_idx, tm_idx)
                print("layer: {}".format(layer_name))
                print("df: {}".format(df))
                print("cost: {}".format(sum(cost_dict.values())))
                print("time: {}".format(layer_time))
                print("---")
                if df is None:
                    return dict(), None, None, None

                seg_df[layer_name]["dataflow"] = df
                seg_df[layer_name]["sched_seq"] = [0, sp_idx, tm_idx]
                for key, value in cost_dict.items():
                    seg_costs[key] = value
                cstr_collections[layer_name] = real_cstr
                seg_times[sp_idx].append(layer_time)
                total_cost += sum(cost_dict.values())

        seg_time = self.cost_model.seg_time_estimation(segment, seg_times, cstr_collections)
        return seg_df, cost_dict, seg_time, total_cost

    def solve_layer_df(self, layer, cstr, resource, sp_idx, tm_idx):
        min_cost = float("inf")
        min_cost_dict = dict()
        min_layer_time = float("inf")
        min_cstr = None
        layer_cand = None

        layer_type = ident_layer_type(layer)
        conv_strds = get_conv_strds(layer_type, layer)
        self.cost_model.set_cur_layer_type(layer_type)

        layer_data_size = [0 for _ in range(de.NUM)]
        if layer_type in (lte.CONV, lte.CONV_BACK_H):
            layer_data_size[de.FIL] = layer.total_filter_size()
        layer_data_size[de.IFM] = layer.total_ifmap_size(batch_size=self.batch_size)
        layer_data_size[de.OFM] = layer.total_ofmap_size(batch_size=self.batch_size)

        loopcnt, origin_regf_repls, regf_unit_tensor, gbuf_unit_tensor, regf_base_stacks, \
            regf_base_updates, origin_stack_step_dict, unit_ops = \
            self.solve_array_mapping(layer_type, layer, conv_strds, \
            resource)

        origin_regf_unit_tensor = self.tdm.format_tensor_dim(layer_type, regf_unit_tensor, conv_strds)
        origin_gbuf_unit_tensor = self.tdm.format_tensor_dim(layer_type, gbuf_unit_tensor, conv_strds)

        # Iterate through all possible orders.
        for bl_ords in itertools.product(*[itertools.permutations(range(le.NUM))
                                         for _ in range(BL.NUM)]):
            regf_unit_tensor = copy.deepcopy(origin_regf_unit_tensor)
            gbuf_unit_tensor = copy.deepcopy(origin_gbuf_unit_tensor)
            regf_repls = [r for r in origin_regf_repls]

            is_valid, top_bl_ts, remain_lcnt = \
                self.cstr_check_prune(layer_type, cstr, loopcnt, bl_ords)
            if not is_valid:
                continue

            # first fit as much as possible into regfile to reduce gbuf access.
            froz_remain_lcnt, froz_regf_unit_tensor, froz_regf_tensor_repl_dict = \
                self.fill_tensor(layer_type,
                                 conv_strds,
                                 layer_data_size,
                                 frozenset(remain_lcnt.items()),
                                 frozenset(regf_unit_tensor.items()),
                                 resource.size_regf,
                                 bl_ords[BL.REGF])
            if froz_remain_lcnt is None:
                continue

            remain_lcnt = dict(froz_remain_lcnt)
            regf_unit_tensor = dict(froz_regf_unit_tensor)
            regf_tensor_repl_dict = dict(froz_regf_tensor_repl_dict)

            # Analyze regfile base stacks.
            _, regf_froz_init_data = shape_init_data_block(self.tdm, regf_unit_tensor)
            regf_base_stacks = tuple(regf_base_stacks)
            logical_dim, _, regf_rmt_shr, _ = \
                self.cost_model.analyze_stacks(regf_froz_init_data, regf_base_stacks,
                resource.dim_array, BL.REGF)

            base_stack_fetch = [0 for _ in range(de.NUM)]
            for dce in range(de.NUM):
                for dtype, shr_num_dict in regf_rmt_shr:
                    if dtype == dce:
                        rmtshr = shr_num_dict
                        break
                noshr_node_num = logical_dim[0] * logical_dim[1]
                for shr_node_num, group_num in rmtshr.items():
                    base_stack_fetch[dce] += group_num
                    noshr_node_num -= shr_node_num * group_num
                base_stack_fetch[dce] += noshr_node_num

            remain_lcnt, regf_stack_repl_dict, regf_repls = \
                self.fill_stack(layer_type, layer_data_size, remain_lcnt, regf_unit_tensor,
                                base_stack_fetch, regf_repls, bl_ords[BL.REGF], multicast=True)

            # Multiply the REGF level blocking factor.
            gbuf_unit_tensor = self.gbuf_tensor_mul(layer_type,
                                                    gbuf_unit_tensor,
                                                    regf_tensor_repl_dict,
                                                    regf_stack_repl_dict)

            gbuf_repls = [resource.proc_region.dim.w, resource.proc_region.dim.h]
            # First if no buffer sharing, it's the same as regf level.
            if not self.options.hw_gbuf_sharing:
                # Fit into tensor.
                froz_remain_lcnt, froz_gbuf_unit_tensor, froz_gbuf_tensor_repl_dict = \
                        self.fill_tensor(layer_type, conv_strds, layer_data_size,
                                         frozenset(remain_lcnt.items()),
                                         frozenset(gbuf_unit_tensor.items()),
                                         resource.size_gbuf, bl_ords[BL.GBUF])
                if froz_remain_lcnt is None:
                    continue
                remain_lcnt = dict(froz_remain_lcnt)
                gbuf_unit_tensor = dict(froz_gbuf_unit_tensor)
                gbuf_tensor_repl_dict = dict(froz_gbuf_tensor_repl_dict)

                base_stack_fetch = [1 for _ in range(de.NUM)]

                remain_lcnt, gbuf_stack_repl_dict, gbuf_repls = \
                    self.fill_stack(layer_type, layer_data_size, remain_lcnt, gbuf_unit_tensor,
                                    base_stack_fetch, gbuf_repls, bl_ords[BL.GBUF],
                                    multicast=self.options.hw_access_forwarding)
                if util.prod(gbuf_repls) != 1:
                    print("Not enough data to be fully paralleled.")
                    continue

            # If buffer sharing is enabled, since the stack dim will affect the actual data stored in
            # each node, it's not true to regrad tensor constraint as a static constraint. Therefore
            # we need to dynamically adjust the constraint and the stack dim.
            else:
                # When buffer sharing is enabled, stacking will not cause data duplication. To fully
                # utilize node array and increase the throughput, we need to stack on all physical nodes
                # and thus the stack schemes will not affect data occupation. Then we can first decide
                # the stack scheme and then fill the node accupation.
                # Fit into stack.
                base_stack_fetch = [1 for _ in range(de.NUM)]
                remain_lcnt, gbuf_stack_repl_dict, gbuf_repls = \
                    self.fill_stack(layer_type, layer_data_size, remain_lcnt, gbuf_unit_tensor,
                                    base_stack_fetch, gbuf_repls, bl_ords[BL.GBUF], multicast=True)
                if util.prod(gbuf_repls) != 1:
                    print("Not enough data to be fully paralleled: {}".format(gbuf_repls))
                    continue

                shr_node_num = [1, 1, 1]
                for drc_stack_dict in gbuf_stack_repl_dict:
                    for dim, repl in drc_stack_dict.items():
                        for dtype in self.tdm.get_dim_irr_dtypes(layer_type, dim):
                            shr_node_num[dtype] *= repl

                # Fit into tensor.
                froz_remain_lcnt, froz_gbuf_unit_tensor, froz_gbuf_tensor_repl_dict = \
                    self.fill_tensor(layer_type, conv_strds, layer_data_size,
                                     frozenset(remain_lcnt.items()),
                                     frozenset(gbuf_unit_tensor.items()),
                                     resource.size_gbuf, bl_ords[BL.GBUF], tuple(shr_node_num))
                if froz_remain_lcnt is None:
                    continue
                remain_lcnt = dict(froz_remain_lcnt)
                gbuf_unit_tensor = dict(froz_gbuf_unit_tensor)
                gbuf_tensor_repl_dict = dict(froz_gbuf_tensor_repl_dict)

            # Check constraint again. The above fill_tensor, fill_stack and fill_with_bufshr algorithm
            # exactly prioritized those constrained dims due to the pipeline order requirement.
            # Thus for those constrained blocking dimension, remain_lcnt should be reduced to 1.
            # Unless the constraint cannot be met.
            gbuf_iter_dict = dict()
            if top_bl_ts[le.BAT] > 0:
                if remain_lcnt["N"] != 1:
                    continue
            else:
                top_bl_ts[le.BAT] = remain_lcnt["N"]

            if top_bl_ts[le.IFM] > 0:
                if remain_lcnt["C"] != 1:
                    continue
            else:
                top_bl_ts[le.IFM] = remain_lcnt["C"]

            if top_bl_ts[le.OFM] > 0:
                if remain_lcnt["K"] != 1:
                    continue
            else:
                top_bl_ts[le.OFM] = remain_lcnt["K"]

            gbuf_iter_dict = self.get_gbuf_iter(layer_type, top_bl_ts, remain_lcnt)
            real_cstr = SimpleCstr(top_bl_ts[le.BAT], top_bl_ts[le.IFM], top_bl_ts[le.OFM])

            ## Finalize the dataflow description and estimate cost.
            regf_workload = self.derive_workload(layer_type, gbuf_unit_tensor, regf_stack_repl_dict,
                conv_strds, origin_stack_step_dict)
            regf_updates = self.derive_update_drc(layer_type, regf_unit_tensor,
                gbuf_tensor_repl_dict, bl_ords[BL.REGF], conv_strds, regf_base_updates)
            regf_stacks = self.derive_stack_drc(layer_type, regf_stack_repl_dict, regf_workload,
                None, conv_strds, regf_base_stacks)

            gbuf_workload = self.derive_workload(layer_type, layer2workload(self.array_mapping, layer, self.batch_size),
                gbuf_stack_repl_dict, conv_strds)
            gbuf_updates = self.derive_update_drc(layer_type, gbuf_unit_tensor, gbuf_iter_dict,
                bl_ords[BL.GBUF], conv_strds)
            gbuf_stacks = self.derive_stack_drc(layer_type, gbuf_stack_repl_dict, gbuf_workload,
                regf_updates, conv_strds)

            # Stats parameters.
            regf_ops_iter = util.prod(regf_tensor_repl_dict.values())
            regf_stack_num = util.prod(stack[-1] for stack in regf_base_stacks)
            gbuf_stack_num = 1
            for regf_stack in regf_stack_repl_dict:
                regf_stack_num *= util.prod(regf_stack.values())
            for gbuf_stack in gbuf_stack_repl_dict:
                gbuf_stack_num *= util.prod(gbuf_stack.values())

            # Get layer dataflow cost.
            accesses_result, noc_hops, cost_dict, total_cost, layer_time = \
                self.get_ldf_cost(layer_type, frozenset(regf_unit_tensor.items()), regf_updates,
                            regf_stacks, regf_workload, len(regf_base_updates),
                            frozenset(gbuf_unit_tensor.items()), gbuf_updates, gbuf_stacks,
                            gbuf_workload, regf_ops_iter, regf_stack_num, gbuf_stack_num,
                            unit_ops, conv_strds, resource)

            if total_cost < min_cost:
                min_cost = total_cost
                min_layer_time = layer_time
                min_cost_dict = cost_dict
                min_cstr = real_cstr
                layer_cand = self.finalize_layer_df(regf_unit_tensor, regf_updates, regf_stacks,
                                            gbuf_unit_tensor, gbuf_updates, gbuf_stacks)

        return layer_cand, min_cstr, min_cost_dict, min_layer_time

    def cstr_check_prune(self, layer_type, constraint, loopcnt, bl_ords):
        is_valid = True
        if loopcnt["N"] < constraint.topbat:
            is_valid = False
        if loopcnt["C"] < constraint.topifm:
            is_valid = False
        if loopcnt["K"] < constraint.topofm:
            is_valid = False

        top_bl_ts = [0 for _ in range(le.NUM)]
        top_bl_ts[le.BAT] = constraint.topbat
        # topifm and topofm cannot be trigger together.
        top_bl_ts[le.IFM] = constraint.topifm
        top_bl_ts[le.OFM] = constraint.topofm

        outermost = le.NUM - 1
        # Require BAT always at the top to eliminate redundant order when topbat is 1.
        if constraint.topbat > 1 and (constraint.topifm > 1 or constraint.topofm > 1) and \
            bl_ords[BL.GBUF].index(outermost) != le.BAT:
            is_valid = False
        outermost -= 1
        if constraint.topifm > 1 and bl_ords[BL.GBUF].index(outermost) != le.IFM:
            is_valid = False
        if constraint.topofm > 1 and bl_ords[BL.GBUF].index(outermost) != le.OFM:
            is_valid = False

        remain_lcnt = dict()
        if top_bl_ts[le.BAT]:
            remain_lcnt["N"] = util.idivc(loopcnt["N"], top_bl_ts[le.BAT])
        else:
            remain_lcnt["N"] = loopcnt["N"]
        if top_bl_ts[le.IFM]:
            remain_lcnt["C"] = util.idivc(loopcnt["C"], top_bl_ts[le.IFM])
        else:
            remain_lcnt["C"] = loopcnt["C"]
        if top_bl_ts[le.OFM]:
            remain_lcnt["K"] = util.idivc(loopcnt["K"], top_bl_ts[le.OFM])
        else:
            remain_lcnt["K"] = loopcnt["K"]

        if self.array_mapping == ame.ROW_STATIONARY:
            if layer_type in (lte.CONV, lte.LOCAL):
                remain_lcnt["Xo"] = loopcnt["Xo"]
                remain_lcnt["Yo"] = loopcnt["Yo"]
            elif layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H):
                remain_lcnt["Xi"] = loopcnt["Xi"]
                remain_lcnt["Yi"] = loopcnt["Yi"]
        elif self.array_mapping == ame.SYSTOLIC:
            remain_lcnt["XY"] = loopcnt["XY"]

        return is_valid, top_bl_ts, remain_lcnt

    def gbuf_tensor_mul(self, layer_type, gbuf_unit_tensor, regf_tensor_repl_dict,
                        regf_stack_repl_dict):
        for dim, r in regf_tensor_repl_dict.items():
            if (layer_type == lte.LOCAL and dim == "K") or \
               (layer_type == lte.LOCAL_BACK_H and dim == "C"):
                gbuf_unit_tensor["C"] *= r
                gbuf_unit_tensor["K"] *= r
            elif self.array_mapping == ame.ROW_STATIONARY and \
                 ((layer_type in (lte.CONV, lte.LOCAL) and dim == "Yo") or \
                 (layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H) and dim == "Yi")):
                gbuf_unit_tensor["Yo"] *= r
                gbuf_unit_tensor["Yi"] *= r
            elif self.array_mapping == ame.ROW_STATIONARY and \
                 (layer_type in (lte.CONV, lte.LOCAL) and dim == "Xo") or \
                 (layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H) and dim == "Xi"):
                gbuf_unit_tensor["Xo"] *= r
                gbuf_unit_tensor["Xi"] *= r
            else:
                gbuf_unit_tensor[dim] *= r

        for reg_stack_dict in regf_stack_repl_dict:
            for dim, r in reg_stack_dict.items():
                if (layer_type == lte.LOCAL and dim == "K") or \
                   (layer_type == lte.LOCAL_BACK_H and dim == "C"):
                    gbuf_unit_tensor["C"] *= r
                    gbuf_unit_tensor["K"] *= r
                elif self.array_mapping == ame.ROW_STATIONARY and \
                     (layer_type in (lte.CONV, lte.LOCAL) and dim == "Yo") or \
                     (layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H) and dim == "Yi"):
                    gbuf_unit_tensor["Yo"] *= r
                    gbuf_unit_tensor["Yi"] *= r
                elif self.array_mapping == ame.ROW_STATIONARY and \
                     (layer_type in (lte.CONV, lte.LOCAL) and dim == "Xo") or \
                     (layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H) and dim == "Xi"):
                    gbuf_unit_tensor["Xo"] *= r
                    gbuf_unit_tensor["Xi"] *= r
                else:
                    gbuf_unit_tensor[dim] *= r

        return gbuf_unit_tensor

    def get_gbuf_iter(self, layer_type, top_bl_ts, remain_lcnt):
        gbuf_iter_dict = dict()
        if self.array_mapping == ame.ROW_STATIONARY:
            gbuf_iter_dict["N"] = top_bl_ts[le.BAT]
            gbuf_iter_dict["C"] = top_bl_ts[le.IFM]
            gbuf_iter_dict["K"] = top_bl_ts[le.OFM]
            if layer_type in (lte.CONV, lte.LOCAL):
                gbuf_iter_dict["Yo"] = remain_lcnt["Yo"]
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                gbuf_iter_dict["Yi"] = remain_lcnt["Yi"]
        elif self.array_mapping == ame.SYSTOLIC:
            gbuf_iter_dict["N"] = top_bl_ts[le.BAT]
            gbuf_iter_dict["C"] = top_bl_ts[le.IFM]
            gbuf_iter_dict["K"] = top_bl_ts[le.OFM]
            gbuf_iter_dict["XY"] = remain_lcnt["XY"]

        return gbuf_iter_dict

    def solve_priortize_segment(self):
        segments = defaultdict(list)
        for seg in self.ilp.gen_segment(self.options):
            if seg not in segments[seg[-1][-1]]:
                segments[seg[-1][-1]].append(seg)

        # priored_segments = gen_segment_set(segments, self.ordered_layer_list, self.network,
        #                                    self.unit_cost, self.options)
        # return priored_segments
        return segments

    def solve_constraint(self, segment):
        for constraint, hints in segment.gen_constraint():
            # if constraint[0][0].topbat != 0 \
            #         and not segment_occp_is_valid(
            #                 seg.seg, seg.network, seg.batch_size,
            #                 constraint, seg.alloc):
            #     continue

            yield constraint, hints

    def solve_array_mapping(self, layer_type, layer, conv_strds, resource):
        # Construct the full layer workload.
        workload = layer2workload(self.array_mapping, layer, self.batch_size)

        # Get the array mapping unit blocks.
        if self.array_mapping == ame.ROW_STATIONARY:
            mapping = RowStationary(layer_type, workload, resource, conv_strds)
        elif self.array_mapping == ame.SYSTOLIC:
            mapping = Systolic(layer_type, workload, resource, conv_strds)
        else:
            raise ValueError("Not yet implement {}".format(self.array_mapping))

        loopcnt, regf_repls, regf_unit_tensor, gbuf_unit_tensor, base_stacks, base_updates, \
            origin_stack_step_dict, unit_ops = mapping.get_unit_block()

        return loopcnt, regf_repls, regf_unit_tensor, gbuf_unit_tensor, base_stacks, base_updates, \
               origin_stack_step_dict, unit_ops

    def fill_tensor(self, layer_type, conv_strds, layer_data_size, froz_lcnt, froz_tensor, buf_size,
                    bl_ord, shr_node_num=None):
        remain_lcnt = dict(froz_lcnt)
        unit_tensor = dict(froz_tensor)
        # print("")
        # print("---begin to fill tensor")
        # print("remain_lcnt: {}".format(remain_lcnt))
        # print("unit_tensor: {}".format(unit_tensor))
        # print("buf_size: {}".format(buf_size))
        # print("shr_node_num: {}".format(shr_node_num))
        if not is_valid(self.tdm, layer_type, unit_tensor, buf_size, shr_node_num):
            print("Bad init unit tensor!")
            return None, None, None
        tensor_repl_dict = defaultdict(lambda: 1)
        inm_ltype = bl_ord.index(0)
        inm_irr_ds = self.tdm.get_ltype_irr_dtypes(layer_type, inm_ltype)
        outer_ltypes = list(range(le.NUM))
        outer_ltypes.pop(inm_ltype)
        outer_irr_ds = list(range(de.NUM))
        for inm_irr_d in inm_irr_ds:
            outer_irr_ds.pop(inm_irr_d)

        # refetch irrelevant iteration.
        outer_refetch_size = [layer_data_size[dce] for dce in outer_irr_ds]
        if de.OFM in outer_irr_ds:
            outer_refetch_size[outer_irr_ds.index(de.OFM)] *= 2
        for outer_idx, dce in enumerate(outer_irr_ds):
            for dim in self.tdm.get_dtype_irr_dims(layer_type, dce):
                outer_refetch_size[outer_idx] *= remain_lcnt[dim]

        # Fit outer_irr lcnt into tensor.
        outer_irr_factors = []
        outer_irr_dims = []
        outer_crrspd_idx = []
        for outer_idx, dce in enumerate(outer_irr_ds):
            for dim in self.tdm.get_dtype_irr_dims(layer_type, dce):
                factors = [x for x, _ in util.factorize(remain_lcnt[dim], 2)]
                # print(factors)
                max_valid_idx = len(factors)
                for idx, f in enumerate(factors):
                    unit_tensor = self.tensor_mul(layer_type, conv_strds, unit_tensor, dim, f)
                    if not is_valid(self.tdm, layer_type, unit_tensor, buf_size, shr_node_num):
                        # print("Not valid")
                        # print(unit_tensor, buf_size, shr_node_num)
                        max_valid_idx = idx
                        unit_tensor = self.tensor_div(layer_type, conv_strds, unit_tensor, dim, f)
                        break
                    unit_tensor = self.tensor_div(layer_type, conv_strds, unit_tensor, dim, f)
                outer_irr_dims.append(dim)
                outer_irr_factors.append(factors[:max_valid_idx])
                outer_crrspd_idx.append(outer_idx)

        # print(outer_irr_dims)
        # print(outer_irr_factors)
        # print(outer_crrspd_idx)
        # print(outer_refetch_size)

        best_factors = None
        min_outer_refetch = float("inf")
        for factors in itertools.product(*outer_irr_factors):
            # print("factors", factors)
            temp_outer_fetch_size = [s for s in outer_refetch_size]
            for dim, o_idx, f in zip(outer_irr_dims, outer_crrspd_idx, factors):
                unit_tensor = self.tensor_mul(layer_type, conv_strds, unit_tensor, dim, f)
                temp_outer_fetch_size[o_idx] = util.idivc(temp_outer_fetch_size[o_idx], f)
            if is_valid(self.tdm, layer_type, unit_tensor, buf_size, shr_node_num):
                sum_refetch = sum(temp_outer_fetch_size)
                # print("sum_refetch: {}, min_outer_refetch: {}".format(sum_refetch, min_outer_refetch))
                if sum_refetch < min_outer_refetch:
                    best_factors = factors
                    min_outer_refetch = sum_refetch
            for dim, f in zip(outer_irr_dims, factors):
                unit_tensor = self.tensor_div(layer_type, conv_strds, unit_tensor, dim, f)

        for dim, f in zip(outer_irr_dims, best_factors):
            unit_tensor = self.tensor_mul(layer_type, conv_strds, unit_tensor, dim, f)
            remain_lcnt[dim] = util.idivc(remain_lcnt[dim], f)
            tensor_repl_dict[dim] *= f

        # print("best_factors: {}".format(best_factors))

        # Try to fit inm_irr lcnt into tensor.
        while True:
            inm_rlvt_dims = self.tdm.get_ltype_rlvt_dims(layer_type, inm_ltype)
            min_factors = [float('inf') for _ in inm_rlvt_dims]
            for dim_idx, dim in enumerate(inm_rlvt_dims):
                min_factor = get_min_factor(remain_lcnt[dim])
                if min_factor != 1:
                    min_factors[dim_idx] = min_factor
            min_factor = min(min_factors)
            if min_factor == float('inf'):
                break
            min_factor_idx = min_factors.index(min_factor)
            min_factor_dim = inm_rlvt_dims[min_factor_idx]

            unit_tensor = self.tensor_mul(layer_type, conv_strds, unit_tensor, min_factor_dim, min_factor)
            if is_valid(self.tdm, layer_type, unit_tensor, buf_size, shr_node_num):
                remain_lcnt[min_factor_dim] = util.idivc(remain_lcnt[min_factor_dim], min_factor)
                tensor_repl_dict[min_factor_dim] *= min_factor
            else:
                unit_tensor = self.tensor_div(layer_type, conv_strds, unit_tensor, min_factor_dim, min_factor)
                break

        return frozenset(remain_lcnt.items()), frozenset(unit_tensor.items()), frozenset(tensor_repl_dict.items())

    def tensor_div(self, layer_type, conv_strds, tensor, dim, factor):
        if (layer_type == lte.LOCAL and dim == "K") or \
           (layer_type == lte.LOCAL_BACK_H and dim == "C"):
            tensor["C"] /= factor
            tensor["K"] /= factor
        else:
            tensor[dim] /= factor

        if self.array_mapping == ame.ROW_STATIONARY:
            # Recalculate ifm/ofm.
            if layer_type in (0, 1):
                tensor["Xi"] = (tensor["Xo"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yi"] = (tensor["Yo"] - 1) * conv_strds[1] + tensor["S"]
            elif layer_type in (2, 3):
                tensor["Xo"] = (tensor["Xi"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yo"] = (tensor["Yi"] - 1) * conv_strds[1] + tensor["S"]

        return tensor

    def tensor_mul(self, layer_type, conv_strds, tensor, dim, factor):
        if (layer_type == lte.LOCAL and dim == "K") or \
           (layer_type == lte.LOCAL_BACK_H and dim == "C"):
            tensor["C"] *= factor
            tensor["K"] *= factor
        else:
            tensor[dim] *= factor

        if self.array_mapping == ame.ROW_STATIONARY:
            # Recalculate ifm/ofm.
            if layer_type in (0, 1):
                tensor["Xi"] = (tensor["Xo"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yi"] = (tensor["Yo"] - 1) * conv_strds[1] + tensor["S"]
            elif layer_type in (2, 3):
                tensor["Xo"] = (tensor["Xi"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yo"] = (tensor["Yi"] - 1) * conv_strds[1] + tensor["S"]

        return tensor

    def fill_stack(self, layer_type, layer_data_size, remain_lcnt, unit_tensor, base_stack_fetch,
                   hw_repls, bl_ord, multicast=False):
        # print("")
        # print("---begin to fill stack")
        # print("hw_repls: {}".format(hw_repls))
        # print("remain_lcnt: {}".format(remain_lcnt))
        # print("unit_tensor: {}".format(unit_tensor))
        stacks_repl_dict = [defaultdict(lambda: 1) for _ in range(len(hw_repls))]
        inm_ltype = bl_ord.index(0)
        inm_irr_ds = self.tdm.get_ltype_irr_dtypes(layer_type, inm_ltype)
        outer_dim = list(range(le.NUM))
        outer_dim.pop(inm_ltype)
        outer_irr_ds = list(range(de.NUM))
        for inm_irr_d in inm_irr_ds:
            outer_irr_ds.pop(inm_irr_d)
        # print("inm_irr_ds: {}".format(inm_irr_ds))
        # print("outer_irr_ds: {}".format(outer_irr_ds))

        # refetch irrelevant iteration.
        refetch_size = [layer_data_size[dce] for dce in range(de.NUM)]
        refetch_size[de.OFM] *= 2
        for dce in range(de.NUM):
            if dce in inm_irr_ds:
                continue
            for dim in self.tdm.get_dtype_irr_dims(layer_type, dce):
                refetch_size[dce] *= remain_lcnt[dim]

        outer_refetch_size = [refetch_size[dce] for dce in outer_irr_ds]

        for drc in range(len(hw_repls)):
            while hw_repls[drc] != 1:
                # First try to fit a factor into the outer dimensions.
                min_factor = get_min_factor(hw_repls[drc])

                # If all outer_refetch dimensions are trivial, then to improve throughput, we can stack on
                # the innermost dimensions.
                outer_trivial = True
                for dce in outer_irr_ds:
                    dce_irr_dims = self.tdm.get_dtype_irr_dims(layer_type, dce)
                    if any(remain_lcnt[dim] > 1 for dim in dce_irr_dims):
                        outer_trivial = False
                # print("is outer trivial: {}".format(outer_trivial))
                if outer_trivial:
                    inm_rlvt_dims = self.tdm.get_ltype_rlvt_dims(layer_type, inm_ltype)
                    if all(remain_lcnt[dim] == 1 for dim in inm_rlvt_dims):
                        break
                    min_cost = float("inf")
                    min_cost_dim = None
                    # # If stack on Yo dimension, there may exist extra ifmap. Therefore we tend to first
                    # # fit to N dimension.
                    # if inm_ltype == le.BAT:
                    #     if remain_lcnt["N"] > 1:
                    #         min_cost_dim = "N"

                    for dim in inm_rlvt_dims:
                        cost = util.idivc(remain_lcnt[dim], min_factor) * min_factor / remain_lcnt[dim]
                        if cost < min_cost:
                            min_cost = cost
                            min_cost_dim = dim
                    remain_lcnt[min_cost_dim] = util.idivc(remain_lcnt[min_cost_dim], min_factor)
                    stacks_repl_dict[drc][min_cost_dim] *= min_factor
                    hw_repls[drc] = util.idivc(hw_repls[drc], min_factor)
                    continue

                # Fit into outer dimensions.
                outer_refetch_size = [refetch_size[dce] for dce in outer_irr_ds]
                outer_idx = outer_refetch_size.index(max(outer_refetch_size))
                dtype = outer_irr_ds[outer_idx]
                irr_ltype = le.NUM - 1 - dtype
                # print("outer_fetch_size", outer_refetch_size)

                dividable = False
                for dim in self.tdm.get_ltype_rlvt_dims(layer_type, irr_ltype):
                    if remain_lcnt[dim] % min_factor == 0:
                        dividable = True
                        remain_lcnt[dim] = util.idivc(remain_lcnt[dim], min_factor)
                        refetch_size[dtype] = util.idivc(refetch_size[dtype], min_factor)
                        stacks_repl_dict[drc][dim] *= min_factor
                        hw_repls[drc] = util.idivc(hw_repls[drc], min_factor)
                        break

                # If the outer dimensions cannot be divided by the min factor, then estimate the cost at
                # each dimension and choose the one with least fetch data size.
                if not dividable:
                    min_cost = float("inf")
                    min_cost_dim = None
                    min_cost_dtype = None

                    for dce in range(de.NUM):
                        irr_ltype = le.NUM - 1 - dce
                        for dim in self.tdm.get_ltype_rlvt_dims(layer_type, irr_ltype):
                            temp_fetch_size = [refetch_size[dce] for dce in range(de.NUM)]
                            extra_ratio = util.idivc(remain_lcnt[dim], min_factor) * min_factor / \
                                        remain_lcnt[dim]
                            for d in self.tdm.get_dim_rlvt_dtypes(layer_type, dim):
                                temp_fetch_size[d] *= extra_ratio
                            temp_fetch_size[dtype] /= min_factor
                            fetch_size_sum = sum(temp_fetch_size)
                            if fetch_size_sum < min_cost:
                                min_cost = fetch_size_sum
                                min_cost_dim = dim
                                min_cost_dtype = dce

                    remain_lcnt[min_cost_dim] = util.idivc(remain_lcnt[min_cost_dim], min_factor)
                    refetch_size[min_cost_dtype] = util.idivc(refetch_size[min_cost_dtype], min_factor)
                    stacks_repl_dict[drc][min_cost_dim] *= min_factor
                    hw_repls[drc] = util.idivc(hw_repls[drc], min_factor)

        return remain_lcnt, stacks_repl_dict, hw_repls

    def derive_workload(self, layer_type, stacked_tensor, stack_repl_dict, conv_strds,
                        origin_stack_step_dict=None):
        workload = dict()
        for key, value in stacked_tensor.items():
            workload[key] = value

        repl_dict = dict()
        for sr_dict in stack_repl_dict:
            for dim, repl in sr_dict.items():
                repl_dict[dim] = repl_dict.setdefault(dim, 1) * repl

        if origin_stack_step_dict is not None:
            for dim, repl in origin_stack_step_dict.items():
                repl_dict[dim] = repl_dict.setdefault(dim, 1) * repl

        for dim, value in repl_dict.items():
            workload[dim] = util.idivc(workload[dim], value)

        if self.array_mapping == ame.ROW_STATIONARY:
            if layer_type in (lte.CONV, lte.LOCAL):
                workload["Xi"] = (workload["Xo"] - 1) * conv_strds[0] + workload["R"]
                workload["Yi"] = (workload["Yo"] - 1) * conv_strds[1] + workload["S"]
            elif layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H):
                workload["Xo"] = (workload["Xi"] - 1) * conv_strds[0] + workload["R"]
                workload["Yo"] = (workload["Yi"] - 1) * conv_strds[1] + workload["S"]

        if layer_type == lte.LOCAL:
            workload["C"] = util.idivc(workload["C"], repl_dict.setdefault("K", 1))
        elif layer_type == lte.LOCAL_BACK_H:
            workload["K"] = util.idivc(workload["K"], repl_dict.setdefault("C", 1))

        return util.HashableDict(workload)

    def derive_update_drc(self, layer_type, unit_tensor, update_cnt_dict, bl_ord, conv_strds, base_updates=()):
        updates = list()
        if base_updates is not None:
            for upd in base_updates:
                updates.append(upd)
        if layer_type == lte.CONV:
            for order in range(le.NUM):
                idx = bl_ord.index(order)
                for dim in self.tdm.get_ltype_rlvt_dims(layer_type, idx):
                    if self.array_mapping == ame.ROW_STATIONARY and \
                        dim in {"Yo", "Xo"} and update_cnt_dict.get(dim, 1) == 1:
                        continue
                    if self.array_mapping == ame.ROW_STATIONARY and dim == "Yo":
                        if not any("Yo" in upd for upd in base_updates):
                            updates.append(("Yi", unit_tensor["Yo"] * conv_strds[1], "Yo", unit_tensor["Yo"]))
                    else:
                        updates.append((dim, unit_tensor[dim]))
        elif layer_type == lte.LOCAL:
            for order in range(le.NUM):
                idx = bl_ord.index(order)
                if idx == le.IFM:
                    continue
                for dim in self.tdm.get_ltype_rlvt_dims(layer_type, idx):
                    if self.array_mapping == ame.ROW_STATIONARY and \
                        dim in {"Yo", "Xo"} and update_cnt_dict.get(dim, 1) == 1:
                        continue
                    if self.array_mapping == ame.ROW_STATIONARY and dim == "Yo":
                        if not any("Yo" in upd for upd in base_updates):
                            updates.append(("Yi", unit_tensor["Yo"] * conv_strds[1], "Yo", unit_tensor["Yo"]))
                    elif dim == "C":
                        continue
                    elif dim == "K":
                        updates.append(("C", unit_tensor["K"] * conv_strds[2], "K", unit_tensor["K"]))
                    else:
                        updates.append((dim, unit_tensor[dim]))
        elif layer_type == lte.CONV_BACK_H:
            for order in range(le.NUM):
                idx = bl_ord.index(order)
                for dim in self.tdm.get_ltype_rlvt_dims(layer_type, idx):
                    if dim in {"Yi", "Xi"} and update_cnt_dict.get(dim, 1) == 1:
                        continue
                    if dim == "Yi":
                        if not any("Yi" in upd for upd in base_updates):
                            updates.append(("Yi", unit_tensor["Yi"], "Yo", unit_tensor["Yi"] * conv_strds[1]))
                    else:
                        updates.append((dim, unit_tensor[dim]))
        elif layer_type == lte.LOCAL_BACK_H:
            for order in range(le.NUM):
                idx = bl_ord.index(order)
                if idx == le.OFM:
                    continue
                for dim in self.tdm.get_ltype_rlvt_dims(layer_type, idx):
                    if dim in {"Yi", "Xi"} and update_cnt_dict.get(dim, 1) == 1:
                        continue
                    if dim == "K":
                        continue
                    elif dim == "C":
                        updates.append(("C", unit_tensor["C"], "K", unit_tensor["C"] * conv_strds[2]))
                    elif dim == "Yi":
                        if not any("Yi" in upd for upd in base_updates):
                            updates.append(("Yi", unit_tensor["Yi"], "Yo", unit_tensor["Yi"] * conv_strds[1]))
                    else:
                        updates.append((dim, unit_tensor[dim]))

        return tuple(updates)

    def derive_stack_drc(self, layer_type, stack_repl_dict, workload, updates, conv_strds, base_stacks=()):
        stacks = list()

        if base_stacks:
            for bs in base_stacks:
                stacks.append(bs)

        strds = defaultdict(lambda: 1)
        for key, value in workload.items():
            strds[key] = value

        repl_dict = dict()
        for sr_dict in stack_repl_dict:
            for dim, repl in sr_dict.items():
                repl_dict[dim] = repl_dict.setdefault(dim, []) + [repl,]

        # Similar loop dims should be placed together to increase the buf sharing region.
        dim_nums = [0 for _ in range(le.NUM)]
        for ltype, dim_list in enumerate(self.tdm.loop_list[layer_type]):
            for dim in dim_list:
                dim_nums[ltype] += len(repl_dict.get(dim, []))
        sorted_ltypes = sorted(range(le.NUM), key=lambda k: dim_nums[k], reverse=True)

        if updates:
            # Optimize no bufshr data type.
            bufshr_rdt_iters = [False for _ in range(de.NUM)]
            start_flags = [False for _ in range(de.NUM)]
            for update in updates:
                relevant_flags = [False for _ in range(de.NUM)]
                for item in update:
                    if isinstance(item, str):
                        for dce in range(de.NUM):
                            if item not in self.tdm.data_list[dce]:
                                continue
                            start_flags[dce] = True
                            relevant_flags[dce] = True
                for dce in range(de.NUM):
                    if start_flags[dce] and not relevant_flags[dce]:
                        bufshr_rdt_iters[dce] = True

            # print("@TEMP: bufshr_rdt_iter: {}".format(bufshr_rdt_iters))

            no_bufshr_ltype_idxs = []
            for dce in range(de.NUM):
                if not bufshr_rdt_iters[dce]:
                    for ltype in self.tdm.get_dtype_rlvt_ltypes(layer_type, dce):
                        no_bufshr_ltype_idxs.append(sorted_ltypes.index(ltype))
                    inm_pos = min(no_bufshr_ltype_idxs)
                    irr_ltypes = self.tdm.get_dtype_irr_ltypes(layer_type, dce)
                    for idx, ltype in enumerate(sorted_ltypes):
                        if ltype in irr_ltypes and idx > inm_pos:
                            sorted_ltypes.pop(idx)
                            sorted_ltypes.insert(inm_pos, ltype)

        sorted_dims = []
        for ltype in sorted_ltypes:
            for dim in self.tdm.loop_list[layer_type][ltype]:
                if dim not in sorted_dims:
                    sorted_dims.append(dim)

        for dim in sorted_dims:
            if dim not in repl_dict:
                continue
            repl_list = repl_dict[dim]
            strd = workload.get(dim, 1)
            for repl in repl_list:
                if layer_type in (lte.CONV, lte.LOCAL):
                    if self.array_mapping == ame.ROW_STATIONARY and dim == "Yo":
                        yo_stack_idx = None
                        for idx, stack in base_stacks:
                            if "Yo" in stack:
                                yo_stack_idx = idx
                                break
                        if yo_stack_idx is None:
                            stacks.append(("Yi", (strd - 1) * conv_strds[1] + workload["S"], "Yo", strd, repl))
                        else:
                            strd = stacks[yo_stack_idx][stacks[yo_stack_idx].index("Yo")+1]
                            origin_repl = stacks[yo_stack_idx][-1]
                            stacks[yo_stack_idx] = ("Yi", (strd - 1) * conv_strds[1] + workload["S"], "Yo", strd, origin_repl * repl)
                    elif dim == "K" and layer_type == 1:
                        stacks.append(("C", strd * conv_strds[2], "K", strd, repl))
                    else:
                        stacks.append((dim, strd, repl))
                elif layer_type in (lte.CONV_BACK_H, lte.LOCAL_BACK_H):
                    assert(self.array_mapping != ame.SYSTOLIC, "Unsupported layer type: {}".format(layer_type))
                    if dim == "Yi":
                        yi_stack_idx = None
                        for idx, stack in base_stacks:
                            if "Yi" in stack:
                                yi_stack_idx = idx
                                break
                        if yi_stack_idx is None:
                            stacks.append(("Yi", strd, "Yo", (strd - 1) * conv_strds[1] + workload["S"], repl))
                        else:
                            strd = stacks[yi_stack_idx][stacks[yi_stack_idx].index("Yi") + 1]
                            origin_repl = stacks[yi_stack_idx][-1]
                            stacks[yi_stack_idx] = ("Yi", strd, "Yo", (strd - 1) * conv_strds[1] + workload["S"], origin_repl * repl)
                    elif dim == "C" and layer_type == 3:
                        stacks.append(("C", strd, "K", strd * conv_strds[2], repl))
                    else:
                        stacks.append((dim, strd, repl))
                strd *= repl

        return tuple(stacks)


    def get_ldf_cost(self, layer_type, regf_froz_tensor, regf_updates, regf_stacks, regf_workload,
                     regf_base_upd_num, gbuf_froz_tensor, gbuf_updates, gbuf_stacks, gbuf_workload,
                     regf_ops_iter, regf_stack_num, gbuf_stack_num, unit_ops, conv_strds, resource):
        regf_unit_tensor = dict(regf_froz_tensor)
        gbuf_unit_tensor = dict(gbuf_froz_tensor)

        accesses_result = [[0 for _ in range(de.NUM)] for _ in range(me.NUM)]
        g_init_datas, g_froz_init_datas = shape_init_data_block(self.tdm, gbuf_unit_tensor)
        g_upd_dims, g_iter_times = self.cost_model.analyze_dim_iters(gbuf_unit_tensor, gbuf_updates, gbuf_workload)
        g_logical_dim, g_buf_sharings, _, _ = self.cost_model.analyze_stacks(g_froz_init_datas, gbuf_stacks,
                                                            resource.proc_region.dim, BL.GBUF)
        gbuf_unit_accesses = self.cost_model.analyze_relevant_accesses(g_init_datas, g_upd_dims,
                                                    g_iter_times, self.options)

        r_init_datas, r_froz_init_datas = shape_init_data_block(self.tdm, regf_unit_tensor)
        r_upd_dims, r_iter_times = self.cost_model.analyze_dim_iters(regf_unit_tensor, regf_updates, regf_workload)
        r_logical_dim, _, r_remote_sharings, r_temporal_sharings = \
                self.cost_model.analyze_stacks(r_froz_init_datas, regf_stacks, resource.dim_array, BL.REGF)
        regf_unit_accesses = self.cost_model.analyze_relevant_accesses(r_init_datas, r_upd_dims,
                                                    r_iter_times, self.options)
        regf_upper_iters = self.cost_model.upper_fetch(r_upd_dims, r_iter_times, g_upd_dims, g_iter_times)

        # Regfile accesses
        regf_accesses = [0 for _ in range(de.NUM)]
        flat_iter_times = util.prod(r_iter_times + g_iter_times) * regf_ops_iter
        if layer_type in (0, 2):
            regf_accesses[de.FIL] = unit_ops * flat_iter_times * regf_stack_num * gbuf_stack_num
        regf_accesses[de.IFM] = unit_ops * flat_iter_times * regf_stack_num * gbuf_stack_num
        regf_accesses[de.OFM] = unit_ops * flat_iter_times * regf_stack_num * gbuf_stack_num * 2
        accesses_result[me.REGF] = regf_accesses

        nndf_ops = unit_ops * flat_iter_times * gbuf_stack_num * regf_stack_num
        proc_time = unit_ops * flat_iter_times

        g_rd_iters = self.cost_model.redundant_iter(g_upd_dims, g_iter_times, range(len(g_upd_dims)))
        r_rd_iters = self.cost_model.redundant_iter(r_upd_dims, r_iter_times, range(len(r_upd_dims) - regf_base_upd_num))

        if resource.no_time_mux:
            trivial_iter = True
            for dim in self.tdm.get_dtype_rlvt_dims(layer_type, de.FIL):
                for upd_idx, dims in enumerate(g_upd_dims):
                    if dim in dims and g_iter_times[upd_idx] != 1:
                        trivial_iter = False
            if trivial_iter:
                g_rd_iters = list(g_rd_iters)
                g_rd_iters[de.FIL] = 0
                g_rd_iters = tuple(g_rd_iters)

        opt_out_bufshr = False
        if is_valid(self.tdm, layer_type, gbuf_unit_tensor, resource.size_gbuf):
            opt_out_bufshr = True

        bufshr_rdt_iters = self.cost_model.bufshr_redundant_iter(g_upd_dims, g_iter_times, r_upd_dims, r_iter_times,
                                                tuple(range(len(r_upd_dims) - regf_base_upd_num)), g_rd_iters, opt_out_bufshr)

        dram_accesses, fwd_hops, buf_shr_hops = \
            self.cost_model.analyze_gbuf_level_access(gbuf_unit_accesses, g_rd_iters, g_logical_dim, g_buf_sharings,
                                    bufshr_rdt_iters, self.options)
        gbuf_accesses, itcn_accesses = \
            self.cost_model.analyze_regf_level_access(regf_unit_accesses, r_rd_iters, r_logical_dim, r_remote_sharings,
                                    r_temporal_sharings, regf_upper_iters, gbuf_stack_num)

        # inter-layer data sharing.
        src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
        dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)
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
        noc_hops = [fwd_hop + bufshr_hop for fwd_hop, bufshr_hop in zip(fwd_hops, buf_shr_hops)]

        # print("g_init_datas:")
        # print(g_init_datas)
        # print("g_upd_dims:")
        # print(g_upd_dims)
        # print("g_iter_times:")
        # print(g_iter_times)
        # print("g_logical_dim:")
        # print(g_logical_dim)
        # print("g_buf_sharings:")
        # print(g_buf_sharings)
        # print("gbuf_unit_accesses:")
        # print(gbuf_unit_accesses)
        # print("g_rd_iters:")
        # print(g_rd_iters)
        # print("")
        # print("r_init_datas:")
        # print(r_init_datas)
        # print("r_upd_dims:")
        # print(r_upd_dims)
        # print("r_iter_times:")
        # print(r_iter_times)
        # print("r_logical_dim:")
        # print(r_logical_dim)
        # print("r_remote_sharings:")
        # print(r_remote_sharings)
        # print("regf_unit_accesses:")
        # print(regf_unit_accesses)
        # print("regf_upper_iters:")
        # print(regf_upper_iters)
        # print("r_rd_iters:")
        # print(r_rd_iters)

        # print("bufshr_rdt_iters:")
        # print(bufshr_rdt_iters)
        # print("src_is_dram: {}".format(src_is_dram))
        # print("dst_is_dram: {}".format(dst_is_dram))

        # calculate the cost
        cost_dict = dict()
        cost_dict["dram_cost"] = sum(accesses_result[me.DRAM]) * self.unit_cost.mem_hier_at(me.DRAM)
        cost_dict["sram_cost"] = sum(accesses_result[me.GBUF]) * self.unit_cost.mem_hier_at(me.GBUF)
        cost_dict["itcn_cost"] = sum(accesses_result[me.ITCN]) * self.unit_cost.mem_hier_at(me.ITCN)
        cost_dict["regf_cost"] = sum(accesses_result[me.REGF]) * self.unit_cost.mem_hier_at(me.REGF)

        cost_dict["remote_sram_cost"] = self.unit_cost.mem_hier_at(me.GBUF) * sum(remote_gbuf_access)
        cost_dict["noc_cost"] = sum(noc_hops) * self.unit_cost.noc_hop
        cost_dict["op_cost"] = nndf_ops * self.unit_cost.mac_op
        total_cost = sum(cost_dict.values())

        # calculate the time
        dram_time = int(math.ceil(sum(accesses_result[me.DRAM]) / resource.dram_bandwidth))
        bus_time = util.idivc(int(math.ceil(1. * max(accesses_result[me.GBUF])
                                / gbuf_stack_num)), resource.array_bus_width)
        layer_time = (proc_time, dram_time, bus_time)

        return accesses_result, noc_hops, cost_dict, total_cost, layer_time

    def finalize_layer_df(self, regf_unit_tensor, regf_updates, regf_stacks, gbuf_unit_tensor,
                          gbuf_updates, gbuf_stacks):
        layer_df = dict()
        gbuf_df = dict()
        gbuf_df['tensor_w'] = {dim: gbuf_unit_tensor[dim] for dim in self.tdm.data_list[de.FIL]}
        gbuf_df['tensor_i'] = {dim: gbuf_unit_tensor[dim] for dim in self.tdm.data_list[de.IFM]}
        gbuf_df['tensor_o'] = {dim: gbuf_unit_tensor[dim] for dim in self.tdm.data_list[de.OFM]}
        gbuf_df['stack'] = gbuf_stacks
        gbuf_df['update'] = gbuf_updates

        regf_df = dict()
        regf_df['tensor_w'] = {dim: regf_unit_tensor[dim] for dim in self.tdm.data_list[de.FIL]}
        regf_df['tensor_i'] = {dim: regf_unit_tensor[dim] for dim in self.tdm.data_list[de.IFM]}
        regf_df['tensor_o'] = {dim: regf_unit_tensor[dim] for dim in self.tdm.data_list[de.OFM]}
        regf_df['stack'] = regf_stacks
        regf_df['update'] = regf_updates

        layer_df['GBUF'] = gbuf_df
        layer_df['REGF'] = regf_df

        return layer_df


def run_func():
    # hw_fp = "nn_dataflow/hardwares/multi_node.json"
    hw_fp = "nn_dataflow/hardwares/single_node.json"
    opt_fp = "nn_dataflow/options/option1.json"
    temp_path = "nn_dataflow/array_mapping_templates/"
    workload = "alex_net"
    # workload = "back_prop"
    # workload = "googlenet"
    # workload = "lstm_gnmt"
    # workload = "mlp_l"
    # workload = "resnet50"
    # workload = "vgg_net"
    # workload = "zfnet"
    network = import_network(workload)

    hw = parse_json(hw_fp)
    resource, unit_cost = parse_hardware(hw)

    opts = parse_json(opt_fp)
    options = parse_options(opts)

    batch_size = 64
    # array_mapping = ame.ROW_STATIONARY
    array_mapping = ame.SYSTOLIC

    solver = KaplaSolver(network, array_mapping, batch_size, resource, unit_cost, options)

    tbeg = time.time()
    df_tops = solver.solve_dataflow()
    tend = time.time()
    telapsed = tend - tbeg

    pp = pprint.PrettyPrinter(indent=2)
    if len(df_tops) == 0:
        pp.pprint("*** No valid schedule found for {}".format(network.net_name))
    for nndf, cost, total_time, total_cost in df_tops:
        for key, value in nndf.items():
            pp.pprint(key)
            pp.pprint(value)
        # pp.pprint("nndf: {}".format(nndf))
        pp.pprint("elapsed: {}".format(telapsed))
        pp.pprint("cost: {}".format(cost))
        pp.pprint("total_cost: {}".format(total_cost))
        pp.pprint("total_time: {}".format(total_time))

if __name__ == "__main__":
    run_func()