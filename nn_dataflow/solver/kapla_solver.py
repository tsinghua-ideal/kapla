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
from nn_dataflow.array_mapping_templates.tensor_dim_map import RSTensorDimMap, ident_layer_type, \
                                                               get_conv_strds, SystolicTensorDimMap
from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import ParallelEnum as pe
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.parser.kapla_cost_model import KaplaCostModel
from nn_dataflow.parser.kapla_parse_utils import parse_options, parse_hardware, parse_json, \
     shape_init_data_block, SegDfCache, SimpleCstr, BL, nn_rearrange, layer2workload, is_valid, \
     get_min_factor, part_workload



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

        self.cost_model = KaplaCostModel(self.tdm)

    def solve_dataflow(self):
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
            seg_counter = 0
            nndf_list = []
            for seg in self.priored_segments[layer_name]:
                print("- {}: {}".format(seg_counter, seg.seg))
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

                for constraint, _ in self.solve_constraint(seg):
                    print("-- constraint: {}".format(constraint))
                    cur_nndf = prev_nndf.copy()
                    seg_df, cost_dict, seg_time, cur_nndf, total_cost = self.solve_segment_df(seg, allocation, constraint, cur_nndf)
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

    def solve_segment_df(self, segment, allocation, constraint, cur_nndf):
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

                df, real_cstr, cost_dict, layer_time, sched_vars = self.solve_layer_df(layer_name, cur_cstr, resource, \
                    sp_idx, tm_idx, ifmap_layout)
                if df is None:
                    return dict(), None, None, None, None
                print("layer: {}".format(layer_name))
                print("df: {}".format(df))
                print("cost: {}".format(sum(cost_dict.values())))
                print("time: {}".format(layer_time))
                print("---")
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

    @functools.lru_cache(maxsize=1024)
    def solve_layer_df(self, layer_name, cstr, resource, sp_idx, tm_idx, ifmap_layout):
        min_cost = float("inf")
        min_cost_dict = dict()
        min_layer_time = float("inf")
        min_cstr = None
        layer_cand = None
        min_sched_var = None

        layer = self.network[layer_name]
        layer_type = ident_layer_type(layer)
        conv_strds = get_conv_strds(layer_type, layer)
        self.cost_model.set_cur_layer_type(layer_type)

        layer_data_size = [0 for _ in range(de.NUM)]
        if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
            layer_data_size[de.FIL] = layer.total_filter_size()
        layer_data_size[de.IFM] = layer.total_ifmap_size(batch_size=self.batch_size)
        layer_data_size[de.OFM] = layer.total_ofmap_size(batch_size=self.batch_size)
        layer_data_size = tuple(layer_data_size)

        loopcnt, origin_regf_repls, regf_unit_tensor, gbuf_unit_tensor, regf_base_stacks, \
            regf_base_updates, origin_stack_step_dict, unit_ops, mapping = \
            self.solve_array_mapping(layer_type, layer, conv_strds, \
            resource)

        origin_regf_unit_tensor = self.tdm.format_tensor_dim(layer_type, regf_unit_tensor, conv_strds)
        origin_gbuf_unit_tensor = self.tdm.format_tensor_dim(layer_type, gbuf_unit_tensor, conv_strds)

        # Iterate through all possible orders.
        for bl_ords in itertools.product(*[itertools.permutations(range(le.NUM))
                                         for _ in range(BL.NUM)]):
            is_valid, top_bl_ts, remain_lcnt = \
                self.cstr_check_prune(layer_type, cstr, loopcnt, bl_ords, resource)
            if not is_valid:
                continue

            regf_unit_tensor = copy.deepcopy(origin_regf_unit_tensor)
            gbuf_unit_tensor = copy.deepcopy(origin_gbuf_unit_tensor)
            regf_repls = [r for r in origin_regf_repls]


            # first fit as much as possible into regfile to reduce gbuf access.
            froz_remain_lcnt, froz_regf_unit_tensor, froz_regf_tensor_repl_dict = \
                self.fill_regf_tensor(layer_type,
                    conv_strds,
                    layer_data_size,
                    frozenset(remain_lcnt.items()),
                    frozenset(regf_unit_tensor.items()),
                    frozenset(gbuf_unit_tensor.items()),
                    tuple(bl_ords[BL.REGF]))
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
                resource.dim_array, tuple(regf_base_updates), BL.REGF)

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
                        self.fill_gbuf_tensor(layer_type, conv_strds, layer_data_size,
                            frozenset(remain_lcnt.items()),
                            frozenset(gbuf_unit_tensor.items()),
                            tuple(bl_ords[BL.GBUF]))
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
                    self.fill_gbuf_tensor(layer_type, conv_strds, layer_data_size,
                        frozenset(remain_lcnt.items()),
                        frozenset(gbuf_unit_tensor.items()),
                        tuple(bl_ords[BL.GBUF]), tuple(shr_node_num))
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

            src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
            dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)
            if layer_type == lte.CONV_BACK_W:
                assert(dst_is_dram, 'KaplaSolver: ConvBackWeightLayer'
                       'shoud be written to dram!')

            if not src_is_dram and bl_ords[BL.GBUF][le.BAT] < bl_ords[BL.GBUF][le.OFM] and \
                top_bl_ts[le.OFM] > 1:
                continue
            if not dst_is_dram and bl_ords[BL.GBUF][le.BAT] < bl_ords[BL.GBUF][le.IFM] and \
                top_bl_ts[le.IFM] > 1:
                continue

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

            part = self.derive_nndf_partition(layer_type, gbuf_stack_repl_dict, gbuf_stacks)

            # Filter nodes. All memory nodes can store filters. Deduplicate.
            filter_nodes = frozenset(resource.dram_region.iter_node())
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
            # Partition NoC hop cost.
            unit_nhops = partition.unit_nhops_to_proc_region(
                layer, self.batch_size, resource.proc_region, part,
                filter_nodes, ifmap_layout, ofmap_layout, self.options)

            print("")
            print("layer")
            print(layer)
            print("resource")
            print(resource)
            print("constraint")
            print(cstr)
            print("loopcnt")
            print(loopcnt)
            print("remain_lcnt")
            print(remain_lcnt)
            print("bl_ords")
            print(bl_ords)
            print("regf_workload")
            print(regf_workload)
            print("gbuf_workload")
            print(gbuf_workload)
            print("regf_tensor_repl_dict:")
            print(regf_tensor_repl_dict)
            print("regf_stack_repl_dict:")
            print(regf_stack_repl_dict)
            print("gbuf_tensor_repl_dict:")
            print(gbuf_tensor_repl_dict)
            print("gbuf_stack_repl_dict:")
            print(gbuf_stack_repl_dict)
            print("")
            print("gbuf_unit_tensor")
            print(gbuf_unit_tensor)
            print("gbuf_stacks")
            print(gbuf_stacks)
            print("gbuf_updates")
            print(gbuf_updates)
            print("gbuf_stack_num")
            print(gbuf_stack_num)
            print("")
            print("regf_unit_tensor")
            print(regf_unit_tensor)
            print("regf_stacks")
            print(regf_stacks)
            print("regf_updates")
            print(regf_updates)
            print("regf_stack_num")
            print(regf_stack_num)
            print("")
            print("regf_ops_iter")
            print(regf_ops_iter)
            print("unit_ops")
            print(unit_ops)

            # Get layer dataflow cost.
            accesses_result, noc_hops, cost_dict, total_cost, layer_time = \
                self.get_ldf_cost(layer_type, frozenset(regf_unit_tensor.items()), regf_updates,
                            regf_stacks, regf_workload, len(regf_base_updates),
                            frozenset(gbuf_unit_tensor.items()), gbuf_updates, gbuf_stacks,
                            gbuf_workload, regf_ops_iter, regf_stack_num, gbuf_stack_num,
                            unit_ops, tuple(unit_nhops), conv_strds, resource)

            print("access_result:")
            print(accesses_result)
            print("noc_hops:")
            print(noc_hops)
            print("cost_dict:")
            print(cost_dict)
            print("layer_cost:")
            print(total_cost)

            if total_cost < min_cost:
                min_cost = total_cost
                min_layer_time = layer_time
                min_cost_dict = cost_dict
                min_cstr = real_cstr
                layer_cand = self.finalize_layer_df(regf_unit_tensor, regf_updates, regf_stacks,
                                            gbuf_unit_tensor, gbuf_updates, gbuf_stacks)
                min_sched_var = (layer_type, layer, gbuf_stacks, mapping,
                    conv_strds,
                    regf_stack_repl_dict, gbuf_stack_repl_dict, regf_tensor_repl_dict,
                    gbuf_tensor_repl_dict, gbuf_iter_dict, bl_ords, unit_ops)


        return layer_cand, min_cstr, min_cost_dict, min_layer_time, min_sched_var

    def cstr_check_prune(self, layer_type, constraint, loopcnt, bl_ords, resource):
        is_valid = True
        remain_lcnt = dict()

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

        # Pipeline constraint check.
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

        # If data regions are not DRAM, can only access once, no spilling.
        src_is_dram = (resource.src_data_region.type == NodeRegion.DRAM)
        dst_is_dram = (resource.dst_data_region.type == NodeRegion.DRAM)
        if not src_is_dram and bl_ords[BL.GBUF][le.IFM] < bl_ords[BL.GBUF][le.OFM]:
            if top_bl_ts[le.OFM] > 1:
                is_valid = False
            else:
                top_bl_ts[le.OFM] = 1
        if not dst_is_dram and bl_ords[BL.GBUF][le.OFM] < bl_ords[BL.GBUF][le.IFM]:
            if top_bl_ts[le.IFM] > 1:
                is_valid = False
            else:
                top_bl_ts[le.IFM] = 1

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
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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
                 (layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H) and dim == "Yi")):
                gbuf_unit_tensor["Yo"] *= r
                gbuf_unit_tensor["Yi"] *= r
            elif self.array_mapping == ame.ROW_STATIONARY and \
                 ((layer_type in (lte.CONV, lte.LOCAL) and dim == "Xo") or \
                 (layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H) and dim == "Xi")):
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
                     ((layer_type in (lte.CONV, lte.LOCAL) and dim == "Yo") or \
                     (layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H) and dim == "Yi")):
                    gbuf_unit_tensor["Yo"] *= r
                    gbuf_unit_tensor["Yi"] *= r
                elif self.array_mapping == ame.ROW_STATIONARY and \
                     ((layer_type in (lte.CONV, lte.LOCAL) and dim == "Xo") or \
                     (layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H) and dim == "Xi")):
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
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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
               origin_stack_step_dict, unit_ops, mapping

    @functools.lru_cache(maxsize=1024)
    def fill_gbuf_tensor(self, layer_type, conv_strds, layer_data_size, froz_lcnt, froz_tensor,
                    bl_ord, shr_node_num=None):
        remain_lcnt = dict(froz_lcnt)
        unit_tensor = dict(froz_tensor)
        buf_size = self.resource.size_gbuf
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

    @functools.lru_cache(maxsize=1024)
    def fill_regf_tensor(self, layer_type, conv_strds, layer_data_size, froz_lcnt, froz_rts, froz_gts,
                    bl_ord, shr_node_num=None):
        remain_lcnt = dict(froz_lcnt)
        regf_tensor = dict(froz_rts)
        gbuf_tensor = dict(froz_gts)
        regf_buf = self.resource.size_regf
        gbuf_buf = self.resource.size_gbuf
        # print("")
        # print("---begin to fill tensor")
        # print("remain_lcnt: {}".format(remain_lcnt))
        # print("regf_tensor: {}".format(regf_tensor))
        # print("gbuf_tensor: {}".format(gbuf_tensor))
        # print("regf_buf: {}".format(regf_buf))
        # print("gbuf_buf: {}".format(gbuf_buf))
        # print("shr_node_num: {}".format(shr_node_num))
        if not is_valid(self.tdm, layer_type, regf_tensor, regf_buf, shr_node_num) or \
           not is_valid(self.tdm, layer_type, gbuf_tensor, gbuf_buf, (1, 1, 1)):
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
                    regf_tensor = self.tensor_mul(layer_type, conv_strds, regf_tensor, dim, f)
                    gbuf_tensor = self.tensor_mul(layer_type, conv_strds, gbuf_tensor, dim, f)
                    if not is_valid(self.tdm, layer_type, regf_tensor, regf_buf, shr_node_num) or \
                       not is_valid(self.tdm, layer_type, gbuf_tensor, gbuf_buf, (1, 1, 1)):
                        # print("Not valid")
                        # print(unit_tensor, buf_size, shr_node_num)
                        max_valid_idx = idx
                        regf_tensor = self.tensor_div(layer_type, conv_strds, regf_tensor, dim, f)
                        gbuf_tensor = self.tensor_div(layer_type, conv_strds, gbuf_tensor, dim, f)
                        break
                    regf_tensor = self.tensor_div(layer_type, conv_strds, regf_tensor, dim, f)
                    gbuf_tensor = self.tensor_div(layer_type, conv_strds, gbuf_tensor, dim, f)
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
                regf_tensor = self.tensor_mul(layer_type, conv_strds, regf_tensor, dim, f)
                gbuf_tensor = self.tensor_mul(layer_type, conv_strds, gbuf_tensor, dim, f)
                temp_outer_fetch_size[o_idx] = util.idivc(temp_outer_fetch_size[o_idx], f)
            if is_valid(self.tdm, layer_type, regf_tensor, regf_buf, shr_node_num) and \
               is_valid(self.tdm, layer_type, gbuf_tensor, gbuf_buf, (1, 1, 1)):
                sum_refetch = sum(temp_outer_fetch_size)
                # print("sum_refetch: {}, min_outer_refetch: {}".format(sum_refetch, min_outer_refetch))
                if sum_refetch < min_outer_refetch:
                    best_factors = factors
                    min_outer_refetch = sum_refetch
            for dim, f in zip(outer_irr_dims, factors):
                regf_tensor = self.tensor_div(layer_type, conv_strds, regf_tensor, dim, f)
                gbuf_tensor = self.tensor_div(layer_type, conv_strds, gbuf_tensor, dim, f)

        for dim, f in zip(outer_irr_dims, best_factors):
            regf_tensor = self.tensor_mul(layer_type, conv_strds, regf_tensor, dim, f)
            gbuf_tensor = self.tensor_mul(layer_type, conv_strds, gbuf_tensor, dim, f)
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

            regf_tensor = self.tensor_mul(layer_type, conv_strds, regf_tensor, min_factor_dim, min_factor)
            gbuf_tensor = self.tensor_mul(layer_type, conv_strds, gbuf_tensor, min_factor_dim, min_factor)
            if is_valid(self.tdm, layer_type, regf_tensor, regf_buf, shr_node_num) and \
               is_valid(self.tdm, layer_type, gbuf_tensor, gbuf_buf, (1, 1, 1)):
                remain_lcnt[min_factor_dim] = util.idivc(remain_lcnt[min_factor_dim], min_factor)
                tensor_repl_dict[min_factor_dim] *= min_factor
            else:
                regf_tensor = self.tensor_div(layer_type, conv_strds, regf_tensor, min_factor_dim, min_factor)
                gbuf_tensor = self.tensor_div(layer_type, conv_strds, gbuf_tensor, min_factor_dim, min_factor)
                break
        # print("Final regf_tensor: ", regf_tensor)
        # print("final gbuf tensor: ", gbuf_tensor)
        # print("remain lcnt:", remain_lcnt)
        return frozenset(remain_lcnt.items()), frozenset(regf_tensor.items()), frozenset(tensor_repl_dict.items())

    def tensor_div(self, layer_type, conv_strds, tensor, dim, factor):
        if (layer_type == lte.LOCAL and dim == "K") or \
           (layer_type == lte.LOCAL_BACK_H and dim == "C"):
            tensor["C"] /= factor
            tensor["K"] /= factor
        else:
            tensor[dim] /= factor

        if self.array_mapping == ame.ROW_STATIONARY:
            # Recalculate ifm/ofm.
            if layer_type in (lte.CONV, lte.LOCAL):
                tensor["Xi"] = (tensor["Xo"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yi"] = (tensor["Yo"] - 1) * conv_strds[1] + tensor["S"]
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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
            if layer_type in (lte.CONV, lte.LOCAL):
                tensor["Xi"] = (tensor["Xo"] - 1) * conv_strds[0] + tensor["R"]
                tensor["Yi"] = (tensor["Yo"] - 1) * conv_strds[1] + tensor["S"]
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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
        elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
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
                elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
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

    @functools.lru_cache(maxsize=1024)
    def get_ldf_cost(self, layer_type, regf_froz_tensor, regf_updates, regf_stacks, regf_workload,
                     regf_base_upd_num, gbuf_froz_tensor, gbuf_updates, gbuf_stacks, gbuf_workload,
                     regf_ops_iter, regf_stack_num, gbuf_stack_num, unit_ops, unit_nhops, conv_strds, resource):
        regf_unit_tensor = dict(regf_froz_tensor)
        gbuf_unit_tensor = dict(gbuf_froz_tensor)

        accesses_result = [[0 for _ in range(de.NUM)] for _ in range(me.NUM)]
        g_init_datas, g_froz_init_datas = shape_init_data_block(self.tdm, gbuf_unit_tensor)
        g_upd_dims, g_iter_times = self.cost_model.analyze_dim_iters(gbuf_unit_tensor, gbuf_updates, gbuf_workload)
        g_logical_dim, g_buf_sharings, _, _ = self.cost_model.analyze_stacks(g_froz_init_datas, gbuf_stacks,
                                                            resource.proc_region.dim, None, BL.GBUF)
        gbuf_unit_accesses = self.cost_model.analyze_relevant_accesses(g_init_datas, g_upd_dims,
                                                    g_iter_times, self.options)

        r_init_datas, r_froz_init_datas = shape_init_data_block(self.tdm, regf_unit_tensor)
        r_upd_dims, r_iter_times = self.cost_model.analyze_dim_iters(regf_unit_tensor, regf_updates, regf_workload)
        r_logical_dim, _, r_remote_sharings, r_temporal_sharings = \
                self.cost_model.analyze_stacks(r_froz_init_datas, regf_stacks, resource.dim_array,
                                               tuple(regf_updates), BL.REGF)
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

            if layer_type == lte.CONV_BACK_W:
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

        node_hops = [fwd_hop + bufshr_hop for fwd_hop, bufshr_hop in zip(fwd_hops, buf_shr_hops)]
        mem_hops = [unh * f for unh, f in zip(unit_nhops, g_rd_iters)]
        noc_hops = [nnh + mnh for nnh, mnh in zip(node_hops, mem_hops)]

        print("g_init_datas:")
        print(g_init_datas)
        print("g_upd_dims:")
        print(g_upd_dims)
        print("g_iter_times:")
        print(g_iter_times)
        print("g_logical_dim:")
        print(g_logical_dim)
        print("g_buf_sharings:")
        print(g_buf_sharings)
        print("gbuf_unit_accesses:")
        print(gbuf_unit_accesses)
        print("g_rd_iters:")
        print(g_rd_iters)
        print("")
        print("r_init_datas:")
        print(r_init_datas)
        print("r_upd_dims:")
        print(r_upd_dims)
        print("r_iter_times:")
        print(r_iter_times)
        print("r_logical_dim:")
        print(r_logical_dim)
        print("r_remote_sharings:")
        print(r_remote_sharings)
        print("regf_unit_accesses:")
        print(regf_unit_accesses)
        print("regf_upper_iters:")
        print(regf_upper_iters)
        print("r_rd_iters:")
        print(r_rd_iters)

        print("bufshr_rdt_iters:")
        print(bufshr_rdt_iters)
        print("src_is_dram: {}".format(src_is_dram))
        print("dst_is_dram: {}".format(dst_is_dram))

        # calculate the cost
        cost_dict = dict()
        cost_dict["dram_cost"] = sum(accesses_result[me.DRAM]) * self.unit_cost.mem_hier_at(me.DRAM)
        cost_dict["sram_cost"] = sum(accesses_result[me.GBUF]) * self.unit_cost.mem_hier_at(me.GBUF)
        cost_dict["itcn_cost"] = sum(accesses_result[me.ITCN]) * self.unit_cost.mem_hier_at(me.ITCN)
        cost_dict["regf_cost"] = sum(accesses_result[me.REGF]) * self.unit_cost.mem_hier_at(me.REGF)

        cost_dict["remote_sram_cost"] = self.unit_cost.mem_hier_at(me.GBUF) * sum(remote_gbuf_access)
        cost_dict["node_hop_cost"] = sum(node_hops) * self.unit_cost.noc_hop
        cost_dict["mem_hop_cost"] = sum(mem_hops) * self.unit_cost.noc_hop
        cost_dict["op_cost"] = nndf_ops * self.unit_cost.mac_op

        # calculate the time
        dram_time = int(math.ceil(sum(accesses_result[me.DRAM]) / resource.dram_bandwidth))
        bus_time = util.idivc(int(math.ceil(1. * max(accesses_result[me.GBUF])
                                / gbuf_stack_num)), resource.array_bus_width)
        layer_time = (proc_time, dram_time, bus_time)
        # cost_dict['static_cost'] = self.unit_cost.idl_unit * max(layer_time)
        total_cost = sum(cost_dict.values())

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

    def derive_sched_result(self, layer_name, seg_idx, sp_idx, tm_idx, resource, ifmap_layout, sched_vars):
        (layer_type, layer, gbuf_stacks, mapping,
            conv_strds,
            regf_stack_repl_dict, gbuf_stack_repl_dict, regf_tensor_repl_dict,
            gbuf_tensor_repl_dict, gbuf_iter_dict, bl_ords, unit_ops) = sched_vars

        proc_region = resource.proc_region
        part = self.derive_nndf_partition(layer_type, gbuf_stack_repl_dict, gbuf_stacks)

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

        bufshr = BufShrScheme(resource.proc_region, part, layer.data_loops())

        lbs = self.derive_nndf_lbs(layer_type, layer, mapping, gbuf_iter_dict, gbuf_tensor_repl_dict,
            gbuf_stack_repl_dict, regf_tensor_repl_dict, regf_stack_repl_dict, conv_strds, bl_ords, unit_ops, resource,
            bufshr)

        sched_seq = (seg_idx, sp_idx, tm_idx)

        sched_result = self._construct_sched_result(lbs, part, ofmap_layout, sched_seq, unit_nhops)

        return sched_result

    def derive_nndf_partition(self, layer_type, gbuf_stack_repl_dict, gbuf_stacks):
        porders = [None for _ in range(nndf_pe.NUM)]

        ord_counter = 0
        for stc in gbuf_stacks:
            for i in range(0, len(stc[:-1]), 2):
                dim = stc[i]
                penum = self.tdm.get_dim_rlvt_part_type(layer_type, dim)
                if penum is not None and porders[penum] is None:
                    porders[penum] = ord_counter
                    ord_counter += 1
                    break

        for idx in range(nndf_pe.NUM):
            if porders[idx] is None:
                porders[idx] = ord_counter
                ord_counter += 1

        porders = tuple(porders)

        pdims = [[1, 1] for _ in range(nndf_pe.NUM)]
        for drc, gbuf_repl in enumerate(gbuf_stack_repl_dict):
            for dim, r in gbuf_repl.items():
                penum = self.tdm.get_dim_rlvt_part_type(layer_type, dim)
                pdims[penum][drc] *= r

        for idx in range(nndf_pe.NUM):
            pdims[idx] = PhyDim2(pdims[idx][1], pdims[idx][0])

        pdims = tuple(pdims)
        return PartitionScheme(porders, pdims)

    def derive_nndf_lbs(self, layer_type, layer, mapping, gbuf_iter_dict, gbuf_tensor_repl_dict,
                        gbuf_stack_repl_dict, regf_tensor_repl_dict, regf_stack_repl_dict, conv_strds,
                        bl_ords, unit_ops, resource, bufshr):
        layer_workload = layer2workload(self.array_mapping, layer, self.batch_size)
        part_workload = defaultdict(lambda: 1)
        gbuf_unit_tensor = defaultdict(lambda: 1)

        for key, value in layer_workload.items():
            part_workload[key] = value
        for stack_dict in gbuf_stack_repl_dict:
            for key, value in stack_dict.items():
                part_workload[key] = util.idivc(part_workload[key], value)

        for key, value in part_workload.items():
            gbuf_unit_tensor[key] = part_workload[key]

        for key, value in gbuf_iter_dict.items():
            gbuf_unit_tensor[key] = util.idivc(part_workload[key], value)
        for key, value in gbuf_tensor_repl_dict.items():
            gbuf_unit_tensor[key] = util.idivc(gbuf_unit_tensor[key], value)
        for key, value in regf_tensor_repl_dict.items():
            gbuf_unit_tensor[key] = util.idivc(gbuf_unit_tensor[key], value)

        gbuf_unit_tensor = self.tdm.format_tensor_dim(layer_type, gbuf_unit_tensor, conv_strds)

        regf_stacks = defaultdict(lambda: 1)
        for s_dict in regf_stack_repl_dict:
            for dim, r in s_dict.items():
                regf_stacks[dim] *= r
        gbuf_stacks = defaultdict(lambda: 1)
        for s_dict in gbuf_stack_repl_dict:
            for dim, r in s_dict.items():
                gbuf_stacks[dim] *= r

        bl_ts = [[1 for _ in range(le.NUM)] for _ in range(BL.NUM+1)]
        loopcnt = [1 for _ in range(le.NUM)]
        for key, r in gbuf_iter_dict.items():
            l = self.tdm.get_dim_crrspd_ltype(layer_type, key)
            if l is None:
                continue
            bl_ts[BL.GBUF][l] *= r
            loopcnt[l] *= r
        bl_ts[BL.GBUF] = tuple(bl_ts[BL.GBUF])
        for key, r in gbuf_tensor_repl_dict.items():
            l = self.tdm.get_dim_crrspd_ltype(layer_type, key)
            if l is None:
                continue
            bl_ts[BL.REGF][l] *= r
            loopcnt[l] *= r
        bl_ts[BL.REGF] = tuple(bl_ts[BL.REGF])
        for key, r in regf_tensor_repl_dict.items():
            l = self.tdm.get_dim_crrspd_ltype(layer_type, key)
            if l is None:
                continue
            bl_ts[BL.REGF+1][l] *= r
            loopcnt[l] *= r
        bl_ts[BL.REGF+1] = tuple(bl_ts[BL.REGF+1])

        bl_ts = tuple(bl_ts)
        loopcnt = tuple(loopcnt)

        if self.array_mapping == ame.ROW_STATIONARY:
            fold_h, fold_w = mapping.fold.h, mapping.fold.w
            # To be compatible with nn_dataflow, we need to fold Yo dim into N dim.
            # This fold is using the whole Yo, so we regrad this Yo to be transformed into
            # temporal iterated N.
            if layer_type in (lte.CONV, lte.LOCAL):
                part_workload["Yo"] = util.idivc(part_workload["Yo"], gbuf_iter_dict.get("Yo", 1))
                fold_w = util.idivc(fold_w, gbuf_iter_dict.get("Yo", 1) * gbuf_stacks.get("Yo", 1))
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
                part_workload["Yi"] = util.idivc(part_workload["Yi"], gbuf_iter_dict.get("Yi", 1))
                fold_w = util.idivc(fold_w, gbuf_iter_dict.get("Yi", 1) * gbuf_stacks.get("Yi", 1))
            part_workload = self.tdm.format_tensor_dim(layer_type, part_workload, conv_strds)

            if layer_type == lte.CONV:
                acclayer = ConvLayer(
                    1, 1,
                    (util.idivc(part_workload["Yo"], fold_w), part_workload["Xo"]),
                    (part_workload["S"], part_workload["R"]),
                    strd=(conv_strds[1], conv_strds[0]))
                print("acclayer.hfim: {}, fold_w: {}, part_workload {}".format(acclayer.hifm, fold_w, part_workload["Yi"]))
                amp_acc_ifm = 1. * acclayer.hifm * fold_w / part_workload["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_workload["S"], fold_h),
                    w=util.idivc(part_workload["Yo"], fold_w))

            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(
                    1,
                    (util.idivc(part_workload["Yo"], fold_w), part_workload["Xo"]),
                    conv_strds[2], (part_workload["S"], part_workload["R"]),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * fold_w / part_workload["Yi"]
                dim_flpeset = PhyDim2(h=util.idivc(part_workload["S"], fold_h),
                    w=util.idivc(part_workload["Yo"], fold_w))

            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                acclayer = ConvLayer(
                    1, 1,
                    (util.idivc(part_workload["Yi"], fold_w), part_workload["Xi"]),
                    (part_workload["S"], part_workload["R"]),
                    strd=(conv_strds[1], conv_strds[0]), rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * fold_w / part_workload["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_workload["S"], fold_h),
                    w=util.idivc(part_workload["Yi"], fold_w))

            elif layer_type == lte.LOCAL_BACK_H:
                acclayer = LocalRegionLayer(
                    1,
                    (util.idivc(part_workload["Yi"], fold_w), part_workload["Xi"]),
                    conv_strds[2], (part_workload["S"], part_workload["R"]),
                    strd=(conv_strds[1], conv_strds[0]), rw_data=layer.rw_data)
                amp_acc_ifm = 1. * acclayer.hifm * fold_w / part_workload["Yo"]
                dim_flpeset = PhyDim2(h=util.idivc(part_workload["S"], fold_h),
                    w=util.idivc(part_workload["Yi"], fold_w))

            flpesets_per_unitpass = fold_h
            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * regf_stacks["K"] * regf_stacks["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * regf_stacks["C"] * regf_stacks["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["C"] * regf_stacks["N"] / amp_acc_ifm
            elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * regf_stacks["C"] * regf_stacks["N"]
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * regf_stacks["C"] * regf_stacks["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["K"] * regf_stacks["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * regf_stacks["K"] * regf_stacks["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["K"] * regf_stacks["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL_BACK_H:
                unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * regf_stacks["C"] * regf_stacks["N"]
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["C"] * regf_stacks["N"] / amp_acc_ifm

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM] * flpesets_per_unitpass
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM] * flpesets_per_unitpass

            unit_access[me.ITCN][de.IFM] = acclayer.wifm * dim_flpeset.size() \
                    * flpesets_per_unitpass * util.prod(regf_stacks.values())
            unit_access[me.ITCN][de.OFM] = acclayer.wofm * dim_flpeset.size() \
                    * flpesets_per_unitpass * util.prod(regf_stacks.values())

            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_access[me.ITCN][de.FIL] = acclayer.wfil * dim_flpeset.size() \
                        * flpesets_per_unitpass * util.prod(regf_stacks.values())
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_access[me.ITCN][de.FIL] = 0

            unit_access[me.REGF] = [acclayer.total_ops() * util.prod(regf_stacks.values())] * de.NUM


            sz_gbuf = self.tdm.get_tensor_size(layer_type, gbuf_unit_tensor)
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
                sz_regf[de.IFM] = acclayer.wreg
                sz_regf[de.OFM] = 1

            ops_lpe = acclayer.total_ops() * util.prod(regf_stacks.values())
            loopcnt = (loopcnt[le.IFM], loopcnt[le.OFM], loopcnt[le.BAT])
            if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
                unit_time = acclayer.wfil * acclayer.hfil
                regf_reusable[de.IFM] = (acclayer.wfil == acclayer.wifm)
            elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
                unit_time = acclayer.nreg * acclayer.wreg * acclayer.hreg
                regf_reusable[de.IFM] = (acclayer.wreg == acclayer.wifm)
            regf_reusable[de.OFM] = (acclayer.wofm == 1)
            regf_reusable[de.FIL] = (mapping.fold.h == 1)

            if layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
                sz_regf[de.IFM], sz_regf[de.OFM] = sz_regf[de.OFM], sz_regf[de.IFM]
                regf_reusable[de.IFM], regf_reusable[de.OFM] = regf_reusable[de.OFM], regf_reusable[de.IFM]
                for i in range(me.NUM):
                    unit_access[i][de.IFM], unit_access[i][de.OFM] = unit_access[i][de.OFM], unit_access[i][de.IFM]

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

            real_resource = resource._replace(size_gbuf=resource.size_gbuf/0.99,
                                              size_regf=resource.size_regf/0.99)

            lbs = LoopBlockingScheme(nld, bl_ts, bl_ords, real_resource, bufshr, self.options)
            if not lbs.is_valid():
                print("LBS INVALID!")
                print("gbuf_stack_repl_dict", gbuf_stack_repl_dict)
                print("part_workload", part_workload)
                print("gbuf_unit_tensor", gbuf_unit_tensor)
                print("amp_acc_ifm", amp_acc_ifm)
                print("gbuf_iter_dict", gbuf_iter_dict)
                print("gbuf_tensor_repl_dict", gbuf_tensor_repl_dict)
                print("regf_tensor_repl_dict", regf_tensor_repl_dict)
                print("regf_stack_repl_dict", regf_stack_repl_dict)
                print("sz_gbuf", sz_gbuf)
                print("sz_regf", sz_regf)
                print("bl_ts", bl_ts)
                print("bl_ords", bl_ords)
                print("bufshr", tuple(bufshr.size(dce) for dce in range(de.NUM)))

        elif self.array_mapping == ame.SYSTOLIC:
            part_layer, p_batch_size, p_occ = bufshr.part.part_layer(layer, self.batch_size)
            fold_h, fold_w = mapping.fold.h, mapping.fold.w
            fold_hofm = util.closest_factor(mapping.fold.h, factor=mapping.fold.h/2)[0]
            fold_wofm = mapping.fold.h / fold_hofm
            if layer_type == lte.CONV:
                acclayer = ConvLayer(
                    1, 1. * layer.nofm / fold_w,
                    (1. * part_layer.hofm / fold_hofm, 1. * part_layer.wofm / fold_wofm),
                    (part_layer.hfil, part_layer.wfil),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / \
                    part_layer.hifm / part_layer.wifm
                dim_flpeset = PhyDim2(h=mapping.logic_region.h, w=mapping.logic_region.w)
                unit_time = layer.wfil * layer.hfil
            elif layer_type == lte.LOCAL:
                acclayer = LocalRegionLayer(
                    1. * layer.nofm / fold_w,
                    (1. * part_layer.hofm / fold_hofm, 1. * part_layer.wofm / fold_wofm),
                    layer.nreg, (layer.hreg, layer.wreg),
                    strd=(conv_strds[1], conv_strds[0]))
                amp_acc_ifm = 1. * acclayer.hifm * acclayer.wifm * fold_h / \
                    part_layer.hifm / part_layer.wifm
                dim_flpeset = PhyDim2(h=mapping.logic_region.h, w=mapping.logic_region.w)
                unit_time = layer.nreg * layer.wreg * layer.hreg

            unit_access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
            regf_reusable = [False for _ in range(de.NUM)]

            unit_access[me.DRAM][de.OFM] = acclayer.total_ofmap_size() * regf_stacks["K"] * regf_stacks["N"]

            if layer_type == lte.CONV:
                unit_access[me.DRAM][de.FIL] = acclayer.total_filter_size() * regf_stacks["C"] * regf_stacks["K"]
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["C"] * regf_stacks["N"] / amp_acc_ifm
            elif layer_type == lte.LOCAL:
                unit_access[me.DRAM][de.FIL] = 0
                unit_access[me.DRAM][de.IFM] = acclayer.total_ifmap_size() * regf_stacks["K"] * regf_stacks["N"] / amp_acc_ifm

            unit_access[me.GBUF][de.FIL] = unit_access[me.DRAM][de.FIL]
            unit_access[me.GBUF][de.IFM] = unit_access[me.DRAM][de.IFM]
            unit_access[me.GBUF][de.OFM] = unit_access[me.DRAM][de.OFM]

            unit_access[me.ITCN][de.IFM] = acclayer.total_ifmap_size() * util.prod(regf_stacks.values())
            unit_access[me.ITCN][de.OFM] = acclayer.total_ofmap_size() * util.prod(regf_stacks.values())
            if layer_type == lte.CONV:
                unit_access[me.ITCN][de.FIL] = acclayer.total_filter_size() * util.prod(regf_stacks.values())
            elif layer_type == lte.LOCAL:
                unit_access[me.ITCN][de.FIL] = 0

            unit_access[me.REGF] = [acclayer.total_ops() * util.prod(regf_stacks.values())] * de.NUM

            sz_gbuf = [0] * de.NUM
            sz_gbuf[de.IFM] = acclayer.total_ifmap_size() * util.prod(regf_stacks.values())
            sz_gbuf[de.OFM] = acclayer.total_ofmap_size() * util.prod(regf_stacks.values())
            if layer_type == lte.CONV:
                sz_gbuf[de.FIL] = acclayer.total_ifmap_size() * util.prod(regf_stacks.values())
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

            ops_lpe = acclayer.total_ops() * util.prod(regf_stacks.values())
            loopcnt = (loopcnt[le.IFM], loopcnt[le.OFM], loopcnt[le.BAT])
            regf_reusable[de.IFM] = False
            regf_reusable[de.FIL] = False
            regf_reusable[de.OFM] = True

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

            real_resource = resource._replace(size_gbuf=resource.size_gbuf/0.99,
                                              size_regf=resource.size_regf/0.99)

            lbs = LoopBlockingScheme(nld, bl_ts, bl_ords, real_resource, bufshr, self.options)
            if not lbs.is_valid():
                print("LBS INVALID!")
                print("gbuf_stack_repl_dict", gbuf_stack_repl_dict)
                print("part_workload", part_workload)
                print("gbuf_unit_tensor", gbuf_unit_tensor)
                print("amp_acc_ifm", amp_acc_ifm)
                print("gbuf_iter_dict", gbuf_iter_dict)
                print("gbuf_tensor_repl_dict", gbuf_tensor_repl_dict)
                print("regf_tensor_repl_dict", regf_tensor_repl_dict)
                print("regf_stack_repl_dict", regf_stack_repl_dict)
                print("sz_gbuf", sz_gbuf)
                print("sz_regf", sz_regf)
                print("bl_ts", bl_ts)
                print("bl_ords", bl_ords)
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
            solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp, args.back_prop)
        else:
            hw_fp = "nn_dataflow/hardwares/multi_node.json"
            opt_fp = "nn_dataflow/options/option_inference.json"
            solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)
    elif array_mapping == ame.SYSTOLIC:
        hw_fp = "nn_dataflow/hardwares/single_node.json"
        opt_fp = "nn_dataflow/options/option_inference.json"
        solve_dataflow(network, batch_size, array_mapping, hw_fp, opt_fp)


if __name__ == "__main__":
    sys.exit(main())