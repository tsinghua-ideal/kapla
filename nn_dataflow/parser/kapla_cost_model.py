import functools
import itertools
from collections import defaultdict
from constraint import Problem

from nn_dataflow import util
import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
from nn_dataflow.core.layer import ConvLayer, LocalRegionLayer, ConvBackActLayer, \
    ConvBackWeightLayer, LocalRegionBackLayer

from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte
from nn_dataflow.array_mapping_templates.tensor_dim_map import SystolicTensorDimMap

class KaplaCostModel():
    def __init__(self, network, tensor_dim_map):
        self.network = network
        self.tdm = tensor_dim_map

    def set_cur_layer_type(self, layer_type):
        if layer_type not in range(lte.NUM):
            raise TypeError("Invalid layer type : {}".format(layer_type))
        self.layer_type = layer_type

        if self.layer_type == lte.CONV_BACK_W:
            self.rw_data = de.FIL
        else:
            self.rw_data = de.OFM

    def parse_update_drc(self, upd_drc):
        dims_strds = []
        dim = None
        strd = None
        for value in upd_drc + ("end",):
            if isinstance(value, str):
                if dim is not None:
                    dims_strds.append((dim, strd))
                    dim = value
                    strd = None
                else:
                    dim = value
            else:
                strd = value
        return dims_strds

    def analyze_dim_iters(self, tensor_dims, updates, workload):
        upd_dims = []
        iter_times = []
        for upd_drc in updates:
            # Find all update data blocks.
            dim_lists = []
            # Set the iter times at current level.
            iter_time = float("inf")
            dims_strds = self.parse_update_drc(upd_drc)
            for dim, strd in dims_strds:
                if workload[dim] < tensor_dims[dim]:
                    iter_time = 1
                else:
                    iter_time = min(1 + util.idivc((workload[dim] - max(tensor_dims[dim], strd)), strd), iter_time)
                dim_lists.append((dim, min(strd, tensor_dims[dim])))
            iter_times.append(iter_time)
            upd_dims.append(tuple(dim_lists))
        return tuple(upd_dims), tuple(iter_times)

    def analyze_relevant_accesses(self, init_datas, upd_dims, iter_times, options):
        unit_accesses = [() for _ in range(de.NUM)]
        for dce in range(de.NUM):
            if self.layer_type in (lte.LOCAL, lte.LOCAL_BACK_H) and dce == de.FIL:
                continue
            rlvt_dims = self.tdm.data_list[dce]
            init_db = init_datas[dce]
            db_dims_dict = defaultdict(lambda: dict())

            irrlvt_iter = 1
            for upd_dim, iter_time in zip(upd_dims, iter_times):
                for dim, dim_size in upd_dim:
                    if dim not in rlvt_dims:
                        continue
                    if len(db_dims_dict[dim]) != 0:
                        if dim_size in db_dims_dict[dim]:
                            for sz in db_dims_dict[dim].keys():
                                db_dims_dict[dim][sz] *= iter_time
                        else:
                            raise ValueError("Invalid update step {} for dim {}, tensor db size: {}".format(dim_size, dim, db_dims_dict[dim].keys()))
                    else:
                        if init_db[dim] != dim_size:
                            db_dims_dict[dim][init_db[dim]] = 1
                            db_dims_dict[dim][dim_size] = iter_time - 1
                        else:
                            db_dims_dict[dim][dim_size] = iter_time

            db_evo_list = [(1, 1),]

            for dim in rlvt_dims:
                new_evo_list = []
                if dim not in db_dims_dict:
                    for mult_size, mult_num in db_evo_list:
                        new_evo_list.append((mult_size * init_db[dim], mult_num))
                else:
                    for mult_size, mult_num in db_evo_list:
                        for dim_size, iter_num in db_dims_dict[dim].items():
                            new_evo_list.append((mult_size * dim_size, mult_num * iter_num))
                db_evo_list = new_evo_list

            unit_accesses[dce] = tuple(db_size * iter_time for db_size, iter_time in db_evo_list)

        return unit_accesses

    @functools.lru_cache(maxsize=1024)
    def redundant_iter(self, upd_dims, iter_times, bl_ord):
        redundant_iters = [1 for _ in range(de.NUM)]
        start_flag = [False for _ in range(de.NUM)]
        ordered_len = len(iter_times) - len(bl_ord)
        for index in range(ordered_len):
            rlvs = [False for _ in range(de.NUM)]
            if iter_times[index] == 1:
                continue
            for dim, _ in upd_dims[index]:
                for dce in range(de.NUM):
                    if dim in self.tdm.data_list[dce]:
                        start_flag[dce] = True
                        rlvs[dce] = True
            for dce in range(de.NUM):
                if start_flag[dce] and not rlvs[dce]:
                    redundant_iters[dce] *= iter_times[index]

        for order in range(len(bl_ord)):
            index = ordered_len + bl_ord.index(order)
            rlvs = [False for _ in range(de.NUM)]
            if iter_times[index] == 1:
                continue
            for dim, _ in upd_dims[index]:
                for dce in range(de.NUM):
                    if dim in self.tdm.data_list[dce]:
                        start_flag[dce] = True
                        rlvs[dce] = True
            for dce in range(de.NUM):
                if start_flag[dce] and not rlvs[dce]:
                    redundant_iters[dce] *= iter_times[index]

        return tuple(redundant_iters)

    def upper_fetch(self, upd_dims, iter_times, upper_upd_dims, upper_iter_times):
        upper_fetches = [util.prod(upper_iter_times) for _ in range(de.NUM)]
        is_trivial = [True for _ in range(de.NUM)]

        for dce in range(de.NUM):
            dce_dims = self.tdm.data_list[dce]
            for upds, itert in zip(upd_dims, iter_times):
                if itert > 1 and any(d in dce_dims for d, _ in upds):
                    is_trivial[dce] = False
                    break

            for updims, up_iter_time in zip(upper_upd_dims, upper_iter_times):
                if up_iter_time > 1 and any(updd in dce_dims for updd, _ in updims):
                    is_trivial[dce] = False
                    break

            if is_trivial[dce]:
                upper_fetches[dce] = 1

        return upper_fetches

    @functools.lru_cache(maxsize=1024)
    def bufshr_redundant_iter(self, gbuf_upd_dims, gbuf_iter_times, regf_upd_dims, regf_iter_times, regf_bl_ord, g_redundant_iters, opt_out_bufshr):
        bufshr_rdt_iters = [1 for _ in range(de.NUM)]

        for g_upd_dims, iter_time in zip(gbuf_upd_dims, gbuf_iter_times):
            related_flag = [False for _ in range(de.NUM)]
            for dim, _ in g_upd_dims:
                for dce in range(de.NUM):
                    if dim in self.tdm.data_list[dce]:
                        related_flag[dce] = True
            for dce in range(de.NUM):
                if not related_flag[dce]:
                    bufshr_rdt_iters[dce] *= iter_time

        outter_encounted_flag = [False for _ in range(de.NUM)]
        ordered_len = len(regf_iter_times) - len(regf_bl_ord)
        for order in reversed(range(len(regf_bl_ord))):
            if all(outter_encounted_flag):
                break
            index = ordered_len + regf_bl_ord.index(order)
            for dim, _ in regf_upd_dims[index]:
                for dce in range(de.NUM):
                    # if (dim in DATA_LIST[dce]) and (regf_iter_times[index] > 1):
                        # outter_encounted_flag[dce] = True
                    if dim in self.tdm.data_list[dce]:
                        outter_encounted_flag[dce] = True
            for dce in range(de.NUM):
                if not outter_encounted_flag[dce]:
                    bufshr_rdt_iters[dce] *= regf_iter_times[index]

        # Judge if one data related dims are trivial.
        # If all the dims are trivial, then the rotrnd is 0.
        is_trivial = [True for _ in range(de.NUM)]
        for order in range(len(regf_bl_ord)):
            index = ordered_len + regf_bl_ord.index(order)
            trvl = (regf_iter_times[index] == 1)
            if trvl:
                continue
            for dim, _ in regf_upd_dims[index]:
                for dce in range(de.NUM):
                    if dim in self.tdm.data_list[dce]:
                        is_trivial[dce] = False
        for dce in range(de.NUM):
            if (bufshr_rdt_iters[dce] == ((g_redundant_iters[dce] + 1) // 2 if dce == self.rw_data
                                        else g_redundant_iters[dce])) or \
                opt_out_bufshr or \
                is_trivial[dce]:
                bufshr_rdt_iters[dce] = 0
        return tuple(bufshr_rdt_iters)

    def analyze_gbuf_level_access(self, unit_accesses, redundant_iters, logical_dim, buf_sharings,
                                bufshr_rdt_iters, options):
        dram_accesses = [0 for _ in range(de.NUM)]
        fwd_hops = [0 for _ in range(de.NUM)]
        buf_shr_hops = [0 for _ in range(de.NUM)]
        logical_node_num = logical_dim[0] * logical_dim[1]

        for dce in range(de.NUM):
            bufshr = None

            for bs in buf_sharings:
                if bs[0] == dce:
                    bufshr = bs
                    break

            if dce == self.rw_data:
                rdt_iter = 2 * redundant_iters[dce] - 1
            else:
                rdt_iter = redundant_iters[dce]
            if bufshr is None or not (options.hw_access_forwarding or options.hw_gbuf_sharing):
                dram_accesses[dce] = sum(unit_accesses[dce]) * rdt_iter * logical_node_num
            else:
                _, shr_node_num, avg_fwd_dist, avg_rot_dist = bufshr
                dram_accesses[dce] = sum(unit_accesses[dce]) * rdt_iter * logical_node_num / shr_node_num
                if options.hw_access_forwarding:
                    fwd_hops[dce] = sum(unit_accesses[dce]) * rdt_iter * avg_fwd_dist * (shr_node_num - 1) * logical_node_num / \
                            shr_node_num
                if options.hw_gbuf_sharing:
                    buf_shr_hops[dce] = sum(unit_accesses[dce]) * bufshr_rdt_iters[dce] * avg_rot_dist / shr_node_num * (shr_node_num - 1) * \
                                logical_node_num / shr_node_num

        return dram_accesses, fwd_hops, buf_shr_hops

    def analyze_regf_level_access(self, unit_accesses, redundant_iters, logical_dim, remote_sharings,
                                temporal_sharings, upper_iter_num, upper_repl_num):
        logical_node_num = logical_dim[0] * logical_dim[1]
        fetches = [0 for _ in range(de.NUM)]
        gbuf_accesses = [0 for _ in range(de.NUM)]
        itcn_accesses = [0 for _ in range(de.NUM)]
        for dce in range(de.NUM):
            fetches[dce] = redundant_iters[dce] * upper_iter_num[dce]
            fetches[dce] = 2 * fetches[dce] - 1 if dce == self.rw_data else fetches[dce]
            # If temporal sharing is not empty, only consider temporal sharings because rempte sharings
            # are contained in temporal sharing.
            if len(temporal_sharings) > 0:
                for uac, tmpshr in zip(unit_accesses[dce], temporal_sharings[dce]):
                    fetch_time = logical_node_num
                    for shr_node_num, group_num in tmpshr.items():
                        fetch_time -= shr_node_num * group_num
                    gbuf_accesses[dce] += uac * fetch_time * upper_repl_num * fetches[dce]
                    itcn_accesses[dce] += uac * fetch_time * upper_repl_num * fetches[dce]
            else:
                rmtshr = None
                for dtype, shr_num_dict in remote_sharings:
                    if dtype == dce:
                        rmtshr = shr_num_dict
                        break

                fetch_time = 0
                noshr_node_num = logical_node_num
                for shr_node_num, group_num in rmtshr.items():
                    fetch_time += group_num
                    noshr_node_num -= shr_node_num * group_num
                fetch_time += noshr_node_num

                gbuf_accesses[dce] = sum(unit_accesses[dce]) * fetch_time * upper_repl_num * fetches[dce]
                itcn_accesses[dce] = sum(unit_accesses[dce]) * logical_node_num * upper_repl_num * fetches[dce]

        return gbuf_accesses, itcn_accesses

    @functools.lru_cache(maxsize=1024)
    def analyze_stacks(self, init_data_tuple, stacks, phy_dims, updates, blk_level):
        init_datas = []
        for init_data_tp in init_data_tuple:
            init_datas.append(dict(init_data_tp))
        local_sharings = [] #(data type, node_nums, avg_hop_dist)
        remote_sharings = [] #(data_type, node_nums, avg_hop_dist)
        temporal_sharings = [] #(data_update_size, node_nums, avg_hops_dist)

        stack_dir = []
        if phy_dims.w > phy_dims.h:
            rdir = 0
        else:
            rdir = 1

        logical_dim = [1, 1]
        stacks_dims = []
        stacks_repls = []
        dim_stack_map = defaultdict(lambda: [])
        unit_dist = [[1, 1]]
        for s_idx, stc in enumerate(stacks):
            repl = stc[-1]
            s_dims = []
            for i in range(0, len(stc[:-1]), 2):
                dim_stack_map[stc[i]].append((s_idx, i))
                s_dims.append(stc[i])
            if rdir:
                ud = [unit_dist[s_idx][0], unit_dist[s_idx][1] * repl]
            else:
                ud = [unit_dist[s_idx][0] * repl, unit_dist[s_idx][1]]
            unit_dist.append(ud)
            stacks_dims.append(s_dims)
            stacks_repls.append(repl)
            logical_dim[rdir] *= repl
            stack_dir.append(rdir)
            if logical_dim[0] / phy_dims.w < logical_dim[1] / phy_dims.h:
                rdir = 0
            else:
                rdir = 1

        if blk_level == 0:
            # At GBUF level only nalyze local sharing, considering dist.
            for didx, data_dims in enumerate(self.tdm.data_list):
                # discnt_dim = set()
                # shr_node_region = [1, 1]
                # for s_idx, s_dims in enumerate(stacks_dims):
                #     sdir = stack_dir[s_idx]
                #     if len(set(data_dims) & set(s_dims)) == 0 and sdir not in discnt_dim:
                #         shr_node_region[sdir] *= stacks_repls[s_idx]
                #     else:
                #         discnt_dim.add(sdir)

                shr_node_region = [1, 1]
                nbr_dist = [1, 1]
                started = False
                for s_idx, s_dims in enumerate(stacks_dims):
                    sdir = stack_dir[s_idx]
                    if len(set(data_dims) & set(s_dims)) == 0:
                        shr_node_region[sdir] *= stacks_repls[s_idx]
                        started = True
                    elif started:
                        break
                    else:
                        nbr_dist[sdir] *= stacks_repls[s_idx]

                avg_nbr_dist = sum(nbr_dist) / 2

                # Allow discontinuous sharing.
                # shr_node_region = [1, 1]
                # for s_idx, s_dims in enumerate(stacks_dims):
                #     sdir = stack_dir[s_idx]
                #     if len(set(data_dims) & set(s_dims)) == 0:
                #         shr_node_region[sdir] *= stacks_repls[s_idx]

                if any(d > 1 for d in shr_node_region):
                    dist = self.get_avg_dist_in_region(shr_node_region)
                    fwd_dist = avg_nbr_dist * ((shr_node_region[0] - 1) // 2 + (shr_node_region[1] - 1) // 2) / 2
                    shr_node_nums = shr_node_region[0] * shr_node_region[1]
                    # rotation follows an zig-zag way.
                    # if shr_node_region[1] % 2 == 0:
                    #     head2tail_dist = shr_node_region[1] - 1 + shr_node_region[0] - 1
                    # else:
                    #     head2tail_dist = shr_node_region[1] - 1
                    head2tail_dist = avg_nbr_dist * (shr_node_region[1] - 1 + shr_node_region[0] - 1)
                    rot_dist = avg_nbr_dist * (head2tail_dist + shr_node_nums - 1) * (shr_node_nums - 1) / shr_node_nums
                    local_sharings.append((didx, shr_node_nums, fwd_dist, rot_dist))
        elif blk_level == 1:
            # At REGF level analyze all possible sharings, without considering dist.
            for didx, data_dims in enumerate(self.tdm.data_list):
                equations = []
                variable_idxs = []
                stack_idx_set = set()
                equation_dim_list = []
                for s_dim in data_dims:
                    if s_dim in dim_stack_map:
                        equation_dim_list.append(s_dim)
                        coefs = []
                        vidxs = []
                        for idx_tuple in dim_stack_map[s_dim]:
                            stack_idx_set.add(idx_tuple[0])
                            coefs.append(stacks[idx_tuple[0]][idx_tuple[1]+1])
                            vidxs.append(idx_tuple[0])
                        if len(coefs) != 0:
                            equations.append(coefs)
                            variable_idxs.append(vidxs)
                # Case 1: All nodes shares the same data.
                if len(stack_idx_set) == 0:
                    remote_sharings.append((didx, {logical_dim[0] * logical_dim[1]: 1}))
                # Case 2: Nodes shares data by local groups.
                elif len(stack_idx_set) == 1:
                    group_num = stacks_repls[stack_idx_set.pop()]
                    shr_num = logical_dim[0] * logical_dim[1] // group_num
                    remote_sharings.append((didx, {shr_num: group_num}))
                # Case 3: Remote sharing.
                else:
                    rmt_shr_nodes = self.solve_same_pairs(equations, variable_idxs, stacks_repls)
                    remote_sharings.append((didx, rmt_shr_nodes))

                # Analyze temporal sharings.
                if isinstance(self.tdm, SystolicTensorDimMap):
                    tmp_shr_list = []
                    for upd in updates:
                        diffs = [0 for _ in equation_dim_list]
                        for s_idx, s_dim in enumerate(equation_dim_list):
                            for dim_idx in range(0, len(upd) - 1, 2):
                                upd_dim = upd[dim_idx]
                                upd_step = upd[dim_idx+1]
                                if s_dim == upd_dim:
                                    diffs[s_idx] = upd_step
                        tmp_shr_nodes = self.solve_same_pairs(equations, variable_idxs,
                            stacks_repls, diffs)
                        tmp_shr_list.append(tmp_shr_nodes)
                    temporal_sharings.append(tmp_shr_list)
        else:
            raise ValueError("Unsupported block level: {}".format(blk_level))

        return logical_dim, local_sharings, remote_sharings, temporal_sharings

    def solve_same_pairs(self, equations, variable_idxs, stacks_repls, diffs=None):
        # print(equations)
        # print(variable_idxs)
        if diffs is None:
            diffs = [0 for _ in equations]
        stack_num = len(stacks_repls)
        variable_list = [("s"+str(i), "t"+str(i)) for i in range(stack_num)]
        s_variables = [var_tp[0] for var_tp in variable_list]
        t_variables = [var_tp[1] for var_tp in variable_list]
        problem = Problem()
        for var_tp, cnstr in zip(variable_list, stacks_repls):
            problem.addVariables(var_tp, range(cnstr))

        def _node_retrieve(node_pair):
            s = tuple(map(node_pair.get, s_variables))
            t = tuple(map(node_pair.get, t_variables))
            vs = tuple(sum(c * s[idx] for c, idx in zip(coefs, vidxs)) for coefs, vidxs in zip(equations, variable_idxs))
            vt = tuple(sum(c * t[idx] for c, idx in zip(coefs, vidxs)) for coefs, vidxs in zip(equations, variable_idxs))
            if vs < vt:
                return [s,], vs
            elif vs > vt:
                return [t,], vt
            else:
                return [s, t], vs

        def _create_eq_func(cfs, diff):
            return lambda *vars: sum(cfs[i] * (vars[2*i] - vars[2*i+1]) for i in range(len(vars) // 2)) == diff

        _exclude_self_func = \
                lambda *vars: any(vars[i] != vars[i+1] for i in range(0, len(vars), 2))
        for coefs, v_idxs, diff in zip(equations, variable_idxs, diffs):
            vs = [variable_list[v_idx] for v_idx in v_idxs]
            problem.addConstraint(_create_eq_func(coefs, diff), tuple(itertools.chain(*vs)))

        # Exclude the same node.
        problem.addConstraint(_exclude_self_func, tuple(itertools.chain(*variable_list)))

        solu = problem.getSolutions()

        shr_dict = defaultdict(lambda: set())
        for node_pair in solu:
            nodes, v = _node_retrieve(node_pair)
            for node_crd in nodes:
                shr_dict[v].add(node_crd)

        node_shr_dict = defaultdict(lambda: 0)
        for value in shr_dict.values():
            node_shr_dict[len(value)] += 1

        return node_shr_dict


    def get_avg_dist_in_region(self, region):
        return (region[0] // 2 + region[1] // 2) / 2


    def prod_data_iter(self, start_idx, data_type, flat_iter_times, flat_upd_sizes):
        up_iters = 1
        for up_idx in range(start_idx, len(flat_iter_times)):
            if flat_upd_sizes[up_idx][data_type] != 0:
                up_iters *= flat_iter_times[up_idx]

        return up_iters


    def approx_inter_layer_dist(self, resource):
        inter_layer_dist = [0] * de.NUM
        src_dist = []
        dst_dist = []
        filter_min = float("inf")

        for proc_node in resource.proc_region.iter_node():
            for src_node in resource.src_data_region.iter_node():
                dist = proc_node.hop_dist(src_node)
                src_dist.append(dist)
            for dst_node in resource.dst_data_region.iter_node():
                dist = proc_node.hop_dist(dst_node)
                dst_dist.append(dist)
            for mem_node in resource.dram_region.iter_node():
                filter_min = min(filter_min, proc_node.hop_dist(mem_node))

        inter_layer_dist[de.IFM] = sum(src_dist) / len(src_dist)
        inter_layer_dist[de.FIL] = filter_min
        inter_layer_dist[de.OFM] = sum(dst_dist) / len(dst_dist)

        return inter_layer_dist

    def seg_time_estimation(self, segment, seg_times, real_cstr_dict):
        bat_ngrp = 0
        ifm_ngrp = 0
        ofm_ngrp = 0

        dram_time = 0
        node_time = 0
        seg_layer_idx = dict()
        seg_timing = list()
        for sp_idx, (ltpl, ltimes) in enumerate(zip(segment, seg_times)):
            # seg_timing item: [time, ngrp, ts_xb, td_xb]
            seg_timing.append([])
            for tm_idx, (layer_name, times) in enumerate(zip(ltpl, ltimes)):
                proc_t, dram_t, bus_t = times
                layer_t = max(proc_t, dram_t, bus_t)
                dram_time += dram_t
                is_conv = isinstance(self.network[layer_name],
                    (ConvLayer, ConvBackActLayer, ConvBackWeightLayer))
                real_cstr = real_cstr_dict[layer_name]
                if not bat_ngrp:
                    bat_ngrp = real_cstr.topbat
                elif bat_ngrp != real_cstr.topbat:
                    bat_ngrp = 1

                ifm_ngrp, ofm_ngrp = real_cstr.topifm, real_cstr.topofm
                ts_xb = 0
                td_xb = 0
                for p in self.network.prevs(layer_name):
                    if p not in seg_layer_idx:
                        continue
                    p_sp_idx, p_tm_idx = seg_layer_idx[p]
                    p_timing = seg_timing[p_sp_idx][p_tm_idx]
                    if p_sp_idx == sp_idx:
                        assert p_tm_idx == tm_idx - 1
                        # Same spatial scheduling.
                        if not is_conv and ofm_ngrp == p_timing[1]:
                            # Fused.
                            start = p_timing[2] + p_timing[0] // p_timing[1]
                        else:
                            # Not fused.
                            start = p_timing[3]
                        td_xb = p_timing[3] + layer_t
                    else:
                        assert p_sp_idx < sp_idx
                        assert p_tm_idx == len(seg_timing[p_sp_idx]) - 1
                        # Previous spatial scheduling.
                        if (ifm_ngrp if is_conv else ofm_ngrp) == p_timing[1]:
                            # I/OFM group forwarding.
                            start = p_timing[2] + p_timing[0] // p_timing[1]
                        else:
                            # All I/OFM double buffering.
                            start = p_timing[3]
                    ts_xb = max(ts_xb, start)
                td_xb = max(td_xb, ts_xb + layer_t)
                # print("layer_name: {}".format(layer_name))
                # print("layer_t: {}, ofm_ngrp: {}, ts_xb: {}, td_xb: {}".format(layer_t, ofm_ngrp, ts_xb, td_xb))
                seg_timing[sp_idx].append((layer_t, ofm_ngrp, ts_xb, td_xb))
                seg_layer_idx[layer_name] = (sp_idx, tm_idx)

        critical_time = max(tlist[-1][3] - tlist[0][2] for tlist in seg_timing)
        node_time = max((tlist[-1][3] + critical_time * (bat_ngrp - 1)) // bat_ngrp
                        for tlist in seg_timing)
        # print("critical_time: {}".format(critical_time))
        # print("bat_ngrp: {}".format(bat_ngrp))
        # print("node_time: {}".format(node_time))

        total_time = max(node_time, dram_time)

        return (node_time, dram_time)