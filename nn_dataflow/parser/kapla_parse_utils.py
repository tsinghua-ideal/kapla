import json
from collections import namedtuple, OrderedDict

from nn_dataflow.core import PhyDim2, NodeRegion, Resource, Option, Cost
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
from nn_dataflow.core.layer import ConvLayer, ConvBackActLayer, ConvBackWeightLayer, \
    LocalRegionLayer, LocalRegionBackLayer
from nn_dataflow import util

from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame
from nn_dataflow.array_mapping_templates.tensor_dim_map import ParallelEnum as pe
from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte


class BL:
    """
    Blocking-level enum. Only used locally.
    """
    GBUF = 0
    REGF = 1
    NUM = 2


CSTR_LIST = ['topbat', 'topifm', 'topofm']


class SimpleCstr(namedtuple('SimpleCstr', CSTR_LIST)):
    """
    Simplified constraint specification.
    """

    def __new__(cls, *args, **kwargs):
        ntp = super(SimpleCstr, cls).__new__(cls, *args, **kwargs)
        return ntp


class SegDfCache:
    """
    Cache segment dataflow and its cost.
    """

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


def shape_init_data_block(tdm, tensor_dims):
    # Shape all init data blocks.
    init_datas = list()
    froz_init_datas = list()
    for ts_dims in tdm.data_list:
        data = dict()
        for dim in ts_dims:
            data[dim] = tensor_dims[dim]
        init_datas.append(data)
        froz_data = frozenset(data.items())
        froz_init_datas.append(froz_data)
    froz_init_datas = tuple(froz_init_datas)
    return init_datas, froz_init_datas


def parse_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


def parse_hardware(hw_desc):
    word_bit = hw_desc['word_bit']
    op_cost = hw_desc['op_cost']
    dram_spec = hw_desc['DRAM']
    gbuf_spec = hw_desc['GBUF']
    itcn_spec = hw_desc['ITCN']
    regf_spec = hw_desc['REGF']

    dim_nodes = PhyDim2(*gbuf_spec['array'])
    dim_array = PhyDim2(*regf_spec['array'])

    word = (word_bit + 7) // 8
    # Due to our ideal abstraction, it may not be precisely the exactly resource usage we'd expect.
    # Therefore we conservatively shrink the available buffer size.
    real_size_gbuf = gbuf_spec['size'] // word
    real_size_regf = regf_spec['size'] // word
    size_gbuf = 0.99 * real_size_gbuf
    size_regf = 0.99 * real_size_regf

    array_bus_width = dram_spec['bus_width'] // word_bit
    if not array_bus_width:
        array_bus_width = float('inf')
    dram_bandwidth = dram_spec['bandwidth'] // word

    proc_region = NodeRegion(dim=dim_nodes,
                             origin=PhyDim2(0, 0),
                             type=NodeRegion.PROC)

    if gbuf_spec['mem_type'] == '2D':
        data_region = NodeRegion(dim=PhyDim2(2, 2),
                                 origin=PhyDim2(0, 0),
                                 dist=dim_nodes - PhyDim2(1, 1),
                                 type=NodeRegion.DRAM)
        assert data_region.rel2abs(PhyDim2(1, 1)) + PhyDim2(1, 1) == proc_region.dim
    elif gbuf_spec['mem_type'] == '3D':
        data_region = NodeRegion(dim=dim_nodes,
                                 origin=PhyDim2(0, 0),
                                 type=NodeRegion.DRAM)
    else:
        raise TypeError("Invalid memory type! {}".format(gbuf_spec['mem_type']))

    resource = Resource(proc_region=proc_region,
                        dram_region=data_region,
                        src_data_region=data_region,
                        dst_data_region=data_region,
                        dim_array=dim_array,
                        size_gbuf=size_gbuf,
                        size_regf=size_regf,
                        array_bus_width=array_bus_width,
                        dram_bandwidth=dram_bandwidth,
                        no_time_mux=False)

    hier_cost = [0] * me.NUM
    hier_cost[me.DRAM] = dram_spec['access_cost']
    hier_cost[me.GBUF] = gbuf_spec['access_cost']
    hier_cost[me.ITCN] = itcn_spec['access_cost']
    hier_cost[me.REGF] = regf_spec['access_cost']

    cost = Cost(mac_op=op_cost,
                mem_hier=tuple(hier_cost),
                noc_hop=gbuf_spec['hop_cost'],
                idl_unit=gbuf_spec['idle_cost'])

    return resource, cost


def parse_options(opts):
    opts = opts["options"]
    opts["sw_gbuf_bypass"] = tuple(opts["sw_gbuf_bypass"])
    return Option(**opts)


def layer2workload(array_mapping, layer, batch_size):
    layer_tensor = dict()
    if array_mapping == ame.ROW_STATIONARY:
        layer_tensor['N'] = batch_size
        layer_tensor['C'] = layer.nifm
        layer_tensor['K'] = layer.nofm
        layer_tensor['Xi'] = layer.hifm
        layer_tensor['Yi'] = layer.wifm
        layer_tensor['Xo'] = layer.hofm
        layer_tensor['Yo'] = layer.wofm
        if isinstance(layer, (ConvLayer, ConvBackActLayer, ConvBackWeightLayer)):
            layer_tensor['R'] = layer.hfil
            layer_tensor['S'] = layer.wfil
        elif isinstance(layer, (LocalRegionLayer, LocalRegionBackLayer)):
            layer_tensor['R'] = layer.hreg
            layer_tensor['S'] = layer.wreg
    elif array_mapping == ame.SYSTOLIC:
        layer_tensor['N'] = batch_size
        layer_tensor['C'] = layer.nifm
        layer_tensor['K'] = layer.nofm
        layer_tensor['XY'] = layer.hofm * layer.wofm
        if isinstance(layer, (ConvLayer, ConvBackActLayer, ConvBackWeightLayer)):
            layer_tensor['F'] = max(layer.wtrd, layer.wfil) * max(layer.htrd, layer.hfil)
        elif isinstance(layer, (LocalRegionLayer, LocalRegionBackLayer)):
            layer_tensor['F'] = max(layer.wtrd, layer.wreg) * max(layer.htrd, layer.hreg)

    return layer_tensor


def is_valid(tensor_dimmap, layer_type, tensor_dim, buf_size, shr_node_num=None):
    total_size = 0
    tensor_sizes = tensor_dimmap.get_tensor_size(layer_type, tensor_dim)
    if shr_node_num is None:
        total_size = sum(tensor_sizes)
    else:
        for d in range(de.NUM):
            tensor_size = util.idivc(tensor_sizes[d], shr_node_num[d])
            total_size += tensor_size
    return total_size < buf_size


def get_min_factor(number):
    min_factor = 1
    for x, _ in util.factorize(number, 2):
        if x > 1:
            min_factor = x
            break
    return min_factor


def nn_rearrange(seg_df, prev_nndf):
    cur_df, cur_cost, cur_seg_time, nndf, cur_total_cost = seg_df
    if prev_nndf is None:
        _df = OrderedDict()
        _df.update(cur_df)
        total_time = max(cur_seg_time)
        cost_dict = dict(cur_cost)
        return _df, cost_dict, total_time, nndf, cur_total_cost
    else:
        _df = OrderedDict()
        _cd = OrderedDict()
        prev_df, prev_cost, prev_total_time, prev_nndf, prev_total_cost = prev_nndf
        total_cost = prev_total_cost + cur_total_cost
        total_time = prev_total_time + max(cur_seg_time)
        _cd.update(prev_cost)
        for key, value in cur_cost.items():
            _cd[key] = value + _cd.setdefault(key, 0)
        _df.update(prev_df)
        _df.update(cur_df)

        return _df, _cd, total_time, nndf, total_cost


def part_workload(array_mapping, part, layer, batch_size):
    pdims = part.pdims
    layer_tensor = dict()
    if array_mapping == ame.ROW_STATIONARY:
        if isinstance(layer, (ConvLayer, LocalRegionLayer)):
            layer_tensor["N"] = util.idivc(batch_size, pdims[pe.BATP].size())
            layer_tensor["K"] = util.idivc(layer.nofm, pdims[pe.OUTP].size())
            layer_tensor["Xo"] = util.idivc(layer.wofm, pdims[pe.OFMP].w)
            layer_tensor["Yo"] = util.idivc(layer.hofm, pdims[pe.OFMP].h)
            if isinstance(layer, ConvLayer):
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.INPP].size())
                layer_tensor["R"] = layer.wfil
                layer_tensor["S"] = layer.hfil
                layer_tensor["Xi"] = (layer_tensor["Xo"] - 1) * layer.wtrd + layer.wfil
                layer_tensor["Yi"] = (layer_tensor["Yo"] - 1) * layer.htrd + layer.hfil
            elif isinstance(layer, LocalRegionLayer):
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.OUTP].size())
                layer_tensor["R"] = layer.wreg
                layer_tensor["S"] = layer.hreg
                layer_tensor["Xi"] = (layer_tensor["Xo"] - 1) * layer.wtrd + layer.wreg
                layer_tensor["Yi"] = (layer_tensor["Yo"] - 1) * layer.htrd + layer.hreg
        elif isinstance(layer, (ConvBackActLayer, ConvBackWeightLayer, LocalRegionBackLayer)):
            layer_tensor["N"] = util.idivc(batch_size, pdims[pe.BATP].size())
            layer_tensor["Xi"] = util.idivc(layer.wifm, pdims[pe.OFMP].w)
            layer_tensor["Yi"] = util.idivc(layer.hifm, pdims[pe.OFMP].h)
            if isinstance(layer, (ConvBackActLayer, ConvBackWeightLayer)):
                layer_tensor["K"] = util.idivc(layer.nofm, pdims[pe.OUTP].size())
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.INPP].size())
                layer_tensor["R"] = layer.wfil
                layer_tensor["S"] = layer.hfil
                layer_tensor["Xo"] = (layer_tensor["Xi"] - 1) * layer.wtrd + layer.wfil
                layer_tensor["Yo"] = (layer_tensor["Yi"] - 1) * layer.htrd + layer.hfil
            elif isinstance(layer, LocalRegionBackLayer):
                # layer_tensor["K"] = util.idivc(layer.nofm, pdims[pe.INPP].size())
                layer_tensor["K"] = util.idivc(layer.nofm, pdims[pe.OUTP].size())
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.OUTP].size())
                layer_tensor["R"] = layer.wreg
                layer_tensor["S"] = layer.hreg
                layer_tensor["Xo"] = (layer_tensor["Xi"] - 1) * layer.wtrd + layer.wreg
                layer_tensor["Yo"] = (layer_tensor["Yi"] - 1) * layer.htrd + layer.hreg
        else:
            raise TypeError("Unsupported layer type: {}".format(type(layer)))
    elif array_mapping == ame.SYSTOLIC:
        if isinstance(layer, (ConvLayer, LocalRegionLayer)):
            layer_tensor["N"] = util.idivc(batch_size, pdims[pe.BATP].size())
            layer_tensor["K"] = util.idivc(layer.nofm, pdims[pe.OUTP].size())
            layer_tensor["XY"] = util.idivc(layer.wofm * layer.hofm, pdims[pe.OFMP].size())
            if isinstance(layer, ConvLayer):
                layer_tensor["F"] = max(layer.hfil, layer.htrd) * max(layer.wfil, layer.wtrd)
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.INPP].size())
            elif isinstance(layer, LocalRegionLayer):
                layer_tensor["F"] = max(layer.hreg, layer.htrd) * max(layer.wreg, layer.wtrd)
                layer_tensor["C"] = util.idivc(layer.nifm, pdims[pe.OUTP].size())
        else:
            raise TypeError("Unsupported layer type: {}".format(type(layer)))

    return layer_tensor


def construct_stack(array_mapping, layer_type, part, workload):
    stacks = []
    if layer_type in (lte.CONV, lte.LOCAL):
        for pae in reversed(part.order):
            pdim = part.pdims[pae]
            if pae == pe.OUTP:
                if layer_type == lte.CONV:
                    if pdim.w != 1:
                        stacks.append(("K", workload["K"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(("K", pdim.w * workload["K"], pdim.h))
                elif layer_type == lte.LOCAL:
                    if pdim.w != 1:
                        stacks.append(("C", workload["C"], "K", workload["K"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(
                            ("C", pdim.w * workload["C"], "K", pdim.w * workload["K"], pdim.h))
            elif pae == pe.INPP:
                if pdim.w != 1:
                    stacks.append(("C", workload["C"], pdim.w))
                if pdim.h != 1:
                    stacks.append(("C", pdim.w * workload["C"], pdim.h))
            elif pae == pe.BATP:
                if pdim.w != 1:
                    stacks.append(("N", workload["N"], pdim.w))
                if pdim.h != 1:
                    stacks.append(("N", pdim.w * workload["N"], pdim.h))
            elif pae == pe.OFMP:
                if array_mapping == ame.ROW_STATIONARY:
                    if pdim.w != 1:
                        stacks.append(("Xo", workload["Xo"], "Xi", workload["Xi"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(("Yo", workload["Yo"], "Yi", workload["Yi"], pdim.h))
                elif array_mapping == ame.SYSTOLIC:
                    if pdim.w != 1:
                        stacks.append(("XY", workload["XY"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(("XY", workload["XY"] * pdim.w, pdim.h))
    elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
        if array_mapping == ame.SYSTOLIC:
            raise TypeError("SYSTOLIC not supports back-prop layer.")
        for pae in reversed(part.order):
            pdim = part.pdims[pae]
            if pae == pe.OUTP:
                if layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                    if pdim.w != 1:
                        stacks.append(("K", workload["K"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(("K", pdim.w * workload["K"], pdim.h))
                elif layer_type == lte.LOCAL_BACK_H:
                    if pdim.w != 1:
                        stacks.append(("C", workload["C"], "K", workload["K"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(
                            ("C", pdim.w * workload["C"], "K", pdim.w * workload["K"], pdim.h))
            elif pae == pe.INPP:
                if layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                    if pdim.w != 1:
                        stacks.append(("C", workload["C"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(("C", pdim.w * workload["C"], pdim.h))
                elif layer_type == lte.LOCAL_BACK_H:
                    if pdim.w != 1:
                        stacks.append(("C", workload["C"], "K", workload["K"], pdim.w))
                    if pdim.h != 1:
                        stacks.append(
                            ("C", pdim.w * workload["C"], "K", pdim.w * workload["K"], pdim.h))
            elif pae == pe.BATP:
                if pdim.w != 1:
                    stacks.append(("N", workload["N"], pdim.w))
                if pdim.h != 1:
                    stacks.append(("N", pdim.w * workload["N"], pdim.h))
            elif pae == pe.OFMP: # pe.IFMP
                if pdim.w != 1:
                    stacks.append(("Xo", workload["Xo"], "Xi", workload["Xi"], pdim.w))
                if pdim.h != 1:
                    stacks.append(("Yo", workload["Yo"], "Yi", workload["Yi"], pdim.h))
    stacks = tuple(stacks)

    return stacks


def layer_rearrange(tdm, gbuf_tensor_dims, gbuf_stack, gbuf_update, regf_tensor_dims,
                    regf_stack, regf_update, buf_sharing):
    layer_df = dict()
    gbuf_df = dict()
    gbuf_df['tensor_w'] = {dim: gbuf_tensor_dims[dim] for dim in tdm.data_list[de.FIL]}
    gbuf_df['tensor_i'] = {dim: gbuf_tensor_dims[dim] for dim in tdm.data_list[de.IFM]}
    gbuf_df['tensor_o'] = {dim: gbuf_tensor_dims[dim] for dim in tdm.data_list[de.OFM]}
    gbuf_df['stack'] = gbuf_stack
    gbuf_df['update'] = gbuf_update

    gbuf_df['tensor_i']['buf_sharing'] = buf_sharing.size(de.IFM) > 1
    gbuf_df['tensor_w']['buf_sharing'] = buf_sharing.size(de.FIL) > 1
    gbuf_df['tensor_o']['buf_sharing'] = buf_sharing.size(de.OFM) > 1

    regf_df = dict()
    regf_df['tensor_w'] = {dim: regf_tensor_dims[dim] for dim in tdm.data_list[de.FIL]}
    regf_df['tensor_i'] = {dim: regf_tensor_dims[dim] for dim in tdm.data_list[de.IFM]}
    regf_df['tensor_o'] = {dim: regf_tensor_dims[dim] for dim in tdm.data_list[de.OFM]}
    regf_df['stack'] = regf_stack
    regf_df['update'] = regf_update

    layer_df['GBUF'] = gbuf_df
    layer_df['REGF'] = regf_df

    return layer_df


def ident_layer_type(layer):
    if isinstance(layer, ConvLayer):
        layer_type = lte.CONV
    elif isinstance(layer, LocalRegionLayer):
        layer_type = lte.LOCAL
    elif isinstance(layer, ConvBackActLayer):
        layer_type = lte.CONV_BACK_H
    elif isinstance(layer, ConvBackWeightLayer):
        layer_type = lte.CONV_BACK_W
    elif isinstance(layer, LocalRegionBackLayer):
        layer_type = lte.LOCAL_BACK_H
    else:
        raise TypeError("Unsupport layer type: {}".format(type(layer)))
    return layer_type


def get_conv_strds(layer_type, layer):
    if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
        conv_strds = (layer.wtrd, layer.htrd, 1)
    elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
        conv_strds = (layer.wtrd, layer.htrd, layer.ntrd)
    else:
        raise TypeError("get_conv_strds: Invalid layer type {}".format(layer_type))
    return conv_strds
