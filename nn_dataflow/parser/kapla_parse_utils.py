import json
from collections import namedtuple, OrderedDict

from nn_dataflow.core import PhyDim2, NodeRegion, Resource, Option, Cost
import nn_dataflow.core.data_category_enum as de
import nn_dataflow.core.mem_hier_enum as me
import nn_dataflow.core.loop_enum as le
from nn_dataflow.core.layer import ConvLayer, ConvBackLayer, LocalRegionLayer, LocalRegionBackLayer
from nn_dataflow import util
from nn_dataflow.array_mapping_templates.tensor_dim_map import ArrayMappingEnum as ame

class BL():
    '''
    Blocking-level enum. Only used locally.
    '''
    GBUF = 0
    REGF = 1
    NUM = 2


CSTR_LIST = ['topbat', 'topifm', 'topofm']
class SimpleCstr(namedtuple('SimpleCstr', CSTR_LIST)):
    '''
    Simplified constraint specification.
    '''
    def __new__(cls, *args, **kwargs):
        ntp = super(SimpleCstr, cls).__new__(cls, *args, **kwargs)
        return ntp


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
    size_gbuf = gbuf_spec['size'] // word * 0.99
    size_regf = regf_spec['size'] // word * 0.99

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
        if isinstance(layer, (ConvLayer, ConvBackLayer)):
            layer_tensor['R'] = layer.hfil
            layer_tensor['S'] = layer.wfil
        elif isinstance(layer, (LocalRegionLayer, LocalRegionBackLayer)):
            layer_tensor['R'] = layer.hreg
            layer_tensor['S'] = layer.wreg * layer.nreg
    elif array_mapping == ame.SYSTOLIC:
        layer_tensor['N'] = batch_size
        layer_tensor['C'] = layer.nifm
        layer_tensor['K'] = layer.nofm
        layer_tensor['XY'] = layer.hofm * layer.wofm
        if isinstance(layer, (ConvLayer, ConvBackLayer)):
            layer_tensor['F'] = layer.wfil * layer.hfil
        elif isinstance(layer, (LocalRegionLayer, LocalRegionBackLayer)):
            layer_tensor['F'] = layer.wreg * layer.wreg

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


def nn_rearrange(seg_no, seg_dfs, prev_nndfs):
    nndf = []
    for seg_df in seg_dfs:
        cur_df, cur_cost, cur_seg_time, cur_total_cost = seg_df
        for layer_df in cur_df.values():
            layer_df["seg_no"] = seg_no
        if len(prev_nndfs) == 0:
            _df = OrderedDict()
            _df.update(cur_df)
            total_cost = cur_total_cost
            total_time = max(cur_seg_time)
            cost_dict = dict(cur_cost)
            nndf.append((_df, cost_dict, total_time, cur_total_cost))
        else:
            for prev_nndf in prev_nndfs:
                _df = OrderedDict()
                _cd = OrderedDict()
                prev_df, prev_cost, prev_total_time, prev_total_cost = prev_nndf
                total_cost = prev_total_cost + cur_total_cost
                total_time = prev_total_time + max(cur_seg_time)
                _cd.update(prev_cost)
                for key, value in cur_cost.items():
                    _cd[key] = value + _cd.setdefault(key, 0)
                _df.update(prev_df)
                _df.update(cur_df)
                nndf.append((_df, _cd, total_time, total_cost))

    return nndf