import itertools
from multiprocessing import Pool
from nn_dataflow.core.buf_shr_scheme import BufShrScheme
from nn_dataflow.core.loop_blocking_scheme import LoopBlockingScheme
from nn_dataflow.util import apply

class BL():
    GBUF = 0
    REGF = 1
    NUM = 2

class Config(object):
    def __init__(self, part, nld, bl_ts_ords, resource, options):
        self.part = part
        bufshr = BufShrScheme(resource.proc_region, part, nld.data_loops)
        self.lbs = LoopBlockingScheme(nld, bl_ts_ords[0], bl_ts_ords[1],
                                      resource, bufshr, options)

    def __str__(self):
        return "part dims: {} bl_ts: {}".format(self.part.pdims, self.lbs.bl_ts)


class DesignSpace(object):
    def __init__(self, partition_array, array_mapping_array, bl_ts_ords_array, lbs_part_len,
                 lbs_am_len, resource, constraint, unit_cost, options):
        self.partition_array = partition_array
        self.array_mapping_array = array_mapping_array
        self.bl_ts_ords_array = bl_ts_ords_array
        self.lbs_part_map = construct_lbs_part_map(lbs_part_len)
        self.lbs_am_map = construct_lbs_am_map(lbs_am_len)
        self.resource = resource
        self.constraint = constraint
        self.unit_cost = unit_cost
        self.options = options

        assert len(self.partition_array) == len(lbs_part_len), \
                "lbs_part_len dim not matching partition array dim!"
        assert len(self.bl_ts_ords_array) == sum(lbs_part_len), \
                "lbs array nums not matching the sum of lbs_part_len!"
        assert len(self.array_mapping_array) == len(lbs_am_len), \
                "lbs_am_len dim not matching partition array dim!"
        assert len(self.bl_ts_ords_array) == sum(lbs_am_len), \
                "lbs array nums not matching the sum of lbs_am_len!"

        self.index_range = len(self.bl_ts_ords_array)

    def index2config(self, index):
        assert index < self.index_range, "index2config: invalid index!"

        bl_ts_ords = self.bl_ts_ords_array[index]
        part = self.find_correspond_part(index)
        array_mapping = self.find_correspond_array_mapping(index)

        return (part, array_mapping, bl_ts_ords, self.resource, self.constraint, self.unit_cost, self.options)

    def indexes2configs(self, indexes):
        if self.options.nprocesses > 1:
            pool = Pool(processes=self.options.nprocesses)
            apply_func = pool.apply_async
            retrieve_func = lambda x: x.get(timeout=3600)
        else:
            pool = None
            apply_func = apply
            retrieve_func = lambda x: x

        handle_list = []
        for index in indexes:
            bl_ts_ords = self.bl_ts_ords_array[index]
            part = self.find_correspond_part(index)
            am = self.find_correspond_array_mapping(index)
            r = apply_func(static_index2config, (part, am, bl_ts_ords,
                                                 self.resource, self.constraint, self.unit_cost, self.options))
            handle_list.append(r)

        configs = tuple([*map(retrieve_func, handle_list)])

        if pool is not None:
            pool.close()
            pool.join()

        return configs

    def config2feature(self, config):
        part = config[0]
        am = config[1]
        bl_ts_ord = config[2]
        partition_feature = part.order + tuple(itertools.chain(*((dim.h, dim.w) \
                                               for dim in part.pdims)))
        loop_blocking_feature = tuple(itertools.chain(*bl_ts_ord[1]))
        size_feature = tuple(am[0].values()) + tuple(am[1][BL.REGF].values()) + tuple(am[1][BL.GBUF].values()) + \
                       (self.resource.size_regf, self.resource.size_gbuf)

        feature = partition_feature + loop_blocking_feature + size_feature

        return feature

    def index2feature(self, index):
        part = self.find_correspond_part(index)
        am = self.find_correspond_array_mapping(index)
        bl_ts_ord = self.bl_ts_ords_array[index]
        partition_feature = part.order + tuple(itertools.chain(*((dim.h, dim.w) \
                                               for dim in part.pdims)))
        loop_blocking_feature = tuple(itertools.chain(*bl_ts_ord[1]))
        size_feature = tuple(am[0].values()) + tuple(am[1][BL.REGF].values()) + tuple(am[1][BL.GBUF].values()) + \
                       (self.resource.size_regf, self.resource.size_gbuf)

        feature = partition_feature + loop_blocking_feature + size_feature
        return feature

    def find_correspond_part(self, index):
        for idx, len_sum in enumerate(self.lbs_part_map):
            if index < len_sum:
                return self.partition_array[idx]

        raise ValueError("find_correspond_part: Invalid index: {}", index)

    def find_correspond_array_mapping(self, index):
        for idx, len_sum in enumerate(self.lbs_am_map):
            if index < len_sum:
                return self.array_mapping_array[idx]

        raise ValueError("find_correspond_array_mapping: Invalid index: {}", index)

    def __len__(self):
        return self.index_range


def construct_lbs_part_map(lbs_part_len):
    lbs_part_map = []
    accum = 0
    for l in lbs_part_len:
        accum += l
        lbs_part_map.append(accum)

    return lbs_part_map

def construct_lbs_am_map(lbs_am_len):
    lbs_am_map = []
    accum = 0
    for l in lbs_am_len:
        accum += l
        lbs_am_map.append(accum)

    return lbs_am_map


def static_index2config(part, array_mapping, bl_ts_ords, resource, constraint, unit_cost, options):
    return (part, array_mapping, bl_ts_ords, resource, constraint, unit_cost, options)
