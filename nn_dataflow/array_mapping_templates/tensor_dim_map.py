import functools
import operator

import nn_dataflow.core.loop_enum as le
import nn_dataflow.core.data_category_enum as de
from nn_dataflow import util


class SearchMethodEnum():
    KAPLA_SEARCHER = 0
    ML_SEARCHER = 1
    RANDOM_SEARCHER = 2


class LayerTypeEnum():
    CONV = 0
    LOCAL = 1
    CONV_BACK_H = 2
    LOCAL_BACK_H = 3
    CONV_BACK_W = 4
    NUM = 5

lte = LayerTypeEnum()


class ArrayMappingEnum():
    ROW_STATIONARY = 0
    SYSTOLIC = 1
    NUM = 2


class ParallelEnum():
    OUTP = 0
    OFMP = 1
    BATP = 2
    INPP = 3
    IFMP = 4
    NUM = 5

pe = ParallelEnum()


class RSTensorDimMap(object):
    '''
    Provide mappings and conversions between data types, loop types and loop dimensions in
    Row-stationary array mapping.
    '''
    def __init__(self):
        self.data_list = [["C", "K", "R", "S"], ["N", "C", "Xi", "Yi"], ["N", "K", "Xo", "Yo"]]
        # Mainly used to bridge to NN_dataflow.
        # For different layer type, we have different loop-dimension mappings.
        self.loop_list = [[["C"], ["K"], ["N", "Yo", "Xo"]],
                          [["K"], ["K"], ["N", "Yo", "Xo"]],
                          [["C"], ["K"], ["N", "Yi", "Xi"]],
                          [["C"], ["C"], ["N", "Yi", "Xi"]],
                          [["C"], ["K"], ["N", "Yi", "Xi"]]]

        self.part_list = [[["K"], ["Yo", "Xo"], ["N"], ["C"]],
                          [["K"], ["Yo", "Xo"], ["N"], ["K"]],
                          [["K"], ["Yi", "Xi"], ["N"], ["C"]],
                          [["C"], ["Yi", "Xi"], ["N"], ["C"]],
                          [["K"], ["Yi", "Xi"], ["N"], ["C"]]]

        self.dim_set = set()
        for dl in self.data_list:
            self.dim_set |= set(dl)

    @functools.lru_cache(maxsize=64)
    def get_dtype_irr_dims(self, layer_type, dtype):
        irr_dims = []
        if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
            if dtype == de.FIL:
                irr_dims = self.loop_list[layer_type][le.BAT]
            elif dtype == de.IFM:
                irr_dims = self.loop_list[layer_type][le.OFM]
            elif dtype == de.OFM:
                irr_dims = self.loop_list[layer_type][le.IFM]
        elif layer_type in (1, 3):
            if dtype == de.FIL:
                irr_dims = self.loop_list[layer_type][le.BAT]
            elif dtype in (de.IFM, de.OFM):
                irr_dims = []
        else:
            raise ValueError("Invalid layer type: {}".format(layer_type))

        return tuple(irr_dims)

    @functools.lru_cache(maxsize=64)
    def get_dtype_rlvt_dims(self, layer_type, dtype):
        rlvt_dims = []
        if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
            if dtype == de.FIL:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.OFM]
            elif dtype == de.IFM:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.BAT]
            elif dtype == de.OFM:
                rlvt_dims = self.loop_list[layer_type][le.OFM] + self.loop_list[layer_type][le.BAT]
        elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
            if dtype == de.FIL:
                rlvt_dims = self.loop_list[layer_type][le.OFM]
            elif dtype == de.IFM:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.BAT]
            elif dtype == de.OFM:
                rlvt_dims = self.loop_list[layer_type][le.OFM] + self.loop_list[layer_type][le.BAT]
        else:
            raise ValueError("Invalid layer type: {}".format(layer_type))

        return tuple(rlvt_dims)

    @functools.lru_cache(maxsize=64)
    def get_dtype_rlvt_ltypes(self, layer_type, dtype):
        ltypes = []
        for l in range(le.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & \
                   set(self.get_ltype_rlvt_dims(layer_type, l))) > 0:
                ltypes.append(l)

        return tuple(ltypes)

    @functools.lru_cache(maxsize=64)
    def get_dtype_irr_ltypes(self, layer_type, dtype):
        ltypes = []
        for l in range(le.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & \
                   set(self.get_ltype_rlvt_dims(layer_type, l))) == 0:
                ltypes.append(l)

        return tuple(ltypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_rlvt_dims(self, layer_type, ltype):
        return tuple(self.loop_list[layer_type][ltype])

    @functools.lru_cache(maxsize=64)
    def get_dim_irr_dtypes(self, layer_type, dim):
        irr_dtypes = []
        for dtype in range(de.NUM):
            if dim not in self.get_dtype_rlvt_dims(layer_type, dtype):
                irr_dtypes.append(dtype)

        return tuple(irr_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_dim_rlvt_dtypes(self, layer_type, dim):
        rlvt_dtypes = []
        for dtype in range(de.NUM):
            if dim in self.get_dtype_rlvt_dims(layer_type, dtype):
                rlvt_dtypes.append(dtype)

        return tuple(rlvt_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_rlvt_dtypes(self, layer_type, ltype):
        rlvt_dtypes = []
        for dim in self.get_ltype_rlvt_dims(layer_type, ltype):
            for dtype in range(de.NUM):
                if dim in self.get_dtype_rlvt_dims(layer_type, dtype):
                    rlvt_dtypes.append(dtype)

        return tuple(rlvt_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_irr_dtypes(self, layer_type, ltype):
        irr_dtypes = []
        dims = self.get_ltype_rlvt_dims(layer_type, ltype)
        for dtype in range(de.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & set(dims)) == 0:
                irr_dtypes.append(dtype)

        return tuple(irr_dtypes)

    def format_tensor_dim(self, layer_type, tensor_dims, conv_strds):
        if layer_type in (lte.CONV, lte.LOCAL):
            for dim in self.dim_set - {"Xi", "Yi"}:
                if dim not in tensor_dims:
                    tensor_dims[dim] = 1

            tensor_dims["Xi"] = (tensor_dims["Xo"] - 1) * conv_strds[0] + tensor_dims["R"]
            tensor_dims["Yi"] = (tensor_dims["Yo"] - 1) * conv_strds[1] + tensor_dims["S"]

            if layer_type == lte.LOCAL:
                tensor_dims["C"] = tensor_dims["K"] * conv_strds[2]
        elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H):
            for dim in self.dim_set - {"Xo", "Yo"}:
                if dim not in tensor_dims:
                    tensor_dims[dim] = 1

            tensor_dims["Xo"] = (tensor_dims["Xi"] - 1) * conv_strds[0] + tensor_dims["R"]
            tensor_dims["Yo"] = (tensor_dims["Yi"] - 1) * conv_strds[1] + tensor_dims["S"]

            if layer_type == lte.LOCAL_BACK_H:
                tensor_dims["K"] = tensor_dims["C"] * conv_strds[2]
        return tensor_dims

    def get_tensor_size(self, layer_type, tensor_dim):
        tensor_sizes = [0 for _ in range(de.NUM)]
        for d in range(len(self.data_list)):
            if (layer_type in (lte.LOCAL, lte.LOCAL_BACK_H)) and (d == de.FIL):
                continue
            tvs = operator.itemgetter(*self.data_list[d])(tensor_dim)
            tensor_sizes[d] = util.prod(tvs)

        return tensor_sizes

    @functools.lru_cache(maxsize=1024)
    def get_dim_rlvt_part_type(self, layer_type, dim):
        rpe = None
        for penum, pdim_list in enumerate(self.part_list[layer_type]):
            if dim in pdim_list:
                rpe = penum
                break
        if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H) and rpe == pe.INPP:
            rpe = pe.OUTP

        return rpe

    @functools.lru_cache(maxsize=1024)
    def get_dim_crrspd_ltype(self, layer_type, dim):
        lpe = None
        for l, dim_list in enumerate(self.loop_list[layer_type]):
            if dim in dim_list:
                lpe = l
        if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H) and lpe == le.IFM:
            lpe = le.OFM
        return lpe


class SystolicTensorDimMap(object):
    """
    Provide mappings and conversions between data types, loop types and loop dimensions.
    """
    def __init__(self):
        self.data_list = [["C", "F", "K"], ["N", "C", "F", "XY"], ["N", "K", "XY"]]
        self.loop_list = [[["C"], ["K"], ["N", "XY"]],
                          [["K"], ["K"], ["N", "XY"]]]
        self.part_list = [[["K"], ["XY"], ["N"], ["C"]],
                          [["K"], ["XY"], ["N"], ["K"]]]

        self.dim_set = set()
        for dl in self.data_list:
            self.dim_set |= set(dl)

    @functools.lru_cache(maxsize=64)
    def get_dtype_irr_dims(self, layer_type, dtype):
        irr_dims = []
        if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
            if dtype == de.FIL:
                irr_dims = self.loop_list[layer_type][le.BAT]
            elif dtype == de.IFM:
                irr_dims = self.loop_list[layer_type][le.OFM]
            elif dtype == de.OFM:
                irr_dims = self.loop_list[layer_type][le.IFM]
        elif layer_type in (1, 3):
            if dtype == de.FIL:
                irr_dims = self.loop_list[layer_type][le.BAT]
            elif dtype in (de.IFM, de.OFM):
                irr_dims = []
        else:
            raise ValueError("Invalid layer type: {}".format(layer_type))

        return tuple(irr_dims)

    @functools.lru_cache(maxsize=64)
    def get_dtype_rlvt_dims(self, layer_type, dtype):
        rlvt_dims = []
        if layer_type in (lte.CONV, lte.CONV_BACK_H, lte.CONV_BACK_W):
            if dtype == de.FIL:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.OFM]
            elif dtype == de.IFM:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.BAT]
            elif dtype == de.OFM:
                rlvt_dims = self.loop_list[layer_type][le.OFM] + self.loop_list[layer_type][le.BAT]
        elif layer_type in (lte.LOCAL, lte.LOCAL_BACK_H):
            if dtype == de.FIL:
                rlvt_dims = self.loop_list[layer_type][le.OFM]
            elif dtype == de.IFM:
                rlvt_dims = self.loop_list[layer_type][le.IFM] + self.loop_list[layer_type][le.BAT]
            elif dtype == de.OFM:
                rlvt_dims = self.loop_list[layer_type][le.OFM] + self.loop_list[layer_type][le.BAT]
        else:
            raise ValueError("Invalid layer type: {}".format(layer_type))

        return tuple(rlvt_dims)

    @functools.lru_cache(maxsize=64)
    def get_dtype_rlvt_ltypes(self, layer_type, dtype):
        ltypes = []
        for l in range(le.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & \
                   set(self.get_ltype_rlvt_dims(layer_type, l))) > 0:
                ltypes.append(l)

        return tuple(ltypes)

    @functools.lru_cache(maxsize=64)
    def get_dtype_irr_ltypes(self, layer_type, dtype):
        ltypes = []
        for l in range(le.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & \
                   set(self.get_ltype_rlvt_dims(layer_type, l))) == 0:
                ltypes.append(l)

        return tuple(ltypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_rlvt_dims(self, layer_type, ltype):
        return tuple(self.loop_list[layer_type][ltype])

    @functools.lru_cache(maxsize=64)
    def get_dim_irr_dtypes(self, layer_type, dim):
        irr_dtypes = []
        for dtype in range(de.NUM):
            if dim not in self.get_dtype_rlvt_dims(layer_type, dtype):
                irr_dtypes.append(dtype)

        return tuple(irr_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_dim_rlvt_dtypes(self, layer_type, dim):
        rlvt_dtypes = []
        for dtype in range(de.NUM):
            if dim in self.get_dtype_rlvt_dims(layer_type, dtype):
                rlvt_dtypes.append(dtype)

        return tuple(rlvt_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_rlvt_dtypes(self, layer_type, ltype):
        rlvt_dtypes = []
        for dim in self.get_ltype_rlvt_dims(layer_type, ltype):
            for dtype in range(de.NUM):
                if dim in self.get_dtype_rlvt_dims(layer_type, dtype):
                    rlvt_dtypes.append(dtype)

        return tuple(rlvt_dtypes)

    @functools.lru_cache(maxsize=64)
    def get_ltype_irr_dtypes(self, layer_type, ltype):
        irr_dtypes = []
        dims = self.get_ltype_rlvt_dims(layer_type, ltype)
        for dtype in range(de.NUM):
            if len(set(self.get_dtype_rlvt_dims(layer_type, dtype)) & set(dims)) == 0:
                irr_dtypes.append(dtype)

        return tuple(irr_dtypes)

    def format_tensor_dim(self, layer_type, tensor_dims, conv_strds):
        if layer_type in (lte.CONV, lte.LOCAL):
            for dim in self.dim_set:
                if dim not in tensor_dims:
                    tensor_dims[dim] = 1

        return tensor_dims

    def get_tensor_size(self, layer_type, tensor_dim):
        tensor_sizes = [0 for _ in range(de.NUM)]
        for d in range(len(self.data_list)):
            if (layer_type in (lte.LOCAL, lte.LOCAL_BACK_H)) and (d == de.FIL):
                continue
            tvs = operator.itemgetter(*self.data_list[d])(tensor_dim)
            tensor_sizes[d] = util.prod(tvs)

        return tensor_sizes

    def get_dim_rlvt_part_type(self, layer_type, dim):
        rpe = None
        for penum, pdim_list in enumerate(self.part_list[layer_type]):
            if dim in pdim_list:
                rpe = penum
                break
        if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H) and rpe == pe.INPP:
            rpe = pe.OUTP

        return rpe

    @functools.lru_cache(maxsize=1024)
    def get_dim_crrspd_ltype(self, layer_type, dim):
        lpe = None
        for l, dim_list in enumerate(self.loop_list[layer_type]):
            if dim in dim_list:
                lpe = l
        if layer_type in (lte.LOCAL, lte.LOCAL_BACK_H) and lpe == le.IFM:
            lpe = le.OFM
        return lpe
