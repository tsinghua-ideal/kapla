from nn_dataflow import util
from nn_dataflow.core import PhyDim2
from nn_dataflow.parser.kapla_parse_utils import BL

from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte

NN_DIM_LIST = ["N", "C", "K", "F", "XY"]


class Systolic(object):
    def __init__(self, layer_type, workload, resource, conv_strds):
        self.layer_type = layer_type
        self.workload = workload
        self.physic_region = resource.dim_array
        self.physic_node_dim = resource.proc_region.dim
        self.conv_strds = conv_strds
        if layer_type in (lte.CONV, lte.LOCAL, lte.DW_CONV):
            self.logic_region = PhyDim2(workload["XY"], workload["K"])
        else:
            raise TypeError("Unsupported layer type: {}".format(layer_type))
        self.repl_fold()
        self.repls = dict()

    def repl_fold(self):
        fold_w = 1
        repl_w = 1
        fold_h = 1
        repl_h = 1

        if self.logic_region.h > self.physic_region.h:
            fold_h = util.idivc(self.logic_region.h, self.physic_region.h)
        else:
            repl_h = self.physic_region.h // self.logic_region.h
        if self.logic_region.w > self.physic_region.w:
            fold_w = util.idivc(self.logic_region.w, self.physic_region.w)
        else:
            repl_w = self.physic_region.w // self.logic_region.w

        f_w2h = min(repl_h, fold_w)
        fold_w = util.idivc(fold_w, f_w2h)
        repl_h = repl_h // f_w2h

        f_h2w = min(repl_w, fold_h)
        fold_h = util.idivc(fold_h, f_h2w)
        repl_w = repl_w // f_h2w

        self.fold = PhyDim2(fold_h, fold_w)
        self.repl = PhyDim2(repl_h, repl_w)

        self.logic_region = PhyDim2(util.idivc(self.logic_region.h, self.fold.h),
                                    util.idivc(self.logic_region.w, self.fold.w))

    def gen_unitpass(self):
        min_cnt_loops = float("inf")
        if self.layer_type == lte.CONV:
            for fact_repl_h in util.factorize(self.repl.h, 2):
                for fact_repl_w in util.factorize(self.repl.w, 2):
                    self.repls["N"] = min(fact_repl_h[0] * fact_repl_w[0], self.workload["N"])
                    self.repls["C"] = min(fact_repl_h[1] * fact_repl_w[1], self.workload["C"])

                    lcnt = dict()
                    for dim in NN_DIM_LIST:
                        lcnt[dim] = 1
                    lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
                    lcnt["C"] = util.idivc(self.workload["C"], self.repls["C"])
                    lcnt["K"] = self.fold.w
                    lcnt["XY"] = self.fold.h

                    cnt_loops = util.prod(lcnt.values())
                    if cnt_loops < min_cnt_loops:
                        min_cnt_loops = cnt_loops
                    elif cnt_loops > min_cnt_loops:
                        continue

                    locc = dict()
                    for dim in NN_DIM_LIST:
                        locc[dim] = 1

                    locc["N"] = 1. * self.workload["N"] / (self.repls["N"] * lcnt["N"])
                    locc["C"] = 1. * self.workload["C"] / (self.repls["C"] * lcnt["C"])
                    locc["K"] = 1. * self.workload["K"] / (self.logic_region.w * lcnt["K"])
                    locc["XY"] = 1. * self.workload["XY"] / (self.logic_region.h * lcnt["XY"])

                    gbuf_unit_tensor = dict()
                    gbuf_unit_tensor["N"] = self.repls["N"]
                    gbuf_unit_tensor["F"] = self.workload["F"]
                    gbuf_unit_tensor["C"] = self.repls["C"]
                    gbuf_unit_tensor["K"] = self.logic_region.w
                    gbuf_unit_tensor["XY"] = self.logic_region.h

                    regf_unit_tensor = dict()
                    regf_unit_tensor["F"] = 1
                    regf_unit_tensor["XY"] = 1

                    yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc
        elif self.layer_type in (lte.LOCAL, lte.DW_CONV):
            self.repls["N"] = min(self.repl.h * self.repl.w, self.workload["N"])
            lcnt = dict()
            for dim in NN_DIM_LIST:
                lcnt[dim] = 1
            lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
            lcnt["K"] = self.fold.w
            lcnt["C"] = 1
            lcnt["XY"] = self.fold.h

            locc = dict()
            for dim in NN_DIM_LIST:
                locc[dim] = 1

            locc["N"] = 1. * self.workload["N"] / (self.repls["N"] * lcnt["N"])
            locc["K"] = 1. * self.workload["K"] / (self.logic_region.w * lcnt["K"])
            locc["XY"] = 1. * self.workload["XY"] / (self.logic_region.h * lcnt["XY"])

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["N"] = self.repls["N"]
            gbuf_unit_tensor["F"] = self.workload["F"]
            gbuf_unit_tensor["C"] = self.logic_region.w * self.conv_strds[2]
            gbuf_unit_tensor["K"] = self.logic_region.w
            gbuf_unit_tensor["XY"] = self.logic_region.h

            regf_unit_tensor = dict()
            regf_unit_tensor["F"] = 1
            regf_unit_tensor["XY"] = 1
            regf_unit_tensor["C"] = self.conv_strds[2]

            yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc

    def gen_array_mapping(self):
        for gbuf_unit_tensor, regf_unit_tensor, lcnt, locc in self.gen_unitpass():
            usize = [None] * BL.NUM
            usize[BL.REGF] = regf_unit_tensor
            usize[BL.GBUF] = gbuf_unit_tensor

            if self.layer_type == lte.CONV:
                regf_stacks = []
                regf_stacks.append(("K", 1, "F", 1, self.logic_region.w))
                regf_stacks.append(("XY", 1, "F", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                regf_updates = []
                regf_updates.append(("F", 1))
                regf_updates = tuple(regf_updates)

                unit_ops = 1

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, self.repls

            elif self.layer_type in (lte.LOCAL, lte.DW_CONV):
                regf_stacks = []
                regf_stacks.append(("K", 1, "F", 1, self.logic_region.w))
                regf_stacks.append(("XY", 1, "F", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                regf_updates = []
                regf_updates.append(("F", 1))
                regf_updates = tuple(regf_updates)

                unit_ops = self.conv_strds[2]

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, self.repls

    def get_unit_block(self):
        regf_repls = [self.repl.w, self.repl.h]
        if self.layer_type == lte.CONV:
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["C"] = self.workload["C"]
            loopcnt["K"] = util.idivc(self.workload["K"], self.logic_region.w)
            loopcnt["F"] = 1
            loopcnt["XY"] = util.idivc(self.workload["XY"], self.logic_region.h)

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["F"] = self.workload["F"]
            gbuf_unit_tensor["XY"] = self.logic_region.h
            gbuf_unit_tensor["K"] = self.logic_region.w

            regf_unit_tensor = dict()
            regf_unit_tensor["F"] = 1
            regf_unit_tensor["XY"] = 1

            base_stacks = []
            base_stacks.append(("K", 1, "F", 1, self.logic_region.w))
            base_stacks.append(("XY", 1, "F", 1, self.logic_region.h))

            origin_stack_step_dict = {"K": self.logic_region.w, "XY": self.logic_region.h}

            base_updates= []
            base_updates.append(("F", 1))

            unit_ops = 1
        elif self.layer_type in (lte.LOCAL, lte.DW_CONV):
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["K"] = util.idivc(self.workload["K"], self.logic_region.w)
            loopcnt["C"] = 1
            loopcnt["F"] = 1
            loopcnt["XY"] = util.idivc(self.workload["XY"], self.logic_region.h)

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["F"] = self.workload["F"]
            gbuf_unit_tensor["XY"] = self.logic_region.h
            gbuf_unit_tensor["K"] = self.logic_region.w
            gbuf_unit_tensor["C"] = self.logic_region.w * self.conv_strds[2]

            regf_unit_tensor = dict()
            regf_unit_tensor["F"] = 1
            regf_unit_tensor["XY"] = 1
            regf_unit_tensor["C"] = self.conv_strds[2]

            base_stacks = []
            base_stacks.append(("K", 1, "F", 1, self.logic_region.w))
            base_stacks.append(("XY", 1, "F", 1, self.logic_region.h))

            origin_stack_step_dict = {"K": self.logic_region.w, "XY": self.logic_region.h}

            base_updates = []
            base_updates.append(("F", 1))

            unit_ops = self.conv_strds[2]
        else:
            raise TypeError("Unsupported layer type: {}".format(self.layer_type))

        return loopcnt, regf_repls, regf_unit_tensor, gbuf_unit_tensor, base_stacks, \
               base_updates, origin_stack_step_dict, unit_ops

