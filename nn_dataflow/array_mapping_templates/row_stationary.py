from nn_dataflow import util
from nn_dataflow.core import PhyDim2
from nn_dataflow.parser.kapla_parse_utils import BL

from nn_dataflow.array_mapping_templates.tensor_dim_map import LayerTypeEnum as lte

PARA_DIM_LIST = ["N", "C", "K", "Xo", "Yo"]
NN_DIM_LIST = ["N", "C", "K", "Xo", "Yo", "Xi", "Yi"]

class RowStationary(object):
    def __init__(self, layer_type, workload, resource, conv_strds):
        self.layer_type = layer_type
        self.workload = workload
        self.physic_region = resource.dim_array
        self.physic_node_dim = resource.proc_region.dim
        self.conv_strds = conv_strds
        if layer_type in (lte.CONV, lte.LOCAL, lte.DW_CONV):
            self.logic_region = PhyDim2(workload["S"], workload["Yo"])
        elif layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W, lte.LOCAL_BACK_H, lte.DW_CONV_H,
                            lte.DW_CONV_W):
            self.logic_region = PhyDim2(workload["S"], workload["Yi"])
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
        # ConvLayer
        if self.layer_type == lte.CONV:
            for fact_repl_h in util.factorize(self.repl.h, 3):
                for fact_repl_w in util.factorize(self.repl.w, 3):
                    self.repls["N"] = min(fact_repl_h[0] * fact_repl_w[0], self.workload["N"])
                    self.repls["C"] = min(fact_repl_h[1] * fact_repl_w[1], self.workload["C"])
                    self.repls["K"] = min(fact_repl_h[2] * fact_repl_w[2], self.workload["K"])

                    lcnt = dict()
                    for dim in NN_DIM_LIST:
                        lcnt[dim] = 1
                    lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
                    lcnt["C"] = util.idivc(self.workload["C"], self.repls["C"])
                    lcnt["K"] = util.idivc(self.workload["K"], self.repls["K"])

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
                    locc["K"] = 1. * self.workload["K"] / (self.repls["K"] * lcnt["K"])

                    gbuf_unit_tensor = dict()
                    gbuf_unit_tensor["R"] = self.workload["R"]
                    gbuf_unit_tensor["S"] = self.workload["S"]
                    gbuf_unit_tensor["Xo"] = self.workload["Xo"]
                    gbuf_unit_tensor["Yo"] = self.workload["Yo"]
                    gbuf_unit_tensor["N"] = self.repls["N"]
                    gbuf_unit_tensor["C"] = self.repls["C"]
                    gbuf_unit_tensor["K"] = self.repls["K"]

                    regf_unit_tensor = dict()
                    regf_unit_tensor["R"] = self.workload["R"]
                    regf_unit_tensor["Xi"] = self.workload["R"]

                    yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc
        elif self.layer_type in (lte.LOCAL, lte.DW_CONV):
            for fact_repl_h in util.factorize(self.repl.h, 2):
                for fact_repl_w in util.factorize(self.repl.w, 2):
                    self.repls["N"] = min(fact_repl_h[0] * fact_repl_w[0], self.workload["N"])
                    self.repls["K"] = min(fact_repl_h[1] * fact_repl_w[1], self.workload["K"])
                    # self.repls["C"] = self.repls["K"]

                    lcnt = dict()
                    for dim in NN_DIM_LIST:
                        lcnt[dim] = 1
                    lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
                    lcnt["K"] = util.idivc(self.workload["K"], self.repls["K"])
                    lcnt["C"] = 1

                    cnt_loops = util.prod(lcnt.values())
                    if cnt_loops < min_cnt_loops:
                        min_cnt_loops = cnt_loops
                    elif cnt_loops > min_cnt_loops:
                        continue

                    locc = dict()
                    for dim in NN_DIM_LIST:
                        locc[dim] = 1

                    locc["N"] = 1. * self.workload["N"] / (self.repls["N"] * lcnt["N"])
                    locc["K"] = 1. * self.workload["K"] / (self.repls["K"] * lcnt["K"])
                    locc["C"] = 1.

                    gbuf_unit_tensor = dict()
                    gbuf_unit_tensor["R"] = self.workload["R"]
                    gbuf_unit_tensor["S"] = self.workload["S"]
                    gbuf_unit_tensor["Xi"] = self.workload["Xi"]
                    gbuf_unit_tensor["Yi"] = self.workload["Yi"]
                    gbuf_unit_tensor["Xo"] = self.workload["Xo"]
                    gbuf_unit_tensor["Yo"] = self.workload["Yo"]
                    gbuf_unit_tensor["N"] = self.repls["N"]
                    gbuf_unit_tensor["C"] = self.repls["K"] * self.conv_strds[2]
                    gbuf_unit_tensor["K"] = self.repls["K"]

                    regf_unit_tensor = dict()

                    # reduction on fmap.
                    if self.layer_type == lte.DW_CONV or self.workload["R"] > 1:
                        regf_unit_tensor["Xi"] = self.workload["R"]
                        regf_unit_tensor["Yi"] = 1
                        regf_unit_tensor["R"] = self.workload["R"]
                    # reduction across channel.
                    else:
                        regf_unit_tensor["Xi"] = 1
                        regf_unit_tensor["Yi"] = 1
                        regf_unit_tensor["C"] = self.conv_strds[2]
                        regf_unit_tensor["K"] = 1

                    yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc
        elif self.layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
            for fact_repl_h in util.factorize(self.repl.h, 3):
                for fact_repl_w in util.factorize(self.repl.w, 3):
                    self.repls["N"] = min(fact_repl_h[0] * fact_repl_w[0], self.workload["N"])
                    self.repls["C"] = min(fact_repl_h[1] * fact_repl_w[1], self.workload["C"])
                    self.repls["K"] = min(fact_repl_h[2] * fact_repl_w[2], self.workload["K"])

                    lcnt = dict()
                    for dim in NN_DIM_LIST:
                        lcnt[dim] = 1
                    lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
                    lcnt["C"] = util.idivc(self.workload["C"], self.repls["C"])
                    lcnt["K"] = util.idivc(self.workload["K"], self.repls["K"])

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
                    locc["K"] = 1. * self.workload["K"] / (self.repls["K"] * lcnt["K"])

                    gbuf_unit_tensor = dict()
                    gbuf_unit_tensor["R"] = self.workload["R"]
                    gbuf_unit_tensor["S"] = self.workload["S"]
                    gbuf_unit_tensor["Xi"] = self.workload["Xi"]
                    gbuf_unit_tensor["Yi"] = self.workload["Yi"]
                    gbuf_unit_tensor["N"] = self.repls["N"]
                    gbuf_unit_tensor["C"] = self.repls["C"]
                    gbuf_unit_tensor["K"] = self.repls["K"]

                    regf_unit_tensor = dict()
                    regf_unit_tensor["R"] = self.workload["R"]
                    regf_unit_tensor["Xo"] = self.workload["R"]

                    yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc
        elif self.layer_type in (lte.LOCAL_BACK_H, lte.DW_CONV_H, lte.DW_CONV_W):
            for fact_repl_h in util.factorize(self.repl.h, 2):
                for fact_repl_w in util.factorize(self.repl.w, 2):
                    self.repls["N"] = min(fact_repl_h[0] * fact_repl_w[0], self.workload["N"])
                    self.repls["C"] = min(fact_repl_h[1] * fact_repl_w[1], self.workload["C"])

                    lcnt = dict()
                    for dim in NN_DIM_LIST:
                        lcnt[dim] = 1
                    lcnt["N"] = util.idivc(self.workload["N"], self.repls["N"])
                    lcnt["C"] = util.idivc(self.workload["C"], self.repls["C"])
                    lcnt["K"] = 1

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
                    locc["K"] = 1.

                    gbuf_unit_tensor = dict()
                    gbuf_unit_tensor["R"] = self.workload["R"]
                    gbuf_unit_tensor["S"] = self.workload["S"]
                    gbuf_unit_tensor["Xi"] = self.workload["Xi"]
                    gbuf_unit_tensor["Yi"] = self.workload["Yi"]
                    gbuf_unit_tensor["Xo"] = self.workload["Xo"]
                    gbuf_unit_tensor["Yo"] = self.workload["Yo"]
                    gbuf_unit_tensor["N"] = self.repls["N"]
                    gbuf_unit_tensor["C"] = self.repls["C"]
                    gbuf_unit_tensor["K"] = self.repls["C"] * self.conv_strds[2]

                    regf_unit_tensor = dict()

                    # reduction on fmap.
                    if self.layer_type in (lte.DW_CONV_H, lte.DW_CONV_W) or self.workload["R"] > 1:
                        regf_unit_tensor["Xo"] = self.workload["R"]
                        regf_unit_tensor["Yo"] = 1
                        regf_unit_tensor["R"] = self.workload["R"]
                    # reduction across channel.
                    else:
                        regf_unit_tensor["Xo"] = 1
                        regf_unit_tensor["Yo"] = 1
                        regf_unit_tensor["C"] = 1
                        regf_unit_tensor["K"] = self.conv_strds[2]

                    yield gbuf_unit_tensor, regf_unit_tensor, lcnt, locc
        else:
            raise ValueError("Invalid layer type: {}".format(self.layer_type))

    def gen_array_mapping(self):
        for gbuf_unit_tensor, regf_unit_tensor, lcnt, locc in self.gen_unitpass():
            usize = [None] * BL.NUM
            usize[BL.REGF] = regf_unit_tensor
            usize[BL.GBUF] = gbuf_unit_tensor

            if self.layer_type == lte.CONV:
                # construct stack
                regf_stacks = []
                regf_stacks.append(("Yi", self.conv_strds[1], "Yo", 1, self.logic_region.w))
                regf_stacks.append(("S", 1, "Yi", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                # construct unitpass updates
                regf_updates = []
                regf_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
                # We simply require a unitpass to calc the full ofm.
                if self.fold.w > 1:
                    regf_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo",
                                         self.logic_region.w))
                if self.fold.h > 1:
                    regf_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))

                regf_updates = tuple(regf_updates)

                # construct unit_ops
                unit_ops = self.workload["R"]

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, \
                      self.repls
            elif self.layer_type in (lte.LOCAL, lte.DW_CONV):
                # construct stack
                regf_stacks = []
                regf_stacks.append(("Yi", self.conv_strds[1], "Yo", 1, self.logic_region.w))
                regf_stacks.append(("S", 1, "Yi", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        if dim == "K":
                            regf_stacks.append(("C", repl * self.conv_strds[2], "K", repl))
                        else:
                            regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                # construct unitpass updates
                regf_updates = []
                if self.layer_type == lte.DW_CONV or self.workload["R"] > 1:
                    regf_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
                    if self.fold.w > 1:
                        regf_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo",
                                             self.logic_region.w))
                    if self.fold.h > 1:
                        regf_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))
                else:
                    regf_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
                    if self.fold.w > 1:
                        regf_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo",
                                             self.logic_region.w))
                    if self.fold.h > 1:
                        regf_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))
                    regf_updates.append(("C", self.conv_strds[2], "K", 1))

                regf_updates = tuple(regf_updates)

                # construct unit_ops
                if self.layer_type == lte.DW_CONV or self.workload["R"] > 1:
                    unit_ops = self.workload["R"]
                else:
                    unit_ops = self.conv_strds[2]

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, \
                      self.repls
            elif self.layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
                # construct stack
                regf_stacks = []
                regf_stacks.append(("Yo", self.conv_strds[1], "Yi", 1, self.logic_region.w))
                regf_stacks.append(("S", 1, "Yo", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                # construct unitpass updates
                regf_updates = []
                regf_updates.append(("Xo", self.conv_strds[0], "Xi", 1))
                # We simply require a unitpass to calc the full ofm.
                if self.fold.w > 1:
                    regf_updates.append(("Yo", self.conv_strds[1] * self.logic_region.w, "Yi",
                                         self.logic_region.w))
                if self.fold.h > 1:
                    regf_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))

                regf_updates = tuple(regf_updates)

                # construct unit_ops
                unit_ops = self.workload["R"]

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, \
                      self.repls
            elif self.layer_type in (lte.LOCAL_BACK_H, lte.DW_CONV_H, lte.DW_CONV_W):
                regf_stacks = []
                regf_stacks.append(("Yo", self.conv_strds[1], "Yi", 1, self.logic_region.w))
                regf_stacks.append(("S", 1, "Yo", 1, self.logic_region.h))
                for dim, repl in self.repls.items():
                    if repl > 1:
                        if dim == "C":
                            regf_stacks.append(("C", repl, "K", repl * self.conv_strds[2]))
                        else:
                            regf_stacks.append((dim, repl))
                regf_stacks = tuple(regf_stacks)

                # construct unitpass updates
                regf_updates = []
                if (self.layer_type in (lte.DW_CONV_H, lte.DW_CONV_W)) or self.workload["R"] > 1:
                    regf_updates.append(("Xo", self.conv_strds[0], "Xi", 1))
                    if self.fold.w > 1:
                        regf_updates.append(("Yo", self.conv_strds[1] * self.logic_region.w, "Yi", self.logic_region.w))
                    if self.fold.h > 1:
                        regf_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))
                else:
                    regf_updates.append(("Xo", self.conv_strds[0], "Xi", 1))
                    if self.fold.w > 1:
                        regf_updates.append(("Yo", self.conv_strds[1] * self.logic_region.w, "Yi", self.logic_region.w))
                    if self.fold.h > 1:
                        regf_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))
                    regf_updates.append(("K", self.conv_strds[2], "C", 1))

                regf_updates = tuple(regf_updates)

                # construct unit_ops
                if (self.layer_type in (lte.DW_CONV_H, lte.DW_CONV_W)) or self.workload["R"] > 1:
                    unit_ops = self.workload["R"]
                else:
                    unit_ops = self.conv_strds[2]

                yield lcnt, usize, self.logic_region, regf_stacks, regf_updates, unit_ops, self.repls
            else:
                raise ValueError("Invalid layer type: {}".format(self.layer_type))

    def get_unit_block(self):
        regf_repls = [self.repl.w, self.repl.h]
        if self.layer_type == lte.CONV:
            elapsed_width = self.logic_region.w * util.idivc(self.workload["Yo"], self.logic_region.w)
            if elapsed_width // self.logic_region.w > 1 and \
               all(util.gcd(elapsed_width // self.logic_region.w, node_dim) == 1 for node_dim in self.physic_node_dim):
                cand = self.logic_region.w - 1
                while cand > (self.logic_region.w * 0.85):
                    cand_upper_iter = util.idivc(self.workload["Yo"], cand)
                    if any(util.gcd(cand_upper_iter, node_dim) > 1 for node_dim in self.physic_node_dim):
                        self.logic_region = PhyDim2(self.logic_region.h, cand)
                        break
                    cand -= 1
            self.fold = PhyDim2(self.fold.h, util.idivc(self.workload["Yo"], self.logic_region.w))
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["C"] = self.workload["C"]
            loopcnt["K"] = self.workload["K"]
            # Fow row-stationary, it's always best to iterate all Xo in each PE, so that the window
            # reuse can be fully utilized. So there is no loop iteration outside array mapping.
            loopcnt["Xo"] = 1
            loopcnt["Yo"] = util.idivc(self.workload["Yo"], self.logic_region.w)
            # loopcnt["Yo"] = 1

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xo to exploit window reuse.
            gbuf_unit_tensor["Xo"] = self.workload["Xo"]
            gbuf_unit_tensor["Yo"] = self.logic_region.w
            # gbuf_unit_tensor["Yo"] = self.workload["Yo"]

            regf_unit_tensor = dict()
            regf_unit_tensor["R"] = self.workload["R"]
            regf_unit_tensor["Xi"] = self.workload["R"]

            base_stacks = []
            base_stacks.append(("Yi", self.conv_strds[1], "Yo", 1, self.logic_region.w))
            base_stacks.append(("S", 1, "Yi", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo", self.logic_region.w))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))

            unit_ops = self.workload["R"]

        elif self.layer_type == lte.LOCAL:
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["K"] = self.workload["K"]
            loopcnt["C"] = 1
            # Fow row-stationary, it's always best to iterate all Xo in each PE, so that the window
            # reuse can be fully utilized. So there is no loop iteration outside array mapping.
            loopcnt["Xo"] = 1
            loopcnt["Yo"] = util.idivc(self.workload["Yo"], self.logic_region.w)
            # loopcnt["Yo"] = 1

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xo to exploit window reuse.
            gbuf_unit_tensor["Xo"] = self.workload["Xo"]
            gbuf_unit_tensor["Yo"] = self.logic_region.w
            # gbuf_unit_tensor["Yo"] = self.workload["Yo"]
            gbuf_unit_tensor["C"] = self.conv_strds[2]
            gbuf_unit_tensor["K"] = 1

            base_stacks = []
            base_stacks.append(("Yi", self.conv_strds[1], "Yo", 1, self.logic_region.w))
            base_stacks.append(("S", 1, "Yi", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo", self.logic_region.w))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))

            regf_unit_tensor = dict()
            # Reduction on fmap.
            if self.workload["R"] > 1:
                regf_unit_tensor["Xi"] = self.workload["R"]
                regf_unit_tensor["R"] = self.workload["R"]
                regf_unit_tensor["Yi"] = 1
                unit_ops = self.workload["R"]
            # Reduction across channel.
            else:
                regf_unit_tensor["Xi"] = 1
                regf_unit_tensor["Yi"] = 1
                regf_unit_tensor["K"] = 1
                regf_unit_tensor["C"] = self.conv_strds[2]
                base_updates.append(("C", self.conv_strds[2], "K", 1))
                unit_ops = self.conv_strds[2]
        elif self.layer_type in (lte.CONV_BACK_H, lte.CONV_BACK_W):
            elapsed_width = self.logic_region.w * util.idivc(self.workload["Yi"], self.logic_region.w)
            if elapsed_width // self.logic_region.w > 1 and \
               all(util.gcd(elapsed_width // self.logic_region.w, node_dim) == 1 for node_dim in self.physic_node_dim):
                cand = self.logic_region.w - 1
                while cand > (self.logic_region.w * 0.85):
                    cand_upper_iter = util.idivc(self.workload["Yi"], cand)
                    if any(util.gcd(cand_upper_iter, node_dim) > 1 for node_dim in self.physic_node_dim):
                        self.logic_region = PhyDim2(self.logic_region.h, cand)
                        break
                    cand -= 1
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["C"] = self.workload["C"]
            loopcnt["K"] = self.workload["K"]
            # Fow row-stationary, it's always best to iterate all Xi in each PE, so that the window
            # reuse can be fully utilized. So there is no loop iteration outside array mapping.
            loopcnt["Xi"] = 1
            loopcnt["Yi"] = util.idivc(self.workload["Yi"], self.logic_region.w)

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xi to exploit window reuse.
            gbuf_unit_tensor["Xi"] = self.workload["Xi"]
            gbuf_unit_tensor["Yi"] = self.logic_region.w

            regf_unit_tensor = dict()
            regf_unit_tensor["R"] = self.workload["R"]
            regf_unit_tensor["Xo"] = self.workload["R"]

            base_stacks = []
            base_stacks.append(("Yi", 1, "Yo", self.conv_strds[1], self.logic_region.w))
            base_stacks.append(("S", 1, "Yo", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", 1, "Xo", self.conv_strds[0], 1))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.logic_region.w, "Yo", self.conv_strds[1] * self.logic_region.w))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))

            unit_ops = self.workload["R"]
        elif self.layer_type == lte.LOCAL_BACK_H:
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["K"] = 1
            loopcnt["C"] = self.workload["C"]
            loopcnt["Xi"] = 1
            loopcnt["Yi"] = util.idivc(self.workload["Yi"], self.logic_region.w)

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xo to exploit window reuse.
            gbuf_unit_tensor["Xi"] = self.workload["Xi"]
            gbuf_unit_tensor["Yi"] = self.logic_region.w
            # gbuf_unit_tensor["Yo"] = self.workload["Yo"]
            gbuf_unit_tensor["C"] = 1
            gbuf_unit_tensor["K"] = self.conv_strds[2]

            base_stacks = []
            base_stacks.append(("Yi", 1, "Yo", self.conv_strds[1], self.logic_region.w))
            base_stacks.append(("S", 1, "Yo", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", 1, "Xo", self.conv_strds[0]))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.logic_region.w, "Yo", self.logic_region.w * self.conv_strds[1]))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))

            regf_unit_tensor = dict()
            # Reduction on fmap.
            if self.workload["R"] > 1:
                regf_unit_tensor["Xo"] = self.workload["R"]
                regf_unit_tensor["R"] = self.workload["R"]
                regf_unit_tensor["Yo"] = 1
                unit_ops = self.workload["R"]
            # Reduction across channel.
            else:
                regf_unit_tensor["Xo"] = 1
                regf_unit_tensor["Yo"] = 1
                regf_unit_tensor["C"] = 1
                regf_unit_tensor["K"] = self.conv_strds[2]
                base_updates.append(("C", 1, "K", self.conv_strds[2]))
                unit_ops = self.conv_strds[2]
        elif self.layer_type == lte.DW_CONV:
            elapsed_width = self.logic_region.w * util.idivc(self.workload["Yo"], self.logic_region.w)
            if elapsed_width // self.logic_region.w > 1 and \
               all(util.gcd(elapsed_width // self.logic_region.w, node_dim) == 1 for node_dim in self.physic_node_dim):
                cand = self.logic_region.w - 1
                while cand > (self.logic_region.w * 0.85):
                    cand_upper_iter = util.idivc(self.workload["Yo"], cand)
                    if any(util.gcd(cand_upper_iter, node_dim) > 1 for node_dim in self.physic_node_dim):
                        self.logic_region = PhyDim2(self.logic_region.h, cand)
                        break
                    cand -= 1
            self.fold = PhyDim2(self.fold.h, util.idivc(self.workload["Yo"], self.logic_region.w))
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["K"] = self.workload["K"]
            loopcnt["C"] = 1
            # Fow row-stationary, it's always best to iterate all Xo in each PE, so that the window
            # reuse can be fully utilized. So there is no loop iteration outside array mapping.
            loopcnt["Xo"] = 1
            loopcnt["Yo"] = util.idivc(self.workload["Yo"], self.logic_region.w)
            # loopcnt["Yo"] = 1

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xo to exploit window reuse.
            gbuf_unit_tensor["Xo"] = self.workload["Xo"]
            gbuf_unit_tensor["Yo"] = self.logic_region.w
            # gbuf_unit_tensor["Yo"] = self.workload["Yo"]
            gbuf_unit_tensor["K"] = 1

            base_stacks = []
            base_stacks.append(("Yi", self.conv_strds[1], "Yo", 1, self.logic_region.w))
            base_stacks.append(("S", 1, "Yi", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", self.conv_strds[0], "Xo", 1))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.conv_strds[1] * self.logic_region.w, "Yo", self.logic_region.w))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yi", self.logic_region.h))

            regf_unit_tensor = dict()
            regf_unit_tensor["Xi"] = self.workload["R"]
            regf_unit_tensor["R"] = self.workload["R"]
            regf_unit_tensor["Yi"] = 1
            unit_ops = self.workload["R"]
        elif self.layer_type in (lte.DW_CONV_H, lte.DW_CONV_W):
            elapsed_width = self.logic_region.w * util.idivc(self.workload["Yi"], self.logic_region.w)
            if elapsed_width // self.logic_region.w > 1 and \
               all(util.gcd(elapsed_width // self.logic_region.w, node_dim) == 1 for node_dim in self.physic_node_dim):
                cand = self.logic_region.w - 1
                while cand > (self.logic_region.w * 0.85):
                    cand_upper_iter = util.idivc(self.workload["Yi"], cand)
                    if any(util.gcd(cand_upper_iter, node_dim) > 1 for node_dim in self.physic_node_dim):
                        self.logic_region = PhyDim2(self.logic_region.h, cand)
                        break
                    cand -= 1
            loopcnt = dict()
            for dim in NN_DIM_LIST:
                loopcnt[dim] = 1
            loopcnt["N"] = self.workload["N"]
            loopcnt["C"] = self.workload["C"]
            loopcnt["K"] = 1
            # Fow row-stationary, it's always best to iterate all Xi in each PE, so that the window
            # reuse can be fully utilized. So there is no loop iteration outside array mapping.
            loopcnt["Xi"] = 1
            loopcnt["Yi"] = util.idivc(self.workload["Yi"], self.logic_region.w)

            gbuf_unit_tensor = dict()
            gbuf_unit_tensor["R"] = self.workload["R"]
            gbuf_unit_tensor["S"] = self.workload["S"]
            # Fully buffer Xi to exploit window reuse.
            gbuf_unit_tensor["Xi"] = self.workload["Xi"]
            gbuf_unit_tensor["Yi"] = self.logic_region.w

            regf_unit_tensor = dict()
            regf_unit_tensor["R"] = self.workload["R"]
            regf_unit_tensor["Xo"] = self.workload["R"]

            base_stacks = []
            base_stacks.append(("Yi", 1, "Yo", self.conv_strds[1], self.logic_region.w))
            base_stacks.append(("S", 1, "Yo", 1, self.logic_region.h))

            base_updates = []
            base_updates.append(("Xi", 1, "Xo", self.conv_strds[0], 1))
            if self.fold.w > 1:
                base_updates.append(("Yi", self.logic_region.w, "Yo", self.conv_strds[1] * self.logic_region.w))
            if self.fold.h > 1:
                base_updates.append(("S", self.logic_region.h, "Yo", self.logic_region.h))

            unit_ops = self.workload["R"]
        else:
            raise ValueError("Invalid layer type: {}".format(self.layer_type))

        origin_stack_step_dict = None

        return loopcnt, regf_repls, regf_unit_tensor, gbuf_unit_tensor, base_stacks, \
               base_updates, origin_stack_step_dict, unit_ops
