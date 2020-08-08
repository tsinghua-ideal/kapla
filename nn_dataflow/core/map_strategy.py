""" $lic$
Copyright (C) 2016-2020 by Tsinghua University and The Board of Trustees of
Stanford University

This program is free software: you can redistribute it and/or modify it under
the terms of the Modified BSD-3 License as published by the Open Source
Initiative.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the BSD-3 License for more details.

You should have received a copy of the Modified BSD-3 License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>.
"""

from . import data_category_enum as de
from . import loop_enum as le
from . import mem_hier_enum as me
from .. import util
from .layer import Layer, ConvLayer, LocalRegionLayer, ConvBackLayer, LocalRegionBackLayer
from .nested_loop_desc import NestedLoopDesc
from .phy_dim2 import PhyDim2

class MapStrategy():
    '''
    Base mapping strategy.

    Map is the procedure to map the 2D convolution computation onto the 2D PE
    array.
    '''

    def __init__(self, layer, batch_size, occupancy, dim_array, reverse_mapping=False):
        if not isinstance(layer, Layer):
            raise TypeError('MapStrategy: layer must be a Layer object.')
        if not 0 < occupancy <= 1:
            raise ValueError('MapStrategy: occupancy must be between 0 and 1.')
        if not isinstance(dim_array, PhyDim2):
            raise TypeError('MapStrategy: dim_array must be a PhyDim2 object.')
        self.layer = layer
        self.batch_size = batch_size
        self.occupancy = occupancy
        self.dim_array = dim_array
        self.reverse_mapping = reverse_mapping
        self._regf_reusable = [False for _ in range(de.NUM)]

    def utilization(self):
        '''
        PE utilization, i.e., average percentage of active PEs.
        '''
        raise NotImplementedError('MapStrategy: derived class must overwrite.')

    def gen_nested_loop_desc(self):
        '''
        Generate all the NestedLoopDesc objects after mapping.
        '''
        raise NotImplementedError('MapStrategy: derived class must overwrite.')


class MapStrategyEyeriss(MapStrategy):
    '''
    Eyeriss mapping scheme, a.k.a, Row-Stationary.

    Chen, Emer, and Sze, ISCA'16.
    '''
    # pylint: disable=too-many-instance-attributes

    def __init__(self, layer, batch_size, occupancy, dim_array, reverse_mapping=False):

        super(MapStrategyEyeriss, self).__init__(layer, batch_size, occupancy,
                                                 dim_array, reverse_mapping)

        # Logic PE set.
        if isinstance(self.layer, ConvLayer):
            # Conv and FC layers.
            self.ops_lpe = self.layer.wfil * self.layer.wofm
            self.dim_lpeset = PhyDim2(self.layer.hfil, self.layer.hofm)
            cnt_lpeset = self.batch_size * self.layer.nofm * self.layer.nifm
        elif isinstance(self.layer, LocalRegionLayer):
            self.ops_lpe = self.layer.nreg * self.layer.wreg * self.layer.wofm
            self.dim_lpeset = PhyDim2(h=self.layer.hreg, w=self.layer.hofm)
            cnt_lpeset = self.batch_size * self.layer.nofm
        elif isinstance(self.layer, ConvBackLayer):
            self.ops_lpe = self.layer.wfil * self.layer.wifm
            self.dim_lpeset = PhyDim2(self.layer.hfil, self.layer.hifm)
            cnt_lpeset = self.batch_size * self.layer.nifm * self.layer.nofm
        elif isinstance(self.layer, LocalRegionBackLayer):
            self.ops_lpe = self.layer.nreg * self.layer.wreg * self.layer.wifm
            self.dim_lpeset = PhyDim2(h=self.layer.hreg, w=self.layer.hifm)
            cnt_lpeset = self.batch_size * self.layer.nifm
        else:
            raise TypeError('MapEyeriss: unrecognized layer type {}.'
                            .format(type(self.layer)))

        ops_logic_total = self.ops_lpe * self.dim_lpeset.size() * cnt_lpeset
        if isinstance(layer, ConvBackLayer):
            print("batch_size: {}".format(self.batch_size))
            print("wfil: {}, wofm: {}".format(self.layer.wfil, self.layer.wofm))
            print("dim_lpeset: {}".format(self.dim_lpeset))
            print("cnt_lpeset: {}".format(cnt_lpeset))
            print("ops_logic_total: {}".format(ops_logic_total))
            print("total_ops: {}".format(self.layer.total_ops(self.batch_size)))
        assert ops_logic_total == self.layer.total_ops(self.batch_size)

        # Physical PE set through replication and folding.
        self._repl_fold()

        # PE utilization.
        # We replicate repl.size() lpesets, and then fold to fold.size()
        # physical array passes.
        self.util = 1. * (self.dim_lpeset.size() * self.repl.size()) \
                / (self.dim_array.size() * self.fold.size())
        assert self.util <= 1. + 1e-6

        assert self.util > 0.5, \
                ('MapEyeriss: PE array resource utilization < 50%. '
                 'Physical PE set {}; array size {}; logic PE set {}; '
                 'folded logic PE set {}. Can\'t we fit more?'
                 .format(self.dim_ppeset, self.dim_array,
                         self.dim_lpeset, self.dim_flpeset))

    def utilization(self):
        return self.util

    def gen_nested_loop_desc(self):
        '''
        Replication and folding:

        - ConvLayer
          - repl.w is only used for ofm; repl.h can be shared by ifm and ofm.
          - fold.h folds fil, which uses different filter parts but same fmaps,
            so do them continuously (innermost loop); fold.w folds fmaps, which
            uses different fmap parts but same filters, so merge into batch.

        - LocalRegionLayer
          - repl is only used for ofm, since ofm/ifm relationship is one-to-one
            rather than all-to-all.
          - fold.h means each 2D region is too high, and needs multiple ppesets
            to process, so do them continuously (innermost loop); fold.w
            divides 2D fmaps into multiple parts that can be processed
            independently, so merge into batch.

        Terminologies:

        - flpeset: folded lpeset, a fraction of lpeset after folding and before
          replication. If a lpeset is divided into separated segments, i.e.,
          using replication first for folding (see Chen, et al, JSSC'17, IV.A.
          pp5), flpeset is the original shape without separating segments.

        - ppeset: physical peset, one physical array pass, replicated flpeset.
          It should fit in the physical array shape. We do not use ppeset in
          mapping.

        - unitpass: unit pass, fold.h flpesets before replication, which deals
          with all folded fils, one folded ifm and one folded ofm. This is one
          full 2D convolution given that the folded ifms/ofms have been merged
          into batch.

          # flpesets in a unitpass = fold.h
          (temporally)

        - procpass: processing pass, unit pass after replication with
          repl.size(). A procpass includes all replication. See Chen, et al,
          ISCA'16, end of V.B.

          # unitpasses in a procpass = repl.size()
          (spatially)

        We first calculate execution stats (ops, time, accesses, sizes) for one
        unit pass, then consider different replications to build a procpass.
        Processing pass is the unit for loop blocking, i.e., the innermost loop
        processes one procpass. So the unit ops/time/accesses are calculated on
        procpass unit.

        Fragmentation:

        When the layer shape parameters are not multipliers of the physical
        dimensions, there are fragmentation. We consider two types of
        fragmentation:

        - ppeset internal fragmentation: due to lpeset folding. E.g., folding
          27 rows by 2 results in two 14 rows, and 1 row is not used.

        - loop occupancy: due to partial full loops. E.g., for total of 32
          ifmaps, if each loop body processes 3 ifmaps, we need 11 ifmap loops,
          but the last one only has 2 ifmaps rather than 3.
        '''

        # Folded filters is not allowed in the original Eyeriss, but we extend
        # to support it with the unit pass concept.
        ops_unitpass, time_unitpass, access_unitpass, \
                sz_gbuf_unitpass, sz_regf_unitpass, amp_acc_ifm = \
                self._calc_unitpass()

        data_loops = self.layer.data_loops()

        # Apply replication.
        for lcnt, locc, rsz, rcnt in self._gen_repl():

            # Number of ops.
            # Replicate to procpass. Also consider external occupancy and loop
            # occupancies.
            unit_ops = ops_unitpass * rsz * self.occupancy * util.prod(locc)

            # Time does not change with replication, and is not affected by
            # loop occupancy.
            unit_time = time_unitpass

            # Buffered data size.
            # Replication uses the single gbuf.
            usize_gbuf = tuple(s * n for s, n in zip(sz_gbuf_unitpass, rcnt))
            # Replication uses different PEs.
            usize_regf = tuple(sz_regf_unitpass)

            # Unit access, i.e., data accesses for one processing pass.
            # Replicate to procpass. Also consider loop occupancies.
            uaccess = [tuple() for _ in range(me.NUM)]
            # Loop occupancies affect accesses.
            aocc = [util.prod(data_loops[dce].take(locc))
                    for dce in range(de.NUM)]
            # Replication uses the single DRAM, gbuf.
            for mhe in [me.DRAM, me.GBUF]:
                uaccess[mhe] = tuple(a * n * o for a, n, o
                                     in zip(access_unitpass[mhe], rcnt, aocc))
            # Itcn access is replicated across all PEs.
            uaccess[me.ITCN] = tuple(a * rsz * o for a, o
                                     in zip(access_unitpass[me.ITCN], aocc))
            # Replication uses different PEs. regf scales with op replication,
            # i.e., affected by all loop occupancies. Also consider external
            # occupancy.
            uaccess[me.REGF] = tuple(a * rsz * self.occupancy * util.prod(locc)
                                     for a in access_unitpass[me.REGF])
            # Finalize.
            unit_access = tuple(uaccess)
            _regf_reusable = tuple(self._regf_reusable)

            # Make nested loop desc.
            nld = NestedLoopDesc(loopcnt=lcnt, unit_access=unit_access,
                                 usize_gbuf=usize_gbuf, usize_regf=usize_regf,
                                 unit_ops=unit_ops, unit_time=unit_time,
                                 data_loops=data_loops,
                                 regf_reusable=_regf_reusable)

            # Check num of ops.
            util.assert_float_eq_int(
                nld.total_ops(),
                self.layer.total_ops(self.batch_size) * self.occupancy,
                'MapEyeriss: total number of physical ops is incorrect.')

            # Check unit access.
            util.assert_float_eq_int(
                nld.total_access_at_of(me.DRAM, de.FIL),
                self.layer.total_filter_size()
                if isinstance(self.layer, (ConvLayer, ConvBackLayer)) else 0,
                'MapEyeriss: total access at DRAM for FIL {} is incorrect.'
                .format(nld.total_access_at_of(me.DRAM, de.FIL)))
            util.assert_float_eq_int(
                # Need to consider amplified access for IFM.
                nld.total_access_at_of(me.DRAM, de.IFM) / amp_acc_ifm,
                self.layer.total_ifmap_size(self.batch_size),
                'MapEyeriss: total access at DRAM for IFM {} is incorrect.'
                .format(nld.total_access_at_of(me.DRAM, de.IFM)))
            util.assert_float_eq_int(
                nld.total_access_at_of(me.DRAM, de.OFM),
                self.layer.total_ofmap_size(self.batch_size),
                'MapEyeriss: total access at DRAM for OFM {} is incorrect.'
                .format(nld.total_access_at_of(me.DRAM, de.OFM)))
            util.assert_float_eq_int(
                nld.unit_access_at_of(me.REGF, de.FIL) * util.prod(nld.loopcnt),
                self.layer.total_ops(self.batch_size) * self.occupancy
                if isinstance(self.layer, (ConvLayer, ConvBackLayer)) else 0,
                'MapEyeriss: unit access at REGF for FIL {} is incorrect.'
                .format(nld.unit_access_at_of(me.REGF)))
            util.assert_float_eq_int(
                nld.unit_access_at_of(me.REGF, de.IFM) * util.prod(nld.loopcnt),
                self.layer.total_ops(self.batch_size) * self.occupancy,
                'MapEyeriss: unit access at REGF for IFM {} is incorrect.'
                .format(nld.unit_access_at_of(me.REGF)))
            util.assert_float_eq_int(
                nld.unit_access_at_of(me.REGF, de.OFM) * util.prod(nld.loopcnt),
                self.layer.total_ops(self.batch_size) * self.occupancy,
                'MapEyeriss: unit access at REGF for OFM {} is incorrect.'
                .format(nld.unit_access_at_of(me.REGF)))

            if self.reverse_mapping:
                usize_gbuf = list(nld.usize_gbuf)
                usize_regf = list(nld.usize_regf)
                loopcnt = list(nld.loopcnt)
                unit_access = [list(ua) for ua in nld.unit_access]
                usize_gbuf[de.IFM], usize_gbuf[de.OFM] = nld.usize_gbuf[de.OFM], nld.usize_gbuf[de.IFM]
                usize_regf[de.IFM], usize_regf[de.OFM] = nld.usize_regf[de.OFM], nld.usize_regf[de.IFM]
                loopcnt[le.IFM], loopcnt[le.OFM] = nld.loopcnt[le.OFM], nld.loopcnt[de.IFM]

                for m in range(me.NUM):
                    unit_access[m][de.IFM], unit_access[m][de.OFM] = nld.unit_access[m][de.OFM], nld.unit_access[m][de.IFM]
                nld._replace(usize_gbuf=tuple(usize_gbuf),
                             usize_regf=tuple(usize_regf),
                             loopcnt=tuple(loopcnt),
                             unit_access=tuple(tuple(ua) for ua in unit_access))

            yield nld

    def _repl_fold(self):
        '''
        Find the replication and folding factors from logic PE set to physical
        array.
        '''
        fold_w = 1
        repl_w = 1
        fold_h = 1
        repl_h = 1

        if self.dim_lpeset.h > self.dim_array.h:
            # Fold on height.
            fold_h = util.idivc(self.dim_lpeset.h, self.dim_array.h)
        else:
            # Replicate on height.
            repl_h = self.dim_array.h // self.dim_lpeset.h
        if self.dim_lpeset.w > self.dim_array.w:
            # Fold on width.
            fold_w = util.idivc(self.dim_lpeset.w, self.dim_array.w)
        else:
            # Replicate on with.
            repl_w = self.dim_array.w // self.dim_lpeset.w

        # Adjust fold and repl, use repl_h to first schedule fold_w.
        # The factor of putting fold_w to repl_h (w to h) is the smaller of the
        # two. Either repl_h cannot accommodate all fold_w, still fold_w; or
        # repl_h has accommodated all fold_w, remain repl_h.
        f_w2h = min(repl_h, fold_w)
        fold_w = util.idivc(fold_w, f_w2h)
        repl_h = repl_h // f_w2h

        # The replication and folding factors for lpeset, considering the
        # adjustment.
        self.fold = PhyDim2(fold_h, fold_w)
        self.repl = PhyDim2(repl_h, repl_w)

        # The folded lpeset size on the ppeset after adjustment. The width may
        # be larger than the array width, but it is actually broken into the
        # height replication.
        self.dim_flpeset = PhyDim2(util.idivc(self.dim_lpeset.h, self.fold.h),
                                   util.idivc(self.dim_lpeset.w, self.fold.w))

        # The physical ppeset size, should fit in the array.
        self.dim_ppeset = PhyDim2(self.dim_flpeset.h * self.repl.h * f_w2h,
                                  util.idivc(self.dim_flpeset.w * self.repl.w,
                                             f_w2h))

        assert (self.dim_ppeset.h <= self.dim_array.h
                and self.dim_ppeset.w <= self.dim_array.w), \
            'MapEyeriss: dim_ppeset {} does not fit in dim_array {}.' \
            .format(self.dim_ppeset, self.dim_array)

    def _calc_unitpass(self):
        '''
        Calculate the ops, time, accessed data size, and buffered data size for
        one unit pass.

        Ops considers ppeset internal fragmentation.

        Time is the maximum value that each unit pass needs.

        Accessed size considers ppeset internal fragmentation.

        Buffered size is the maximum value that buffer needs to support.

        Return ops, time, accessed size for all hierarchies, and buffered size
        in gbuf and regf. Also return the amplified access ratio for ifmaps.
        '''
        ops = float('nan')
        time = float('nan')
        access = [[float('nan')] * de.NUM for _ in range(me.NUM)]
        sz_gbuf = [float('nan')] * de.NUM
        sz_regf = [float('nan')] * de.NUM

        flpesets_per_unitpass = self.fold.h

        if isinstance(self.layer, ConvLayer):

            # A unitpass processes all folded fils, and one folded ifm/ofm.
            # Row size is not affected since a row is within one PE.
            acclayer = ConvLayer(
                1, 1,
                (1. * self.layer.hofm / self.fold.w, self.layer.wofm),
                (self.layer.hfil, self.layer.wfil),
                strd=(self.layer.htrd, self.layer.wtrd))
            buflayer = ConvLayer(
                1, 1,
                (util.idivc(self.layer.hofm, self.fold.w), self.layer.wofm),
                (self.layer.hfil, self.layer.wfil),
                strd=(self.layer.htrd, self.layer.wtrd))

            ops = acclayer.total_ops()

            time = flpesets_per_unitpass * self.ops_lpe

            # Data are accessed once from DRAM into gbuf.
            access[me.DRAM][de.FIL] = acclayer.total_filter_size()
            access[me.DRAM][de.IFM] = acclayer.total_ifmap_size()
            access[me.DRAM][de.OFM] = acclayer.total_ofmap_size()

            # To iterate all folded fils over the fmaps, we have two choices
            # for ConvLayer:
            # a) only store one folded fil in regf and access fmaps multiple
            # times from gbuf;
            # b) store all folded fils in regf and only access fmaps once from
            # gbuf.
            # To save regf size, we choose a).
            access[me.GBUF][de.FIL] = access[me.DRAM][de.FIL]
            access[me.GBUF][de.IFM] = access[me.DRAM][de.IFM] \
                    * flpesets_per_unitpass
            access[me.GBUF][de.OFM] = access[me.DRAM][de.OFM] \
                    * flpesets_per_unitpass

            # All data from/to regf go through itcn.
            # Data per PE * number of PEs * number of rounds (flpsets).
            access[me.ITCN][de.FIL] = acclayer.wfil * self.dim_flpeset.size() \
                    * flpesets_per_unitpass
            access[me.ITCN][de.IFM] = acclayer.wifm * self.dim_flpeset.size() \
                    * flpesets_per_unitpass
            access[me.ITCN][de.OFM] = acclayer.wofm * self.dim_flpeset.size() \
                    * flpesets_per_unitpass

            # regf access is based on num of ops.
            access[me.REGF] = [ops] * de.NUM

            sz_gbuf[de.FIL] = buflayer.total_filter_size()
            sz_gbuf[de.IFM] = buflayer.total_ifmap_size()
            sz_gbuf[de.OFM] = buflayer.total_ofmap_size()

            # Entire fil row of one folded fil per PE.
            sz_regf[de.FIL] = buflayer.wfil
            # For 1D conv in each PE, ifm and ofm are both accessed in a
            # streaming fashion (sliding window). Only capturing wfil ifm
            # elements and 1 ofm element is adequate.
            sz_regf[de.IFM] = buflayer.wfil
            sz_regf[de.OFM] = 1

            # Since we choose to only store a sliding window of data, the
            # data may be fetched multipled times.
            self._regf_reusable[de.IFM] = (buflayer.wfil == buflayer.hifm)
            self._regf_reusable[de.OFM] = (buflayer.wofm == 1)
            self._regf_reusable[de.FIL] = (self.fold.h == 1)

        elif isinstance(self.layer, LocalRegionLayer):
            # A unitpass processes all folded regions, and one folded ifm/ofm.
            # Row size is not affected since a row is within one PE.
            acclayer = LocalRegionLayer(
                1,
                (1. * self.layer.hofm / self.fold.w, self.layer.wofm),
                self.layer.nreg, (self.layer.hreg, self.layer.wreg),
                strd=(self.layer.htrd, self.layer.wtrd))

            buflayer = LocalRegionLayer(
                1,
                (util.idivc(self.layer.hofm, self.fold.w), self.layer.wofm),
                self.layer.nreg, (self.layer.hreg, self.layer.wreg),
                strd=(self.layer.htrd, self.layer.wtrd))

            ops = acclayer.total_ops()

            time = flpesets_per_unitpass * self.ops_lpe

            # Data are accessed once from DRAM into gbuf.
            access[me.DRAM][de.FIL] = 0
            access[me.DRAM][de.IFM] = acclayer.total_ifmap_size()
            access[me.DRAM][de.OFM] = acclayer.total_ofmap_size()

            # For LocalRegionLayer or LocalRegionBackLayer, ofm needs to access multiple times, each
            # with a different ifm range.
            access[me.GBUF][de.FIL] = 0
            access[me.GBUF][de.IFM] = access[me.DRAM][de.IFM]
            access[me.GBUF][de.OFM] = access[me.DRAM][de.OFM] \
                    * flpesets_per_unitpass

            # All data from/to regf go through itcn.
            # Data per PE * number of PEs * number of rounds (flpsets).
            access[me.ITCN][de.FIL] = 0
            access[me.ITCN][de.IFM] = acclayer.wifm * self.dim_flpeset.size()
            access[me.ITCN][de.OFM] = acclayer.wofm * self.dim_flpeset.size() \
                    * flpesets_per_unitpass

            # regf access is based on num of ops.
            access[me.REGF][de.FIL] = 0
            access[me.REGF][de.IFM] = ops
            access[me.REGF][de.OFM] = ops

            sz_gbuf[de.FIL] = 0
            sz_gbuf[de.IFM] = buflayer.total_ifmap_size()
            sz_gbuf[de.OFM] = buflayer.total_ofmap_size()

            sz_regf[de.FIL] = 0
            # In each PE, ifm and ofm are both accessed in a streaming fashion
            # (sliding window). Only capturing wreg ifm elements and 1 ofm
            # element is adequate.
            sz_regf[de.IFM] = buflayer.wreg * buflayer.nreg
            sz_regf[de.OFM] = 1

            # Since we choose to only store a sliding window of data, the
            # data may be fetched multipled times.
            self._regf_reusable[de.IFM] = (buflayer.wreg == buflayer.hifm)
            self._regf_reusable[de.OFM] = (buflayer.wofm == 1)
            self._regf_reusable[de.FIL] = (self.fold.h == 1)

        elif isinstance(self.layer, ConvBackLayer):
            # A unitpass processes all folded fils, and one folded ifm/ofm.
            # Row size is not affected sicne a row is within one PE.
            acclayer = ConvLayer(
                1, 1,
                (1. * self.layer.hifm / self.fold.w, self.layer.wifm),
                (self.layer.hfil, self.layer.wfil),
                strd=(self.layer.htrd, self.layer.wtrd))
            buflayer = ConvLayer(
                1, 1,
                (util.idivc(self.layer.hifm, self.fold.w), self.layer.wifm),
                (self.layer.hfil, self.layer.wfil),
                strd=(self.layer.htrd, self.layer.wtrd))

            ops = acclayer.total_ops()

            time = flpesets_per_unitpass * self.ops_lpe

            # Data are accessed once from DRAM into gbuf.
            access[me.DRAM][de.FIL] = acclayer.total_filter_size()
            access[me.DRAM][de.IFM] = acclayer.total_ofmap_size()
            access[me.DRAM][de.OFM] = acclayer.total_ifmap_size()

            access[me.GBUF][de.FIL] = access[me.DRAM][de.FIL]
            access[me.GBUF][de.IFM] = access[me.DRAM][de.IFM] * flpesets_per_unitpass
            access[me.GBUF][de.OFM] = access[me.DRAM][de.OFM] * flpesets_per_unitpass

            access[me.ITCN][de.FIL] = acclayer.wfil * self.dim_flpeset.size() * flpesets_per_unitpass
            access[me.ITCN][de.IFM] = acclayer.wofm * self.dim_flpeset.size() * flpesets_per_unitpass
            access[me.ITCN][de.OFM] = acclayer.wifm * self.dim_flpeset.size() * flpesets_per_unitpass

            access[me.REGF] = [ops] * de.NUM

            sz_gbuf[de.FIL] = buflayer.total_filter_size()
            sz_gbuf[de.IFM] = buflayer.total_ofmap_size()
            sz_gbuf[de.OFM] = buflayer.total_ifmap_size()

            sz_regf[de.FIL] = buflayer.wfil
            sz_regf[de.IFM] = 1
            sz_regf[de.OFM] = buflayer.wfil

        elif isinstance(self.layer, LocalRegionBackLayer):
            # A unitpass processes all folded regions, and one folded ifm/ofm.
            # Row size is not affected since a row is within one PE.
            acclayer = LocalRegionLayer(
                1,
                (1. * self.layer.hifm / self.fold.w, self.layer.wifm),
                self.layer.nreg, (self.layer.hreg, self.layer.wreg),
                strd=(self.layer.htrd, self.layer.wtrd))

            buflayer = LocalRegionLayer(
                1,
                (util.idivc(self.layer.hifm, self.fold.w), self.layer.wifm),
                self.layer.nreg, (self.layer.hreg, self.layer.wreg),
                strd=(self.layer.htrd, self.layer.wtrd))

            ops = acclayer.total_ops()

            time = flpesets_per_unitpass * self.ops_lpe

            # Data are accessed once from DRAM into gbuf.
            access[me.DRAM][de.FIL] = 0
            access[me.DRAM][de.IFM] = acclayer.total_ofmap_size()
            access[me.DRAM][de.OFM] = acclayer.total_ifmap_size()

            # For LocalRegionLayer or LocalRegionBackLayer, ofm needs to access multiple times, each
            # with a different ifm range.
            access[me.GBUF][de.FIL] = 0
            access[me.GBUF][de.IFM] = access[me.DRAM][de.IFM]
            access[me.GBUF][de.OFM] = access[me.DRAM][de.OFM] \
                    * flpesets_per_unitpass

            # All data from/to regf go through itcn.
            # Data per PE * number of PEs * number of rounds (flpsets).
            access[me.ITCN][de.FIL] = 0
            access[me.ITCN][de.IFM] = acclayer.wofm * self.dim_flpeset.size()
            access[me.ITCN][de.OFM] = acclayer.wifm * self.dim_flpeset.size() \
                    * flpesets_per_unitpass

            # regf access is based on num of ops.
            access[me.REGF][de.FIL] = 0
            access[me.REGF][de.IFM] = ops
            access[me.REGF][de.OFM] = ops

            sz_gbuf[de.FIL] = 0
            sz_gbuf[de.IFM] = buflayer.total_ofmap_size()
            sz_gbuf[de.OFM] = buflayer.total_ifmap_size()

            sz_regf[de.FIL] = 0
            # In each PE, ifm and ofm are both accessed in a streaming fashion
            # (sliding window). Only capturing wreg ifm elements and 1 ofm
            # element is adequate.
            sz_regf[de.IFM] = 1
            sz_regf[de.OFM] = buflayer.wreg * buflayer.nreg

        else:
            raise TypeError("map_strategy: Invalid layer type: {}".format(self.layer.__class__.__name__))

        # All utilized PEs run `time` to execute replicated `ops`
        assert util.isclose(time * self.dim_array.size() * self.util,
                            ops * self.repl.size(),
                            abs_tol=1e-3)

        # Due to folding, the overlapping ifmaps may need to be re-fetched,
        # resulting in amplified access for ifmaps.
        # Consider one flpeset, hifm rows are folded by fold.w.
        amp_acc_ifm = 1. * acclayer.hifm * self.fold.w / self.layer.hifm

        return ops, time, access, sz_gbuf, sz_regf, amp_acc_ifm

    def _gen_repl(self):
        '''
        Generate all replication with ifmaps/ofmaps, to build procpass from
        unitpass.

        Return the total loop count tuple, the loop occupancy list, the actual
        replication size, and the replicated data counts.
        '''
        if isinstance(self.layer, ConvLayer):

            # repl.w is only used for ofmaps, and repl.h can be used either for
            # ifmaps or ofmaps.

            # Loop body unit time is constant w.r.t. the split of repl.h, so we
            # pick the smallest total number of loops.
            min_cnt_loops = float('inf')

            for t_repl_h in util.factorize(self.repl.h, 2):

                ifms = t_repl_h[0]
                ofms = t_repl_h[1] * self.repl.w

                ifms = min(ifms, self.layer.nifm)
                ofms = min(ofms, self.layer.nofm)
                repl_size = ifms * ofms

                # Loop trip counts.
                lcnt = [float('nan')] * le.NUM
                lcnt[le.IFM] = util.idivc(self.layer.nifm, ifms)
                lcnt[le.OFM] = util.idivc(self.layer.nofm, ofms)
                # fold.w is equivalent to increasing batch size.
                lcnt[le.BAT] = self.batch_size * self.fold.w

                cnt_loops = util.prod(lcnt)
                if cnt_loops < min_cnt_loops:
                    min_cnt_loops = cnt_loops
                elif cnt_loops > min_cnt_loops:
                    continue

                # Loop occupancy.
                locc = [1.] * le.NUM
                locc[le.IFM] = 1. * self.layer.nifm / ifms / lcnt[le.IFM]
                locc[le.OFM] = 1. * self.layer.nofm / ofms / lcnt[le.OFM]

                # Replicated data counts.
                repl_cnt = [0] * de.NUM
                repl_cnt[de.FIL] = ifms * ofms
                repl_cnt[de.IFM] = ifms
                repl_cnt[de.OFM] = ofms

                yield tuple(lcnt), locc, repl_size, repl_cnt

        elif isinstance(self.layer, (LocalRegionLayer, LocalRegionBackLayer)):

            # repl is only used for ofmaps.
            ofms = self.repl.size()

            ofms = min(ofms, self.layer.nofm)
            repl_size = ofms

            # Loop trip counts.
            lcnt = [float('nan')] * le.NUM
            # Loop ifm is corresponding to loop ofm, so always 1.
            lcnt[le.IFM] = 1
            lcnt[le.OFM] = util.idivc(self.layer.nofm, ofms)
            # fold.w is equivalent to increasing batch size.
            lcnt[le.BAT] = self.batch_size * self.fold.w

            # Loop occupancy.
            locc = [1.] * le.NUM
            locc[le.OFM] = 1. * self.layer.nofm / ofms / lcnt[le.OFM]

            # Replicated data counts.
            repl_cnt = [0] * de.NUM
            repl_cnt[de.FIL] = 0
            repl_cnt[de.IFM] = ofms  # ifm and ofm is one-to-one.
            repl_cnt[de.OFM] = ofms

            yield tuple(lcnt), locc, repl_size, repl_cnt

        elif isinstance(self.layer, ConvBackLayer):
            # repl.w is only used for ifmaps, and repl.h can be used either for ifmaps or ofmaps.

            min_cnt_loops = float('inf')

            for t_repl_h in util.factorize(self.repl.h, 2):
                ifms = t_repl_h[0] * self.repl.w
                ofms = t_repl_h[1]

                ifms = min(ifms, self.layer.nifm)
                ofms = min(ofms, self.layer.nofm)
                repl_size = ifms * ofms

                lcnt = [float('nan')] * le.NUM
                lcnt[le.IFM] = util.idivc(self.layer.nifm, ifms)
                lcnt[le.OFM] = util.idivc(self.layer.nofm, ofms)
                lcnt[le.BAT] = self.batch_size * self.fold.w

                cnt_loops = util.prod(lcnt)
                if cnt_loops < min_cnt_loops:
                    min_cnt_loops = cnt_loops
                elif cnt_loops > min_cnt_loops:
                    continue

                locc = [1.] * le.NUM
                locc[le.IFM] = 1. * self.layer.nifm / ifms / lcnt[le.IFM]
                locc[le.OFM] = 1. * self.layer.nofm / ofms / lcnt[le.OFM]

                repl_cnt = [0] * de.NUM
                repl_cnt[de.FIL] = ifms * ofms
                repl_cnt[de.IFM] = ifms
                repl_cnt[de.OFM] = ofms

                yield tuple(lcnt), locc, repl_size, repl_cnt
        else:
            raise TypeError("map_strategy: Invalid layer type: {}".format(self.layer.__class__.__name__))
