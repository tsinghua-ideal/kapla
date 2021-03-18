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
from .. import util
from .data_dim_loops import DataDimLoops

class Layer(util.ContentHashClass):
    '''
    Base NN layer.

    Includes only the output neuron parameters.

    nofm: # ofmap channels
    hofm, wofm: ofmap height/width
    htrd, wtrd: stride height/width
    '''

    def __init__(self, nofm, sofm, strd=1):
        if isinstance(sofm, int):
            hofm = sofm
            wofm = sofm
        elif len(sofm) == 2:
            hofm = sofm[0]
            wofm = sofm[1]
        else:
            raise ValueError('Layer: sofm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sofm))
        assert hofm > 0 and wofm > 0

        if isinstance(strd, int):
            htrd = strd
            wtrd = strd
        elif len(strd) == 2:
            htrd = strd[0]
            wtrd = strd[1]
        else:
            raise ValueError('Layer: strd is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(strd))
        assert htrd > 0 and wtrd > 0

        self.nofm = nofm
        self.hofm = hofm
        self.wofm = wofm

        self.htrd = htrd
        self.wtrd = wtrd

    @staticmethod
    def data_loops():
        ''' Dimension loops of the data. '''
        raise NotImplementedError

    def input_layer(self):
        ''' Get the input layer parameters. '''
        raise NotImplementedError(self.__class__.__name__)

    @property
    def nifm(self):
        ''' Number of fmap channels of input layer. '''
        return self.input_layer().nofm

    @property
    def hifm(self):
        ''' Fmap height of input layer. '''
        return self.input_layer().hofm

    @property
    def wifm(self):
        ''' Fmap width of input layer. '''
        return self.input_layer().wofm

    def ofmap_size(self, batch_size=1, word_size=1):
        '''
        Get size of one output fmap with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hofm * self.wofm * batch_size * word_size

    def total_ofmap_size(self, batch_size=1, word_size=1):
        '''
        Get total size of all output fmaps with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nofm * self.ofmap_size(batch_size, word_size)

    def ifmap_size(self, batch_size=1, word_size=1):
        '''
        Get size of one input fmap with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().ofmap_size(batch_size, word_size)

    def total_ifmap_size(self, batch_size=1, word_size=1):
        '''
        Get total size of all input fmaps with `batch_size`.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.input_layer().total_ofmap_size(batch_size, word_size)

    # def ops_per_neuron(self):
    #     ''' Number of operations per neuron. '''
    #     raise NotImplementedError(self.__class__.__name__)

    def total_ops(self, batch_size=1):
        ''' Get total number of operations. '''
        if callable(getattr(self, "ops_per_neuron", None)):
            return self.total_ofmap_size() * self.ops_per_neuron() * batch_size
        else:
            raise NotImplementedError(self.__class__.__name__)

    def is_valid_padding_sifm(self, sifm):
        ''' Whether the given `sifm` is valid when allowing padding. '''
        if isinstance(sifm, int):
            hifm = sifm
            wifm = sifm
        elif len(sifm) == 2:
            hifm = sifm[0]
            wifm = sifm[1]
        else:
            raise ValueError('Layer: sifm is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sifm))

        h_padding_rng = sorted((self.hofm * self.htrd, self.hifm))
        w_padding_rng = sorted((self.wofm * self.wtrd, self.wifm))
        return (h_padding_rng[0] <= hifm <= h_padding_rng[1]
                and w_padding_rng[0] <= wifm <= w_padding_rng[1])

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class InputLayer(Layer):
    '''
    NN input layer parameters.
    '''

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops()
        dls[de.IFM] = DataDimLoops()
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return None

    def ops_per_neuron(self):
        return 0


class ConvLayer(Layer):
    '''
    NN convolutional layer parameters.

    nifm (C): # ifmap channels
    nofm (M): # ofmap channels
    hifm, wifm (H): ifmap height/width
    hofm, wofm (E): ofmap height/width
    hfil, wfil (R): weight filter width/height
    htrd, wtrd (U): stride height/width
    '''

    def __init__(self, nifm, nofm, sofm, sfil, strd=1, rw_data=de.OFM):
        super(ConvLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('ConvLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil

        hifm = self.hfil + (self.hofm - 1) * self.htrd
        wifm = self.wfil + (self.wofm - 1) * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = rw_data

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.IFM, le.OFM)
        dls[de.IFM] = DataDimLoops(le.IFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # 2D convolution across all ifmap channels.
        return self.hfil * self.wfil * self.nifm

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class FCLayer(ConvLayer):
    '''
    NN fully-connected layer parameters.

    As a special case of CONVLayer.

    hifm = hfil, wifm = wfil, strd = 1, hofm = wofm = 1
    '''

    def __init__(self, nifm, nofm, sfil=1):
        super(FCLayer, self).__init__(nifm, nofm, 1, sfil)
        assert self.hofm == 1 and self.wofm == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class DepthwiseConvolutionLayer(Layer):
    '''
    Depthwise convolution applies a single filter to each input channel.
    Depthwise convolution is extremely efficient relative to standard convolution.
    However it only filters input channels and it does not combine them to create new features.
    '''
    def __init__(self, nofm, sofm, sfil, strd=1, rw_data=de.OFM):
        super(DepthwiseConvolutionLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            self.hfil = sfil
            self.wfil = sfil
        elif len(sfil) == 2:
            self.hfil = sfil[0]
            self.wfil = sfil[1]
        else:
            raise ValueError('DepthwiseConvolutionLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))
        nifm = self.nofm
        hifm = self.hfil + (self.hofm - 1) * self.htrd
        wifm = self.wfil + (self.wofm - 1) * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = rw_data

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.OFM)
        dls[de.IFM] = DataDimLoops(le.OFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        return self.hfil * self.wfil

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nofm * self.filter_size(word_size)


    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sreg={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class LocalRegionLayer(Layer):
    '''
    NN layer which computes on a local region. The layer has no or limited
    shared weights, whose impact can be ignored during scheduling.

    Includes pooling layer, normalization layer, and element-wise layer.
    '''

    def __init__(self, nofm, sofm, nreg, sreg, ntrd=1, strd=1, rw_data=de.OFM):
        super(LocalRegionLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sreg, int):
            hreg = sreg
            wreg = sreg
        elif len(sreg) == 2:
            hreg = sreg[0]
            wreg = sreg[1]
        else:
            raise ValueError('LocalRegionLayer: sreg is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sreg))
        if nreg > 1 and (hreg * wreg) > 1:
            raise ValueError('LocalRegionLayer: local region cannot be a mix '
                             'of both n ({}) and h & w ({}, {})'
                             .format(nreg, hreg, wreg))
        self.nreg = nreg
        self.hreg = hreg
        self.wreg = wreg
        self.ntrd = ntrd

        nifm = self.nofm * self.ntrd  # ignore all-zero padding channels.
        hifm = self.hreg + (self.hofm - 1) * self.htrd
        wifm = self.wreg + (self.wofm - 1) * self.wtrd
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = rw_data

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops()
        dls[de.IFM] = DataDimLoops(le.OFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # Each output point corresponds to merging a local region.
        return self.region_size()

    def region_size(self):
        ''' The size of the local region corresponding to one output point. '''
        return self.nreg * self.hreg * self.wreg

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg)),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'ntrd={}'.format(repr(self.ntrd)),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class PoolingLayer(LocalRegionLayer):
    '''
    NN pooling layer parameters.

    As a special case of LocalRegionLayer.

    nreg = ntrd = 1
    '''

    def __init__(self, nofm, sofm, sreg, strd=None):
        if strd is None:
            strd = sreg
        super(PoolingLayer, self).__init__(nofm, sofm, 1, sreg,
                                           ntrd=1, strd=strd)
        assert self.nreg == 1
        assert self.ntrd == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class EltwiseLayer(LocalRegionLayer):
    '''
    NN element-wise layer parameters.

    As a special case of LocalRegionLayer.

    nreg = ntrd, sreg = 1
    '''

    def __init__(self, nofm, sofm, nreg):
        super(EltwiseLayer, self).__init__(nofm, sofm, nreg, 1,
                                           ntrd=nreg, strd=1)
        assert self.hreg == self.wreg == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg))]))


# class ConvBackLayer(Layer):
#     '''
#     NN convolutional layer in back-propagation.
#     '''

#     def __init__(self, nifm, nofm, sofm, sfil, strd=1, crop=False, raw_sofm=None):
#         super(ConvBackLayer, self).__init__(nofm, sofm, strd=strd)

#         if isinstance(sfil, int):
#             hfil = sfil
#             wfil = sfil
#         elif len(sfil) == 2:
#             hfil = sfil[0]
#             wfil = sfil[1]
#         else:
#             raise ValueError('ConvLayer: sfil is invalid ({}), '
#                              'needs to be either one integer or '
#                              'a pair of integers'.format(sfil))

#         self.hfil = hfil
#         self.wfil = wfil
#         self.crop = crop
#         if crop:
#             if isinstance(raw_sofm, int):
#                 hro = raw_sofm
#                 wro = raw_sofm
#             elif len(raw_sofm) == 2:
#                 hro = raw_sofm[0]
#                 wro = raw_sofm[1]
#             else:
#                 raise ValueError('ConvLayer: sfil is invalid ({}), '
#                                  'needs to be either one integer or '
#                                  'a pair of integers'.format(sfil))
#             self.r_hofm = hro
#             self.r_wofm = wro
#             hifm = util.reverse_high(self.hofm-1, hro, self.hfil, self.htrd) + 1
#             wifm = util.reverse_high(self.wofm-1, wro, self.wfil, self.wtrd) + 1
#         else:
#             hifm = (self.hofm - self.hfil) // self.htrd + 1
#             wifm = (self.wofm - self.wfil) // self.wtrd + 1
#         self.inlayer = Layer(nifm, (hifm, wifm))

#     @staticmethod
#     def data_loops():
#         dls = [None] * de.NUM
#         dls[de.FIL] = DataDimLoops(le.IFM, le.OFM)
#         dls[de.IFM] = DataDimLoops(le.IFM, le.BAT)
#         dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
#         return tuple(dls)

#     def input_layer(self):
#         return self.inlayer

#     def total_ops(self, batch_size=1):
#         return ConvLayer(self.nofm, self.nifm, (self.hifm, self.wifm),
#                          (self.hfil, self.wfil), (self.htrd, self.wtrd)).total_ops(batch_size)

#     def filter_size(self, word_size=1):
#         '''
#         Get size of one weight filter.

#         If `word_size` is set to word byte size, return size in bytes.
#         '''
#         return self.hfil * self.wfil * word_size

#     def total_filter_size(self, word_size=1):
#         '''
#         Get total size of all weight filters.

#         If `word_size` is set to word byte size, return size in bytes.
#         '''
#         return self.nifm * self.nofm * self.filter_size(word_size)

#     def __repr__(self):
#         return '{}({})'.format(
#             self.__class__.__name__,
#             ', '.join([
#                 'nifm={}'.format(repr(self.nifm)),
#                 'nofm={}'.format(repr(self.nofm)),
#                 'sofm={}'.format(repr((self.hofm, self.wofm))),
#                 'sfil={}'.format(repr((self.hfil, self.wfil))),
#                 'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class ConvBackActLayer(Layer):
    '''
    NN convolutional activation layer in back-propagation.
    '''
    def __init__(self, nifm, nofm, sofm, sfil, strd=1, crop=False, raw_sofm=None):
        super(ConvBackActLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('ConvLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil
        self.crop = crop
        if crop:
            if isinstance(raw_sofm, int):
                hro = raw_sofm
                wro = raw_sofm
            elif len(raw_sofm) == 2:
                hro = raw_sofm[0]
                wro = raw_sofm[1]
            else:
                raise ValueError('ConvLayer: sfil is invalid ({}), '
                                 'needs to be either one integer or '
                                 'a pair of integers'.format(sfil))
            self.r_hofm = hro
            self.r_wofm = wro
            hifm = util.reverse_high(self.hofm-1, hro, self.hfil, self.htrd) + 1
            wifm = util.reverse_high(self.wofm-1, wro, self.wfil, self.wtrd) + 1
        else:
            hifm = (self.hofm - self.hfil) // self.htrd + 1
            wifm = (self.wofm - self.wfil) // self.wtrd + 1
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = de.OFM

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.IFM, le.OFM)
        dls[de.IFM] = DataDimLoops(le.IFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def total_ops(self, batch_size=1):
        return ConvLayer(self.nofm, self.nifm, (self.hifm, self.wifm),
                         (self.hfil, self.wfil), (self.htrd, self.wtrd)).total_ops(batch_size)

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class ConvBackWeightLayer(Layer):
    '''
    NN convolutional weight layer in back-propagation.
    '''
    def __init__(self, nifm, nofm, sofm, sfil, strd=1, crop=False, raw_sofm=None):
        super(ConvBackWeightLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('ConvLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil
        self.crop = crop
        if crop:
            if isinstance(raw_sofm, int):
                hro = raw_sofm
                wro = raw_sofm
            elif len(raw_sofm) == 2:
                hro = raw_sofm[0]
                wro = raw_sofm[1]
            else:
                raise ValueError('ConvLayer: sfil is invalid ({}), '
                                 'needs to be either one integer or '
                                 'a pair of integers'.format(sfil))
            self.r_hofm = hro
            self.r_wofm = wro
            hifm = util.reverse_high(self.hofm-1, hro, self.hfil, self.htrd) + 1
            wifm = util.reverse_high(self.wofm-1, wro, self.wfil, self.wtrd) + 1
        else:
            hifm = (self.hofm - self.hfil) // self.htrd + 1
            wifm = (self.wofm - self.wfil) // self.wtrd + 1
        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = de.FIL

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.IFM, le.OFM)
        dls[de.IFM] = DataDimLoops(le.IFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def total_ops(self, batch_size=1):
        return ConvLayer(self.nofm, self.nifm, (self.hifm, self.wifm),
                         (self.hfil, self.wfil), (self.htrd, self.wtrd)).total_ops(batch_size)

    def filter_size(self, word_size=1):
        '''
        Get size of one weight filter.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        '''
        Get total size of all weight filters.

        If `word_size` is set to word byte size, return size in bytes.
        '''
        return self.nifm * self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))

# class FCBackLayer(ConvBackLayer):
#     '''
#     NN fully-connected layer parameters.

#     As a special case of CONVLayer.

#     hifm = hfil, wifm = wfil, strd = 1, hifm = wifm = 1
#     '''

#     def __init__(self, nifm, nofm, sfil=1):
#         super(FCBackLayer, self).__init__(nifm, nofm, sfil, sfil)
#         assert self.hifm == 1 and self.wifm == 1

#     def __repr__(self):
#         return '{}({})'.format(
#             self.__class__.__name__,
#             ', '.join([
#                 'nifm={}'.format(repr(self.nifm)),
#                 'nofm={}'.format(repr(self.nofm)),
#                 'sifm={}'.format(repr((self.hifm, self.wifm))),
#                 'sofm={}'.format(repr((self.hofm, self.wofm))),
#                 'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class FCBackActLayer(ConvBackActLayer):
    '''
    NN fully-connected layer parameters.

    As a special case of CONVLayer.

    hifm = hfil, wifm = wfil, strd = 1, hifm = wifm = 1
    '''

    def __init__(self, nifm, nofm, sfil=1):
        super(FCBackActLayer, self).__init__(nifm, nofm, sfil, sfil)
        assert self.hifm == 1 and self.wifm == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sifm={}'.format(repr((self.hifm, self.wifm))),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class FCBackWeightLayer(ConvBackWeightLayer):
    '''
    NN fully-connected layer parameters.

    As a special case of CONVLayer.

    hifm = hfil, wifm = wfil, strd = 1, hifm = wifm = 1
    '''

    def __init__(self, nifm, nofm, sfil=1):
        super(FCBackWeightLayer, self).__init__(nifm, nofm, sfil, sfil)
        assert self.hifm == 1 and self.wifm == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nifm={}'.format(repr(self.nifm)),
                'nofm={}'.format(repr(self.nofm)),
                'sifm={}'.format(repr((self.hifm, self.wifm))),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil)))]))


class LocalRegionBackLayer(Layer):
    '''
    NN localregion layer in back-propagation parameters.
    '''
    def __init__(self, nofm, sofm, nreg, sreg, ntrd=1, strd=1, crop=False, raw_sofm=None, raw_nofm=None):
        super(LocalRegionBackLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sreg, int):
            hreg = sreg
            wreg = sreg
        elif len(sreg) == 2:
            hreg = sreg[0]
            wreg = sreg[1]
        else:
            raise ValueError('LocalRegionBackLayer: sreg is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sreg))
        if nreg > 1 and (hreg * wreg) > 1:
            raise ValueError('LocalRegionBackLayer: local region cannot be a mix '
                             'of both n ({}) and h & w ({}, {})'
                             .format(nreg, hreg, wreg))
        self.nreg = nreg
        self.hreg = hreg
        self.wreg = wreg
        self.ntrd = ntrd
        self.crop = crop
        self.raw_sofm = raw_sofm

        if crop:
            if isinstance(raw_sofm, int):
                hro = raw_sofm
                wro = raw_sofm
            elif len(raw_sofm) == 2:
                hro = raw_sofm[0]
                wro = raw_sofm[1]
            else:
                raise ValueError('LocalRegionBackLayer: sfil is invalid ({}), '
                                 'needs to be either one integer or '
                                 'a pair of integers'.format(raw_sofm))
            assert isinstance(raw_nofm, int), 'invalid raw_nofm: {}'.format(raw_nofm)
            nifm = util.reverse_high(self.nofm-1, raw_nofm, self.nreg, self.ntrd) + 1
            hifm = util.reverse_high(self.hofm-1, hro, self.hreg, self.htrd) + 1
            wifm = util.reverse_high(self.wofm-1, wro, self.wreg, self.wtrd) + 1
        else:
            nifm = self.nofm // self.ntrd  # ignore all-zero padding channels.
            hifm = (self.hofm - self.hreg) // self.htrd + 1
            wifm = (self.wofm - self.wreg) // self.wtrd + 1

        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = de.OFM

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops()
        dls[de.IFM] = DataDimLoops(le.OFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        # Each output point corresponds to merging a local region.
        return self.region_size()
    def total_ops(self, batch_size=1):
        return LocalRegionLayer(self.nofm, (self.hifm, self.wifm), self.nreg, (self.hreg, self.wreg),
                                self.ntrd, (self.htrd, self.wtrd)).total_ops(batch_size)


    def region_size(self):
        ''' The size of the local region corresponding to one output point. '''
        return self.nreg * self.hreg * self.wreg

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg)),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'ntrd={}'.format(repr(self.ntrd)),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class PoolingBackLayer(LocalRegionBackLayer):
    '''
    NN pooling layer in back-propagation.

    As a special case of LocalRegionBackLayer.

    nreg = ntrd = 1
    '''

    def __init__(self, nofm, sofm, sreg, strd=None):
        if strd is None:
            strd = sreg
        super(PoolingBackLayer, self).__init__(nofm, sofm, 1, sreg,
                                           ntrd=1, strd=strd)
        assert self.nreg == 1
        assert self.ntrd == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sreg={}'.format(repr((self.hreg, self.wreg))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class EltwiseBackLayer(LocalRegionBackLayer):
    '''
    NN element-wise layer in back-propagation.

    As a special case of LocalRegionBackLayer.

    nreg = ntrd, sreg = 1
    '''

    def __init__(self, nofm, sofm, nreg):
        super(EltwiseBackLayer, self).__init__(nofm, sofm, nreg, 1,
                                           ntrd=nreg, strd=1)
        assert self.hreg == self.wreg == 1

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'nreg={}'.format(repr(self.nreg))]))


class DepthwiseConvolutionBackActLayer(Layer):
    '''
    Depthwise convolution layer in back-propagation.
    '''
    def __init__(self, nofm, sofm, sfil, strd=1, crop=False, raw_sofm=None, raw_nofm=None):
        super(DepthwiseConvolutionBackActLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('DepthwiseConvolutionBackLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil
        self.crop = crop
        self.raw_sofm = raw_sofm

        if crop:
            if isinstance(raw_sofm, int):
                hro = raw_sofm
                wro = raw_sofm
            elif len(raw_sofm) == 2:
                hro = raw_sofm[0]
                wro = raw_sofm[1]
            else:
                raise ValueError('DepthwiseConvolutionBackLayer: sfil is invalid ({}), '
                                 'needs to be either one integer or '
                                 'a pair of integers'.format(raw_sofm))
            nifm = util.reverse_high(self.nofm-1, raw_nofm, 1, 1) + 1
            hifm = util.reverse_high(self.hofm-1, hro, self.hfil, self.htrd) + 1
            wifm = util.reverse_high(self.wofm-1, wro, self.wfil, self.wtrd) + 1
        else:
            nifm = self.nofm
            hifm = (self.hofm - self.hfil) // self.htrd + 1
            wifm = (self.wofm - self.wfil) // self.wtrd + 1

        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = de.OFM

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.OFM)
        dls[de.IFM] = DataDimLoops(le.OFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        return self.region_size()

    def total_ops(self, batch_size=1):
        return DepthwiseConvolutionLayer(self.nofm, (self.hifm, self.wifm),
                                            (self.hfil, self.wfil),
                                            (self.htrd, self.wtrd)).total_ops(batch_size)

    def region_size(self):
        return self.hfil * self.wfil

    def filter_size(self, word_size=1):
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        return self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))


class DepthwiseConvolutionBackWeightLayer(Layer):
    '''
    Depthwise convolution layer in back-propagation.
    '''
    def __init__(self, nofm, sofm, sfil, strd=1, crop=False, raw_sofm=None, raw_nofm=None):
        super(DepthwiseConvolutionBackWeightLayer, self).__init__(nofm, sofm, strd=strd)

        if isinstance(sfil, int):
            hfil = sfil
            wfil = sfil
        elif len(sfil) == 2:
            hfil = sfil[0]
            wfil = sfil[1]
        else:
            raise ValueError('DepthwiseConvolutionBackLayer: sfil is invalid ({}), '
                             'needs to be either one integer or '
                             'a pair of integers'.format(sfil))

        self.hfil = hfil
        self.wfil = wfil
        self.crop = crop
        self.raw_sofm = raw_sofm

        if crop:
            if isinstance(raw_sofm, int):
                hro = raw_sofm
                wro = raw_sofm
            elif len(raw_sofm) == 2:
                hro = raw_sofm[0]
                wro = raw_sofm[1]
            else:
                raise ValueError('DepthwiseConvolutionBackLayer: sfil is invalid ({}), '
                                 'needs to be either one integer or '
                                 'a pair of integers'.format(raw_sofm))
            nifm = util.reverse_high(self.nofm-1, raw_nofm, 1, 1) + 1
            hifm = util.reverse_high(self.hofm-1, hro, self.hfil, self.htrd) + 1
            wifm = util.reverse_high(self.wofm-1, wro, self.wfil, self.wtrd) + 1
        else:
            nifm = self.nofm
            hifm = (self.hofm - self.hfil) // self.htrd + 1
            wifm = (self.wofm - self.wfil) // self.wtrd + 1

        self.inlayer = Layer(nifm, (hifm, wifm))
        self.rw_data = de.FIL

    @staticmethod
    def data_loops():
        dls = [None] * de.NUM
        dls[de.FIL] = DataDimLoops(le.OFM)
        dls[de.IFM] = DataDimLoops(le.OFM, le.BAT)
        dls[de.OFM] = DataDimLoops(le.OFM, le.BAT)
        return tuple(dls)

    def input_layer(self):
        return self.inlayer

    def ops_per_neuron(self):
        return self.region_size()

    def total_ops(self, batch_size=1):
        return DepthwiseConvolutionLayer(self.nofm, (self.hifm, self.wifm),
                                            (self.hfil, self.wfil),
                                            (self.htrd, self.wtrd)).total_ops(batch_size)

    def region_size(self):
        return self.hfil * self.wfil

    def filter_size(self, word_size=1):
        return self.hfil * self.wfil * word_size

    def total_filter_size(self, word_size=1):
        return self.nofm * self.filter_size(word_size)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            ', '.join([
                'nofm={}'.format(repr(self.nofm)),
                'sofm={}'.format(repr((self.hofm, self.wofm))),
                'sfil={}'.format(repr((self.hfil, self.wfil))),
                'strd={}'.format(repr((self.htrd, self.wtrd)))]))