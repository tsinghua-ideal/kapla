from collections import defaultdict
from multiprocessing import Pool
import itertools
import fastcache
import heapq

from nn_dataflow.core import data_category_enum as de
from nn_dataflow.core import loop_enum as le
from nn_dataflow.core import mem_hier_enum as me
from nn_dataflow.core import partition
from nn_dataflow import util
from nn_dataflow.core import BufShrScheme
from nn_dataflow.core import LocalRegionLayer, ConvLayer, LocalRegionBackLayer, ConvBackActLayer, \
    ConvBackWeightLayer, DepthwiseConvolutionLayer, DepthwiseConvolutionBackActLayer, \
    DepthwiseConvolutionBackWeightLayer
from nn_dataflow.core.map_strategy import MapStrategyEyeriss
from nn_dataflow.core import NodeRegion
from nn_dataflow.core import PhyDim2


'''
Fast explorer for a quick schedule on nn dataflow.
'''


def gen_segment_set(segments, ordered_layer_list, network, cost, options, explore_n_seg_sets=4,
                    nprocesses=8):
    """
    Generate a set of best segments that are preferred to schedule.
    """

    # Put all segments in order.
    ordered_segments = list()
    for layer_name in ordered_layer_list:
        ordered_segments.extend(segments[layer_name])

    num_top_segs = explore_n_seg_sets
    # Too many preferred segments, simply return all.
    if num_top_segs > len(ordered_segments):
        return segments

    opt_segments = dict()
    for layer in network:
        opt_segments[layer] = [(float('inf'), []) for _ in range(num_top_segs)]

    # Enable multiprocess.
    if nprocesses > 1:
        pool = Pool(processes=nprocesses)
        apply_func = pool.apply_async
        retrieve_func = lambda x: x.get(timeout=3600)
    else:
        pool = None
        apply_func = util.apply
        retrieve_func = lambda x: x

    handler_list = []

    for idx, segment in enumerate(ordered_segments):
        r = apply_func(estimate_seg_cost, (segment, network, options, cost))
        handler_list.append(r)

    seg_cost_list = list(map(retrieve_func, handler_list))

    if pool is not None:
        pool.close()
        pool.join()

    for idx, segment in enumerate(ordered_segments):
        # Solve the constraint with least buffer occupation.
        min_cost = seg_cost_list[idx]

        # Initialize the cost of current segment.
        cur_cands = [(0, [])]
        prev_idx = ordered_layer_list.index(segment[0][0]) - 1
        if prev_idx != -1:
            prev_layer = ordered_layer_list[prev_idx]
            cur_cands = list(opt_segments[prev_layer])

        # Update candidate costs.
        cur_cands = [(cand[0] + min_cost, cand[1] + [idx]) for cand in cur_cands]

        # Update dp tracker.
        last_layer = segment.seg[-1][-1]
        opt_segments[last_layer] = sorted(opt_segments[last_layer] + cur_cands)[:num_top_segs]

    seg_set = set()
    for cost_seg in opt_segments[ordered_layer_list[-1]]:
        seg_set.update(cost_seg[-1])

    # Group the segments by the ending layers.
    new_segments = defaultdict(set)
    for idx in seg_set:
        seg = ordered_segments[idx]
        if seg not in new_segments[seg[-1][-1]]:
            new_segments[seg[-1][-1]].add(seg)

    # Get all single-layer segments as backup.
    for l in segments:
        for seg in segments[l]:
            if len(seg) == 1 and len(seg[0]) == 1:
                new_segments[l].add(seg)

    return new_segments


def estimate_seg_cost(segment, network, options, cost):
    """
    Estimate the cost of the segment.
    """
    print('estimate_seg_cost', segment.seg, flush=True)
    batch_size = segment.batch_size

    def _estimate_per_cstr_cost(constraint):
        """ Estimate the cost of the segment under a given constraint. """
        min_cost = 0
        for layer_name, rsrc, cstr in zip(
                itertools.chain(*segment.seg),
                itertools.chain(*segment.allocation()),
                itertools.chain(*constraint)):
            layer = network[layer_name]
            # Use the minimum cost among different partition schemes as the
            # corresponding layer's cost.
            min_part_cost = min(
                estimate_layer_cost(
                    layer, batch_size, p, rsrc, cstr, cost, options)
                for p in partition.gen_partition(
                    layer, batch_size, rsrc.proc_region.dim, options,
                    guaranteed=True))
            if min_part_cost == float('inf'):
                return float('inf')
            min_cost += min_part_cost
        return min_cost

    # Sequentially search constraints.
    # As we check topifm/ofm/bat satisfiability during the cost calculation, we cannot assume a
    # single point dividing invalid/valid constraints. So advanced search methods cannot use.
    min_cost = float('inf')
    for cstr, _ in segment.gen_constraint():
        if cstr[0][0].topbat != 0 and not segment_occp_is_valid(segment.seg, segment.network,
                                                                segment.batch_size, cstr,
                                                                segment.alloc):
            continue
        min_cost = _estimate_per_cstr_cost(cstr)
        if min_cost < float('inf'):
            break

    return min_cost


def gen_partition_list(layer, batch_size, resource, constraint, cost, options, explore_n_parts=5):
    """
    Generate the best partition schemes that are preferred to schedule.
    """
    return SortedIterator(
        partition.gen_partition(layer, batch_size, resource.proc_region.dim, options,
                                guaranteed=True),
        counter=explore_n_parts,
        key=lambda part: estimate_layer_cost(layer, batch_size, part, resource,
                                             constraint, cost, options))


def segment_occp_is_valid(seg_tuple, network, batch_size, constraint,
                          allocation):
    """
    Check if a segment is valid with the given scheduling constraint and
    resource allocation, in terms of the buffer occupation.
    """

    for lyr_tpl, cstr_tpl, alloc_tpl in zip(seg_tuple, constraint, allocation):
        for lyr, cstr, rsrc in zip(lyr_tpl, cstr_tpl, alloc_tpl):
            layer = network[lyr]

            layer_occp = _estimate_layer_occp(layer, batch_size, rsrc, cstr)

            if layer_occp > rsrc.proc_region.dim.size() * rsrc.size_gbuf:
                return False

    return True


def _estimate_buf_ifm_ofm_lb(layer, buf_batch, resource):
    """
    Estimate a lower bound of the product of the numbers of ifm-ofm pairs that
    can execute in parallel on the allocated resource, which is equal to the
    ratio of the total PE number over the number of ops per ifm-ofm pair.
    """
    total_pe = resource.proc_region.dim.size() * resource.dim_array.size()
    if isinstance(layer, (ConvLayer, LocalRegionLayer, DepthwiseConvolutionLayer)):
        layer_ops = layer.filter_size() * layer.ofmap_size(buf_batch)
    elif isinstance(layer, (ConvBackActLayer, ConvBackWeightLayer, LocalRegionBackLayer,
                            DepthwiseConvolutionBackActLayer, DepthwiseConvolutionBackWeightLayer)):
        layer_ops = layer.filter_size() * layer.ifmap_size(buf_batch)
    return max(1, total_pe // layer_ops)


def _estimate_layer_occp(layer, batch_size, resource, constraint):
    """
    Estimate the buffer occupation for the layer executing on the given
    resource with the constraint.
    """

    buf_bat = batch_size // constraint.topbat if constraint.topbat else 1

    if isinstance(layer, LocalRegionLayer):
        buf_ofm = layer.nofm // constraint.topofm if constraint.topofm else 1
        occp = (layer.ofmap_size(buf_bat) * buf_ofm +
                layer.ifmap_size(buf_bat) * buf_ofm)
        return occp

    elif isinstance(layer, (ConvLayer, ConvBackActLayer, ConvBackWeightLayer)):
        top_ifm_ofm_cands = []

        # This is an upper bound of the product of topifm and topofm.
        top_io = util.idivc(layer.nifm * layer.nofm,
                            _estimate_buf_ifm_ofm_lb(layer, buf_bat, resource))

        # Require one of topifm and topofm must be 1, unless both are
        # given from the constraint.
        if constraint.topifm and constraint.topofm:
            # Both are given. Use them.
            top_ifm_ofm_cands.append((constraint.topifm, constraint.topofm))

        elif not constraint.topifm and not constraint.topofm:
            # Both are not given. Estimate from the product upper bound.
            top_ifm_ofm_cands.append((1, top_io))
            top_ifm_ofm_cands.append((top_io, 1))

        elif not constraint.topifm:
            top_ifm = top_io if constraint.topofm == 1 else 1
            top_ifm_ofm_cands.append((top_ifm, constraint.topofm))

        else:
            assert not constraint.topofm
            top_ofm = top_io if constraint.topifm == 1 else 1
            top_ifm_ofm_cands.append((constraint.topifm, top_ofm))

        occp = float('inf')
        for top_ifm, top_ofm in top_ifm_ofm_cands:
            buf_ifm = layer.nifm // top_ifm
            buf_ofm = layer.nofm // top_ofm

            sz = (buf_ifm * layer.ifmap_size(buf_bat) +
                  buf_ofm * layer.ofmap_size(buf_bat) +
                  buf_ifm * buf_ofm * layer.filter_size())
            occp = min(sz, occp)

        return occp

    elif isinstance(layer, LocalRegionBackLayer):
        buf_ifm = layer.nifm // constraint.topifm if constraint.topifm else 1
        occp = (layer.ofmap_size(buf_bat) * buf_ifm +
                layer.ifmap_size(buf_bat) * buf_ifm)

        return occp

    elif isinstance(layer, DepthwiseConvolutionLayer):
        buf_ofm = layer.nofm // constraint.topofm if constraint.topofm else 1
        occp = (layer.ofmap_size(buf_bat) * buf_ofm +
                layer.ifmap_size(buf_bat) * buf_ofm +
                buf_ofm * layer.filter_size())

        return occp

    elif isinstance(layer, (DepthwiseConvolutionBackActLayer, DepthwiseConvolutionBackWeightLayer)):
        buf_ifm = layer.nifm // constraint.topifm if constraint.topifm else 1
        occp = (layer.ofmap_size(buf_bat) * buf_ifm +
                layer.ifmap_size(buf_bat) * buf_ifm +
                buf_ifm * layer.filter_size())

        return occp

    else:
        raise TypeError('Invalid layer type.')


def partition_occp_is_valid(part, layer, batch_size, resource, constraint,
                            options, map_strategy="eyeriss"):
    """
    Check if a layer partition is valid with the given scheduling constraint
    and resource, in terms of the buffer occupation.
    """
    p_layer, p_batch, p_occ = part.part_layer(layer, batch_size)

    if options.hw_gbuf_sharing:
        bufshr = BufShrScheme(resource.proc_region, part, layer.data_loops())
        bufshr_size = tuple(bufshr.size(dce) for dce in range(de.NUM))
    else:
        bufshr_size = tuple([1] * de.NUM)

    if map_strategy == 'eyeriss':
        map_strategy = MapStrategyEyeriss(p_layer, p_batch, p_occ,
                                          resource.dim_array)
    # elif map_strategy == 'tpu':
    #     map_strategy = MapStrategyTPU(p_layer, p_batch, p_occ,
    #                                       resource.dim_array)
    else:
        raise ValueError("Invalid option.map_strategy: {}".format(
                         options.map_strategy))

    occp_list = []

    for nld in map_strategy.gen_nested_loop_desc():
        loop_cnt = nld.loopcnt
        buf_bl = [0] * le.NUM

        if constraint.topifm == 0:
            buf_bl[le.IFM] = 1
        else:
            q, r = divmod(loop_cnt[le.IFM], constraint.topifm)
            if r != 0:
                continue
            buf_bl[le.IFM] = q

        if constraint.topofm == 0:
            buf_bl[le.OFM] = 1
        else:
            q, r = divmod(loop_cnt[le.OFM], constraint.topofm)
            if r != 0:
                continue
            buf_bl[le.OFM] = q

        if constraint.topbat == 0:
            buf_bl[le.BAT] = 1
        else:
            buf_bl[le.BAT] = loop_cnt[le.BAT] // constraint.topbat

        occp = [util.prod(nld.data_loops[dce].take(buf_bl)) *
                nld.usize_gbuf_of(dce) // bufshr_size[dce]
                for dce in range(de.NUM)]
        occp_list.append(sum(occp))

    return occp_list and min(occp_list) <= resource.size_gbuf


@fastcache.clru_cache(maxsize=1024)
def estimate_layer_cost(layer, batch_size, part, resource, constraint, cost,
                        options):
    """
    Estimate the cost of the layer under the given partition scheme.
    """
    if not partition_occp_is_valid(part, layer, batch_size, resource,
                                   constraint, options):
        return float('inf')

    # Estimate accesses.
    ac = _estimate_layer_accesses(layer, batch_size, part, resource,
                                  constraint, options)
    # Estimate NoC hops.
    mh = _estimate_layer_mem_nhops(layer, batch_size, part, resource,
                                   constraint, options)

    layer_cost = sum(ac[mhe] * cost.mem_hier_at(mhe)
                     for mhe in range(me.NUM)) + sum(mh) * cost.noc_hop
    return layer_cost


def _estimate_layer_accesses(layer, batch_size, part, resource, constraint,
                             options):
    p_layer, p_batch, _ = part.part_layer(layer, batch_size)

    if options.hw_gbuf_sharing:
        bufshr = BufShrScheme(resource.proc_region, part, layer.data_loops())
        bufshr_size = tuple(bufshr.size(dce) for dce in range(de.NUM))
    else:
        bufshr_size = tuple([1] * de.NUM)

    layer_acc = [0] * me.NUM

    # DRAM access.
    if resource.src_data_region.type == NodeRegion.DRAM:
        acc = p_layer.total_ifmap_size(p_batch) * \
                part.size() // bufshr_size[de.IFM]
        layer_acc[me.DRAM] += acc
    if resource.dst_data_region.type == NodeRegion.DRAM:
        acc = p_layer.total_ofmap_size(p_batch) * \
                part.size() // bufshr_size[de.OFM]
        layer_acc[me.DRAM] += acc
    try:
        acc = p_layer.total_filter_size() * \
                part.size() // bufshr_size[de.FIL]
        if not _filter_is_fully_buffered(p_layer, p_batch, constraint,
                                         bufshr_size, resource.size_gbuf):
            acc *= constraint.topbat
        layer_acc[me.DRAM] += acc
    except AttributeError:
        pass

    # GBUF access.
    layer_acc[me.GBUF] += p_layer.total_ifmap_size(p_batch) * part.size()
    layer_acc[me.GBUF] += p_layer.total_ofmap_size(p_batch) * part.size()
    try:
        layer_acc[me.GBUF] += p_layer.total_filter_size() * part.size()
    except AttributeError:
        pass

    return layer_acc


def _estimate_layer_mem_nhops(layer, batch_size, part, resource, constraint,
                              options):
    p_layer, p_batch, _ = part.part_layer(layer, batch_size)

    def _centralize_node(region):
        central_origin = region.rel2abs(PhyDim2(region.dim.h/2, region.dim.w/2))
        return central_origin

    def _find_nearest_node(nodes, target_node):
        return min(nodes, key=lambda node: node.hop_dist(target_node))

    # To simplify unit nhops computation, we use the central node to represent
    # each layout and the process region.
    src_central_node = _centralize_node(resource.src_data_region)
    dst_central_node = _centralize_node(resource.dst_data_region)
    proc_central_node = _centralize_node(resource.proc_region)

    # Filter nodes. All memory nodes can store filters.
    filter_nodes = frozenset(resource.dram_region.iter_node())
    nearest_filter_node = _find_nearest_node(filter_nodes, proc_central_node)

    # Use the central region of the `proc_region`.
    dists = [None] * de.NUM
    dists[de.IFM] = proc_central_node.hop_dist(src_central_node)
    dists[de.OFM] = proc_central_node.hop_dist(dst_central_node)
    dists[de.FIL] = proc_central_node.hop_dist(nearest_filter_node)

    # Data sizes considered duplication.
    data_sizes = [0] * de.NUM
    data_sizes[de.IFM] = p_layer.total_ifmap_size(p_batch) * part.size()
    data_sizes[de.OFM] = p_layer.total_ofmap_size(p_batch) * part.size()
    try:
        data_sizes[de.FIL] = p_layer.total_filter_size() * part.size()
    except AttributeError:
        pass

    if options.hw_access_forwarding or options.hw_gbuf_sharing:
        bufshr = BufShrScheme(resource.proc_region, part, layer.data_loops())
        bufshr_size = tuple(bufshr.size(dce) for dce in range(de.NUM))

        # Use bufshr_size to estimate data-forwarding hops within
        # the proc region.
        dists = [d + bss for d, bss in zip(dists, bufshr_size)]
        # Divide bufshr_size to eliminate the duplicate.
        data_sizes = [s / bss for s, bss in zip(data_sizes, bufshr_size)]

    # Set fetch times.
    fetches = [1] * de.NUM
    if not _filter_is_fully_buffered(p_layer, p_batch, constraint, bufshr_size,
                                     resource.size_gbuf):
        fetches[de.FIL] *= constraint.topbat

    layer_mh = [s * d * f for s, d, f in zip(data_sizes, dists, fetches)]
    return layer_mh


def _filter_is_fully_buffered(p_layer, p_batch, constraint, bufshr_size,
                              size_gbuf):

    if constraint.topbat == 0:
        # Single-layer segment. Do not assume re-access filters multiple times.
        return True

    # If filter is fully buffered, topifm and topofm must both be 1.
    if constraint.topifm > 1 or constraint.topofm > 1:
        return False

    buf_batch = p_batch // constraint.topbat
    try:
        buf_occp = p_layer.total_ifmap_size(buf_batch) // bufshr_size[de.IFM] \
                + p_layer.total_ofmap_size(buf_batch) // bufshr_size[de.OFM] \
                + p_layer.total_filter_size() // bufshr_size[de.FIL]
    except AttributeError:
        return True

    return buf_occp <= size_gbuf


class SortedIterator(object):
    """
    An adaptive iterator that yields a given number of elements as sorted,
    according to the key of the element. The number of the yielded elements can
    be dynamically adjusted.
    """
    def __init__(self, iterator, counter=float('inf'), key=None):
        try:
            _ = iter(iterator)
        except TypeError:
            raise TypeError('SortedIterator: iterator should be iterable.')

        if counter <= 0:
            raise ValueError('counter should be positive.')

        self.key_elem_list = []
        for elem in iterator:
            ke = (key(elem) if key else None, elem)
            heapq.heappush(self.key_elem_list, ke)

        self.counter = counter

    def __iter__(self):
        return self

    def next(self):
        """ __next__. """
        if self.counter == 0:
            raise StopIteration
        self.counter -= 1
        try:
            return heapq.heappop(self.key_elem_list)[1]
        except IndexError:
            raise StopIteration

    def increment_counter(self):
        """ Dynamically increment the number of yielded elements. """
        self.counter += 1
