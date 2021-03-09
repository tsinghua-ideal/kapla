'''
Print layer pipeline.
'''
import sys
import json
from collections import OrderedDict

from natsort import natsorted

def main(argv):
    ''' Main. '''
    for logfname in natsorted(argv[1:]):

        try:
            with open(logfname, 'r') as fh:
                res = json.load(fh, object_pairs_hook=OrderedDict)
        except Exception:
            print('Failed to parse file {}'.format(logfname))
            raise

        print(res['net'])

        mappings = res['schedules']

        seg_time = res['segment_time']

        for layer_name, sched in mappings.items():

            # print layer_name, sched[-1]

            sched_seq = sched[-1]

            if sched_seq[2] == 0:

                if sched_seq[1] == 0:

                    print('')
                    print(seg_time.pop(0), '---', end='')

                else:

                    print(',', end='')

            else:

                print('+', end='')

            # print sched[0]['time'],
            print(layer_name, '(', sched[0]['time'], ')', end='')

        print('')
        print('')


if __name__ == '__main__':
    main(sys.argv)

