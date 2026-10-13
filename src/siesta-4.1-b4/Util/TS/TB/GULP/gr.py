#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append("../")

# Our tight-binding model
import pht_tb as PH, tbt_tb as TB

#TB.graphene_uc(alat=1.42).tile(2,1).tile(4,0).tile(4,1).xyz('ac.xyz')
#TB.graphene_uc(alat=1.42).tile(2,1).tile(4,1).tile(4,0).xyz('zz.xyz')

# create output-object
for out in ['ac','zz']:
    gulp_out = PH.GULP(out + '.gout')

    print('Reading output')
    tb = gulp_out.read_model()
    # In GULP correcting for Newtons second law is already obeyed
    tb.correct_Newton()

    # Save full
    if out == 'ac':
        el = tb.cut(axis=1,seps=2).cut(1,2)
    else:
        el = tb.cut(axis=0,seps=4)
    el.save('DEVICE_'+out+'.nc')

    if out == 'ac':
        el = el.cut(axis=0,seps=2).cut(0,2)
    else:
        el = el.cut(axis=1,seps=4)
    el.save('ELEC_'+out+'.nc')

