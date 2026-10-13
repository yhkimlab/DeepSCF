#!/usr/bin/env python
from __future__ import print_function

import sys
sys.path.append("../")

# Our tight-binding model
import pht_tb as PH, numpy as np

# Define Si square unit-cell
xa = np.array([
        [ 0.000000000, 0.000000000, 0.000000000],
        [ 0.000000000, 2.715989480, 2.715989480],
        [ 2.715989480, 0.000000000, 2.715989480],
        [ 2.715989480, 2.715989480, 0.000000000],
        [ 1.357994740, 1.357994740, 1.357994740],
        [ 1.357994740, 4.073984220, 4.073984220],
        [ 4.073984220, 1.357994740, 4.073984220],
        [ 4.073984220, 4.073984220, 1.357994740]]) + np.array([1.35/2]*3)[None,:]

Si = PH.PHT_Geom(cell=np.diag([5.4]*3),xa=xa,Z=14)
# Si.tile(4,0).tile(4,1).tile(4,2).xyz('SiGeom.xyz')

gulp_out = PH.GULP('Si.gout')
print('Reading output')
tb = gulp_out.read_model()
print('Correcting for Newtons laws')
# In GULP correcting for Newtons second law is already obeyed
tb.correct_Newton()
# Save full
print('Cutting first')
el = tb.cut(axis=2,seps=4).cut(axis=1,seps=4)
el.save('DEVICE_Si.nc')
print('Cutting second')
el = el.cut(axis=0,seps=4)
el.save('ELEC_Si.nc')

