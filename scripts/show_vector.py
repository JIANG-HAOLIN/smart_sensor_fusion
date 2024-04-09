import matplotlib.pyplot as plt
import numpy as np
import copy


len = 100
array = np.load("example_traj.npy")
x1, y1, z1 = (- array[:len, 1], - array[:len, 0], array[:len, 2])
x2, y2, z2 = (- array[:len, 1], - array[:len, 0], array[:len, 2])

fig, ax = plt.subplots(2, 1,subplot_kw=dict(projection='3d'), )
markerline, stemlines, baseline = ax[0].stem(
    x1, y1, z1, linefmt='none', markerfmt='.-', orientation='y', )
markerline.set_markerfacecolor('none')
ax[0].set_aspect('auto')


markerline, stemlines, baseline = ax[0].stem(
    x2, y2, z2, linefmt='none', markerfmt='.-', orientation='x', )
markerline.set_markerfacecolor('none')
ax[0].set_aspect('auto')

# _, __, ___ = ax.stem(
#     -y, -x, z, linefmt='none', markerfmt='', orientation='x', )
# ax.set_aspect('equalxy')


plt.show()