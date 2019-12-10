# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits import mplot3d
from math import exp

# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt  # noqa: F401 unused import

fig = plt.figure()
ax = plt.axes(projection='3d')

# Fixing random state for reproducibility

x = np.arange(1,20)
y = np.arange(1,20)
z = np.arange(0,0.6)
x,y = np.meshgrid(x,y)
# exp(10)
z = 0.6 * exp((-0.003 * (x-20)**2) - (0.015 * (y - 14)**2))
plt.xlabel("X Label")
plt.ylabel("Y Label")
# surf = ax.plot_surface(x, y, z)
ax.plot_wireframe(x,y,z, rstride = 10, cstride = 10)


# plt.zlabel("Z Label")



plt.show()

