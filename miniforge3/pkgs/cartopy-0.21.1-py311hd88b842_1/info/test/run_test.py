#  tests for cartopy-0.21.1-py311hd88b842_1 (this is a generated file);
print('===== testing package: cartopy-0.21.1-py311hd88b842_1 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import cartopy.crs as ccrs

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

ax = plt.axes(projection=ccrs.Robinson())
ax.coastlines()
plt.savefig('working.png')
#  --- run_test.py (end) ---

print('===== cartopy-0.21.1-py311hd88b842_1 OK =====');
print("import: 'cartopy'")
import cartopy

print("import: 'cartopy.mpl.geoaxes'")
import cartopy.mpl.geoaxes

print("import: 'cartopy.crs'")
import cartopy.crs

