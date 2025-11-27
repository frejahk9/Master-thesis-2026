#  tests for pyproj-3.7.2-py311h4e6619b_2 (this is a generated file);
print('===== testing package: pyproj-3.7.2-py311h4e6619b_2 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import os
import sys

import pyproj
from pyproj import Proj

Proj(init="epsg:4269")


# Test pyproj_datadir.
if not os.path.isdir(pyproj.datadir.get_data_dir()):
    sys.exit(1)
#  --- run_test.py (end) ---

print('===== pyproj-3.7.2-py311h4e6619b_2 OK =====');
print("import: 'pyproj'")
import pyproj

