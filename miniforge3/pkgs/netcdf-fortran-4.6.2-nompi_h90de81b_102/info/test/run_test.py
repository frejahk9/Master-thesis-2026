#  tests for netcdf-fortran-4.6.2-nompi_h90de81b_102 (this is a generated file);
print('===== testing package: netcdf-fortran-4.6.2-nompi_h90de81b_102 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
# Load the libraries using ctypes.
import os
import sys
import ctypes

platform = sys.platform

if platform.startswith('linux'):
    path = os.path.join(sys.prefix, 'lib', 'libnetcdff.so')
    lib = ctypes.CDLL(path)
elif platform == 'darwin':
    path = os.path.join(sys.prefix, 'lib', 'libnetcdff.dylib')
    lib = ctypes.CDLL(path)
elif platform == 'win32':
    path = os.path.join(sys.prefix, 'Library', 'bin', 'libnetcdff.dll')
    lib = ctypes.CDLL(path)
else:
    raise ValueError('Unrecognized platform: {}'.format(platform))
#  --- run_test.py (end) ---

print('===== netcdf-fortran-4.6.2-nompi_h90de81b_102 OK =====');
