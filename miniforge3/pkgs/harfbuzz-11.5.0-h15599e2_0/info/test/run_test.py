#  tests for harfbuzz-11.5.0-h15599e2_0 (this is a generated file);
print('===== testing package: harfbuzz-11.5.0-h15599e2_0 =====');
print('running run_test.py');
#  --- run_test.py (begin) ---
import gi
gi.require_version('HarfBuzz', '0.0')
from gi.repository import HarfBuzz as hb
import sys

if hb.buffer_create () is None:
    sys.exit(1)
#  --- run_test.py (end) ---

print('===== harfbuzz-11.5.0-h15599e2_0 OK =====');
