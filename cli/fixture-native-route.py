import sys,runpy
from pathlib import Path
import catlass.tla as tla
check,source=sys.argv[1:]
counts=[0,0]
native=tla.compile
def compile_spy(*args,**kwargs):
 op=native(*args,**kwargs)
 counts[0]+=1
 def launch(*a,**kw):
  result=op(*a,**kw);counts[1]+=1;return result
 return launch
tla.compile=compile_spy
sys.argv=[check,source]
try:
 runpy.run_path(check,run_name='__main__')
except BaseException as failure:
 if not (isinstance(failure,SystemExit) and failure.code in (None,0)):
  if all(x>0 for x in counts):print('VERIFIED_NATIVE_ROUTE',*counts,flush=True)
  raise
assert all(x>0 for x in counts),f'no native Catlass launch: {counts}'
print('VERIFIED_NATIVE_ROUTE',*counts)

