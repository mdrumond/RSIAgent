import runpy, types, sys
from pathlib import Path
p=Path(sys.argv[1])/'kernel.py';m=types.SimpleNamespace(**runpy.run_path(str(p)))
compiled=[]
def compiler(n):
 compiled.append(n)
 def kernel(x,scale):
  assert len(x)==n
  return [v*scale for v in x]
 return kernel
launch=m.make_launcher(compiler)
x=[1,2,3]
assert launch(x,2)==[2,4,6]
assert launch([7,8,9],3)==[21,24,27]
x[0]=11
assert launch(x,-1)==[-11,-2,-3]
assert launch([5],4)==[20]
assert launch([],10)==[]
assert sorted(compiled)==[0,1,3]
print('fresh compiled-launch arguments passed')
