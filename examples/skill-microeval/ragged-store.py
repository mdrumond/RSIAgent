import runpy, types, sys, random
from pathlib import Path
p=Path(sys.argv[1])/'kernel.py';m=types.SimpleNamespace(**runpy.run_path(str(p)))
r=random.Random(9901)
for rows,columns,pitch in [(0,3,4),(1,0,4),(1,1,1),(3,5,8),(9,7,7),(7,1,13)]:
 x=[r.randrange(-99,99) for _ in range(rows*pitch+11)];before=x[:]
 expected=x[:]
 for row in range(rows):
  for column in range(columns):expected[row*pitch+column]+=17
 assert m.add_rows(x,rows,columns,pitch,17)==expected
 assert x==before
print('ragged row padding and tail preservation passed')
