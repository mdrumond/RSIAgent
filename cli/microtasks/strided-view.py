import runpy, types, sys, itertools, random
from pathlib import Path
p=Path(sys.argv[1])/'kernel.py';m=types.SimpleNamespace(**runpy.run_path(str(p)))
r=random.Random(1407)
for shape,strides,offset in [((2,3),(3,1),0),((3,2),(1,3),0),((2,2),(4,2),1),((3,),(-1,),4),((2,3),(0,1),0),((),(),5),((0,3),(3,1),0)]:
 storage=[r.random() for _ in range(64)];before=storage[:]
 expected=[storage[offset+sum(i*s for i,s in zip(index,strides))] for index in itertools.product(*(range(n) for n in shape))]
 assert m.logical_flatten(storage,shape,strides,offset)==expected
 assert storage==before
print('strided indexing contract passed')
