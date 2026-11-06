"""All commands run locally; transport and device admission belong to the caller."""
import argparse
import json
import signal
from pathlib import Path
from . import WORKLOADS, describe

parser=argparse.ArgumentParser()
sub=parser.add_subparsers(dest='command',required=True)
p=sub.add_parser('describe');p.add_argument('--workload',choices=WORKLOADS,required=True)
p=sub.add_parser('qualify');p.add_argument('--workload',choices=WORKLOADS,required=True)
p.add_argument('--target',choices=['a3','a5'],required=True)
p.add_argument('--identity',type=Path,required=True);p.add_argument('--output',required=True)
p.add_argument('--max-seconds',type=float,default=7200)
p=sub.add_parser('case');p.add_argument('workload',choices=WORKLOADS);p.add_argument('case_id',type=int)
p.add_argument('--device',required=True);p.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.command=='describe':
    result=describe(args.workload)
elif args.command=='qualify':
    from .qualification import qualify
    def interrupt(_signal, _frame):
        raise KeyboardInterrupt('Qualification cancelled')
    signal.signal(signal.SIGTERM,interrupt)
    result=qualify(args.workload,args.target,json.loads(args.identity.read_text()),args.output,args.max_seconds)
else:
    from .qualification import atomic, execute_case
    if args.device.startswith('npu'):
        import torch_npu
    result=execute_case(args.workload,args.case_id,args.device)
    atomic(args.output,result)
print(json.dumps(result))
