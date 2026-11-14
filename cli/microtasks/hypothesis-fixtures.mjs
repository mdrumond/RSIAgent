/** Public CPU fixture data; it contains no candidate guidance or expected repairs. */
export function constructionFixture(prefix=''){
 const p=name=>prefix+name;
 const bench=`import argparse,sys\np=argparse.ArgumentParser();p.add_argument('--mode',choices=['correctness','performance','baseline','cpu'],required=True);p.parse_args()\nsys.path.insert(0,'.')\nfrom implementation import value\nsys.exit(0 if value()==42 else 1)\n`;
 return {files:{[p('source/implementation.py')]:'def value(): return 42\n',[p('reference/implementation.py')]:'def value(): return sum(range(7))*2\n',[p('negative/implementation.py')]:'def value(): return 0\n',[p('bench.py')]:bench,[p('quick.py')]:bench},
  proposal:{goal:'CPU construction protocol fixture',source:p('source'),reference:p('reference'),negative_controls:[p('negative')],performance_goal:'Preserve the declared command protocol; this fixture makes no device-speed claim',
   verifier:{files:[p('bench.py'),p('quick.py')],command:{entrypoint:p('bench.py'),args:['--mode','correctness']},performance_command:{entrypoint:p('bench.py'),args:['--mode','performance']},baseline_command:{entrypoint:p('bench.py'),args:['--mode','baseline']}},
   construction_command:{entrypoint:p('bench.py'),args:['--mode','cpu']},baseline_construction_command:{entrypoint:p('bench.py'),args:['--mode','cpu']}}};
}
