// Invoked only by the worker-loss case; the native Pi controller owns execution.
import {join} from 'node:path';
import {readConfig} from '../../../cli/instance.mjs';
import {State} from '../../../controller/state.mjs';
import {processIdentity} from '../../../controller/lifecycle.mjs';
import {runLiveActor} from './live-harness.mjs';
const config=await readConfig(process.argv[2]),state=new State(join(config.root,'state.sqlite'));
const attempt=state.admit('numeric',await processIdentity());
try{await runLiveActor({config,state,attempt});}finally{state.close();}
