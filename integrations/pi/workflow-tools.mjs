import {createNativeResources as filesystem} from './native-files.mjs';
import {resolve} from 'node:path';
export function createNativeResources(context) {
  const resources = filesystem(context);
  const tools = resources.tools.map(tool => ({...tool,
    execute:async (id, args, ...rest) => {
      let result;
      try {
        result = await tool.execute(id, args, ...rest);
        if (tool.name === 'read') await context.host('rsi_workflow_read_observed', {
          ...args, path:resolve(context.cwd, args.path), details:result.details}, id);
      }
      catch (error) { result = {isError:true, content:[{type:'text', text:error.message}]}; }
      const state = await context.host('rsi_workflow_state', {}, id);
      return {...result, terminate:state.terminate || result.terminate, content:[...result.content, ...state.content]};
    }}));
  return {...resources, tools, resourceOptions:{noSkills:true,
    skillsOverride:() => ({skills:context.options.skills, diagnostics:[]})}};
}
