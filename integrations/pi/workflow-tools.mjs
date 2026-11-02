import {createNativeResources as filesystem} from './native-files.mjs';
export function createNativeResources(context) {
  const resources = filesystem(context);
  const tools = resources.tools.map(tool => ({...tool,
    execute:async (id, ...args) => {
      const result = await tool.execute(id, ...args);
      const state = await context.host('rsi_workflow_state', {}, id);
      return {...result, content:[...result.content, ...state.content]};
    }}));
  return {...resources, tools, resourceOptions:{noSkills:true,
    skillsOverride:() => ({skills:context.options.skills, diagnostics:[]})}};
}
