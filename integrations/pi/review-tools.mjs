import {createNativeResources as filesystem} from './native-files.mjs';
import {resolve} from 'node:path';
export function createNativeResources(context) {
  const resources = filesystem(context);
  return {...resources, tools:resources.tools.map(tool => ({...tool,
    execute:async (id, args, ...rest) => {
      const result = await tool.execute(id, args, ...rest);
      if (tool.name === 'read') await context.host('rsi_review_read_observed', {
        ...args, path:resolve(context.cwd, args.path), details:result.details}, id);
      return result;
    }}))};
}
