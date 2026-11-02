import {createNativeResources as filesystem} from './native-files.mjs';
export function createNativeResources(context) {
  const resources = filesystem(context);
  return {...resources, resourceOptions:{noSkills:true,
    skillsOverride:() => ({skills:context.options.skills, diagnostics:[]})}};
}
