/** Native Pi owns edit matching and read pagination. Host owns filesystem admission. */
export function createNativeResources({sdk, host, cwd, options}) {
  let callId;
  async function invoke(name, args) {
    const result = await host(name, args, callId);
    if (result.isError) throw new Error(result.content?.map(item => item.text || '').join('\n'));
    return result.details;
  }
  const readFile = async path => Buffer.from((await invoke('rsi_fs_read', {path})).data, 'base64');
  const writeFile = async (path, content) => invoke('rsi_fs_write', {path, content});
  const tools = [sdk.createReadToolDefinition(cwd, {operations:{readFile,
    access:path => invoke('rsi_fs_access', {path}), detectImageMimeType:async () => null}})];
  if (options.writable) {
    tools.push(sdk.createEditToolDefinition(cwd, {operations:{readFile, writeFile,
      access:path => invoke('rsi_fs_access', {path, write:true})}}));
    tools.push(sdk.createWriteToolDefinition(cwd, {operations:{writeFile,
      mkdir:path => invoke('rsi_fs_mkdir', {path})}}));
  }
  return {tools:tools.map(tool => ({...tool, executionMode:'sequential',
    execute:async (id, ...args) => {
      callId = id;
      try { return await tool.execute(id, ...args); }
      finally { callId = undefined; }
    }}))};
}
