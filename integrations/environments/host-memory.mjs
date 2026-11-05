import {readFile} from 'node:fs/promises';

export const hostMemorySource = await readFile(new URL('./host-memory.py', import.meta.url), 'utf8');
export function hostMemoryConfiguration(options = {}) {
  const slice = options.slice ?? 'rsi.slice';
  if (typeof slice !== 'string' || !/^[A-Za-z0-9_][A-Za-z0-9_.-]*\.slice$/.test(slice))
    throw new Error('host_memory.slice must name a shared system or user slice unit');
  const max_bytes = options.max_bytes ?? 16 * 1024 ** 3;
  if (!Number.isSafeInteger(max_bytes) || max_bytes <= 0) throw new Error('host_memory.max_bytes must be a positive safe integer');
  if (options.container_root !== undefined && (typeof options.container_root !== 'string' ||
      !options.container_root.startsWith('/') || options.container_root.split('/').some(p => p === '..' || p === '.')))
    throw new Error('host_memory.container_root must be an absolute registered cgroup root');
  return {max_bytes, slice, ...(options.container_root ? {container_root:options.container_root} : {})};
}
export function requireHostMemory(proof, configuration) {
  if (proof?.available !== true || !Number.isSafeInteger(proof.max_bytes) || proof.max_bytes <= 0 ||
      proof.max_bytes > configuration.max_bytes || !Number.isSafeInteger(proof.current_bytes) || proof.current_bytes < 0 ||
      !['rsi.slice', 'container-root'].includes(proof.kind) || !/^[a-f0-9]{64}$/.test(proof.fingerprint ?? '') ||
      (configuration.fingerprint && configuration.fingerprint !== proof.fingerprint)) {
    throw Object.assign(new Error(proof?.detail ?? 'Verified aggregate target memory enforcement required; register a memory-bounded context'),
      {code:'MEMORY_ENFORCEMENT_UNAVAILABLE', host_memory:proof ?? null});
  }
  return proof;
}
