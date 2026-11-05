// Dependency for CPU Ascend fixtures outside the managed CLI cgroup. Never a
// profile option: production always reads kernel files in its actual context.
export const hostMemorySource = `def host_memory_proof(configuration):
 return {'available':True,'kind':'rsi.slice','scope':'/fixture/rsi.slice','fingerprint':'${'a'.repeat(64)}','max_bytes':4096,'current_bytes':123}
`;
