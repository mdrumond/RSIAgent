// Required workload, not retry/iteration stopping limits. Individual replay selects one case.
export const requiredCases=[
  ...['existing-regressions','workspace-snapshots','lifecycle-matrix','lifecycle-native-compaction',
    'campaign-recovery','installed-cli-dashboard','seeded-workspace','seeded-soak'].map(id=>({id,tier:'local'})),
  ...['deepseek-readiness','deepseek-workspace-revision','deepseek-construction-resume',
    'deepseek-worker-loss','deepseek-compaction','deepseek-wave-recovery'].map(id=>({id,tier:'deepseek'})),
  ...['remote-workspace','remote-construction-resume','remote-observer-resume','installed-cli-dashboard-cpl','registered-cpu-transport']
    .map(id=>({id,tier:'remote'})),
];
