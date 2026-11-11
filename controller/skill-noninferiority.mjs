// An observed resource-cost gate, not a statistical non-inferiority test.
// Callers freeze projectIds, repetitions and tolerances before collecting results.
export function evaluateNonInferiority(results, {projectIds, runs = 2, margin = 0.05,
  metrics = ['elapsedSeconds', 'tokens']} = {}) {
  if (!Array.isArray(projectIds) || !projectIds.length ||
      projectIds.some(id => typeof id !== 'string' || !id) || new Set(projectIds).size !== projectIds.length)
    throw new Error('Distinct nonempty project IDs required');
  if (!Number.isInteger(runs) || runs < 1) throw new Error('Positive integer repetitions required');
  if (!Number.isFinite(margin) || margin < 0 || margin > 1) throw new Error('Margin must be between zero and one');
  if (!Array.isArray(metrics) || !metrics.length || new Set(metrics).size !== metrics.length ||
      metrics.some(metric => !['elapsedSeconds', 'tokens'].includes(metric)))
    throw new Error('Select elapsedSeconds and/or tokens resource metrics');
  const report = {promote: false, reason: 'inconclusive', evidence: 'observed', margin,
    runs, projectIds: [...projectIds], metrics: {}};
  if (!Array.isArray(results) || results.length !== projectIds.length * runs * 2) return report;
  const pairs = [];
  for (const projectId of projectIds) for (let run = 0; run < runs; run++) {
    const pair = {projectId, run};
    for (const arm of ['active', 'candidate']) {
      const matches = results.filter(result => result?.projectId === projectId && result.run === run && result.arm === arm);
      if (matches.length !== 1 || typeof matches[0].correct !== 'boolean') return report;
      pair[arm] = matches[0];
    }
    pairs.push(pair);
  }
  if (pairs.some(pair => !pair.candidate.correct)) return {...report, reason: 'correctness-regression'};
  if (runs < 2) return {...report, reason: 'insufficient-repetitions'};
  report.correctnessImproved = pairs.some(pair => !pair.active.correct);
  for (const metric of metrics) {
    // Zero/missing costs cannot define a multiplicative comparison. Never turn
    // unknown token accounting into a free candidate or a synthetic zero cost.
    if (pairs.some(pair => ['active', 'candidate'].some(arm =>
      !Number.isFinite(pair[arm][metric]) || pair[arm][metric] <= 0)))
      return {...report, reason: 'missing-cost-telemetry', missingMetric: metric};
    const tasks = projectIds.map(projectId => {
      const observations = pairs.filter(pair => pair.projectId === projectId).map(pair => ({
        run: pair.run, active: pair.active[metric], candidate: pair.candidate[metric],
        logRatio: Math.log(pair.candidate[metric]) - Math.log(pair.active[metric]),
      }));
      const logRatio = observations.reduce((sum, row) => sum + row.logRatio, 0) / runs;
      return {projectId, ratio: Math.exp(logRatio), logRatio,
        pairs: observations.map(({logRatio, ...row}) => ({...row, ratio: Math.exp(logRatio)}))};
    });
    const logRatio = tasks.reduce((sum, task) => sum + task.logRatio, 0) / tasks.length;
    // Compare in log space so extreme finite observations cannot overflow the gate.
    report.metrics[metric] = {ratio: Math.exp(logRatio), withinMargin: logRatio <= Math.log1p(margin) + 1e-12,
      tasks: tasks.map(({logRatio: _logRatio, ...task}) => task)};
  }
  return {...report, promote: Object.values(report.metrics).every(metric => metric.withinMargin),
    reason: Object.values(report.metrics).every(metric => metric.withinMargin)
      ? 'observed-non-inferiority' : 'aggregate-cost-regression'};
}
