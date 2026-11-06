// This is a fixed local experiment, even when its private profile came from a campaign.
export function localPythonConfig(profile, projects) {
  const { campaign_plan: _plan, phase_sequence: _sequence, phase_targets: _targets,
    completion_target: _target, resource_ids: _resources, ...settings } = profile;
  const ids = projects.map(project => project.id);
  return { ...settings, projects, mode: 'phase2', tune_every: 2, target: 'cpu', concurrency: 1,
    curriculum: { phase2: ids },
    skillSettings: { ...settings.skillSettings, tuningProjectIds: ids },
    environment: { module: 'machinery/integrations/environments/index.mjs', options: { mode: 'local' } } };
}
