# Fresh dual campaigns

`cli/campaign-manifest.mjs` builds portable bootstrap options with the native
`campaign_plan`. Supply a new cohort ID, `a3` or `a5`, a private bootstrap profile,
and a map of frozen Phase 2 project specs keyed by the exported `workloads`.
Profiles supply baseline KDB/skills paths, providers, environment routes and named
runtimes. No machine installation or credential settings are embedded in the plan.

There are sixteen eight-project Phase 1 batches and twenty-five Phase 2 originals.
Each Phase 1 entry freezes eight construction topics: one SIMT usage/benchmarking
project and seven rotating topics. SIMT has sixteen projects per target; every
other topic has fourteen. Its requirements demand performance comparison and
target capability evidence. `campaignEntryConfig(config, entryId)` selects those
entry-specific construction settings and the phase's actor allocation. Generation
must enforce these requirements; CPU construction is not target timing evidence.

`bootstrapFreshCampaign` calls the normal fresh-instance bootstrap only after
`qualificationGate` accepts local validation for `expected.sourceRevision` and all
seven numerical/timing receipts on both targets. Expected contract hashes, runtime
fingerprints and case IDs must come from the frozen qualification packs. The normal
bootstrap copies the pristine KDB and baseline skills into each new root.

Background A/B is opt-in through `skillSettings.background`. The first FA and GDN
project IDs are fixed evaluation controls; repetitions keep their frozen contracts
but get new project IDs. The scheduler supplies remaining campaign time for A/B.

`projectCatalogExtension({cohort, campaigns})` exposes `rsi_previous_projects` with
paginated `list`, `files` and `read` operations. Each member is an explicit
`{root, campaignId, target}` binding to the fresh cohort and its storage ledger.
Only acknowledged, postprocessed original attempts with closed settlement enter
the catalog. Reads use immutable source Git trees, including dirty boundary
snapshots, and never change the source campaign. Actors may borrow any useful
prior source; a failed result is labeled as failed, not recommended as a starter.
