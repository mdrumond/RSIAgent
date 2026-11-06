import {bootstrap} from './instance.mjs';

export const workloads=['fa','gdn','bsa','kimi_delta_attention','lact_ttt_chunk',
  'sparse_lightning_indexer_grad_kl_loss_enhance','tensor_program_vm'];
const topics=['movement/layout','vector fusion/dtypes','reductions/scans','normalization/softmax',
  'matmul/tiling','pipelining/synchronization','sparse indexing','stateful recurrence'];
const repeats=[[0,1],[2,3],[4,5],[6,0],[1,2],[3,4],[5,6],[0,1],[2,3]];
const seconds=30*24*60*60;
const simt='SIMT usage/benchmarking';

// Machine routes, installations, credentials and baseline paths come from the private profile.
export function buildCampaignManifest({campaignId,target,profile,phase2Projects}) {
  if(!/^[a-z0-9][a-z0-9-]*$/.test(campaignId??'')||!['a3','a5'].includes(target))
    throw new Error('Campaign ID and supported target required');
  if(typeof profile?.baseline!=='string'||typeof profile.skills!=='string'||!profile.environment)
    throw new Error('Fresh baseline KDB, baseline skills and environment profile required');
  for(const workload of workloads)if(!phase2Projects?.[workload]?.source||!phase2Projects[workload].verifier)
    throw new Error('Missing frozen Phase 2 contract: '+workload);
  const {campaign_plan:_plan,phase_sequence:_sequence,phase_targets:_targets,completion_target:_completion,
    projects:_projects,curriculum:_curriculum,storage:_storage,root:_root,revision:_revision,
    baselineKDB:_baseline,skillSettings,...settings}=structuredClone(profile);
  const {evaluationSeconds:_evaluation,...skillPolicy}=skillSettings??{};
  const plan=[],projects=[];
  for(let round=1;round<=16;round++) {
    const prefix=`${campaignId}-${target}-r${String(round).padStart(2,'0')}`;
    plan.push({id:prefix+'-p1',phase:'phase1',new_projects:8,generation:{
      topics:[simt,...topics.filter((_topic,index)=>index!==Math.floor((round-1)/2))],
      requirements:{[simt]:{performance:true,capabilityEvidence:true,comparison:'non-simt-or-vector'}}}});
    const selected=round<=7?[round-1]:repeats[round-8];
    const entry={id:prefix+'-p2',phase:'phase2',new_projects:selected.length};plan.push(entry);
    for(const index of selected)projects.push({...structuredClone(phase2Projects[workloads[index]]),
      id:entry.id+'-'+workloads[index],phase:'phase2',campaign_entry:entry.id,
      campaign_workload:workloads[index],actor_seconds:28800});
  }
  return {...settings,mode:'phase1',target,campaign_cohort:campaignId,
    kernel_language:target==='a3'?'triton':'catlass-dsl',actor_seconds:7200,curriculum_seconds:1800,
    campaign_seconds:seconds,recovery:false,concurrency:8,tune_every:1,campaign_plan:plan,projects,
    skillSettings:{...skillPolicy,activeVersion:'baseline',tuningDisabled:false,background:true,runsPerArm:2,
      tuningProjectIds:projects.slice(0,2).map(project=>project.id)},
    curriculum:{phase1:[],phase2:projects.map(project=>project.id),generation:{enabled:true,topics,
      by_phase:{phase1:{enabled:true},phase2:{enabled:false}}}}};
}

export function campaignEntryConfig(config,entryId) {
  const entry=config.campaign_plan.find(item=>item.id===entryId);
  if(!entry)throw new Error('Unknown campaign entry');
  return {...config,mode:entry.phase,actor_seconds:entry.phase==='phase1'?7200:28800,
    curriculum:{...config.curriculum,generation:{...config.curriculum.generation,...entry.generation}}};
}

export function qualificationGate({localValidation,receipts,expected}) {
  if(localValidation?.passed!==true||!localValidation.sourceRevision||
    localValidation.sourceRevision!==expected?.sourceRevision)
    throw new Error('Passing local validation bound to a source revision required');
  for(const target of ['a3','a5'])for(const workload of workloads) {
    const matches=(receipts??[]).filter(item=>item.target===target&&item.workload===workload);
    if(matches.length!==1)throw new Error(`One qualification required: ${target}/${workload}`);
    const receipt=matches[0],contract=expected?.[target]?.[workload];
    if(!contract||receipt.format!==1||receipt.qualified!==true||
      ['sourceHash','caseHash','verifierHash'].some(key=>!contract[key]||receipt[key]!==contract[key])||
      !contract.runtimeFingerprint||receipt.runtime?.fingerprint!==contract.runtimeFingerprint)
      throw new Error(`Qualification binding mismatch: ${target}/${workload}`);
    const ids=contract.caseIds;
    if(!Array.isArray(ids)||!ids.length||new Set(ids).size!==ids.length||
      !Array.isArray(receipt.cases)||receipt.cases.length!==ids.length||
      new Set(receipt.cases.map(item=>item.id)).size!==ids.length||
      receipt.cases.some(item=>!ids.includes(item.id)||item.passed!==true||!item.referenceHash||!item.outputHash||
        !Number.isFinite(item.maxAbsError)||item.maxAbsError<0))
      throw new Error(`Numerical qualification incomplete: ${target}/${workload}`);
    const samples=receipt.timing?.samples;
    if(receipt.timing?.passed!==true||!Array.isArray(samples)||samples.length!==ids.length||
      new Set(samples.map(item=>item.id)).size!==ids.length||samples.some(item=>!ids.includes(item.id)||
        !Array.isArray(item.seconds)||!item.seconds.length||item.seconds.some(value=>!Number.isFinite(value)||value<=0)))
      throw new Error(`Timing qualification incomplete: ${target}/${workload}`);
  }
  return {sourceRevision:localValidation.sourceRevision,qualified:14};
}

// All callers use the same fresh-instance copying/storage contract as the normal CLI.
export async function bootstrapFreshCampaign({destination,qualification,...input},{create=bootstrap}={}) {
  qualificationGate(qualification);
  return create(destination,buildCampaignManifest(input));
}
