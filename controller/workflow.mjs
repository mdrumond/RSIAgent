export const stages = Object.freeze(['preflight','task-construction','design','implementation',
  'test','debug','optimisation','validation','acceptance']);

export class Workflow {
  constructor({event, fingerprint, verify, clock, readSkill, search}) {
    Object.assign(this,{event,fingerprint,verify,clock,readSkill,search});
    this.index = 0;
    this.facts = {};
    this.validation = null;
    this.history = [];
  }
  restore(events) {
    for (const {type,data} of events) {
      if (type === 'stage-completed') {
        this.history.push(data);
        this.index = stages.indexOf(data.next);
        if (this.index < 0) throw new Error('Unknown persisted workflow stage');
        this.facts = {};
        if (['design','implementation'].includes(data.next)) this.validation = null;
      } else if (type === 'skill-read' && data.stage === this.stage) this.facts.skill = true;
      else if (type === 'knowledge-retrieved' && data.stage === this.stage) this.facts[data.kind] = true;
      else if (type === 'verification') this.validation = data;
    }
    return this;
  }
  get stage() { return stages[this.index]; }
  async skill() {
    const result = await this.readSkill(this.stage);
    this.facts.skill = true;
    await this.event('skill-read',{stage:this.stage,...result.evidence});
    return result.content;
  }
  async query(query,kind,ctx) {
    if (!['good','negative'].includes(kind)) throw new Error('Query good and negative evidence separately');
    const result = await this.search({query,kind},ctx);
    this.facts[kind] = true;
    await this.event('knowledge-retrieved',{stage:this.stage,query,kind,evidence:result.evidence});
    return result.content;
  }
  async validate() {
    this.clock.check();
    if (this.stage !== 'validation') throw new Error('Verification belongs to validation');
    const before = await this.fingerprint();
    const result = await this.verify();
    const after = await this.fingerprint();
    this.validation = {passed:result.passed === true && before === after, fingerprint:after};
    await this.event('verification',{...this.validation,evidence:result.evidence});
    return this.validation;
  }
  async advance({reflection = '', next} = {}) {
    this.clock.check();
    if (!this.facts.skill) throw new Error('Read the current step skill first');
    if (['design','optimisation'].includes(this.stage)) {
      if (!this.facts.good || !this.facts.negative) throw new Error('Retrieve positive and negative knowledge first');
      if (!reflection.trim()) throw new Error('Record what previous evidence supports and rules out');
    }
    if (this.stage === 'acceptance') {
      if (!this.validation?.passed || this.validation.fingerprint !== await this.fingerprint())
        throw new Error('Acceptance requires a passing verifier on the current source');
      await this.event('accepted',{});
      return {accepted:true};
    }
    const destination = next ?? stages[this.index+1];
    const allowed = [stages[this.index+1]];
    if (['debug','optimisation','validation'].includes(this.stage)) allowed.push('design','implementation');
    if (!allowed.includes(destination)) throw new Error(`Current stage ${this.stage}; permitted destinations: ${allowed.join(', ')}. Use rsi_advance for the next step.`);
    if (this.stage === 'validation' && destination === 'acceptance' && !this.validation?.passed)
      throw new Error('Validation did not pass');
    this.history.push({stage:this.stage,reflection,next:destination});
    await this.event('stage-completed',this.history.at(-1));
    this.index = stages.indexOf(destination);
    this.facts = {};
    if (['design','implementation'].includes(destination)) this.validation = null;
    return {stage:this.stage,history:this.history};
  }
}
