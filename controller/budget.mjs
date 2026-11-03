import { performance } from 'node:perf_hooks';

// Only explicit setup and transfer scopes pause the actor clock.
export class ActorClock {
  constructor(seconds, elapsed = 0, now = () => performance.now()) {
    if (!(seconds > 0) || elapsed < 0) throw new Error('Invalid elapsed allocation');
    this.limit = seconds * 1000;
    this.saved = elapsed * 1000;
    this.now = now;
    this.started = now();
    this.depth = 0;
  }
  get elapsed() { return (this.saved + (this.depth ? 0 : this.now() - this.started)) / 1000; }
  get remaining() { return Math.max(0, this.limit / 1000 - this.elapsed); }
  check() { if (!this.remaining) throw Object.assign(new Error('Actor elapsed allocation exhausted'),{code:'ACTOR_ELAPSED_EXHAUSTED'}); }
  async exclude(label, operation) {
    if (!['setup', 'transfer'].includes(label)) throw new Error('Unsupported clock exclusion');
    if (!this.depth++) this.saved += this.now() - this.started;
    try { return await operation(); }
    finally { if (!--this.depth) this.started = this.now(); }
  }
}
