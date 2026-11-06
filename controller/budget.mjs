import { performance } from 'node:perf_hooks';

// Admission holds unpublished time until its receipt distinguishes work from queue wait.
export class ActorClock {
  constructor(seconds, elapsed = 0, now = () => performance.now()) {
    if (!(seconds > 0) || elapsed < 0) throw new Error('Invalid elapsed allocation');
    this.limit = seconds * 1000;
    this.saved = elapsed * 1000;
    this.now = now;
    this.started = now();
    this.depth = 0;
    this.pending = [];
  }
  advance() {
    const now = this.now(), duration = now - this.started;
    this.started = now;
    if (this.depth) return;
    if (this.pending.length) this.pending.at(-1).billable += duration;
    else this.saved += duration;
  }
  get elapsed() { this.advance(); return this.saved / 1000; }
  get remaining() { return Math.max(0, this.limit / 1000 - this.elapsed); }
  check() { if (!this.remaining) throw Object.assign(new Error('Actor elapsed allocation exhausted'),{code:'ACTOR_ELAPSED_EXHAUSTED'}); }
  async exclude(label, operation) {
    if (label === 'admission') {
      this.advance();
      const pending = { billable: 0 };
      this.pending.push(pending);
      let charge = true;
      try {
        const result = await operation();
        charge = result?.queueWait !== true;
        return result;
      } finally {
        this.advance();
        this.pending.pop();
        if (charge) {
          if (this.pending.length) this.pending.at(-1).billable += pending.billable;
          else this.saved += pending.billable;
        }
      }
    }
    if (!['setup', 'transfer', 'queue-wait'].includes(label)) throw new Error('Unsupported clock exclusion');
    this.advance();
    this.depth++;
    try { return await operation(); }
    finally { this.advance(); this.depth--; }
  }
}
