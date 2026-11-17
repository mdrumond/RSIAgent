// State is synchronous: waiting here blocks its owner, just as SQLite's native
// busy handler does. Keep that wait elapsed-bounded and never replay a workflow
// or transaction body. Competing writers must be separate processes/connections.
const sleeper=new Int32Array(new SharedArrayBuffer(4));

export function sqliteBackoff(db,{seconds=30}={}) {
  if(!Number.isFinite(seconds)||seconds<=0)throw new Error('SQLite contention seconds must be positive and finite');
  const exec=db.exec.bind(db),prepare=db.prepare.bind(db);
  // Short native waits allow the remaining elapsed allowance to be checked.
  exec(`PRAGMA busy_timeout=${Math.min(100,Math.ceil(seconds*1000))}`);
  function retry(operation,{commit=false}={}) {
    const deadline=performance.now()+seconds*1000;let delay=25;
    for(;;) {
      try{return operation();}
      catch(error) {
        // A stale WAL snapshot cannot become writable by waiting. Retrying a
        // locked transaction would retain its snapshot and may prevent release.
        if(error?.code!=='ERR_SQLITE_ERROR'||error.errcode!==5||
            (db.isTransaction&&!commit))throw error;
        const remaining=deadline-performance.now();
        if(remaining<=0)throw error;
        Atomics.wait(sleeper,0,0,Math.min(delay,remaining));
        if(performance.now()>=deadline)throw error;
        delay=Math.min(delay*2,500);
      }
    }
  }
  db.exec=function(sql) {
    // exec accepts multiple statements, whose earlier effects could have
    // committed before a later BUSY. Only these single transaction controls
    // are safe to retry without parsing or replaying an arbitrary script.
    if(/^\s*BEGIN\s+IMMEDIATE\s*;?\s*$/i.test(sql))return retry(()=>exec(sql));
    if(/^\s*COMMIT\s*;?\s*$/i.test(sql))return retry(()=>exec(sql),{commit:true});
    return exec(sql);
  };
  db.prepare=function(...args) {
    const statement=retry(()=>prepare(...args));
    for(const method of ['run','get','all']) {
      const invoke=statement[method].bind(statement);
      statement[method]=(...parameters)=>retry(()=>invoke(...parameters));
    }
    // Iterators may already have yielded rows; never restart their execution.
    return statement;
  };
  // Schema setup contains only idempotent CREATE IF NOT EXISTS and PRAGMAs.
  // Give it the same contention allowance without making arbitrary exec safe.
  return {db,initialize:sql=>retry(()=>exec(sql))};
}
