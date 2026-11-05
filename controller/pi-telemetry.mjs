import { appendFileSync, closeSync, existsSync, fsyncSync, openSync } from 'node:fs';
import { dirname } from 'node:path';

function exception(error, seen = new Set()) {
  if (!(error instanceof Error)) return error;
  if (seen.has(error)) return {name:error.name,message:error.message,circularCause:true};
  seen.add(error);
  return {name:error.name,message:error.message,code:error.code,errcode:error.errcode,
    errstr:error.errstr,stack:error.stack,
    ...(error.cause === undefined ? {} : {cause:exception(error.cause,seen)})};
}

// A journal is an alternate telemetry destination, never a queue for SQL replay.
export function createPiTelemetryJournal(path, attempt, write) {
  let journal = existsSync(path);
  let directorySynced = false;
  return data => {
    const record = {attempt,type:'pi-event',created:new Date().toISOString(),data};
    if (!journal) {
      try { write(data); return; }
      catch (error) {
        record.persistence = {operation:'State.event INSERT INTO events',
          sqliteMirror:'disabled-for-attempt',error:exception(error)};
        journal = true;
      }
    }
    // Keep this outside the SQL catch: journal failures must escape to the caller.
    const fd = openSync(path,'a',0o600);
    try {
      appendFileSync(fd,JSON.stringify(record)+'\n');
      fsyncSync(fd);
    } finally { closeSync(fd); }
    // Persist the directory entry too, including the first newly-created journal.
    if (!directorySynced) {
      const directory = openSync(dirname(path),'r');
      try { fsyncSync(directory); } finally { closeSync(directory); }
      directorySynced = true;
    }
  };
}
