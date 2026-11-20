/** Search replies can expose a note even without a separate entry open. */
export function searchableExposure(entries,args={}){
 return entries.flatMap(entry=>{
  if(typeof entry.text!=='string')return [];
  const ids=[...new Set([...entry.text.matchAll(/searchable-skill:([a-f0-9]{64})\b/g)].map(match=>match[1]))];
  return ids.map(id=>({id,collection:entry.collection??args.collection??null,entryId:entry.chunk_id??entry.entry_id??args.entry_id??null}));
 });
}
