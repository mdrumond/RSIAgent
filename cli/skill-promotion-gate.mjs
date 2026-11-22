// A persisted admission gate, released only by successful skill publication.
export function skillPromotionPending(config) {
  const gate=config.skills?.promotionGate;
  if(!gate)return false;
  if(typeof gate.baseVersion!=='string'||!gate.baseVersion||!['waiting','released'].includes(gate.status)||
      gate.status==='released'&&(typeof gate.acceptedVersion!=='string'||!gate.acceptedVersion||gate.acceptedVersion===gate.baseVersion))
    throw new Error('Invalid skill promotion gate');
  return gate.status==='waiting';
}

export function publishedPromotionGate(skills,{active,candidate}) {
  const gate=skills.promotionGate;
  if(!skillPromotionPending({skills})||gate.baseVersion!==active||candidate===active)return gate;
  return {...gate,status:'released',acceptedVersion:candidate};
}
