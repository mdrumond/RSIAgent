export const recoveryRepair=`
def tool_seconds(events, operation):
 starts=[e['at'] for e in events if e['operation']==operation and e['phase']=='start']
 ends=[e['at'] for e in events if e['operation']==operation and e['phase']=='end']
 if len(starts)!=1 or len(ends)!=1 or ends[0]<starts[0]:return None
 return ends[0]-starts[0]

def next_action(case, actions):
 if case['expected']!=case['observed']:
  actions.report('runtime-identity');name='blocked'
 elif case['lease_error'] in ('missing-receipt','host-fence-unresolved'):
  actions.report('harness-lease');name='blocked'
 elif case['current_verified']:
  actions.accept();name='accept'
 elif case['remaining_seconds']<case['verification_seconds']:
  actions.checkpoint();name='checkpoint'
 elif case['locally_correct']:
  actions.verify();name='validate'
 else:
  actions.repair();name='repair'
 return {'action':name,'tool_seconds':tool_seconds(case['events'],'retrieve')}
`;
