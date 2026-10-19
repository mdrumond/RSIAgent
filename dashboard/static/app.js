// Keep live status useful without reloading long code pages.
if (document.querySelector('.status.running, .operation.running, .operation.queued')) setTimeout(() => location.reload(), 5000);

const csrf = document.body.dataset.csrf;
async function control(action, subject, body) {
  if (!confirm(`Confirm ${action} for ${subject}?`)) return;
  body.confirmation = `${action}:${subject}`;
  const response = await fetch(`/api/control/${action}`, {
    method: 'POST', headers: {'Content-Type': 'application/json', 'X-CSRF-Token': csrf,
      'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify(body)
  });
  if (!response.ok) alert((await response.json()).error || `Request failed: ${response.status}`);
  else location.reload();
}
document.querySelectorAll('button[data-action]').forEach(button => button.addEventListener('click', () => {
  const action = button.dataset.action, subject = button.dataset.subject;
  control(action, subject, action === 'start' ? {preset_id: button.dataset.presetId} : {campaign_id: button.dataset.campaignId});
}));
document.querySelectorAll('[data-proposal-action]').forEach(button => button.addEventListener('click', event => {
  event.preventDefault();
  const form = button.closest('form'), action = button.dataset.proposalAction;
  const body = {campaign_id: form.dataset.campaignId, proposal_id: form.dataset.proposalId};
  if (action === 'approve') body.project_ids = [...form.querySelectorAll('input:checked')].map(item => item.value);
  control(action, form.dataset.proposalId, body);
}));

document.querySelectorAll('.trace-event[data-trace-url]').forEach(section => section.addEventListener('toggle', async () => {
  if (!section.open || section.dataset.loaded) return;
  const output = section.querySelector('.trace-payload');
  try {
    const response = await fetch(section.dataset.traceUrl);
    if (!response.ok) throw new Error(`payload unavailable (${response.status})`);
    output.textContent = (await response.json()).text;
    section.dataset.loaded = 'true';
  } catch (error) {
    output.textContent = error.message;
  }
}));
