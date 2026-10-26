document.querySelectorAll('[data-curriculum]').forEach(section => {
  const status = section.querySelector('[data-curriculum-status]');
  const form = section.querySelector('[data-curriculum-preview]');
  async function submit(action, body, subject) {
    body.campaign_id = section.dataset.curriculum;
    body.confirmation = `${action}:${subject}`;
    const response = await fetch(`/api/control/${action}`, {method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf,
      'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify(body)});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Request failed (${response.status})`);
    status.textContent = 'Request accepted; waiting for completion…';
    for (;;) {
      const operationResponse = await fetch(`/api/operations/${result.operation_id}`);
      if (!operationResponse.ok) throw new Error('Operation status unavailable; reload to inspect retained proposals');
      const operation = await operationResponse.json();
      if (operation.status === 'failed') throw new Error(operation.error);
      if (operation.status === 'succeeded') { location.reload(); return; }
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  form?.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button'); button.disabled = true;
    try {
      const body = Object.fromEntries(new FormData(form));
      body.format_seconds = Number(body.format_seconds);
      await submit('curriculum-preview', body, section.dataset.curriculum);
    } catch (error) { status.textContent = error.message; button.disabled = false; }
  });
  section.querySelectorAll('[data-curriculum-approve]').forEach(button => button.addEventListener('click', async () => {
    const body = {proposal_id: button.dataset.curriculumApprove};
    if (button.dataset.restart === 'yes') {
      body.additional_seconds = Number(button.closest('article').querySelector('[data-restart-seconds]').value);
      if (!(body.additional_seconds > 0)) { status.textContent = 'Enter a positive restart allowance.'; return; }
    }
    if (!confirm('Apply this reviewed curriculum proposal?')) return;
    button.disabled = true;
    try { await submit('curriculum-approve', body, body.proposal_id); }
    catch (error) { status.textContent = error.message; button.disabled = false; }
  }));
});
