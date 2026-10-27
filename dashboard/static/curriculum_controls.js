document.querySelectorAll('[data-curriculum]').forEach(section => {
  const status = section.querySelector('[data-curriculum-status]');
  const form = section.querySelector('[data-curriculum-preview]');
  const reconcile = section.querySelector('[data-curriculum-reconcile]');
  const storageKey = `curriculum:${section.dataset.curriculum}:${section.dataset.curriculumProject || ''}`;
  let pending = JSON.parse(sessionStorage.getItem(storageKey) || 'null'), submitting = false;
  function persist() {
    if (pending) sessionStorage.setItem(storageKey, JSON.stringify(pending));
    else sessionStorage.removeItem(storageKey);
  }
  function refresh() {
    section.querySelectorAll('input, textarea, select, button').forEach(input => {
      input.disabled = submitting || Boolean(pending);
    });
    reconcile.hidden = !pending;
    reconcile.disabled = submitting;
  }
  async function submit(action, body, subject) {
    if (submitting) return;
    submitting = true;
    try {
      if (!pending) {
        body.campaign_id = section.dataset.curriculum;
        body.confirmation = `${action}:${subject}`;
        pending = {action, key: crypto.randomUUID(), body: JSON.stringify(body), operation: null};
      }
      // Persist before dispatch so a reload can replay the exact accepted request.
      persist(); refresh();
      if (!pending.operation) {
        const response = await fetch(`/api/control/${pending.action}`, {method: 'POST', headers: {
          'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf,
          'Idempotency-Key': pending.key}, body: pending.body});
        if (!response.ok) {
          if (response.status >= 400 && response.status < 500) { pending = null; persist(); }
          const failure = await response.json();
          throw new Error(failure.error || `Request failed (${response.status})`);
        }
        const result = await response.json();
        pending.operation = result.operation_id;
        persist();
      }
      status.textContent = 'Request accepted; waiting for completion…';
      for (;;) {
        const response = await fetch(`/api/operations/${pending.operation}`);
        if (!response.ok) throw new Error('Operation status unavailable');
        const operation = await response.json();
        if (operation.status === 'failed') {
          pending = null; persist();
          throw new Error(operation.error || 'Curriculum operation failed');
        }
        if (operation.status === 'succeeded') {
          pending = null; persist(); location.reload(); return;
        }
        await new Promise(resolve => setTimeout(resolve, 1000));
      }
    } catch (error) {
      status.textContent = error.message + (pending ? ' Reconcile the original request before submitting another change.' : '');
    } finally { submitting = false; refresh(); }
  }
  form?.addEventListener('submit', event => {
    event.preventDefault();
    const body = Object.fromEntries(new FormData(form));
    body.format_seconds = Number(body.format_seconds);
    return submit('curriculum-preview', body, section.dataset.curriculum);
  });
  section.querySelectorAll('[data-curriculum-approve]').forEach(button => button.addEventListener('click', () => {
    const body = {proposal_id: button.dataset.curriculumApprove};
    if (button.dataset.restart === 'yes') {
      body.additional_seconds = Number(button.closest('article').querySelector('[data-restart-seconds]').value);
      if (!(body.additional_seconds > 0)) { status.textContent = 'Enter a positive restart allowance.'; return; }
    }
    if (!confirm('Apply this reviewed curriculum proposal?')) return;
    return submit('curriculum-approve', body, body.proposal_id);
  }));
  reconcile.addEventListener('click', () => pending && submit());
  if (pending) status.textContent = 'A prior request needs reconciliation.';
  refresh();
});

// Restart workers outlive their dashboard operation; follow their retained status.
document.querySelectorAll('[data-restart-status-url]').forEach(output => {
  let terminal = false;
  async function refresh() {
    if (terminal) return;
    try {
      const response = await fetch(output.dataset.restartStatusUrl);
      if (!response.ok) throw new Error('Restart status unavailable; reload to reconcile');
      const {request} = await response.json();
      output.textContent = request ? `Restart: ${request.status}${request.error ? ` · ${request.error}` : ''}` : 'Restart: not dispatched; use Resume/reconcile';
      if (request?.status === 'observation-required') output.textContent += ' · Observe the prior execution before resubmitting.';
      if (request && ['succeeded', 'observation-required'].includes(request.status)) {
        const retry = output.closest('article').querySelector('[data-curriculum-approve]');
        if (retry) retry.hidden = true;
      }
      terminal = request && ['succeeded', 'failed', 'observation-required'].includes(request.status);
    } catch (error) { output.textContent = error.message; }
  }
  refresh();
  setInterval(refresh, 2000);
});
