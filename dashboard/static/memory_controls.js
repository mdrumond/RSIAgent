document.querySelectorAll('[data-force-memory]').forEach(form => {
  let pending = null, submitting = false;
  const reason = form.elements.reason;
  form.addEventListener('submit', async event => {
    event.preventDefault();
    if (submitting || !confirm(`Merge ${form.dataset.project} into memory, overriding its usefulness admission?`)) return;
    const output = form.querySelector('[data-force-status]'), button = form.querySelector('button');
    submitting = true; button.disabled = true; reason.disabled = true;
    pending ||= {key: crypto.randomUUID(), body: JSON.stringify({reason: reason.value,
      confirmation: `force-memory:${form.dataset.project}`}), operation: null};
    try {
      let result;
      if (!pending.operation) {
        const response = await fetch(form.dataset.forceMemory, {method: 'POST', headers: {
          'Content-Type': 'application/json', 'X-CSRF-Token': form.dataset.csrf,
          'Idempotency-Key': pending.key}, body: pending.body});
        if (!response.ok) {
          if (response.status >= 400 && response.status < 500) pending = null;
          const failure = await response.json();
          throw new Error(failure.error || 'Memory merge failed');
        }
        result = await response.json();
        pending.operation = result.operation_id;
      }
      while (pending.operation) {
        const response = await fetch(`/api/operations/${pending.operation}`);
        if (!response.ok) throw new Error('Memory operation status is unavailable');
        const operation = await response.json();
        if (operation.status === 'failed') {
          pending = null;
          throw new Error(operation.error || 'Memory merge failed');
        }
        if (operation.status === 'succeeded') { result = operation.result; break; }
        output.textContent = `Memory publication ${operation.status}…`;
        await new Promise(resolve => setTimeout(resolve, 750));
      }
      if (!result?.admission?.version) throw new Error('Memory publication has not been confirmed');
      pending = null;
      output.textContent = `Published memory version ${result.admission.version}`;
      location.reload();
    } catch (error) {
      output.textContent = error.message + (pending ? ' Submit again to reconcile the original request.' : '');
      button.disabled = false; reason.disabled = Boolean(pending);
    } finally { submitting = false; }
  });
});
