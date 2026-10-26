document.querySelectorAll('[data-force-memory]').forEach(form => form.addEventListener('submit', async event => {
  event.preventDefault();
  if (!confirm(`Merge ${form.dataset.project} into memory, overriding its usefulness admission?`)) return;
  const output = form.querySelector('[data-force-status]'), button = form.querySelector('button');
  button.disabled = true;
  try {
    form.dataset.key ||= crypto.randomUUID();
    const response = await fetch(form.dataset.forceMemory, {method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-CSRF-Token': form.dataset.csrf,
      'Idempotency-Key': form.dataset.key}, body: JSON.stringify({reason: form.elements.reason.value,
        confirmation: `force-memory:${form.dataset.project}`})});
    let result = await response.json();
    if (!response.ok) throw new Error(result.error || 'Memory merge failed');
    while (result.operation_id) {
      const operation = await (await fetch(`/api/operations/${result.operation_id}`)).json();
      if (operation.status === 'failed') throw new Error(operation.error);
      if (operation.status === 'succeeded') { result = operation.result; break; }
      output.textContent = `Memory publication ${operation.status}…`;
      await new Promise(resolve => setTimeout(resolve, 750));
    }
    output.textContent = `Published memory version ${result.admission.version}`;
    location.reload();
  } catch (error) { output.textContent = error.message; button.disabled = false; }
}));
