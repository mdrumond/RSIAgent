document.querySelectorAll('[data-actor-feedback]').forEach(section => {
  const url = section.dataset.actorFeedback, form = section.querySelector('form');
  const status = section.querySelector('[data-feedback-status]');
  const receipts = section.querySelector('[data-feedback-receipts]');
  let requestUrl;
  async function poll() {
    try {
      const response = await fetch(url), body = await response.json();
      if (response.ok) {
        receipts.replaceChildren(...body.messages.map(message => {
          const row = document.createElement('li');
          row.textContent = `${message.consumed ? 'Consumed' : 'Pending'}: ${message.text}`;
          return row;
        }));
      }
      if (requestUrl) {
        const response = await fetch(requestUrl), operation = await response.json();
        status.textContent = operation.status + (operation.error ? ': ' + operation.error : '');
      }
    } catch (error) { status.textContent = 'Connection unavailable; retained request status will reconnect.'; }
  }
  form.addEventListener('submit', async event => {
    event.preventDefault();
    const button = form.querySelector('button'); button.disabled = true;
    try {
      const csrf = await (await fetch('/api/csrf')).json();
      const response = await fetch(url, {method: 'POST', headers: {
        'Content-Type': 'application/json', 'X-CSRF-Token': csrf.token,
        'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify({
          text: form.elements.text.value, additional_seconds: Number(form.elements.additional_seconds.value)})});
      const body = await response.json();
      if (!response.ok) throw new Error(body.error || 'Feedback could not be submitted');
      status.textContent = body.request ? 'Continuation queued at a safe checkpoint.' : 'Feedback sent to live actor.';
      if (body.request) requestUrl = `/api/campaigns/${encodeURIComponent(location.pathname.split('/')[2])}/manual-requests/${body.request.id}`;
      await poll();
    } catch (error) { status.textContent = error.message; }
    finally { button.disabled = false; }
  });
  poll(); setInterval(poll, 2000);
});
