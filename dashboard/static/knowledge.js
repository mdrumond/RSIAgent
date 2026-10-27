const questionForm = document.querySelector('#knowledge-question');
questionForm?.addEventListener('submit', async event => {
  event.preventDefault();
  const status = document.querySelector('#knowledge-status'), result = document.querySelector('#knowledge-answer');
  const button = questionForm.querySelector('button');
  status.textContent = 'Searching sources and asking the librarian…'; result.hidden = true; button.disabled = true;
  try {
    const response = await fetch(questionForm.dataset.url, {method: 'POST', headers: {
      'Content-Type': 'application/json', 'X-CSRF-Token': document.body.dataset.csrf, 'Idempotency-Key': crypto.randomUUID()}, body: JSON.stringify({
      campaign_id: questionForm.elements.campaign_id.value, question: questionForm.elements.question.value,
      max_seconds: Number(questionForm.elements.max_seconds.value)})});
    let data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Knowledge query failed');
    while (data.operation_id) {
      const operation = await (await fetch(`/api/operations/${data.operation_id}`)).json();
      if (operation.status === 'failed') throw new Error(operation.error);
      if (operation.status === 'succeeded') { data = operation.result; break; }
      status.textContent = `Librarian request ${operation.status}…`;
      await new Promise(resolve => setTimeout(resolve, 750));
    }
    result.querySelector('[data-answer]').textContent = data.answer;
    const sources = result.querySelector('[data-sources]'); sources.replaceChildren();
    for (const source of data.sources) {
      const article = document.createElement('article'), citation = document.createElement('pre'), excerpt = document.createElement('pre');
      citation.textContent = JSON.stringify(source.citation, null, 2); excerpt.textContent = source.text;
      article.append(citation, excerpt); sources.append(article);
    }
    status.textContent = ''; result.hidden = false;
  } catch (error) { status.textContent = error.message; }
  finally { button.disabled = false; }
});
