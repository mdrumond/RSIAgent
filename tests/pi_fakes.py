"""Scripted typed sessions for composition tests; real Pi is tested separately."""
import json
from coding.pi.session import ActionError


class ScriptedPiSession:
    def __init__(self, provider, profile):
        self.provider, self.profile = provider, profile
        self.completion_tokens = 0
        self.messages = []

    def next_action(self, context, allowed_actions, limits=None, record=None, **kwargs):
        system = context['contract']
        instruction = context.get('instruction', json.dumps(context.get('task')))
        if not self.messages:
            self.messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': instruction}]
        elif context.get('latest_host_observation') is not None:
            self.messages.append({'role': 'user', 'content': context['latest_host_observation']})
        completion = self.provider.complete(self.profile, self.messages)
        if record:
            record("pi-provider-request", {"request": self.profile.request(self.messages)})
        self.completion_tokens += completion.completion_tokens
        self.messages.append({'role': 'assistant', 'content': completion.text})
        try:
            value = json.loads(completion.text)
            name, args = next(iter(value.items()))
            if len(value) != 1 or name not in allowed_actions:
                raise ValueError('wrong action')
        except (ValueError, StopIteration):
            raise ActionError('scripted invalid action') from None
        return name, args

    def close(self): pass
    def __enter__(self): return self
    def __exit__(self, *_): self.close()
