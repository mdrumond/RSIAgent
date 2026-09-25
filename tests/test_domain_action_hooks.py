"""Functional coverage for the opt-in domain action loop seam."""

from dataclasses import dataclass

import pytest

import core.loop as loop
from config.settings import load
from core.trace import ArtifactSink


@dataclass(frozen=True)
class KernelAction:
    kind: str
    payload: str = ""
    dup: int = 1


class VM:
    """Domain actions must not escape through the legacy VM action channels."""

    def run_script(self, *_args, **_kwargs):
        raise AssertionError("domain action reached vm.run_script")

    def run_command(self, *_args, **_kwargs):
        # Core's unchanged surface snapshot uses this before any model action.
        return ""


def _cfg():
    cfg = load(None)
    cfg.max_iters = 4
    cfg.wall_clock_secs = 3600
    cfg.history_keep_pairs = 0
    cfg.independent_verify = False
    cfg.practice_mode = True
    cfg.practice_done_requires = ""
    return cfg


def _run(monkeypatch, tmp_path, replies, **hooks):
    pending = list(replies)

    def fake_chat(*_args, **_kwargs):
        return pending.pop(0)

    monkeypatch.setattr(loop, "chat", fake_chat)
    return loop.run_attempt(
        "author a kernel", VM(), _cfg(), ArtifactSink(str(tmp_path)), **hooks)


def test_domain_parser_and_executor_exchange_observations(monkeypatch, tmp_path):
    parsed = []
    executed = []
    users = []

    def fake_chat(_model, _system, user, **_kwargs):
        users.append(user)
        return ["compile candidate", "submit candidate"][len(users) - 1]

    def parser(text):
        parsed.append(text)
        return KernelAction("submit" if text.startswith("submit") else "compile", text)

    def executor(action):
        executed.append(action)
        if action.kind == "submit":
            return loop.DomainActionResult("host verification passed", terminal=True)
        return loop.DomainActionResult("trusted compiler result: success")

    monkeypatch.setattr(loop, "chat", fake_chat)
    result, history = loop.run_attempt(
        "author a kernel", VM(), _cfg(), ArtifactSink(str(tmp_path)),
        turn_parser=parser, action_executor=executor)

    assert result.status == "done"
    assert result.iters == 2
    assert result.programs_run == result.looks == result.asks == 0
    assert parsed == ["compile candidate", "submit candidate"]
    assert [a.kind for a in executed] == ["compile", "submit"]
    assert users[1] == "trusted compiler result: success"
    assert history[-1]["content"] == "submit candidate"


def test_terminal_domain_stall_skips_legacy_recovery(monkeypatch, tmp_path):
    cfg = _cfg()
    cfg.independent_verify = True
    monkeypatch.setattr(loop, "chat", lambda *_args, **_kwargs: "stop")
    monkeypatch.setattr(
        loop,
        "verify_independent",
        lambda *_args, **_kwargs: pytest.fail("terminal domain result was reverified"),
    )

    result, _ = loop.run_attempt(
        "author a kernel",
        VM(),
        cfg,
        ArtifactSink(str(tmp_path)),
        turn_parser=lambda _text: KernelAction("submit"),
        action_executor=lambda _action: loop.DomainActionResult(
            "host verification stalled", terminal=True, status="stalled"
        ),
    )

    assert result.status == "stalled"
    assert result.surface_delta is None
    assert result.inspections == []


def test_custom_parser_can_delegate_builtin_actions(monkeypatch, tmp_path):
    # Integrations may extend the grammar while retaining ordinary Done semantics.
    result, _ = _run(
        monkeypatch, tmp_path, ['{"done":null}'],
        turn_parser=loop.parse_turn,
        action_executor=lambda _action: pytest.fail("built-in action was delegated"),
        allow_noop_done=True)

    assert result.status == "done"


def test_custom_parser_receives_text_for_empty_reply_and_retries(monkeypatch, tmp_path):
    seen = []

    def parser(text):
        seen.append(text)
        assert isinstance(text, str)
        return loop.parse_turn(text)

    result, history = _run(
        monkeypatch,
        tmp_path,
        [None, '{"done":null}'],
        turn_parser=parser,
        action_executor=lambda _action: pytest.fail("built-in action was delegated"),
        allow_noop_done=True,
    )

    assert result.status == "done"
    assert result.turns == 2
    assert seen == ["", '{"done":null}']
    assert history[1]["content"] == "(empty reply)"


def test_domain_action_without_executor_fails_closed(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="without action_executor"):
        _run(monkeypatch, tmp_path, ["compile"],
             turn_parser=lambda _text: KernelAction("compile"))


@pytest.mark.parametrize(
    "outcome, error",
    [
        ("untyped", "must return DomainActionResult"),
        (loop.DomainActionResult(b"bytes"), "observation must be a string"),
        (loop.DomainActionResult(None), "observation must be a string"),
        (loop.DomainActionResult(""), "observation must not be empty"),
        (loop.DomainActionResult("x", terminal="false"), "terminal must be a bool"),
        (loop.DomainActionResult("x", terminal=0), "terminal must be a bool"),
        (loop.DomainActionResult("x", terminal=1), "terminal must be a bool"),
        (loop.DomainActionResult("x", terminal=True, status="passed"),
         "invalid terminal domain action status"),
    ],
)
def test_executor_result_contract_fails_closed(monkeypatch, tmp_path, outcome, error):
    with pytest.raises((TypeError, ValueError), match=error):
        _run(monkeypatch, tmp_path, ["compile"],
             turn_parser=lambda _text: KernelAction("compile"),
             action_executor=lambda _action: outcome)


def test_default_path_still_uses_original_parser(monkeypatch, tmp_path):
    seen = []
    original = loop.parse_turn

    def recording_parser(text):
        seen.append(text)
        return original(text)

    monkeypatch.setattr(loop, "parse_turn", recording_parser)
    result, _ = _run(monkeypatch, tmp_path, ['{"done":null}'], allow_noop_done=True)

    assert result.status == "done"
    assert seen == ['{"done":null}']
