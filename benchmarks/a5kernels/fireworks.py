"""Fail-closed Fireworks inference and managed SFT integration for A5 studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any, Mapping

import requests


FIREWORKS_BASE_MODEL = "accounts/fireworks/models/deepseek-v4-flash"
FIREWORKS_API_URL = "https://api.fireworks.ai"
FIREWORKS_INFERENCE_URL = f"{FIREWORKS_API_URL}/inference/v1"
FIREWORKS_CREDENTIAL_ENV = "FIREWORKS_API_KEY"
FIREWORKS_SUBMIT_ENV = "RSI_ALLOW_FIREWORKS_SFT"
_PROFILE_PATH = Path(__file__).resolve().parents[2] / "config/a5/fireworks-deepseek-v4-flash.json"
_RESOURCE_ID = re.compile(r"^[a-z][a-z0-9-]{0,62}$")
_TERMINAL_SUCCESS = {"JOB_STATE_COMPLETED"}
_TERMINAL_FAILURE = {
    "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED",
    "JOB_STATE_EARLY_STOPPED", "JOB_STATE_DELETED", "JOB_STATE_ARCHIVED",
}


class FrozenDict(dict):
    """JSON-serializable dictionary that rejects every mutation path."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("provenance is immutable")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _immutable


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return FrozenDict({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"invalid {label} fields: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


@dataclass(frozen=True)
class FireworksInferenceProfile:
    profile_id: str
    model: str
    base_url: str
    credential_env: str
    max_tokens: int
    temperature: float
    top_p: float

    def __post_init__(self) -> None:
        if self.profile_id != "fireworks-deepseek-v4-flash-v1":
            raise ValueError("unsupported Fireworks profile_id")
        if self.model != FIREWORKS_BASE_MODEL:
            raise ValueError("Fireworks profile requires the exact DeepSeek V4 Flash base")
        if self.base_url != FIREWORKS_INFERENCE_URL:
            raise ValueError("Fireworks profile requires the direct inference URL")
        if self.credential_env != FIREWORKS_CREDENTIAL_ENV:
            raise ValueError("Fireworks profile requires FIREWORKS_API_KEY")
        if self.max_tokens != 32768 or self.temperature != 0.0 or self.top_p != 1.0:
            raise ValueError("unsupported Fireworks generation settings")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FireworksInferenceProfile":
        expected = {
            "profile_id", "model", "base_url", "credential_env", "max_tokens",
            "temperature", "top_p",
        }
        _exact_keys(value, expected, "Fireworks profile")
        return cls(**value)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_fireworks_profile(path: Path | None = None) -> FireworksInferenceProfile:
    source = _PROFILE_PATH if path is None else Path(path)
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unable to load Fireworks profile: {source}") from exc
    return FireworksInferenceProfile.from_mapping(value)


@dataclass(frozen=True)
class FireworksCompletion:
    text: str
    provenance: Mapping[str, Any]


def complete_fireworks(
    system: str,
    user: str,
    *,
    history: list[dict[str, Any]] | None = None,
    profile: FireworksInferenceProfile | None = None,
    environment: Mapping[str, str] | None = None,
    client_factory=None,
) -> FireworksCompletion:
    """Call the exact Fireworks model without retaining its credential."""
    selected = profile or load_fireworks_profile()
    selected.__post_init__()
    environ = os.environ if environment is None else environment
    api_key = environ.get(selected.credential_env, "")
    if not api_key:
        raise ValueError(f"{selected.credential_env} is required")
    if client_factory is None:
        from openai import OpenAI

        client_factory = OpenAI
    client = client_factory(api_key=api_key, base_url=selected.base_url)
    messages = [{"role": "system", "content": system}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user})
    response = client.chat.completions.create(
        model=selected.model,
        messages=messages,
        max_tokens=selected.max_tokens,
        temperature=selected.temperature,
        top_p=selected.top_p,
    )
    choices = getattr(response, "choices", None)
    if not choices or not isinstance(choices[0].message.content, str):
        raise RuntimeError("Fireworks completion returned no text")
    return FireworksCompletion(
        text=choices[0].message.content,
        provenance=_freeze({
            "profile_id": selected.profile_id,
            "profile_sha256": selected.fingerprint,
            "model": selected.model,
            "base_url": selected.base_url,
            "generation": {
                "max_tokens": selected.max_tokens,
                "temperature": selected.temperature,
                "top_p": selected.top_p,
            },
        }),
    )


@dataclass(frozen=True)
class FireworksSFTContract:
    lora_rank: int = 32
    seed: int = 42
    epochs: int = 1
    learning_rate: float = 1e-5
    max_context_length: int = 32768
    seed_api_support: bool = False

    def __post_init__(self) -> None:
        if self.lora_rank != 32:
            raise ValueError("SFT contract requires LoRA rank 32")
        if self.seed != 42:
            raise ValueError("SFT contract requires seed 42")
        if self.epochs != 1:
            raise ValueError("SFT contract requires epochs=1")
        if self.learning_rate != 1e-5:
            raise ValueError("SFT contract requires learning rate 1e-5")
        if self.max_context_length != 32768:
            raise ValueError("SFT contract requires 32K context")
        if self.seed_api_support is not False:
            raise ValueError("managed SFT does not expose a seed field")

    def request_fields(self, dataset: str, output_model: str) -> dict[str, Any]:
        return {
            "baseModel": FIREWORKS_BASE_MODEL,
            "dataset": dataset,
            "outputModel": output_model,
            "epochs": self.epochs,
            "learningRate": self.learning_rate,
            "maxContextLength": self.max_context_length,
            "loraRank": self.lora_rank,
        }


@dataclass(frozen=True)
class DatasetIdentity:
    path: Path
    sha256: str
    byte_count: int
    example_count: int


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_function(value: Any, label: str, *, require_arguments: bool) -> None:
    if not isinstance(value, dict) or not _nonempty_string(value.get("name")):
        raise ValueError(f"{label} requires a function name")
    if require_arguments and not isinstance(value.get("arguments"), str):
        raise ValueError(f"{label} function arguments must be a JSON string")
    if "parameters" in value and not isinstance(value["parameters"], dict):
        raise ValueError(f"{label} function parameters must be an object")


def _validate_tools(value: Any, row_number: int) -> set[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"row {row_number} tools must be a non-empty array")
    names = set()
    for index, tool in enumerate(value):
        label = f"row {row_number} tools[{index}]"
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError(f"{label} must declare type=function")
        _validate_function(tool.get("function"), label, require_arguments=False)
        name = tool["function"]["name"]
        if name in names:
            raise ValueError(f"row {row_number} tools has duplicate function {name!r}")
        names.add(name)
    return names


def _validate_message(
    message: Any,
    row_number: int,
    *,
    declared: set[str],
    pending: dict[str, str],
    seen_call_ids: set[str],
) -> None:
    if not isinstance(message, dict):
        raise ValueError(f"row {row_number} messages must be objects")
    role = message.get("role")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError(f"row {row_number} has an unsupported message role")
    if "weight" in message and (
        type(message["weight"]) is not int or message["weight"] not in {0, 1}
    ):
        raise ValueError(f"row {row_number} message weight must be 0 or 1")
    content_valid = isinstance(message.get("content"), (str, list))
    tool_calls = message.get("tool_calls")
    if role == "assistant" and tool_calls is not None:
        if not isinstance(tool_calls, list) or not tool_calls:
            raise ValueError(f"row {row_number} assistant tool_calls must be non-empty")
        for index, call in enumerate(tool_calls):
            label = f"row {row_number} tool_calls[{index}]"
            if not isinstance(call, dict) or call.get("type") != "function":
                raise ValueError(f"{label} must declare type=function")
            _validate_function(call.get("function"), label, require_arguments=True)
            call_id = call.get("id")
            if not _nonempty_string(call_id):
                raise ValueError(f"{label} requires a nonempty unique call ID")
            if call_id in seen_call_ids:
                raise ValueError(f"{label} has a duplicate call ID")
            function = call["function"]
            if function["name"] not in declared:
                raise ValueError(f"{label} function must be declared in row tools")
            try:
                arguments = json.loads(function["arguments"])
            except json.JSONDecodeError as exc:
                raise ValueError(f"{label} arguments must contain valid JSON") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"{label} arguments JSON must be an object")
            seen_call_ids.add(call_id)
            pending[call_id] = function["name"]
        # Fireworks' documented tool-call example intentionally has no content.
        return
    if role == "tool":
        if not (_nonempty_string(message.get("tool_call_id"))
                or _nonempty_string(message.get("name"))):
            raise ValueError(f"row {row_number} tool message requires tool_call_id or name")
        if not content_valid:
            raise ValueError(f"row {row_number} tool message requires content")
        call_id = message.get("tool_call_id")
        name = message.get("name")
        if _nonempty_string(call_id):
            if call_id not in pending:
                raise ValueError(f"row {row_number} tool result has no pending call")
            expected_name = pending[call_id]
        else:
            matches = [key for key, value in pending.items() if value == name]
            if len(matches) != 1:
                raise ValueError(f"row {row_number} tool result has no unique pending call")
            call_id = matches[0]
            expected_name = name
        if _nonempty_string(name) and name != expected_name:
            raise ValueError(f"row {row_number} tool result name does not match pending call")
        pending.pop(call_id)
        return
    if not content_valid:
        raise ValueError(f"row {row_number} has invalid message content")


def _validate_chat_bytes(source: Path, data: bytes) -> DatasetIdentity:
    byte_count = len(data)
    if byte_count > 150 * 1024 * 1024:
        raise ValueError("managed upload supports at most 150 MiB")
    rows = []
    try:
        lines = data.decode("utf-8").splitlines()
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL line {line_number}")
            rows.append(json.loads(line))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid training JSONL: {exc}") from exc
    if len(rows) < 3:
        raise ValueError("Fireworks training requires at least 3 examples")
    weighted = [isinstance(row, dict) and "weight" in row for row in rows]
    if any(weighted) and not all(weighted):
        raise ValueError("root sample weight must be present on every row when used")
    if all(weighted):
        for row_number, row in enumerate(rows, 1):
            weight = row["weight"]
            if type(weight) not in (int, float):
                raise ValueError(f"row {row_number} root sample weight must be numeric")
            if not math.isfinite(weight):
                raise ValueError(f"row {row_number} root sample weight must be finite")
    for row_number, row in enumerate(rows, 1):
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or not messages:
            raise ValueError(f"row {row_number} requires a messages array")
        declared = _validate_tools(row["tools"], row_number) if "tools" in row else set()
        pending: dict[str, str] = {}
        seen_call_ids: set[str] = set()
        for message in messages:
            _validate_message(
                message, row_number, declared=declared, pending=pending,
                seen_call_ids=seen_call_ids,
            )
    digest = hashlib.sha256(data).hexdigest()
    return DatasetIdentity(source, digest, byte_count, len(rows))


def validate_chat_jsonl(path: Path) -> DatasetIdentity:
    source = Path(path)
    if source.suffix != ".jsonl" or not source.is_file():
        raise ValueError("training dataset must be an existing .jsonl file")
    try:
        data = source.read_bytes()
    except OSError as exc:
        raise ValueError(f"unable to read training JSONL: {source}") from exc
    return _validate_chat_bytes(source, data)


def _snapshot_chat_jsonl(path: Path, provenance_path: Path) -> DatasetIdentity:
    source = Path(path)
    if source.suffix != ".jsonl" or not source.is_file():
        raise ValueError("training dataset must be an existing .jsonl file")
    data = source.read_bytes()
    checked = _validate_chat_bytes(source, data)
    snapshot_dir = Path(provenance_path).parent / "datasets"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot = snapshot_dir / f"{checked.sha256}.jsonl"
    if snapshot.exists():
        if snapshot.read_bytes() != data:
            raise RuntimeError("content-addressed dataset snapshot collision")
    else:
        temporary = snapshot.with_suffix(".jsonl.tmp")
        temporary.write_bytes(data)
        temporary.replace(snapshot)
    return DatasetIdentity(snapshot, checked.sha256, checked.byte_count, checked.example_count)


class FireworksHTTPTransport:
    """Small requests boundary so lifecycle behavior is testable without network."""

    def __init__(
        self, api_key: str, *, session=None, timeout: float = 60.0,
        get_retries: int = 3, retry_backoff: float = 1.0,
    ):
        if not api_key:
            raise ValueError("FIREWORKS_API_KEY is required")
        self._api_key = api_key
        self._session = session or requests.Session()
        self._timeout = timeout
        if type(get_retries) is not int or get_retries < 0:
            raise ValueError("get_retries must be a nonnegative integer")
        if not isinstance(retry_backoff, (int, float)) or not math.isfinite(retry_backoff) or retry_backoff < 0:
            raise ValueError("retry_backoff must be finite and nonnegative")
        self._get_retries = get_retries
        self._retry_backoff = retry_backoff

    def request(self, method, path, *, json_body=None, file_path=None):
        headers = {"Authorization": f"Bearer {self._api_key}"}
        kwargs: dict[str, Any] = {"headers": headers, "timeout": self._timeout}
        opened = None
        try:
            if file_path is not None:
                opened = Path(file_path).open("rb")
                kwargs["files"] = {"file": (Path(file_path).name, opened, "application/jsonl")}
            else:
                headers["Content-Type"] = "application/json"
                if json_body is not None:
                    kwargs["json"] = json_body
            for attempt in range(self._get_retries + 1):
                try:
                    response = self._session.request(method, FIREWORKS_API_URL + path, **kwargs)
                    response.raise_for_status()
                    break
                except (requests.Timeout, requests.ConnectionError, requests.HTTPError) as exc:
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    retryable = method == "GET" and (
                        isinstance(exc, (requests.Timeout, requests.ConnectionError))
                        or status == 429 or (isinstance(status, int) and 500 <= status <= 599)
                    )
                    if not retryable or attempt == self._get_retries:
                        raise
                    if self._retry_backoff:
                        time.sleep(self._retry_backoff * (2 ** attempt))
            value = response.json()
            if not isinstance(value, dict):
                raise RuntimeError("Fireworks API returned a non-object response")
            return value
        finally:
            if opened is not None:
                opened.close()


def _resource_id(value: str, label: str) -> str:
    if not isinstance(value, str) or not _RESOURCE_ID.fullmatch(value):
        raise ValueError(f"invalid Fireworks {label}: {value!r}")
    return value


class FireworksManagedSFT:
    def __init__(
        self,
        transport,
        *,
        environment: Mapping[str, str] | None = None,
        contract: FireworksSFTContract | None = None,
        poll_seconds: float = 30.0,
        max_polls: int = 720,
    ):
        self.transport = transport
        self.environment = os.environ if environment is None else environment
        self.contract = contract or FireworksSFTContract()
        if (not isinstance(poll_seconds, (int, float))
                or not math.isfinite(poll_seconds) or poll_seconds < 0):
            raise ValueError("poll_seconds must be finite and nonnegative")
        if type(max_polls) is not int or max_polls <= 0:
            raise ValueError("max_polls must be a positive integer")
        self.poll_seconds = poll_seconds
        self.max_polls = max_polls

    def _submission_guard(self, submit: bool) -> None:
        if not submit:
            raise RuntimeError("billable training requires the explicit --submit flag")
        if self.environment.get(FIREWORKS_SUBMIT_ENV) != "1":
            raise RuntimeError(f"billable training requires {FIREWORKS_SUBMIT_ENV}=1")

    def _assert_job(self, job: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
        for field, value in expected.items():
            if job.get(field) != value:
                raise RuntimeError(
                    f"Fireworks job contract drift for {field}: "
                    f"expected {value!r}, got {job.get(field)!r}"
                )

    def submit(
        self,
        dataset_path: Path,
        *,
        account_id: str,
        dataset_id: str,
        output_model_id: str,
        job_id: str,
        submit: bool,
        provenance_path: Path,
    ) -> tuple[DatasetIdentity, dict[str, Any]]:
        self._submission_guard(submit)
        identity = _snapshot_chat_jsonl(dataset_path, provenance_path)
        account_id = _resource_id(account_id, "account ID")
        dataset_id = _resource_id(dataset_id, "dataset ID")
        output_model_id = _resource_id(output_model_id, "output model ID")
        job_id = _resource_id(job_id, "job ID")
        account_path = f"/v1/accounts/{account_id}"
        dataset_name = f"accounts/{account_id}/datasets/{dataset_id}"
        output_model = f"accounts/{account_id}/models/{output_model_id}"
        created = self.transport.request(
            "POST", f"{account_path}/datasets",
            json_body={
                "datasetId": dataset_id,
                "dataset": {
                    "displayName": dataset_id,
                    "exampleCount": str(identity.example_count),
                    "userUploaded": {},
                    "format": "CHAT",
                },
            },
        )
        if created.get("name") != dataset_name:
            raise RuntimeError("Fireworks returned an unexpected dataset resource")
        uploaded = self.transport.request(
            "POST", f"{account_path}/datasets/{dataset_id}:upload",
            file_path=identity.path,
        )
        if uploaded.get("id") not in {dataset_id, dataset_name}:
            raise RuntimeError("Fireworks upload did not resolve the requested dataset")
        if "bytes" in uploaded and uploaded["bytes"] != identity.byte_count:
            raise RuntimeError(
                "Fireworks upload byte count differs from the validated snapshot"
            )
        fields = self.contract.request_fields(dataset_name, output_model)
        job = self.transport.request(
            "POST", f"{account_path}/supervisedFineTuningJobs?supervisedFineTuningJobId={job_id}",
            json_body={"displayName": job_id, **fields},
        )
        expected_name = f"accounts/{account_id}/supervisedFineTuningJobs/{job_id}"
        if job.get("name") != expected_name:
            raise RuntimeError("Fireworks returned an unexpected SFT job resource")
        self._assert_job(job, fields)
        provisional = self._provenance(
            lifecycle="submitted", identity=identity, job=job,
            account_id=account_id, dataset_id=dataset_id,
            output_model_id=output_model_id, job_id=job_id,
        )
        self._write_provenance(provenance_path, provisional)
        return identity, dict(job)

    def _provenance(
        self, *, lifecycle: str, identity: DatasetIdentity,
        job: Mapping[str, Any], account_id: str, dataset_id: str,
        output_model_id: str, job_id: str, model: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        output_model = f"accounts/{account_id}/models/{output_model_id}"
        result = {
            "schema_version": 2,
            "provider": "fireworks",
            "lifecycle": lifecycle,
            "state": job.get("state"),
            "account_id": account_id,
            "job_id": job_id,
            "job_name": f"accounts/{account_id}/supervisedFineTuningJobs/{job_id}",
            "output_model_id": output_model_id,
            "trained_model": model.get("name") if model else None,
            "checkpoint_id": model.get("name") if model else None,
            "base_model": FIREWORKS_BASE_MODEL,
            "dataset_id": dataset_id,
            "dataset_name": f"accounts/{account_id}/datasets/{dataset_id}",
            "dataset_sha256": identity.sha256,
            "dataset_bytes": identity.byte_count,
            "dataset_examples": identity.example_count,
            "dataset_snapshot": str(identity.path),
            "contract": asdict(self.contract),
            "completed_time": job.get("completedTime"),
            "model_create_time": model.get("createTime") if model else None,
        }
        if model is not None and model.get("name") != output_model:
            raise RuntimeError("completed SFT did not resolve the expected trained model")
        return result

    @staticmethod
    def _write_provenance(path: Path, value: Mapping[str, Any]) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        temporary.replace(destination)

    def _wait(
        self, *, identity: DatasetIdentity, job: Mapping[str, Any],
        account_id: str, dataset_id: str, output_model_id: str, job_id: str,
        provenance_path: Path,
    ) -> dict[str, Any]:
        expected = self.contract.request_fields(
            f"accounts/{account_id}/datasets/{dataset_id}",
            f"accounts/{account_id}/models/{output_model_id}",
        )
        current = dict(job)
        self._assert_job(current, expected)
        for _ in range(self.max_polls):
            if current.get("state") in _TERMINAL_SUCCESS | _TERMINAL_FAILURE:
                break
            if self.poll_seconds:
                time.sleep(self.poll_seconds)
            current = self.transport.request(
                "GET", f"/v1/accounts/{account_id}/supervisedFineTuningJobs/{job_id}"
            )
            self._assert_job(current, expected)
            if current.get("state") in _TERMINAL_SUCCESS | _TERMINAL_FAILURE:
                break
        if current.get("state") not in _TERMINAL_SUCCESS | _TERMINAL_FAILURE:
            raise TimeoutError(f"Fireworks SFT job did not finish after {self.max_polls} polls")
        if current.get("state") not in _TERMINAL_SUCCESS:
            failed = self._provenance(
                lifecycle="failed", identity=identity, job=current,
                account_id=account_id, dataset_id=dataset_id,
                output_model_id=output_model_id, job_id=job_id,
            )
            self._write_provenance(provenance_path, failed)
            raise RuntimeError(f"Fireworks SFT ended in {current.get('state')}")
        model = self.transport.request(
            "GET", f"/v1/accounts/{account_id}/models/{output_model_id}"
        )
        peft = model.get("peftDetails")
        if not isinstance(peft, Mapping) or peft.get("baseModel") != FIREWORKS_BASE_MODEL:
            raise RuntimeError("trained model reports a missing or mismatched PEFT base")
        result = self._provenance(
            lifecycle="completed", identity=identity, job=current,
            account_id=account_id, dataset_id=dataset_id,
            output_model_id=output_model_id, job_id=job_id, model=model,
        )
        self._write_provenance(provenance_path, result)
        return result

    def run(
        self,
        dataset_path: Path,
        *,
        account_id: str,
        dataset_id: str,
        output_model_id: str,
        job_id: str,
        submit: bool,
        provenance_path: Path,
    ) -> dict[str, Any]:
        identity, job = self.submit(
            dataset_path, account_id=account_id, dataset_id=dataset_id,
            output_model_id=output_model_id, job_id=job_id, submit=submit,
            provenance_path=provenance_path,
        )
        return self._wait(
            identity=identity, job=job, account_id=account_id,
            dataset_id=dataset_id, output_model_id=output_model_id,
            job_id=job_id, provenance_path=provenance_path,
        )

    def observe(
        self,
        *,
        account_id: str,
        job_id: str,
        output_model_id: str,
        provenance_path: Path,
    ) -> dict[str, Any]:
        account_id = _resource_id(account_id, "account ID")
        job_id = _resource_id(job_id, "job ID")
        output_model_id = _resource_id(output_model_id, "output model ID")
        source = Path(provenance_path)
        try:
            retained = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("observe requires valid provisional provenance") from exc
        expected_identity = {
            "schema_version": 2, "provider": "fireworks", "account_id": account_id,
            "job_id": job_id, "output_model_id": output_model_id,
        }
        for field, value in expected_identity.items():
            if retained.get(field) != value:
                raise ValueError(f"provisional provenance mismatch for {field}")
        if retained.get("base_model") != FIREWORKS_BASE_MODEL:
            raise ValueError("provisional provenance base model drift")
        if retained.get("contract") != asdict(self.contract):
            raise ValueError("provisional provenance SFT contract drift")
        snapshot = Path(retained.get("dataset_snapshot", ""))
        if not snapshot.is_file():
            raise ValueError("provisional provenance dataset snapshot is unavailable")
        data = snapshot.read_bytes()
        identity = _validate_chat_bytes(snapshot, data)
        if (identity.sha256 != retained.get("dataset_sha256")
                or identity.byte_count != retained.get("dataset_bytes")
                or identity.example_count != retained.get("dataset_examples")):
            raise ValueError("provisional provenance dataset identity drift")
        dataset_id = _resource_id(retained.get("dataset_id"), "dataset ID")
        job = self.transport.request(
            "GET", f"/v1/accounts/{account_id}/supervisedFineTuningJobs/{job_id}"
        )
        return self._wait(
            identity=identity, job=job, account_id=account_id,
            dataset_id=dataset_id, output_model_id=output_model_id,
            job_id=job_id, provenance_path=provenance_path,
        )
