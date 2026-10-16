"""Bound model feedback while retaining complete evidence on disk."""

def compact_observation(observation):
    """Bound prompt diagnostics without changing the full evidence on disk."""
    result = {key: observation[key] for key in (
        "status", "passed", "source_sha256", "reason", "failed_case", "cases",
    ) if key in observation}
    diagnostic = observation.get("diagnostic", "")
    if diagnostic:
        lines = diagnostic.splitlines()
        errors = []
        if observation.get("status") == "compile_failed":
            for index, line in enumerate(lines):
                if "error:" in line or "TimeoutExpired" in line:
                    errors.extend(lines[index:index + 3])
        text = "\n".join(dict.fromkeys(errors)) if errors else diagnostic
        result["diagnostic"] = text if len(text) <= 2000 else text[:1000] + " ... " + text[-1000:]
    return result
