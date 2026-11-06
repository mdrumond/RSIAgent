"""Local dashboard: all instance operations belong to the Node CLI."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import threading

from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import BadRequest


class CLIError(Exception):
    pass


class Driver:
    def __init__(self, instance, machinery=None, node="node"):
        self.instance = str(Path(instance).resolve())
        self.machinery = Path(machinery or Path(self.instance) / "machinery").resolve()
        self.node = node

    def call(self, command, payload=None, *, detach=False, mode=None):
        argv = [self.node, str(self.machinery / "cli" / "main.mjs"),
                command, "--instance", self.instance]
        if detach:
            argv.append("--detach")
        if mode:
            argv.extend(["--mode", mode])
        # A private, unique file survives until the CLI has consumed its input.
        with tempfile.TemporaryDirectory(prefix="ascend-dashboard-") as directory:
            if payload is not None:
                path = Path(directory) / "input.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                argv.extend(["--project" if command == "project" else "--config", str(path)])
            try:
                result = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                                        check=False)
            except subprocess.TimeoutExpired as error:
                raise CLIError("CLI timed out; refresh status before retrying an action") from error
            except OSError as error:
                raise CLIError(str(error)) from error
            if result.returncode:
                raise CLIError(result.stderr.strip() or f"CLI exited {result.returncode}")
            try:
                return json.loads(result.stdout)
            except (ValueError, TypeError) as error:
                raise CLIError("CLI returned invalid JSON") from error


def create_app(instance, machinery=None, node="node"):
    app = Flask(__name__, static_folder=None)
    driver = Driver(instance, machinery, node)
    web = driver.machinery / "web"
    # Serialize mutating requests, including config + tune, within this app.
    lock = threading.Lock()

    def body():
        value = request.get_json()
        if not isinstance(value, dict):
            raise BadRequest("Expected a JSON object")
        return value

    @app.errorhandler(CLIError)
    def cli_error(error):
        return jsonify(error=str(error)), 502

    @app.errorhandler(BadRequest)
    def bad_request(error):
        return jsonify(error=error.description), 400

    @app.get("/")
    def index():
        return send_from_directory(web, "index.html")

    @app.get("/assets/<path:name>")
    def assets(name):
        return send_from_directory(web, name)

    @app.get("/api/status")
    def status():
        return jsonify(driver.call("status"))

    @app.get("/api/config")
    def config():
        return jsonify(driver.call("config"))

    @app.put("/api/config")
    def save_config():
        value = body()
        with lock:
            return jsonify(driver.call("config", value))

    @app.post("/api/project")
    def project():
        value = body()
        with lock:
            return jsonify(driver.call("project", value))

    @app.post("/api/skill")
    def skill():
        value = body()
        with lock:
            return jsonify(driver.call("skill", value))

    @app.post("/api/browse")
    def browse():
        return jsonify(driver.call("browse", body()))

    @app.post("/api/wave")
    @app.post("/api/campaign")
    def wave():
        value = body()
        mode = value.get("mode")
        if mode not in ("phase1", "phase2", "inference", "revisit"):
            raise BadRequest("Unknown wave mode")
        with lock:
            return jsonify(driver.call(request.path.rsplit("/", 1)[-1], detach=True, mode=mode))

    @app.post("/api/stop")
    @app.post("/api/pause")
    def stop():
        with lock:
            return jsonify(driver.call(request.path.rsplit("/", 1)[-1]))

    @app.post("/api/resume")
    def resume():
        with lock:
            return jsonify(driver.call("resume", detach=True))

    @app.post("/api/kdb")
    def kdb():
        value = body()
        with lock:
            return jsonify(driver.call("kdb", value, detach=True))

    @app.get("/api/storage")
    def storage_status():
        return jsonify(driver.call("storage", {"action": "inspect", "limit": 100}))

    @app.post("/api/storage")
    def storage_action():
        value = body()
        if value.get("action") not in ("inspect", "verify", "compress", "pack", "prune"):
            raise BadRequest("Unknown dashboard storage action")
        with lock:
            return jsonify(driver.call("storage", value,
                                      detach=value["action"] in ("compress", "pack", "prune")))

    @app.post("/api/tune")
    def tune():
        value = body()
        if "userinfo" not in value:
            raise BadRequest("userinfo is required")
        with lock:
            config = driver.call("config")
            config["userinfo"] = value["userinfo"]
            driver.call("config", config)
            return jsonify(driver.call("tune", value, detach=True))

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True)
    parser.add_argument("--machinery")
    parser.add_argument("--node", default="node")
    parser.add_argument("--port", type=int, default=46731)
    args = parser.parse_args()
    create_app(args.instance, args.machinery, args.node).run(
        host="127.0.0.1", port=args.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
