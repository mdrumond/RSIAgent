"""Materialize the dashboard's read views without retaining a server or credentials."""
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile

from .app import create_app
from .store import DashboardStore


class _ExportStore(DashboardStore):
    """Share authenticated ledgers across this export's many lazy payload reads."""

    def __init__(self, *args):
        super().__init__(*args)
        self._ledgers = {}

    @staticmethod
    def _ledger_fingerprint(path):
        try:
            stat = path.stat()
            return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        except OSError:
            return None

    def _authenticated_ledger(self, path):
        path = Path(path)
        fingerprint = self._ledger_fingerprint(path)
        previous = self._ledgers.get(path)
        if previous is not None and previous[0] == fingerprint:
            return previous[1]
        result = super()._authenticated_ledger(path)
        if self._ledger_fingerprint(path) == fingerprint:
            self._ledgers[path] = fingerprint, result
        return result


def _compact(path):
    """Only human-readable compact evidence belongs in a portable demo."""
    if any(word in path.name.lower() for word in ('timeline', '.pt', '.so', '.db')):
        return False
    try:
        if path.stat().st_size > 20 * 1024 * 1024:
            return False
        return '\x00' not in path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return False


def export_demo(config_path, cache_dir, output):
    """Export into a *new* directory. Never replace or modify a source directory."""
    output = Path(output).absolute()
    if output.exists():
        raise ValueError('output already exists; choose a new snapshot directory')
    output.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(config_path).read_text())
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    with tempfile.TemporaryDirectory(prefix='rsi-demo-') as temporary:
        temporary = Path(temporary)
        database = temporary / 'dashboard.sqlite3'
        original = Path(cache_dir) / 'dashboard.sqlite3'
        if original.is_file():
            with sqlite3.connect(original.resolve().as_uri() + '?mode=ro', uri=True) as source:
                with sqlite3.connect(database) as destination:
                    source.backup(destination)
        config['controls'] = {'enabled': False}
        optimizer = config.get('optimizer', {})
        jobs = Path(optimizer.get('root', ''))
        copied_jobs = temporary / 'jobs'
        copied_jobs.mkdir()
        if optimizer.get('root') and jobs.is_dir():
            for path in jobs.rglob('*'):
                # No worker configs, lock files, logs, or arbitrary root files.
                relative = path.relative_to(jobs)
                allowed = (len(relative.parts) == 2 and path.name in
                           {'submission.json', 'manifest.json', 'events.jsonl'}) or (
                           len(relative.parts) == 3 and relative.parts[1] == 'artifacts')
                metadata = len(relative.parts) == 2 and allowed
                if allowed and path.is_file() and (metadata or _compact(path)):
                    target = copied_jobs / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        worker = temporary / 'worker.json'
        worker.write_text('{}')
        config['optimizer'] = {'enabled': True, 'root': str(copied_jobs),
                               'worker_config': str(worker), 'memory_roots': []}
        private_config = temporary / 'config.json'
        private_config.write_text(json.dumps(config))
        store = _ExportStore(private_config, database)
        snapshot = store.refresh()
        app = create_app(private_config, database, refresh_seconds=float('inf'), dashboard_store=store)
        app.context_processor(lambda: {'demo_timestamp': timestamp})
        client = app.test_client()
        pages = ['/', '/memory', '/optimizer']
        endpoints = ['/api/status', '/api/optimizer/jobs']
        for campaign in snapshot['campaigns']:
            phase = campaign['phase']
            pages.append(f'/phase/{phase}')
            for project in campaign['projects']:
                base = f'/api/phase/{phase}/project/{project["id"]}'
                pages.append(f'/phase/{phase}/project/{project["id"]}')
                endpoints.extend([base + '/performance', base + '/trace'])
                trace = store.trace_index(phase, project['id'])
                for event in (trace or {}).get('events', []):
                    if event.get('available'):
                        endpoints.extend([event['payload_url'], event['download_url']])
        manager = app.extensions['optimizer']
        for job in manager.store.list():
            key = job['job_id']
            pages.append(f'/optimizer/{key}')
            base = f'/api/optimizer/jobs/{key}'
            endpoints.extend([base, base + '/events', base + '/submission', base + '/results'])
            for event in manager.store.events(key):
                endpoints.append(base + f'/events/{event["sequence"]}')
            from .optimizer_results import sources
            for index, _ in enumerate(sources(manager.store, key)):
                endpoints.extend([base + f'/sources/{index}', base + f'/sources/{index}/download'])
            for artifact in manager.store.artifacts(key):
                # Content-addressed files have no extension; check registered logical names too.
                if not any(word in artifact['name'].lower()
                           for word in ('timeline', '.pt', '.so', '.db')):
                    endpoints.append(base + '/artifacts/' + artifact['sha256'])
        stage = temporary / 'output'
        (stage / 'data').mkdir(parents=True)
        shutil.copytree(Path(__file__).parent / 'static', stage / 'static')
        manifest, bodies = {}, {}
        for route in dict.fromkeys(pages + endpoints):
            response = client.get(route)
            body, status = response.data, response.status_code
            try:
                metadata = '/events' in route or route.endswith('/submission')
                compact = (metadata or len(body) <= 20 * 1024 * 1024) and '\x00' not in body.decode('utf-8')
            except UnicodeError:
                compact = False
            if not compact:
                body, status = b'Evidence excluded: binary or larger than 20 MiB.', 404
            filename = ('index.html' if route == '/' else 'data/' +
                        hashlib.sha256(route.encode()).hexdigest() +
                        ('.html' if route in pages else '.txt'))
            manifest[route] = {'file': '/' + filename, 'status': status,
                               'type': response.content_type}
            if route.endswith('/download'):
                from werkzeug.http import parse_options_header
                _, options = parse_options_header(response.headers.get('Content-Disposition', ''))
                manifest[route]['download_name'] = options.get('filename', 'retained-evidence.txt')
            bodies[route] = body
        for route, body in bodies.items():
            if route in pages:
                text = body.decode()
                def link(match):
                    prefix, url, suffix = match.groups()
                    entry = manifest.get(html.unescape(url))
                    download = (' download="' + html.escape(entry['download_name'], quote=True) + '"'
                                if entry and 'download_name' in entry else '')
                    return prefix + (entry['file'] if entry else url) + suffix + download
                text = re.sub(r'((?:href|src)=")([^"]+)(")', link, text)
                body = text.encode()
            (stage / manifest[route]['file'].lstrip('/')).write_bytes(body)
        (stage / 'static/demo.js').write_text(_PLAYBACK.replace('__MANIFEST__', json.dumps(manifest)))
        (stage / 'snapshot.json').write_text(json.dumps({'exported_at': timestamp,
            'routes': manifest, 'campaigns': len(snapshot['campaigns']),
            'note': 'Captured statuses only. Missing evidence remains unavailable.'}, indent=2))
        (stage / 'serve.py').write_text(_SERVER)
        (stage / 'requirements.txt').write_text('# Playback uses Python 3 standard library only.\n')
        (stage / 'README.md').write_text(_README)
        app.extensions['dashboard_controls'].close()
        # Cross-filesystem-safe final publication; destination is exclusively ours.
        shutil.copytree(stage, output)
    return output


_PLAYBACK = r'''window.RSI_DEMO = true;
(() => {
  const routes = __MANIFEST__;
  const nativeFetch = window.fetch.bind(window);
  window.fetch = async (input, options = {}) => {
    const url = new URL(typeof input === 'string' ? input : input.url, location.href);
    const entry = routes[url.pathname];
    if ((options.method || 'GET').toUpperCase() !== 'GET' || !entry)
      return new Response('Evidence not included in this snapshot', {status: 404});
    const response = await nativeFetch(entry.file);
    return new Response(await response.arrayBuffer(), {status: entry.status,
      headers: {'Content-Type': entry.type}});
  };
  function links() {
    document.querySelectorAll('a[href]').forEach(a => {
      const path = new URL(a.href, location.href).pathname;
      if (routes[path]) {
        a.href = routes[path].file;
        if (routes[path].download_name) a.download = routes[path].download_name;
      }
    });
  }
  document.addEventListener('DOMContentLoaded', () => {
    links(); new MutationObserver(links).observe(document.body, {childList: true, subtree: true});
  });
})();
'''

_SERVER = '''"""Serve this snapshot with Python's standard library only."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--port', type=int, default=8080)
parser.add_argument('--host', default='127.0.0.1')
args = parser.parse_args()
handler = partial(SimpleHTTPRequestHandler, directory=str(Path(__file__).resolve().parent))
print(f'Demo: http://{args.host}:{args.port}', flush=True)
ThreadingHTTPServer((args.host, args.port), handler).serve_forever()
'''

_README = '''# RSI dashboard demo snapshot

Run `python serve.py --port 8080` and open http://127.0.0.1:8080.
No installation, Flask, credentials, RSI checkout, or remote connection is needed.
Serve at the HTTP root (not a subdirectory); opening index.html as file:// is unsupported.
Use `--host 0.0.0.0` only if you intend to expose the retained private evidence.

This is read-only. Running statuses are frozen, not live. Empty phases/jobs are
genuinely empty. Missing or excluded evidence stays unavailable, never fabricated.
Timelines, tensor/binary payloads, worker logs/configuration and files over 20 MiB
are excluded. Prompts, source and retained results may be private: review before sharing.

To import later real progress, run from the demo tooling branch in RSI:

```
python -m dashboard export-demo --config /path/to/dashboard-config.json \\
  --cache-dir /path/to/dashboard-cache --output /path/to/new-snapshot
```

Choose a new output directory each time; existing output is never overwritten.
The exporter reads local artifacts and a private backup of the cache; it does not
launch campaigns, models or remote work. Stop the old server and serve the new
folder to switch snapshots. Package with `tar -czf demo.tar.gz -C /path/to new-snapshot`.
Extract the entire folder elsewhere and run its serve.py to play it back.
'''
