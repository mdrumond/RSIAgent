"""Bounded production proof against an existing job, using a supplied real kernel.

PYTHONPATH=. python tests/optimizer_worker_proof.py --root ROOT --job-id ID --config CONFIG
    --candidate KERNEL.cpp
Omit --candidate to exercise the real configured provider instead.
"""
import argparse
import json
from pathlib import Path

from optimizer.worker import load_config, run_worker


class ProofAgent:
    identity = {'model': 'production-transport-proof-v1'}

    def __init__(self, source):
        self.source = source

    def complete(self, stage, context, record):
        if stage == 'cases':
            action = {'action': 'cases', 'signatures': [], 'summary': 'retain representative cases; host generates bounded corners'}
        elif not context['history']:
            action = {'action': 'candidate', 'source': self.source, 'block_count': 1, 'summary': 'bounded supplied production-proof candidate'}
        else:
            action = {'action': 'finish', 'summary': 'bounded production proof complete'}
        record('request', {'stage': stage, 'context': context})
        record('response', action)
        return action


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--job-id', required=True)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--candidate', type=Path)
    args = parser.parse_args()
    agent = ProofAgent(args.candidate.read_text()) if args.candidate else None
    print(json.dumps(run_worker(args.root, args.job_id, load_config(args.config), agent=agent)))
