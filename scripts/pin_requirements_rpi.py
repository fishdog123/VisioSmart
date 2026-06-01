#!/usr/bin/env python3
"""
Pin installed package versions into requirements.txt.

Usage (on the Raspberry Pi in the project's venv):

    python scripts/pin_requirements_rpi.py

This script reads `requirements.in`, runs `pip freeze`, and writes exact
matched lines for the packages listed in `requirements.in` into
`requirements.txt`.
"""
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REQ_IN = ROOT / "requirements.in"
OUT = ROOT / "requirements.txt"

if not REQ_IN.exists():
    print(f"{REQ_IN} not found", file=sys.stderr)
    sys.exit(2)

with open(REQ_IN) as f:
    pkgs = [line.strip().split('[',1)[0].lower() for line in f if line.strip() and not line.strip().startswith('#')]

proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True, check=True)
freeze = proc.stdout.splitlines()

out_lines = []
for p in pkgs:
    found = False
    for line in freeze:
        if line.lower().startswith(p + "=="):
            out_lines.append(line)
            found = True
            break
    if not found:
        print(f"Warning: {p} not found in pip freeze; install it first", file=sys.stderr)

if out_lines:
    out_lines = sorted(set(out_lines))
    with open(OUT, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')
    print(f"Wrote {OUT} with {len(out_lines)} pinned packages")
else:
    print("No packages pinned; check that required packages are installed in the venv.")
