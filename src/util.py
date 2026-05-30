"""Shared utility functions for cftk."""

import os
import sys
import time
import re
import subprocess


def disp(msg):
    print(f"@{time.asctime()}\t{msg}", file=sys.stderr)


def run_command(cmd, label="", check=True):
    disp(f"CMD [{label}]: {cmd[:120]}" if label else f"CMD: {cmd[:120]}")
    ret = subprocess.run(cmd, shell=True)
    if check and ret.returncode != 0:
        sys.exit(f"[util] ERROR: command failed — {label or cmd[:80]}")
    return ret.returncode


def is_number(s):
    return bool(re.match(r"^-?\d+(?:\.\d+)?$", str(s)))
