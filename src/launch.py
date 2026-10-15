"""Restart-safe parallel job launcher.
python launch.py jobs.txt 4
Each line: <expected_output_path> ||| <command>. Jobs whose output exists are skipped; failed jobs retried twice.
"""
import os
import sys
import time
import subprocess
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
LOG = os.path.join(ROOT, "logs", "launch")
os.makedirs(LOG, exist_ok=True)


def run(job):
    out, cmd = job
    if os.path.exists(out):
        return out, "skip"
    name = os.path.basename(out).replace(".json", "")
    lock = out + ".lock"
    try:  # atomic claim: skip jobs another launcher is running (locks older than 8 h are considered stale)
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > 8 * 3600:
            os.remove(lock)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        return out, "claimed"
    try:
        return _run(out, cmd, name)
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def _run(out, cmd, name):
    for attempt in range(3):
        if os.path.exists(out):
            return out, "ok"
        with open(os.path.join(LOG, name + ".log"), "a") as f:
            f.write(f"\n### attempt {attempt} {time.ctime()}\n{cmd}\n")
            f.flush()
            rc = subprocess.call(cmd, shell=True, stdout=f, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.abspath(__file__)),
                                 creationflags=0x08000000 if os.name == "nt" else 0)   # CREATE_NO_WINDOW
        if os.path.exists(out):
            print(f"OK {name}", flush=True)
            return out, "ok"
    print(f"FAIL {name} rc={rc}", flush=True)
    return out, "fail"


if __name__ == "__main__":
    jobs = []
    for line in open(sys.argv[1], encoding="utf-8-sig"):
        line = line.strip().lstrip("﻿")
        if line and not line.startswith("#"):
            out, cmd = [x.strip() for x in line.split("|||")]
            jobs.append((os.path.normpath(os.path.join(ROOT, out)), cmd))
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    todo = [j for j in jobs if not os.path.exists(j[0])]
    print(f"{len(jobs)} jobs, {len(todo)} to run, {k} workers", flush=True)
    with ThreadPoolExecutor(k) as ex:
        res = list(ex.map(run, todo))
    print("LAUNCH_DONE", sum(r[1] == "ok" for r in res), "ok", sum(r[1] == "fail" for r in res), "fail", flush=True)
