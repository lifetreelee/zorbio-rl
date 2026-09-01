#!/usr/bin/env python3
"""Standalone, zero-LLM hourly health check for the Zorbio RL run.

Runs via cron (see health_check.sh), appends one line to health_report.log,
and a CLEARLY-FLAGGED alert line if something looks wrong. Uses only the two
metrics that have ever actually been diagnostic in this project (see
rl_phase_progress.md memory) - discrimination spread and stuck-fraction -
plus basic process liveness. No AI involved; this is pure arithmetic and
log-parsing, the same checks a Claude Code check-in was doing by hand.
"""
import subprocess
import re
import collections
import datetime
import os
import sys

RL_DIR = os.path.dirname(os.path.abspath(__file__))
SCRATCH_SERVER_LOG = "/tmp/claude-1000/-home-lee-zorbio-zorbio/7ddc8d51-19a7-4490-b46c-9fe3b1b9e497/scratchpad/server.log"
REPORT_PATH = os.path.join(RL_DIR, "health_report.log")

sys.path.insert(0, RL_DIR)


def check_processes():
    out = subprocess.run(["ps", "aux"], capture_output=True, text=True).stdout
    procs = {
        "server.js": "server.js" in out and "grep" not in out.split("server.js")[0][-50:],
        "zorbio_agent.py": "zorbio_agent.py" in out,
        "ppo_train.py": "ppo_train.py" in out,
    }
    alive = {k: (k in out) for k in ["server.js", "zorbio_agent.py", "ppo_train.py"]}
    return alive


def check_discrimination():
    import torch
    import zorbio_agent as za

    net = za.PolicyNet()
    net.load_state_dict(torch.load(os.path.join(RL_DIR, "policy.pt")))
    net.eval()

    def mk(iv):
        x = torch.zeros(38)
        x[3] = 0.3
        for k, v in iv.items():
            x[k] = v
        return x

    scen = {
        "wander": mk({}), "corner": mk({22: .05, 23: .05}),
        "prey-N": mk({19: 1, 21: .6, 9: .2}), "prey-S": mk({19: 1, 21: .6, 9: -.2}),
        "prey-E": mk({19: 1, 21: .6, 8: .2}), "prey-W": mk({19: 1, 21: .6, 8: -.2}),
        "threat-N": mk({35: .2, 37: .5}), "threat-S": mk({35: -.2, 37: .5}),
    }
    md = [[] for _ in range(4)]
    for x in scen.values():
        mean, _, _, _ = net(x)
        m = mean.squeeze(0).tolist()
        for i in range(4):
            md[i].append(m[i])
    spreads = [round(max(d) - min(d), 3) for d in md]
    return dict(zip(["dx", "dy", "dz", "speed"], spreads))


def check_stuck_fraction():
    if not os.path.exists(SCRATCH_SERVER_LOG):
        return None, 0
    pat = re.compile(
        r"ended after \d+ steps.*(stuck|terminated)"
    )
    n = 0
    stuck = 0
    # only look at the tail to keep this cheap
    with open(SCRATCH_SERVER_LOG, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 2_000_000))
        data = f.read().decode(errors="ignore")
    for line in data.splitlines()[-2000:]:
        m = pat.search(line)
        if not m:
            continue
        n += 1
        if m.group(1) == "stuck":
            stuck += 1
    if n == 0:
        return None, 0
    return round(100 * stuck / n, 2), n


def check_hangs():
    if not os.path.exists(SCRATCH_SERVER_LOG):
        return 0
    with open(SCRATCH_SERVER_LOG, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 500_000))
        data = f.read().decode(errors="ignore")
    return data.count("RLBridge") and (data.count("destroy") + data.count("watchdog") + data.count("timed out"))


def main():
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    alerts = []

    procs = check_processes()
    for name, alive in procs.items():
        if not alive:
            alerts.append(f"PROCESS DOWN: {name}")

    spreads = {}
    try:
        spreads = check_discrimination()
        # per-dimension thresholds, set below the lowest value ever observed for
        # each dim (speed's whole historical range is ~0.18-0.62, well under the
        # old uniform 0.3 - that was firing on ordinary noise, not saturation)
        dim_thresholds = {"dx": 1.0, "dy": 1.0, "dz": 0.5, "speed": 0.10}
        for dim, val in spreads.items():
            thresh = dim_thresholds.get(dim, 0.3)
            if val < thresh:
                alerts.append(f"DISCRIMINATION COLLAPSE: {dim} spread={val} (below {thresh})")
    except Exception as e:
        alerts.append(f"discrimination check failed: {e}")

    stuck_pct, n = check_stuck_fraction()
    if stuck_pct is not None and stuck_pct > 10.0:
        alerts.append(f"STUCK-FRACTION SPIKE: {stuck_pct}% (n={n})")

    hangs = check_hangs()

    line = (
        f"{ts} procs={procs} spreads={spreads} "
        f"stuck={stuck_pct}%(n={n}) hangs={hangs}"
    )
    if alerts:
        line += "  *** ALERT: " + " | ".join(alerts) + " ***"

    with open(REPORT_PATH, "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    main()
