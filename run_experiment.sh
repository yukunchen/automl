#!/usr/bin/env bash
# Run ONE autoresearch-cv experiment: train + deploy + emit a candidate
# results.tsv row.
#
# Usage:
#   bash run_experiment.sh "<short description>"
#
# Caller (the agent) is expected to have:
#   - made changes in student.py and/or deploy.py
#   - git committed those changes
#   - cd'd into the repo root
#
# This script does NOT touch git. The agent decides keep/discard/revert
# based on the printed RESULTS_ROW vs the current best in results.tsv.

set -uo pipefail
export LANG=C

DESCRIPTION="${1:-(no description)}"

if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
    # shellcheck disable=SC1091
    source /root/miniconda3/etc/profile.d/conda.sh
    conda activate base
fi

COMMIT=$(git rev-parse --short HEAD)
echo "=== experiment $COMMIT  $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== description: $DESCRIPTION ==="

rm -f run.log metrics_train.json metrics_deploy.json

run_phase() {
    local phase="$1" venv="$2" script="$3"
    echo "--- $phase: $script ---"
    # shellcheck disable=SC1090,SC1091
    source "$venv/bin/activate"
    if ! python "$script" 2>&1 | tee -a run.log; then
        echo "$phase phase crashed; tail of run.log:"
        tail -n 60 run.log
        deactivate
        return 1
    fi
    deactivate
}

if ! run_phase "train" ".venv-train" "student.py"; then
    PHASE_FAILED=train
elif ! run_phase "deploy" ".venv-deploy" "deploy.py"; then
    PHASE_FAILED=deploy
else
    PHASE_FAILED=""
fi

# Emit a TSV-ready row to stdout. The agent reads this and updates results.tsv.
python3 <<PY
import json, math, os
commit = "$COMMIT"
desc = """$DESCRIPTION""".replace("\t", " ").replace("\n", " ")
phase_failed = "$PHASE_FAILED"

if phase_failed:
    print()
    print(f"RESULTS_ROW\t{commit}\t-inf\t0\t0\t0\t0\t0\tfalse\t-\tcrash\t{phase_failed} crashed: {desc}")
    raise SystemExit(0)

try:
    m = json.loads(open("metrics_deploy.json").read())
except Exception as e:
    print(f"RESULTS_ROW\t{commit}\t-inf\t0\t0\t0\t0\t0\tfalse\t-\tcrash\tno deploy metrics: {e!r}: {desc}")
    raise SystemExit(0)

def f(key, fmt="{:.4f}", default=0.0):
    v = m.get(key, default)
    if v is None: return "0"
    if isinstance(v, float) and v == float("-inf"): return "-inf"
    try: return fmt.format(v)
    except: return str(v)

score = m.get("score", float("-inf"))
score_s = "-inf" if (score == float("-inf") or score is None) else f"{score:.4f}"
qnn_ok = "true" if m.get("qnn_export_ok") else "false"

# status = "?" -- agent decides keep/discard against the current best
row = "\t".join([
    commit,
    score_s,
    f("fp32_map"),
    f("quantized_map"),
    f("latency_ms_p50", "{:.2f}"),
    f("size_mb", "{:.2f}"),
    f("peak_mem_mb", "{:.1f}"),
    qnn_ok,
    str(m.get("cpu_fallback_ops", 0)),
    "?",
    desc,
])
print()
print(f"RESULTS_ROW\t{row}")
print(f"CONSTRAINT_STATUS\t{m.get('constraint_status','?')}")
PY

if [ -n "$PHASE_FAILED" ]; then
    exit 1
fi
