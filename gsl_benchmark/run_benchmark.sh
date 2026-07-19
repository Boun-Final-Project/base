#!/usr/bin/env bash
# gsl_benchmark — one command to run the full 29-scenario GSL benchmark.
#
# Everything the benchmark needs lives in this folder:
#   - oracle_budgets_nav2_reliable.json : the per-scenario oracle distance/time budgets
#   - this script                       : wires an agent + the benchmark29 suite + the
#                                         5x/10x/20x envelope into `ros2 run gsl_bench eval`
#   - aggregate.py                      : re-aggregate any results dir with 95% CIs
#
# The actual engine is the gsl_bench package (harness owns sim + Nav2 + metrics).
# This script just picks the robot (agent), the scenarios, and the repeat count.
#
# Usage:
#   ./run_benchmark.sh <agent> [runs] [extra gsl_bench eval flags...]
#
# Examples:
#   ./run_benchmark.sh gpulaika                 # 29 scenarios x 5 runs, defaults
#   ./run_benchmark.sh gpulaika 3               # 3 runs each
#   ./run_benchmark.sh gpulaika 5 --scenario ultimate_1,ultimate_2   # subset
#   ./run_benchmark.sh adsm 5 --realistic       # ADSM needs realistic perception
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"

AGENT="${1:?usage: run_benchmark.sh <agent> [runs] [extra flags...]}"
RUNS="${2:-5}"
shift || true; shift 2>/dev/null || true   # remaining args pass through

# The robot / method under test. "Selecting a robot" == selecting an agent + its
# config; the checkpoint (if any) is loaded by the agent from its config YAML.
AGENT_CONFIG="$WS/src/gsl_bench/configs/agents/${AGENT}.yaml"
CONFIG_FLAG=()
[[ -f "$AGENT_CONFIG" ]] && CONFIG_FLAG=(--agent-config "$AGENT_CONFIG")

ORACLE="$HERE/oracle_budgets_nav2_reliable.json"

# ROS_DOMAIN_ID keeps this sweep isolated from anything else on the host. Never run
# two sweeps at once on the same host without distinct ids (see repo memory).
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-42}"

# shellcheck disable=SC1091
set +u; source "$WS/install/setup.bash"; set -u   # setup.bash uses unbound vars

echo "gsl_benchmark: agent=$AGENT  suite=benchmark29 (29)  runs=$RUNS  domain=$ROS_DOMAIN_ID"
echo "  success<=5x oracle dist | terminate at 10x dist / 20x time / stuck"

exec ros2 run gsl_bench eval \
  --agent "$AGENT" \
  "${CONFIG_FLAG[@]}" \
  --suite benchmark29 \
  --runs "$RUNS" \
  --oracle-budgets "$ORACLE" \
  --budget-multiplier 10 \
  --time-budget-multiplier 20 \
  --success-budget-multiplier 5 \
  --success-radius 0.5 \
  "$@"
