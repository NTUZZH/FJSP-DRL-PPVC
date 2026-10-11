#!/usr/bin/env bash
# run_revision_gpu_queue.sh -- GPU training queue for the journal revision.
#
# Every arm: PPVC data, default factory (25 stations), mixed class mix,
# 2000 PPO updates, --vali_size 100 --ppvc_vali_sizes 10,20, DANIEL unless
# stated. H = the headline configuration chosen at the gate:
#   --headline mix    -> --ppvc_size_mix 10,15,20  (names mix10-15-20x25+ppvc-mixed+...)
#   --headline single -> --n_j 10                  (names 10x25+ppvc-mixed+...)
#
#  #   label          backbone arm   extra flags                        seed suffix             params  comparator snapshot (allowed diffs)
#  1   pilot-mix      DANIEL   A3    --ppvc_size_mix 10,15,20           301  full2k-s301         32530   mix10-20-30x25+ppvc-mixed+full-sizemix (size mix, max_updates, vali sizes, seed, suffix)
#  2   pilot-single   DANIEL   A3    --n_j 10                           301  full2k-s301         32530   10x25+ppvc-mixed+full (max_updates, vali sizes, seed, suffix)
#      GATE: greedy vs SPT at 20/30/50/80 modules; the queue STOPS. Continue with --from 3 --headline mix|single
#  3   H-seed302      DANIEL   A3    H                                  302  full2k-s302         32530   H A3 s301 (seed, suffix)
#  4   H-seed303      DANIEL   A3    H                                  303  full2k-s303         32530   H A3 s301 (seed, suffix)
#  5   H-A0           DANIEL   A0    H --ppvc_lagblind True             301  a0-2k-s301          28834   H A3 s301 (lagblind, lag features, embeddings, fea_j_input_dim, suffix)
#  6   H-A1           DANIEL   A1    H                                  301  a1-2k-s301          28834   H A3 s301 (lag features, embeddings, fea_j_input_dim, suffix)
#  7   H-A2           DANIEL   A2    H                                  301  a2-2k-s301          29346   H A3 s301 (embeddings, suffix)
#  8   H-A1-lagfreeLB DANIEL   A1    H --bound_lag_free True            301  a1nolagbound-2k-s301 28834  H A1 s301 (bound_lag_free, suffix)
#  9   H-PLAN-A1      PLAN     A1    H --minibatch_size 256             301  a1-2k-s301 (+plan-) 13506   H A1 s301 (framework, minibatch)
#  10  H-PLAN-A3      PLAN     A3    H --minibatch_size 256             301  full2k-s301 (+plan-) 13762  H A3 s301 (framework, minibatch)
#  11  H-PLAN-A0      PLAN     A0    H --minibatch_size 256             301  a0-2k-s301 (+plan-) 13506   H A0 s301 (framework, minibatch)
#  12  HGNN-A0        HGNN     A0    --lag_mode blind --bound_pt mean   300  run hgnn-a0-s300    204250  none (first HGNN run)
#  13  HGNN-A1        HGNN     A1    --lag_mode aware --bound_pt min    300  run hgnn-a1-s300    204250  HGNN A0 run_config (lag_mode, bound_pt, run_name)
#  14  var-routes     DANIEL   A3    --n_j 10 --ppvc_route_variant variable 301 full2k-s301 (-var) 32530 pilot-single (route variant)
#  15  lag-cv02       DANIEL   A3    H --ppvc_lag_cv 0.2                301  full2k-cv02-s301    32530   H A3 s301 (lag cv, suffix)      [SKIP if the flag is absent]
#  16a ft-tight       DANIEL   A3    --init_from H-A3-s301 --ppvc_factory tight --n_j 10 --max_updates 400
#                                                                        301  ft-tight-s301       32530   H A3 s301 (init_from, factory, n_m, size mix, max_updates, suffix)
#  16b ft-lag15       DANIEL   A3    H --init_from H-A3-s301 --lag_scale 1.5 --max_updates 400
#                                                                        301  ft-lag15-s301       32530   H A3 s301 (init_from, lag_scale, max_updates, suffix) [SKIP if flags absent]
#
# Per arm: (1) skip if the run is complete, abort if a partial run exists;
# (2) the four-question block is written to the log; (3) preflight_train.py
# resolves the config on the CPU, asserts the parameter count, diffs it
# against the comparator, checks checkpoint and data; the arm is aborted on
# any failure; (4) wait for the GPU share marker and for no other training of
# this repository; (5) train pinned to CORES with thread caps, logging
# wall-clock, load average and GPU processes at start and end; (6) greedy
# evaluation (eval_ppvc.py, cuda) on the arm's test sets, re-checking the
# GPU marker before every job; A0 arms also run eval_a0_repair.py.
#
# Usage:
#   bash run_revision_gpu_queue.sh                          # arms 1-2, gate, stop
#   bash run_revision_gpu_queue.sh --from 3 --headline mix  # arms 3-16
#   options: --from N  --to N  --only N  --headline mix|single  --dry_run
# Environment: CORES (default 16-17), GPU_SHARE_FILE (default empty =
#   no wait; else a file whose presence permits each GPU job), POLL_SEC (default 600), PY.
# Log: train_log/revision_gpu_queue.log; per-job logs train_log/revq_*.log
set -uo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

FROM=1; TO=""; ONLY=""; HEADLINE=""; DRY_RUN=0
while [ $# -gt 0 ]; do
    case "$1" in
        --from) FROM="$2"; shift 2 ;;
        --to) TO="$2"; shift 2 ;;
        --only) ONLY="$2"; shift 2 ;;
        --headline) HEADLINE="$2"; shift 2 ;;
        --dry_run|--dry-run) DRY_RUN=1; shift ;;
        -h|--help) sed -n '2,50p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -n "$ONLY" ] && { FROM="$ONLY"; TO="$ONLY"; }
if [ -z "$TO" ]; then
    if [ "$FROM" -le 2 ]; then TO=2; else TO=16; fi
fi
if [ "$TO" -ge 3 ] && [ "$HEADLINE" != "mix" ] && [ "$HEADLINE" != "single" ]; then
    echo "arms >= 3 need --headline mix|single (chosen at the gate)" >&2; exit 2
fi

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate fjsp 2>/dev/null || true
fi
PY="${PY:-python}"

CORES="${CORES:-16-17}"
GPU_SHARE_FILE="${GPU_SHARE_FILE:-}"
POLL_SEC="${POLL_SEC:-600}"
LOG="train_log/revision_gpu_queue.log"
mkdir -p train_log
[ "$DRY_RUN" = 1 ] && LOG=/dev/null   # a dry run prints only

# number of cores in CORES ("16-17", "0,2,4-7") -> thread caps
count_cores() {
    local n=0 part a b
    IFS=',' read -ra parts <<< "$1"
    for part in "${parts[@]}"; do
        if [[ "$part" == *-* ]]; then a="${part%-*}"; b="${part#*-}"; n=$((n + b - a + 1))
        else n=$((n + 1)); fi
    done
    echo "$n"
}
NCORES="$(count_cores "$CORES")"
PIN=(env OMP_NUM_THREADS="$NCORES" MKL_NUM_THREADS="$NCORES" OPENBLAS_NUM_THREADS="$NCORES"
     NUMEXPR_NUM_THREADS="$NCORES" taskset -c "$CORES")

ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*" | tee -a "$LOG"; }

# one queue instance at a time
if [ "$DRY_RUN" = 0 ]; then
    exec 9>"train_log/revision_gpu_queue.lock"
    if ! flock -n 9; then echo "another run_revision_gpu_queue.sh holds the lock" >&2; exit 3; fi
fi

# ---------------------------------------------------------------------------
# configuration shared by the arms
# ---------------------------------------------------------------------------
A3="--use_lag_features True --use_type_embedding True"
A2="--use_lag_features True --use_type_embedding False"
A1="--use_lag_features False --use_type_embedding False"
A0="--ppvc_lagblind True --use_lag_features False --use_type_embedding False"
base_flags() {  # base_flags <max_updates> <factory>
    echo "--data_source PPVC --device cuda --ppvc_factory $2 --ppvc_mix mixed --max_updates $1 --vali_size 100 --ppvc_vali_sizes 10,20"
}
COMMON="$(base_flags 2000 default)"
MIX_FLAGS="--ppvc_size_mix 10,15,20"; MIX_DATA="mix10-15-20x25+ppvc-mixed"
SINGLE_FLAGS="--n_j 10";              SINGLE_DATA="10x25+ppvc-mixed"
case "$HEADLINE" in
    mix)    H_FLAGS="$MIX_FLAGS";    H_DATA="$MIX_DATA" ;;
    single) H_FLAGS="$SINGLE_FLAGS"; H_DATA="$SINGLE_DATA" ;;
    *)      H_FLAGS="";              H_DATA="" ;;
esac
SNAP="train_log/PPVC"
H_A3_S301="${H_DATA}+full2k-s301"

EVAL_GATE="20x25+ppvc-mixed 30x25+ppvc-mixed 50x25+ppvc-mixed 80x25+ppvc-mixed"
EVAL_MAIN="10x25+ppvc-mixed 20x25+ppvc-mixed 30x25+ppvc-mixed 50x25+ppvc-mixed 80x25+ppvc-mixed"

# ---------------------------------------------------------------------------
# the four questions, one row per arm: id | purpose | ideal result | contamination | data accuracy
# ---------------------------------------------------------------------------
read -r -d '' FOUR_Q <<'EOF_Q'
1|Pilot for the headline regime: size-mixed training (10/15/20 modules) at 2000 updates. Lands in the gate table and decides H for every later arm and the scale table (20-80 modules).|Greedy mean makespan below SPT at 20, 30, 50 and 80 modules. If it loses at 50/80 while pilot-single wins, H=single; if both lose at 50/80, stop before the seed and ablation arms and rethink the training regime.|New suffix full2k-s301: no existing checkpoint is resumed or overwritten; complete/partial runs detected before launch; one training at a time on a GPU shared only through the marker file; load average and GPU processes logged at start and end.|Validation in memory, seed0=100 (seeds 100-199) at 10 and 20 modules, 100 each; test sets 20x25 (seed0 80000, 50), 30x25 (90000, 20), 50x25 (100000, 50), 80x25 (110000, 50), counts checked against dataset_meta.json by the preflight.
2|Pilot for the headline regime: single-size training at 10 modules, 2000 updates (the regime of the submitted paper, at twice the budget). Lands in the gate table next to pilot-mix.|Greedy below SPT at 20-80 modules. If single matches or beats mix at 50/80, H=single (fewer moving parts); if it loses where mix wins, H=mix.|New suffix full2k-s301 on the 10x25 name: the existing 10x25+ppvc-mixed+full*.pth files are untouched; same GPU/partial-run guards as arm 1.|Validation on disk at data/data_train_vali/PPVC/10x25+ppvc-mixed, seed0=100; the preflight checks the provenance record and regenerates instance_000 against the file. Test sets as arm 1.
3|Second seed of H (A3). Lands in the main table as mean and spread over seeds 301-303.|Within about 2% of the seed-301 makespan at every size. A larger spread means single-seed ablation gaps below that spread cannot be claimed.|Only seed_train and suffix differ from H seed 301 (asserted by the diff); name full2k-s302 is new.|Same validation seeds (100..) and test sets as the pilot; only the training stream changes.
4|Third seed of H (A3). Lands in the main table as mean and spread over seeds 301-303.|As arm 3.|As arm 3 (suffix full2k-s303).|As arm 3.
5|Ablation row A0 under H: trained with all lags zeroed, the industry-practice stand-in. Lands in the ablation table (repaired makespan) and the plan-versus-reality inflation.|A0 repaired makespan clearly worse than A3. If A0 ties A3, lag awareness does not pay at this budget and the ablation claim must be restated.|Differs from H A3 s301 only in lagblind, lag features, embeddings, fea_j_input_dim, suffix (asserted); new suffix a0-2k-s301.|Training and validation lags are zeroed in memory; the on-disk validation files keep true lags; evaluation uses true lags (repair + validator).
6|Ablation row A1 under H: bare DANIEL with the lag-aware environment and bound. Lands in the ablation table.|A1 worse than A2 worse than A3. Another ordering is reported as found and the text on the adaptations changes accordingly.|Differs from H A3 s301 only in lag features, embeddings, fea_j_input_dim, suffix (asserted).|Same validation seeds and test sets as H.
7|Ablation row A2 under H: lag features without type embeddings. Lands in the ablation table.|Between A1 and A3. If A2 equals A3, the embeddings add nothing at H.|Differs from H A3 s301 only in embeddings and suffix (asserted).|Same validation seeds and test sets as H.
8|Isolates the lag-augmented lower bound (Adaptation 2a): A1 with the original lag-free bound in reward and features. Compared to arm 6 only; lands in the ablation table as one extra row.|Worse than arm 6. If equal, the bound is not what makes A1 work and the paper stops crediting it.|Differs from arm 6 only in bound_lag_free and suffix (asserted); the flag is recorded in the snapshot and applied by eval_ppvc.py.|Same validation seeds and test sets as H; dynamics keep the true lags (bound flag only changes the estimate).
9|PLAN backbone, A1, under H: shows the adaptations plug into a second backbone. Lands in the multi-backbone table.|PLAN A1 worse than PLAN A3 (arm 10). If PLAN A1 matches A3, the adaptations are backbone-specific.|Differs from H A1 (arm 6) only in framework and minibatch (asserted); name carries +plan-, so no DANIEL file collides.|Same validation seeds and test sets as H.
10|PLAN backbone, A3, under H. Lands in the multi-backbone table.|Better than PLAN A1 and PLAN A0.|Differs from H A3 s301 only in framework and minibatch (asserted).|Same validation seeds and test sets as H.
11|PLAN backbone, A0, under H. Lands in the multi-backbone table.|Worse than PLAN A3 after repair.|Differs from H A0 (arm 5) only in framework and minibatch (asserted).|Lags zeroed in training/validation only, as arm 5.
12|HGNN backbone, lag-blind with the upstream mean-time estimate (A0 of the third backbone). Lands in the multi-backbone table.|Worse than HGNN A1 after repair at 10 and 20 modules.|New run directory hgnn-a0-s300 under backbones/hgnn/save; completion marker written by the queue; partial directories abort the arm.|Validation in memory, seeds 100-199 at 10 modules; tests 10x25 (seed0 50000, 100) and 20x25 (80000, 50).
13|HGNN backbone, lag-aware with the min-time lower bound (A1 of the third backbone). Lands in the multi-backbone table.|Better than HGNN A0 at 10 and 20 modules.|Differs from the HGNN A0 run only in lag_mode, bound_pt, run_name (asserted).|As arm 12.
14|Variable routes (modules of 18-22 operations), A3 at 10 modules. Lands in the variable-route table.|Greedy below SPT on the variable-route test sets.|Differs from pilot-single only in the route variant (asserted); names carry -var.|Validation on disk under 10x25+ppvc-mixed-var (seed0 100); tests 10x25-var (seed0 120000, 100) and 20x25-var (130000, 50).
15|Training with randomized lags (lognormal CV 0.2) under H. Lands in the stochastic-lag robustness table.|Equal or better than H A3 s301 under perturbed lags, no worse than 1% at nominal lags.|Differs from H A3 s301 only in ppvc_lag_cv and suffix (asserted).|Validation keeps nominal lags (seed0 100); test sets as H.
16|Fine-tuning curve from H A3 s301 to the tight factory (15 stations) and to 1.5x lags, 400 updates each. Lands in the transfer figure (quality versus updates, from valicurve_*.json).|Quality reaches the from-scratch level within a fraction of the 2000-update budget. If not, transfer by fine-tuning is not cheaper and the figure says so.|The source checkpoint is loaded strictly and its sha256 recorded; source flags must match (checked in train.py); new suffixes ft-tight-s301 / ft-lag15-s301.|Tight: validation on disk at 10x15 (seed0 100), test 10x15+ppvc-mixed. Lag 1.5x: validation stored under a -lagx1.5 directory; its test set must exist with scaled lags, otherwise its evaluation is skipped and logged.
EOF_Q

four_questions() {
    local id="$1" line
    line="$(printf '%s\n' "$FOUR_Q" | awk -F'|' -v id="$id" '$1 == id')"
    [ -z "$line" ] && { say "  (no four-question entry for arm $id)"; return; }
    IFS='|' read -r _ q1 q2 q3 q4 <<< "$line"
    {
        echo "  Purpose       : $q1"
        echo "  Ideal result  : $q2"
        echo "  Contamination : $q3"
        echo "  Data accuracy : $q4"
    } | tee -a "$LOG"
}

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
run_cmd() {  # run_cmd <logfile> <cmd...>
    local lf="$1"; shift
    if [ "$DRY_RUN" = 1 ]; then echo "    + $*"; return 0; fi
    echo "    + $*" >> "$LOG"
    "$@" >> "$lf" 2>&1
}

flag_exists() {  # flag_exists --name
    "$PY" - "$1" <<'EOF_F'
import sys
flag = sys.argv[1]
sys.argv = [sys.argv[0]]
import params
sys.exit(0 if any(flag in a.option_strings for a in params.parser._actions) else 1)
EOF_F
}

wait_gpu_share() {
    [ -z "$GPU_SHARE_FILE" ] && return 0
    if [ "$DRY_RUN" = 1 ]; then
        if [ -e "$GPU_SHARE_FILE" ]; then echo "    [gpu] share marker present"; else echo "    [gpu] share marker ABSENT: the job would wait, polling every ${POLL_SEC}s"; fi
        return 0
    fi
    while [ ! -e "$GPU_SHARE_FILE" ]; do
        say "  GPU share marker absent ($GPU_SHARE_FILE); waiting ${POLL_SEC}s"
        sleep "$POLL_SEC"
    done
}

# another training of THIS repository (cwd = repo) still running?
other_training_running() {
    local pid args cwd
    while read -r pid args; do
        [ "$pid" = "$$" ] && continue
        set -- $args
        case "$(basename "${1:-}")" in python*) ;; *) continue ;; esac
        local tok hit=0
        for tok in "$@"; do
            case "$(basename "$tok")" in train.py|train_ppvc.py) hit=1 ;; esac
        done
        [ "$hit" = 1 ] || continue
        cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
        [ "$cwd" = "$REPO_DIR" ] && return 0
    done < <(ps -eo pid=,args=)
    return 1
}

wait_training_free() {
    [ "$DRY_RUN" = 1 ] && return 0
    while other_training_running; do
        say "  another training of this repository is running; waiting 60s"
        sleep 60
    done
}

gpu_procs() {
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null \
            | tr '\n' ';' ; echo
    else
        echo "nvidia-smi unavailable"
    fi
}

stamp() {  # stamp START|END <label>
    say "  $1 $2  epoch=$(date +%s)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)  cores=$CORES threads=$NCORES"
    say "  $1 $2  gpu-procs: $(gpu_procs)  share-marker: $([ -e "$GPU_SHARE_FILE" ] && echo present || echo ABSENT)"
}

timed_gpu_job() {  # timed_gpu_job <label> <logfile> <cmd...>
    local label="$1" lf="$2"; shift 2
    wait_gpu_share
    if [ "$DRY_RUN" = 1 ]; then echo "    + $*"; return 0; fi
    wait_training_free
    stamp START "$label"
    local t0; t0=$(date +%s)
    echo "    + $*" >> "$LOG"
    "$@" >> "$lf" 2>&1
    local st=$?
    stamp END "$label"
    say "  $label exit=$st wall=$(( $(date +%s) - t0 ))s log=$lf"
    return $st
}

preflight() {  # preflight <model> <expect> <compare|-> <allow> <testsets> <flags...>
    local model="$1" expect="$2" cmp="$3" allow="$4" tsets="$5"; shift 5
    local out="train_log/revq_preflight_${model}.json" plog="train_log/revq_preflight_${model}.log"
    local args=(--expect_params "$expect" --check_ckpt --check_data --out "$out")
    [ "$cmp" != "-" ] && args+=(--compare "$cmp" --allow_diff "$allow")
    [ -n "$tsets" ] && args+=(--testsets "$tsets")
    if [ "$DRY_RUN" = 1 ]; then
        echo "    + ${PIN[*]} $PY preflight_train.py ${args[*]} $*"
        return 0
    fi
    "${PIN[@]}" "$PY" preflight_train.py "${args[@]}" "$@" > "$plog" 2>&1
    local st=$?
    grep -E '^(PREFLIGHT|DIFF|  [a-z_.]+ .*\[(allowed|NOT ALLOWED)\]|  absent|  allowed|  \(no diff|  \[(data|test)\])' "$plog" | tee -a "$LOG"
    if [ $st -ne 0 ]; then say "  preflight exit $st (full output: $plog)"; return $st; fi
    local resolved
    resolved="$("$PY" -c "import json,sys; d=json.load(open(sys.argv[1]))['derived']; print(d.get('model_name', d.get('run_name')))" "$out")"
    if [ "$resolved" != "$model" ]; then
        say "  preflight resolved name '$resolved' != queue name '$model'"; return 9
    fi
    return 0
}

eval_done() {  # eval_done <model> <ds> <ckpt> : result exists and is newer than the checkpoint
    local f="test_results/PPVC/$2/Result_greedy+$1_$2.npy"
    [ -f "$f" ] && [ "$f" -nt "$3" ]
}

eval_sets() {  # eval_sets <model> <kind: greedy|a0> <datasets...>
    local model="$1" kind="$2"; shift 2
    local ckpt="trained_network/PPVC/${model}.pth" ds
    for ds in "$@"; do
        if [ ! -d "data/PPVC/$ds" ]; then say "  SKIP eval ${model} on ${ds} (test set absent)"; continue; fi
        if [ "$DRY_RUN" = 0 ] && eval_done "$model" "$ds" "$ckpt"; then
            say "  SKIP eval ${model} on ${ds} (result newer than checkpoint)"
        else
            say "  EVAL ${model} on ${ds} (greedy)"
            timed_gpu_job "eval ${model} ${ds}" "train_log/revq_eval_${model}.log" \
                "${PIN[@]}" "$PY" eval_ppvc.py --model_name "$model" --data_path "data/PPVC/$ds" \
                --methods greedy --device cuda \
                || say "  WARN eval ${model} on ${ds} failed"
        fi
        if [ "$kind" = "a0" ]; then
            say "  EVAL ${model} on ${ds} (A0 lag-blind plan + right-shift repair)"
            timed_gpu_job "a0eval ${model} ${ds}" "train_log/revq_eval_${model}.log" \
                "${PIN[@]}" "$PY" eval_a0_repair.py --model_name "$model" --data_path "data/PPVC/$ds" \
                || say "  WARN A0 repair eval ${model} on ${ds} failed"
        fi
    done
}

# train_arm <id> <label> <model> <expect> <compare|-> <allow> <kind> <eval sets> -- <train flags...>
ARM_STATUS=()
train_arm() {
    local id="$1" label="$2" model="$3" expect="$4" cmp="$5" allow="$6" kind="$7" sets="$8"; shift 8
    [ "${1:-}" = "--" ] && shift
    local ckpt="trained_network/PPVC/${model}.pth" rlog="train_log/PPVC/reward_${model}.txt"
    say "=== ARM ${id} ${label}: ${model}"
    four_questions "${id%[ab]}"
    if [ -f "$ckpt" ] && [ -f "$rlog" ]; then
        say "  SKIP train ${model} (complete: checkpoint and final reward log exist)"
    elif [ -f "$ckpt" ]; then
        say "  ABORT arm ${id}: partial run (checkpoint without final reward log). Inspect, then delete ${ckpt} to retrain."
        ARM_STATUS+=("${id} ${model}: ABORTED (partial run)"); return 1
    else
        # test sets that exist are verified by the preflight; absent ones are
        # logged as SKIP at evaluation time
        local present="" s
        for s in $sets; do [ -d "data/PPVC/$s" ] && present="${present:+$present,}$s"; done
        if ! preflight "$model" "$expect" "$cmp" "$allow" "$present" "$@"; then
            say "  ABORT arm ${id}: preflight failed"
            ARM_STATUS+=("${id} ${model}: ABORTED (preflight)"); return 1
        fi
        say "  TRAIN ${model}"
        if ! timed_gpu_job "train ${model}" "train_log/revq_train_${model}.log" \
                "${PIN[@]}" "$PY" -u train.py "$@"; then
            say "  FAIL train ${model}"; ARM_STATUS+=("${id} ${model}: TRAIN FAILED"); return 1
        fi
        if [ "$DRY_RUN" = 0 ] && { [ ! -f "$ckpt" ] || [ ! -f "$rlog" ]; }; then
            say "  FAIL train ${model}: checkpoint or final reward log missing"
            ARM_STATUS+=("${id} ${model}: TRAIN INCOMPLETE"); return 1
        fi
    fi
    # shellcheck disable=SC2086
    eval_sets "$model" "$kind" $sets
    ARM_STATUS+=("${id} ${model}: $([ "$DRY_RUN" = 1 ] && echo listed || echo done)")
}

# hgnn_arm <id> <label> <run> <expect> <compare|-> <allow> <eval arm> <eval sets> -- <train_ppvc flags...>
hgnn_arm() {
    local id="$1" label="$2" run="$3" expect="$4" cmp="$5" allow="$6" earm="$7" sets="$8"; shift 8
    [ "${1:-}" = "--" ] && shift
    local dir="backbones/hgnn/save/${run}" ds
    say "=== ARM ${id} ${label}: ${run}"
    four_questions "$id"
    if [ -f "$dir/queue_done.txt" ]; then
        say "  SKIP train ${run} (complete)"
    elif [ -f "$dir/best.pt" ]; then
        say "  ABORT arm ${id}: partial HGNN run in ${dir} (no completion marker)"
        ARM_STATUS+=("${id} ${run}: ABORTED (partial run)"); return 1
    else
        local args=(--hgnn --expect_params "$expect" --check_ckpt --check_data
                    --out "train_log/revq_preflight_${run}.json")
        [ "$cmp" != "-" ] && args+=(--compare "$cmp" --allow_diff "$allow")
        if [ "$DRY_RUN" = 1 ]; then
            echo "    + ${PIN[*]} $PY preflight_train.py ${args[*]} $*"
        else
            local plog="train_log/revq_preflight_${run}.log"
            if ! "${PIN[@]}" "$PY" preflight_train.py "${args[@]}" "$@" > "$plog" 2>&1; then
                grep -E '^(PREFLIGHT|DIFF|  )' "$plog" | grep -v '^  "' | tee -a "$LOG"
                say "  ABORT arm ${id}: preflight failed (full output: $plog)"
                ARM_STATUS+=("${id} ${run}: ABORTED (preflight)"); return 1
            fi
            grep -E '^(PREFLIGHT|DIFF|  [a-z_.]+ .*\[(allowed|NOT ALLOWED)\]|  \[data\])' "$plog" | tee -a "$LOG"
        fi
        say "  TRAIN ${run}"
        if ! timed_gpu_job "train ${run}" "train_log/revq_train_${run}.log" \
                "${PIN[@]}" "$PY" -u backbones/hgnn/train_ppvc.py "$@"; then
            say "  FAIL train ${run}"; ARM_STATUS+=("${id} ${run}: TRAIN FAILED"); return 1
        fi
        [ "$DRY_RUN" = 0 ] && date '+%Y-%m-%d %H:%M:%S' > "$dir/queue_done.txt"
    fi
    for ds in $sets; do
        local res="test_results/PPVC/${ds}/Result_greedy+hgnn-${earm}+${run}_${ds}.npy"
        if [ "$DRY_RUN" = 0 ] && [ -f "$res" ] && [ "$res" -nt "$dir/best.pt" ]; then
            say "  SKIP eval ${run} on ${ds} (result newer than checkpoint)"; continue
        fi
        say "  EVAL hgnn-${earm}+${run} on ${ds}"
        timed_gpu_job "eval ${run} ${ds}" "train_log/revq_eval_${run}.log" \
            "${PIN[@]}" "$PY" backbones/hgnn/eval_ppvc_hgnn.py --ckpt "$dir/best.pt" --arm "$earm" \
            --data_path "data/PPVC/$ds" --tag "$run" --device cuda --threads "$NCORES" \
            || say "  WARN eval ${run} on ${ds} failed"
    done
    ARM_STATUS+=("${id} ${run}: $([ "$DRY_RUN" = 1 ] && echo listed || echo done)")
}

# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
arm_1() {
    train_arm 1 pilot-mix "${MIX_DATA}+full2k-s301" 32530 \
        "$SNAP/config_mix10-20-30x25+ppvc-mixed+full-sizemix.json" \
        ppvc_size_mix,max_updates,ppvc_vali_sizes,vali_size,seed_train,model_suffix \
        greedy "10x25+ppvc-mixed $EVAL_GATE" -- \
        $COMMON $MIX_FLAGS $A3 --seed_train 301 --model_suffix full2k-s301
}
arm_2() {
    train_arm 2 pilot-single "${SINGLE_DATA}+full2k-s301" 32530 \
        "$SNAP/config_10x25+ppvc-mixed+full.json" \
        max_updates,ppvc_vali_sizes,vali_size,seed_train,model_suffix \
        greedy "10x25+ppvc-mixed $EVAL_GATE" -- \
        $COMMON $SINGLE_FLAGS $A3 --seed_train 301 --model_suffix full2k-s301
}
arm_3() {
    train_arm 3 H-seed302 "${H_DATA}+full2k-s302" 32530 "$SNAP/config_${H_A3_S301}.json" \
        seed_train,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A3 --seed_train 302 --model_suffix full2k-s302
}
arm_4() {
    train_arm 4 H-seed303 "${H_DATA}+full2k-s303" 32530 "$SNAP/config_${H_A3_S301}.json" \
        seed_train,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A3 --seed_train 303 --model_suffix full2k-s303
}
arm_5() {
    train_arm 5 H-A0 "${H_DATA}+a0-2k-s301" 28834 "$SNAP/config_${H_A3_S301}.json" \
        ppvc_lagblind,use_lag_features,use_type_embedding,fea_j_input_dim,model_suffix \
        a0 "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A0 --seed_train 301 --model_suffix a0-2k-s301
}
arm_6() {
    train_arm 6 H-A1 "${H_DATA}+a1-2k-s301" 28834 "$SNAP/config_${H_A3_S301}.json" \
        use_lag_features,use_type_embedding,fea_j_input_dim,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A1 --seed_train 301 --model_suffix a1-2k-s301
}
arm_7() {
    train_arm 7 H-A2 "${H_DATA}+a2-2k-s301" 29346 "$SNAP/config_${H_A3_S301}.json" \
        use_type_embedding,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A2 --seed_train 301 --model_suffix a2-2k-s301
}
arm_8() {
    train_arm 8 H-A1-lagfreeLB "${H_DATA}+a1nolagbound-2k-s301" 28834 \
        "$SNAP/config_${H_DATA}+a1-2k-s301.json" bound_lag_free,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A1 --bound_lag_free True --seed_train 301 --model_suffix a1nolagbound-2k-s301
}
arm_9() {
    train_arm 9 H-PLAN-A1 "${H_DATA}+plan-a1-2k-s301" 13506 \
        "$SNAP/config_${H_DATA}+a1-2k-s301.json" framework,minibatch_size greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A1 --framework PLAN --minibatch_size 256 --seed_train 301 --model_suffix a1-2k-s301
}
arm_10() {
    train_arm 10 H-PLAN-A3 "${H_DATA}+plan-full2k-s301" 13762 \
        "$SNAP/config_${H_A3_S301}.json" framework,minibatch_size greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A3 --framework PLAN --minibatch_size 256 --seed_train 301 --model_suffix full2k-s301
}
arm_11() {
    train_arm 11 H-PLAN-A0 "${H_DATA}+plan-a0-2k-s301" 13506 \
        "$SNAP/config_${H_DATA}+a0-2k-s301.json" framework,minibatch_size a0 "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A0 --framework PLAN --minibatch_size 256 --seed_train 301 --model_suffix a0-2k-s301
}
arm_12() {
    hgnn_arm 12 HGNN-A0 hgnn-a0-s300 204250 - - blind "10x25+ppvc-mixed 20x25+ppvc-mixed" -- \
        --n_modules 10 --factory default --mix mixed --lag_mode blind --bound_pt mean \
        --seed 300 --run_name hgnn-a0-s300 --device cuda --threads "$NCORES"
}
arm_13() {
    hgnn_arm 13 HGNN-A1 hgnn-a1-s300 204250 "backbones/hgnn/save/hgnn-a0-s300/run_config.json" \
        args.lag_mode,args.bound_pt,args.run_name,env_paras.lag_mode,env_paras.bound_pt \
        aware "10x25+ppvc-mixed 20x25+ppvc-mixed" -- \
        --n_modules 10 --factory default --mix mixed --lag_mode aware --bound_pt min \
        --seed 300 --run_name hgnn-a1-s300 --device cuda --threads "$NCORES"
}
arm_14() {
    train_arm 14 var-routes "10x25+ppvc-mixed-var+full2k-s301" 32530 \
        "$SNAP/config_${SINGLE_DATA}+full2k-s301.json" ppvc_route_variant \
        greedy "10x25+ppvc-mixed-var 20x25+ppvc-mixed-var" -- \
        $COMMON --n_j 10 --ppvc_route_variant variable $A3 --seed_train 301 --model_suffix full2k-s301
}
arm_15() {
    if ! flag_exists --ppvc_lag_cv; then
        say "=== ARM 15 lag-cv02: SKIP (train.py has no --ppvc_lag_cv)"; ARM_STATUS+=("15: SKIPPED (flag absent)"); return 0
    fi
    train_arm 15 lag-cv02 "${H_DATA}+full2k-cv02-s301" 32530 "$SNAP/config_${H_A3_S301}.json" \
        ppvc_lag_cv,model_suffix greedy "$EVAL_MAIN" -- \
        $COMMON $H_FLAGS $A3 --ppvc_lag_cv 0.2 --seed_train 301 --model_suffix full2k-cv02-s301
}
arm_16() {
    if ! flag_exists --init_from; then
        say "=== ARM 16 fine-tune: SKIP (train.py has no --init_from)"; ARM_STATUS+=("16: SKIPPED (flag absent)"); return 0
    fi
    train_arm 16a ft-tight "10x15+ppvc-mixed+ft-tight-s301" 32530 "$SNAP/config_${H_A3_S301}.json" \
        init_from,init_from_sha256,ppvc_factory,n_m,ppvc_size_mix,max_updates,model_suffix \
        greedy "10x15+ppvc-mixed" -- \
        $(base_flags 400 tight) --n_j 10 $A3 --init_from "$H_A3_S301" --validate_timestep 10 \
        --seed_train 301 --model_suffix ft-tight-s301
    if ! flag_exists --lag_scale; then
        say "=== ARM 16b ft-lag15: SKIP (train.py has no --lag_scale)"; ARM_STATUS+=("16b: SKIPPED (flag absent)"); return 0
    fi
    train_arm 16b ft-lag15 "${H_DATA}+ft-lag15-s301" 32530 "$SNAP/config_${H_A3_S301}.json" \
        init_from,init_from_sha256,lag_scale,max_updates,model_suffix \
        greedy "10x25+ppvc-mixed-lagx1.5" -- \
        $(base_flags 400 default) $H_FLAGS $A3 --init_from "$H_A3_S301" --lag_scale 1.5 \
        --validate_timestep 10 --seed_train 301 --model_suffix ft-lag15-s301
}

# ---------------------------------------------------------------------------
# gate: pilot greedy vs SPT at 20/30/50/80 modules
# ---------------------------------------------------------------------------
gate() {
    say "=== GATE: pilot greedy vs SPT"
    local ds
    for ds in $EVAL_GATE; do
        if ! ls "test_results/PPVC/$ds/"Result_SPT+*"_$ds.npy" >/dev/null 2>&1; then
            say "  SPT reference missing on $ds; computing it on the CPU (tag pdr-spt)"
            run_cmd "train_log/revq_spt_${ds}.log" "${PIN[@]}" "$PY" eval_ppvc.py \
                --model_name 10x25+ppvc-mixed+full --out_tag pdr-spt --methods SPT --device cpu \
                --data_path "data/PPVC/$ds" || say "  WARN SPT on $ds failed"
        fi
    done
    if [ "$DRY_RUN" = 1 ]; then
        echo "    [dry-run] gate table from test_results/PPVC/<ds>/Result_greedy+<pilot>_<ds>.npy and Result_SPT+*_<ds>.npy"
        return 0
    fi
    "$PY" - "${MIX_DATA}+full2k-s301" "${SINGLE_DATA}+full2k-s301" $EVAL_GATE <<'EOF_G' | tee -a "$LOG"
import glob, os, sys
import numpy as np
models, sets = sys.argv[1:3], sys.argv[3:]
def spt(ds):
    d = f'test_results/PPVC/{ds}'
    for pref in (f'{d}/Result_SPT+pdr_{ds}.npy', f'{d}/Result_SPT+pdr-spt_{ds}.npy'):
        if os.path.exists(pref):
            return pref
    c = sorted(glob.glob(f'{d}/Result_SPT+*_{ds}.npy'))
    return c[0] if c else None
print(f'{"model":<38}{"dataset":<20}{"n":>4}{"policy":>10}{"SPT":>10}{"gap%":>9}')
for m in models:
    for ds in sets:
        f = f'test_results/PPVC/{ds}/Result_greedy+{m}_{ds}.npy'
        s = spt(ds)
        if not os.path.exists(f) or s is None:
            print(f'{m:<38}{ds:<20}{"":>4}{"missing" if not os.path.exists(f) else "":>10}'
                  f'{"missing" if s is None else "":>10}')
            continue
        p, r = np.load(f)[:, 0], np.load(s)[:, 0]
        if len(p) != len(r):
            print(f'{m:<38}{ds:<20} instance count differs: policy {len(p)} SPT {len(r)}')
            continue
        print(f'{m:<38}{ds:<20}{len(p):>4}{p.mean():>10.1f}{r.mean():>10.1f}'
              f'{100 * (p.mean() / r.mean() - 1):>+9.2f}')
print('gap% = policy / SPT - 1 (negative: policy shorter). SPT file per set: '
      + ', '.join(f'{ds}: {os.path.basename(spt(ds) or "none")}' for ds in sets))
EOF_G
}

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
say "##### revision GPU queue: arms ${FROM}-${TO} headline=${HEADLINE:-unset} cores=${CORES} threads=${NCORES} dry_run=${DRY_RUN}"
say "      GPU share marker: ${GPU_SHARE_FILE} ($([ -e "$GPU_SHARE_FILE" ] && echo present || echo absent))"
for id in $(seq "$FROM" "$TO"); do
    "arm_${id}" || true
    if [ "$id" = 2 ]; then
        gate
        say "##### GATE reached: choose H, then run: bash run_revision_gpu_queue.sh --from 3 --headline mix|single"
        break
    fi
done
say "##### queue summary"
for s in "${ARM_STATUS[@]}"; do say "  $s"; done
