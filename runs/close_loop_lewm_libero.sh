#!/usr/bin/env bash
# Official LIBERO closed-loop evaluation of LeWM-based checkpoints (BC, PPP): 40 tasks x 50 episodes.
#
#   bash runs/close_loop_lewm_libero.sh RUN_DIR "checkpoint_120 [...]" [JOBS_PER_GPU] [GPUS] [EP_SEED]
#
# RUN_DIR holds dump_cfg.yaml and checkpoints/<ckpt>/. (ckpt, task) jobs are spread over GPUS with
# JOBS_PER_GPU concurrent jobs per GPU; a task whose result JSON already holds 50 episodes is skipped.
# Env vars: TASKS (default 0-39), CELL_TAG / EXTRA_ARGS (extra evaluation flags and a tag for their
# output folder), VIDEO_ARGS (default "--save_video --video_arms gg_p"; VIDEO_ARGS="" saves no video),
# PY (python executable, default `python`).
# Output: RUN_DIR/closed_loop/epseed<EP_SEED>_[<CELL_TAG>_]<ckpt>/task<N>.json (logs in .../logs/)
set -u
RUN_DIR="$1"
CKPTS="$2"
PER_GPU="${3:-2}"
GPUS="${4:-0 1 2 3}"
EP_SEED="${5:-}"
PY="${PY:-python}"
read -r -a GPU_ARR <<< "$GPUS"
NG=${#GPU_ARR[@]}
CFG="$RUN_DIR/dump_cfg.yaml"
[ -f "$CFG" ] || { echo "no config: $CFG" >&2; exit 1; }

cell_name () {
  local c=$1 tag=""
  [ -n "${CELL_TAG:-}" ] && tag="${CELL_TAG}_"
  if [ -n "$EP_SEED" ]; then echo "epseed${EP_SEED}_${tag}$c"; else echo "${tag}$c"; fi
}

JOBS=$(mktemp)
i=0
for t in ${TASKS:-$(seq 0 39)}; do
  for c in $CKPTS; do
    [ -d "$RUN_DIR/checkpoints/$c" ] || { echo "no checkpoint dir: $RUN_DIR/checkpoints/$c" >&2; exit 1; }
    cell=$(cell_name "$c")
    out="$RUN_DIR/closed_loop/$cell/task$t.json"
    if [ -f "$out" ] && "$PY" -c "import json,sys; sys.exit(0 if len(json.load(open('$out'))['episode_results']['gg'])==50 else 1)" 2>/dev/null; then
      continue
    fi
    echo "${GPU_ARR[$((i % NG))]} $c $t" >> "$JOBS"
    i=$((i + 1))
  done
done
echo "$(wc -l < "$JOBS") jobs over GPUs [$GPUS], $PER_GPU per GPU, run $RUN_DIR, ckpts [$CKPTS]"

run_one () {
  local gpu=$1 c=$2 t=$3
  local cell extra=()
  cell=$(cell_name "$c")
  [ -n "$EP_SEED" ] && extra=(--episode_seed_base "$EP_SEED")
  read -r -a more <<< "${EXTRA_ARGS:-}"
  extra+=("${more[@]}")
  local vid=(--save_video --video_arms gg_p)
  [ -n "${VIDEO_ARGS+x}" ] && read -r -a vid <<< "$VIDEO_ARGS"
  local logdir="$RUN_DIR/closed_loop/$cell/logs"
  mkdir -p "$logdir"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" scripts/close_loop_lewm_libero.py \
    --ckpt "$RUN_DIR/checkpoints/$c" --cfg "$CFG" --out_dir "$RUN_DIR/closed_loop/$cell" \
    --task_idx "$t" "${vid[@]}" --gpu_id 0 "${extra[@]}" \
    > "$logdir/task$t.log" 2>&1
  echo "$(date +%H:%M:%S) gpu $gpu $c task $t rc=$?"
}
export -f run_one cell_name
export RUN_DIR CFG PY EP_SEED CELL_TAG EXTRA_ARGS VIDEO_ARGS
xargs -r -P $((PER_GPU * NG)) -L 1 bash -c 'run_one "$0" "$1" "$2"' < "$JOBS"
rm -f "$JOBS"
echo "queue finished $(date)"
