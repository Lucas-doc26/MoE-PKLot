#!/usr/bin/env bash
set -euo pipefail

# Greedy hyperparameter search for MoE.py
# - Trains over: PUC, UFPR04, UFPR05, camera1..camera9
# - Tests on the full list (PUC,UFPR04,UFPR05,camera1..camera9,PKLot,CNR)
# - Grid candidates:
#     batch_size: 64,128,256,512,1024
#     num_workers: 3,5,10,20
#     num_epochs: 3,10,20
#     top_k: 2,3
#
# Usage: ./scripts/greedy_search.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# find repository root by locating MoE.py (support running script from root or scripts/)
if [ -f "$SCRIPT_DIR/MoE.py" ]; then
  ROOT_DIR="$SCRIPT_DIR"
elif [ -f "$SCRIPT_DIR/../MoE.py" ]; then
  ROOT_DIR="$SCRIPT_DIR/.."
else
  # fallback: use script dir
  ROOT_DIR="$SCRIPT_DIR"
fi
cd "$ROOT_DIR"

LOG_DIR="logs/greedy"
mkdir -p "$LOG_DIR"

TRAINS=(PUC UFPR04 UFPR05 camera1 camera2 camera3 camera4 camera5 camera6 camera7 camera8 camera9)
TESTS='PUC,UFPR04,UFPR05,camera1,camera2,camera3,camera4,camera5,camera6,camera7,camera8,camera9,PKLot,CNR'

BATCH_SIZES=(64 128 256 512 1024)
WORKERS=(3 5 10 20)
EPOCHS=(3 10 20)
TOP_K=(2 3)

# initial config
cur_batch=64
cur_workers=3
cur_epochs=3
cur_topk=2

RESULTS_FILE="greedy_results.txt"
echo "Greedy search started at $(date)" > "$RESULTS_FILE"
echo "Tests list: $TESTS" >> "$RESULTS_FILE"

run_and_get_val() {
  local train=$1
  local batch=$2
  local workers=$3
  local epochs=$4
  local topk=$5

  local prev_lines=0
  if [ -f metrics.csv ]; then
    prev_lines=$(wc -l < metrics.csv)
  fi

  local logfile="$LOG_DIR/${train}_B${batch}_W${workers}_E${epochs}_K${topk}.log"
  # Log start to logfile only (do not print to stdout)
  echo "Running: train=$train batch=$batch workers=$workers epochs=$epochs topk=$topk" >> "$logfile"

  # Run training (valid_data set to same as train) and append all output to logfile
  python MoE.py --train_data "$train" \
    --valid_data "$train" \
    --batch_size "$batch" --num_workers "$workers" \
    --num_epochs "$epochs" --top_k "$topk" \
    --test_datasets "$TESTS" &>> "$logfile" || true
  local exit_code=$?
  if [ $exit_code -ne 0 ]; then
    # training failed; record failure in logfile and return 0 as validation metric
    echo "ERROR: MoE.py exited with code $exit_code for train=$train (see $logfile)" >> "$logfile"
    echo "0"
    return
  fi
  local new_lines=0
  local appended=0
  if [ -f metrics.csv ]; then
    new_lines=$(wc -l < metrics.csv)
    appended=$((new_lines - prev_lines))
  fi
  if [ "$appended" -le 0 ]; then
    echo "0"
    return
  fi

  # final_val_acc is the same for all appended rows of the run; extract the last occurrence
  local val
  # extract final_val_acc from the appended rows (handle header cases)
  val=$(tail -n "$appended" metrics.csv | awk -F',' 'NR==1{for(i=1;i<=NF;i++){if($i=="final_val_acc")c=i}} NR>1{v=$c} END{print (v==""?"0":v)}')
  # sanitize numeric value
  if ! printf '%s' "$val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
    val=0
  fi
  printf "%s" "$val"
}

search_param() {
  local name=$1
  shift
  local candidates=($@)

  echo "\nSearching $name..." | tee -a "$RESULTS_FILE"

  local best_val=-1
  local best_choice=${!name}

  for cand in "${candidates[@]}"; do
    echo " Testing $name=$cand" | tee -a "$RESULTS_FILE"
    sum=0
    cnt=0
    for train in "${TRAINS[@]}"; do
      val=$(run_and_get_val "$train" "$cur_batch" "$cur_workers" "$cur_epochs" "$cur_topk")
      # If the parameter currently being tested is one of the current four, override it for this run
      case "$name" in
        cur_batch) val=$(run_and_get_val "$train" "$cand" "$cur_workers" "$cur_epochs" "$cur_topk") ;;
        cur_workers) val=$(run_and_get_val "$train" "$cur_batch" "$cand" "$cur_epochs" "$cur_topk") ;;
        cur_epochs) val=$(run_and_get_val "$train" "$cur_batch" "$cur_workers" "$cand" "$cur_topk") ;;
        cur_topk) val=$(run_and_get_val "$train" "$cur_batch" "$cur_workers" "$cur_epochs" "$cand") ;;
      esac
      # ensure val is numeric before summing
      if ! printf '%s' "$val" | grep -Eq '^[0-9]+(\.[0-9]+)?$'; then
        val=0
      fi
      sum=$(awk "BEGIN{print $sum + $val}")
      cnt=$((cnt+1))
    done
    avg=$(awk "BEGIN{print ($cnt==0?0:$sum/$cnt)}")
    echo "  -> avg val_acc for $name=$cand : $avg" | tee -a "$RESULTS_FILE"
    if awk "BEGIN{exit !($avg > $best_val)}"; then
      best_val=$avg
      best_choice=$cand
    fi
  done

  # update the named parameter
  eval "$name=$best_choice"
  echo "Best $name = $best_choice (avg val_acc=$best_val)" | tee -a "$RESULTS_FILE"
}

# Greedy order: batch -> workers -> epochs -> topk
search_param cur_batch "${BATCH_SIZES[@]}"
search_param cur_workers "${WORKERS[@]}"
search_param cur_epochs "${EPOCHS[@]}"
search_param cur_topk "${TOP_K[@]}"

echo "\nGreedy search finished at $(date)" | tee -a "$RESULTS_FILE"
echo "Final config: batch=$cur_batch workers=$cur_workers epochs=$cur_epochs topk=$cur_topk" | tee -a "$RESULTS_FILE"

echo "Detailed logs: $LOG_DIR" | tee -a "$RESULTS_FILE"

echo "Done. Review $RESULTS_FILE and metrics.csv for results." | tee -a "$RESULTS_FILE"
