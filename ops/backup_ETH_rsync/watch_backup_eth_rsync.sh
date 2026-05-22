#!/usr/bin/env bash
set -u

SRC="/mnt/backup_ETH/"
DEST="/mnt/soclim0/backup_ETH/"
SRC_MATCH="/mnt/backup_ETH"
DEST_MATCH="/mnt/soclim0/backup_ETH"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
WATCHDOG_LOG="${LOG_DIR}/watchdog.log"
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-3600}"
LOCK_FILE="${LOG_DIR}/watch_backup_eth_rsync.lock"

mkdir -p "${LOG_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${WATCHDOG_LOG}"
}

find_matching_rsync_pids() {
  ps -eo pid=,args= | awk -v src="${SRC_MATCH}" -v dest="${DEST_MATCH}" '
    $0 ~ /[r]sync/ && index($0, src) && index($0, dest) { print $1 }
  '
}

check_mounts() {
  if ! mountpoint -q /mnt/backup_ETH; then
    log "Source mountpoint /mnt/backup_ETH is not mounted; skipping this check."
    return 1
  fi

  if ! mountpoint -q /mnt/soclim0; then
    log "Destination mountpoint /mnt/soclim0 is not mounted; skipping this check."
    return 1
  fi

  if [[ ! -d "${SRC}" ]]; then
    log "Source directory ${SRC} is missing; skipping this check."
    return 1
  fi

  if [[ ! -d "${DEST}" ]]; then
    log "Destination directory ${DEST} is missing; skipping this check."
    return 1
  fi
}

start_rsync_and_wait() {
  local run_log
  local rc

  run_log="${LOG_DIR}/copy_log_retry_$(date +%F_%H%M%S).txt"
  log "No matching rsync process found; starting resume run. Log: ${run_log}"

  rsync -avh --info=progress2 --partial --append-verify "${SRC}" "${DEST}" > "${run_log}" 2>&1
  rc=$?

  log "rsync exited with code ${rc}. Run log: ${run_log}"
  return "${rc}"
}

if ! command -v rsync >/dev/null 2>&1; then
  log "rsync is not available on PATH; exiting."
  exit 127
fi

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  log "Another watchdog is already running; exiting."
  exit 0
fi

log "Watchdog started. Checking every ${CHECK_INTERVAL_SECONDS}s."

while true; do
  if ! check_mounts; then
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  mapfile -t pids < <(find_matching_rsync_pids)
  if (( ${#pids[@]} > 0 )); then
    log "rsync is running with PID(s): ${pids[*]}"
    sleep "${CHECK_INTERVAL_SECONDS}"
    continue
  fi

  if start_rsync_and_wait; then
    log "Incremental rsync completed successfully. Watchdog exiting."
    exit 0
  fi

  log "rsync did not finish successfully; retrying after ${CHECK_INTERVAL_SECONDS}s."
  sleep "${CHECK_INTERVAL_SECONDS}"
done
