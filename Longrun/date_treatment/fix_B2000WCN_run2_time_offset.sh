#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# Batch copy + internal-time-fix for extracted yearly CAM h3 files
#
# 功能：
#   1) 复制 extracted 年文件到 *_timefixed 目录
#   2) 对 run002 对应文件（默认文件名年份 >= 0105）修正内部：
#        - time      += 104 * 365
#        - date      += 1040000
#        - time_bnds  重建为 [time-1, time]
#
# 特点：
#   - 若某实验组已存在 OUT_ROOT/.TIMEFIX_DONE，则直接跳过整个实验组
#   - 若某文件已存在且非空，则跳过该文件（支持断点续跑）
#   - 不删除任何年份
#   - 不做 NaN 填充（partial year 后续单独处理）
#
# 依赖：
#   ncap2, ncks, ncrename, ncatted, xargs
# ==============================================================================

MAX_JOBS="${MAX_JOBS:-16}"
OVERWRITE="${OVERWRITE:-0}"

CORE_VARS=("U" "V" "T" "OMEGA" "PS" "Z3" "O3")

# ------------------------------------------------------------------------------
# 实验组配置
# ------------------------------------------------------------------------------
# 说明：
#   EXP_NAMES    只是日志里显示的名字
#   FILE_PREFIX  是文件名前缀，不含年份和变量，例如：
#                  B2000WCN.sample.cam.h3
#                  B2000WCN.NOCOUPL.sample.cam.h3
#   IN_ROOT      是你提取后的目录（含 U/V/T... 子目录）
#   OUT_ROOT     是输出 timefixed 目录
#   RUN2_START   文件名年份从几几年开始视作 run002，需要做内部时间平移
#   SHIFT_YEARS  平移多少年（这里按你现在的约定都是 104）
# ------------------------------------------------------------------------------

EXP_NAMES=(
  "B2000WCN001002"
  "B2000WCN_NOCOUPL001002"
)

FILE_PREFIXES=(
  "B2000WCN.sample.cam.h3"
  "B2000WCN.NOCOUPL.sample.cam.h3"
)

IN_ROOTS=(
  "/mnt/soclim0/public_data/weiji/B2000WCN001002"
  "/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002"
)

OUT_ROOTS=(
  "/mnt/soclim0/public_data/weiji/B2000WCN001002_timefixed"
  "/mnt/soclim0/public_data/weiji/B2000WCN_NOCOUPL001002_timefixed"
)

RUN2_START_YEARS=(
  "105"
  "105"
)

SHIFT_YEARS_LIST=(
  "104"
  "104"
)

# ------------------------------------------------------------------------------
# 工具函数
# ------------------------------------------------------------------------------

log() {
  echo "[$(date '+%F %T')] $*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[FATAL] command not found: $1" >&2
    exit 1
  }
}

experiment_done() {
  local out_root="$1"
  [[ -s "${out_root}/.TIMEFIX_DONE" ]]
}

prepare_out_dirs() {
  local out_root="$1"
  mkdir -p "${out_root}"
  for var in "${CORE_VARS[@]}"; do
    mkdir -p "${out_root}/${var}"
  done
}

parse_year_from_filename() {
  local base="$1"
  local var="$2"

  if [[ "${base}" =~ \.cam\.h3\.([0-9]{4})\.${var}\.nc$ ]]; then
    echo "${BASH_REMATCH[1]}"
  else
    echo ""
  fi
}

process_one_file() {
  local exp_name="$1"
  local file_prefix="$2"
  local in_root="$3"
  local out_root="$4"
  local run2_start_year="$5"
  local shift_years="$6"
  local var="$7"
  local in_file="$8"

  local shift_days=$((shift_years * 365))
  local shift_date=$((shift_years * 10000))

  local base
  base="$(basename "${in_file}")"

  local year_str
  year_str="$(parse_year_from_filename "${base}" "${var}")"

  if [[ -z "${year_str}" ]]; then
    echo "[ERROR] ${exp_name} ${var}: cannot parse year from ${base}" >&2
    return 1
  fi

  local year_int=$((10#${year_str}))
  local out_file="${out_root}/${var}/${base}"

  if [[ "${OVERWRITE}" != "1" && -s "${out_file}" ]]; then
    echo "[SKIP] ${exp_name} ${var} ${year_str}: output exists"
    return 0
  fi

  local tmp1="${out_file}.tmp1.$$"
  local tmp2="${out_file}.tmp2.$$"
  local has_date="0"
  local has_time_bnds="0"

  rm -f "${tmp1}" "${tmp2}"

  if (( year_int < run2_start_year )); then
    # run001-like: 直接复制，不改内部时间
    echo "[COPY ] ${exp_name} ${var} ${year_str}: ${base}"
    cp -p "${in_file}" "${tmp1}"
    mv "${tmp1}" "${out_file}"
    return 0
  fi

  # run002-like: 复制并修正内部时间
  if ncks -m "${in_file}" | grep -q " date("; then
    has_date="1"
  fi
  if ncks -m "${in_file}" | grep -q "time_bnds"; then
    has_time_bnds="1"
  fi

  local expr="time=time+${shift_days}.0;"

  if [[ "${has_date}" == "1" ]]; then
    expr="${expr} date=date+${shift_date};"
  fi

  if [[ "${has_time_bnds}" == "1" ]]; then
    # 先复制出一个新变量 tb_new，避免直接原地写 time_bnds 的 warning
    expr="${expr} tb_new=time_bnds; tb_new(:,0)=time-1.0; tb_new(:,1)=time;"
  fi

  echo "[SHIFT] ${exp_name} ${var} ${year_str}: ${base}"

  ncap2 -O -s "${expr}" "${in_file}" "${tmp1}"

  if [[ "${has_time_bnds}" == "1" ]]; then
    ncks -O -x -v time_bnds "${tmp1}" "${tmp2}"
    ncrename -O -v tb_new,time_bnds "${tmp2}"
    mv "${tmp2}" "${tmp1}"
  fi

  # 统一 time 属性
  if ncks -m "${tmp1}" | grep -q " time("; then
    if [[ "${has_time_bnds}" == "1" ]]; then
      ncatted -O \
        -a units,time,o,c,"days since 0001-01-01 00:00:00" \
        -a calendar,time,o,c,"noleap" \
        -a bounds,time,o,c,"time_bnds" \
        "${tmp1}"
    else
      ncatted -O \
        -a units,time,o,c,"days since 0001-01-01 00:00:00" \
        -a calendar,time,o,c,"noleap" \
        "${tmp1}"
    fi
  fi

  # 轻压缩
  ncks -4 -L 1 -O "${tmp1}" "${tmp2}"
  mv "${tmp2}" "${out_file}"
  rm -f "${tmp1}"

  return 0
}

export -f log
export -f parse_year_from_filename
export -f process_one_file
export OVERWRITE

# ------------------------------------------------------------------------------
# 前置检查
# ------------------------------------------------------------------------------

need_cmd ncap2
need_cmd ncks
need_cmd ncrename
need_cmd ncatted
need_cmd xargs

log "MAX_JOBS=${MAX_JOBS}"
log "OVERWRITE=${OVERWRITE}"
echo

# ------------------------------------------------------------------------------
# 主循环：逐实验组处理
# ------------------------------------------------------------------------------

NEXP=${#EXP_NAMES[@]}

for ((i=0; i<NEXP; i++)); do
  EXP_NAME="${EXP_NAMES[$i]}"
  FILE_PREFIX="${FILE_PREFIXES[$i]}"
  IN_ROOT="${IN_ROOTS[$i]}"
  OUT_ROOT="${OUT_ROOTS[$i]}"
  RUN2_START_YEAR="${RUN2_START_YEARS[$i]}"
  SHIFT_YEARS="${SHIFT_YEARS_LIST[$i]}"

  echo "=============================================================================="
  log "Experiment: ${EXP_NAME}"
  log "FILE_PREFIX      = ${FILE_PREFIX}"
  log "IN_ROOT          = ${IN_ROOT}"
  log "OUT_ROOT         = ${OUT_ROOT}"
  log "RUN2_START_YEAR  = ${RUN2_START_YEAR}"
  log "SHIFT_YEARS      = ${SHIFT_YEARS}"
  log "SHIFT_DAYS       = $((SHIFT_YEARS * 365))"
  echo "=============================================================================="

  if [[ ! -d "${IN_ROOT}" ]]; then
    log "Input root missing, skip: ${IN_ROOT}"
    echo
    continue
  fi

  if experiment_done "${OUT_ROOT}"; then
    log "Found marker ${OUT_ROOT}/.TIMEFIX_DONE ; skip whole experiment."
    echo
    continue
  fi

  prepare_out_dirs "${OUT_ROOT}"

  TASK_FILE="$(mktemp)"
  trap 'rm -f "${TASK_FILE}"' EXIT

  # 构建任务列表
  for VAR in "${CORE_VARS[@]}"; do
    if [[ ! -d "${IN_ROOT}/${VAR}" ]]; then
      log "Missing var dir, skip var: ${IN_ROOT}/${VAR}"
      continue
    fi

    shopt -s nullglob
    files=( "${IN_ROOT}/${VAR}"/*.${VAR}.nc )
    shopt -u nullglob

    for F in "${files[@]}"; do
      echo "${EXP_NAME}"$'\t'"${FILE_PREFIX}"$'\t'"${IN_ROOT}"$'\t'"${OUT_ROOT}"$'\t'"${RUN2_START_YEAR}"$'\t'"${SHIFT_YEARS}"$'\t'"${VAR}"$'\t'"${F}" >> "${TASK_FILE}"
    done
  done

  NTASK=$(wc -l < "${TASK_FILE}" | awk '{print $1}')
  log "Tasks prepared: ${NTASK}"

  if [[ "${NTASK}" == "0" ]]; then
    log "No tasks found, skip."
    rm -f "${TASK_FILE}"
    trap - EXIT
    echo
    continue
  fi

  # 并行执行
  cat "${TASK_FILE}" | xargs -P "${MAX_JOBS}" -d $'\n' -I {} bash -c '
    IFS=$'\''\t'\'' read -r exp_name file_prefix in_root out_root run2_start shift_years var in_file <<< "{}"
    process_one_file "$exp_name" "$file_prefix" "$in_root" "$out_root" "$run2_start" "$shift_years" "$var" "$in_file"
  '

  # 简单检查：每个变量目录至少有一个非空文件
  all_ok="1"
  for VAR in "${CORE_VARS[@]}"; do
    shopt -s nullglob
    out_files=( "${OUT_ROOT}/${VAR}"/*.${VAR}.nc )
    shopt -u nullglob

    if [[ ${#out_files[@]} -eq 0 ]]; then
      log "Check failed: no output files in ${OUT_ROOT}/${VAR}"
      all_ok="0"
      continue
    fi

    found_nonempty="0"
    for FF in "${out_files[@]}"; do
      if [[ -s "${FF}" ]]; then
        found_nonempty="1"
        break
      fi
    done

    if [[ "${found_nonempty}" != "1" ]]; then
      log "Check failed: all files empty in ${OUT_ROOT}/${VAR}"
      all_ok="0"
    fi
  done

  if [[ "${all_ok}" == "1" ]]; then
    {
      echo "experiment=${EXP_NAME}"
      echo "file_prefix=${FILE_PREFIX}"
      echo "in_root=${IN_ROOT}"
      echo "out_root=${OUT_ROOT}"
      echo "run2_start_year=${RUN2_START_YEAR}"
      echo "shift_years=${SHIFT_YEARS}"
      echo "completed_at=$(date '+%F %T')"
    } > "${OUT_ROOT}/.TIMEFIX_DONE"
    log "Marker written: ${OUT_ROOT}/.TIMEFIX_DONE"
  else
    log "Output check not fully passed; marker not written."
  fi

  rm -f "${TASK_FILE}"
  trap - EXIT

  # 快速抽检一个文件
  SAMPLE_FILE="${OUT_ROOT}/O3/${FILE_PREFIX}.0105.O3.nc"
  if [[ -s "${SAMPLE_FILE}" ]]; then
    log "Quick check sample: ${SAMPLE_FILE}"
    ncks -H -C -v date "${SAMPLE_FILE}" | head || true
    ncks -H -C -v time "${SAMPLE_FILE}" | head || true
    ncks -H -C -v time_bnds "${SAMPLE_FILE}" | head || true
  fi

  echo
done

log "All experiments finished."