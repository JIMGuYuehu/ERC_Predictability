#!/usr/bin/env bash
set -euo pipefail

# Extract the Hindcast variables needed by the Longrun-style diagnostics.
#
# Required output layout:
#   /mnt/soclim0/public_data/weiji/Hindcast/<case>/<VAR>/<member>.cam.h3.<VAR>.nc
#
# The script is restart-safe: existing non-empty outputs are skipped. OMEGA is
# optional because raw NOCOUPL Hindcast h3 files do not contain it. Set
# EXTRACT_OMEGA=1 to archive any available coupled OMEGA segments.

INPUT_BASE="${INPUT_BASE:-/mnt/backup_ETH/lens}"
OUTPUT_BASE="${OUTPUT_BASE:-/mnt/soclim0/public_data/weiji/Hindcast}"
TMP_BASE="${TMP_BASE:-${OUTPUT_BASE}/_tmp_extract_longrun_vars}"
MAX_JOBS="${MAX_JOBS:-16}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
EXTRACT_OMEGA="${EXTRACT_OMEGA:-0}"

CORE_VARS=(U V T PS Z3 O3)
if [[ "${EXTRACT_OMEGA}" == "1" ]]; then
    CORE_VARS+=(OMEGA)
fi
COORD_VARS="P0,hyai,hyam,hybi,hybm,date,time,datesec,time_bnds,lat,lon,lev,ilev,gw"
LOG_DIR="${OUTPUT_BASE}/_logs"
MISSING_LOG="${LOG_DIR}/hindcast_missing_vars_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${TMP_BASE}" "${LOG_DIR}"

process_member() {
    local prefix_path="$1"
    local case_name="$2"
    local member_name
    member_name="$(basename "${prefix_path}")"

    shopt -s nullglob
    local files=( "${prefix_path}".*.nc "${prefix_path}".*.nc4 )
    shopt -u nullglob

    if [[ ${#files[@]} -eq 0 ]]; then
        return
    fi

    local var_name
    for var_name in "${CORE_VARS[@]}"; do
        local dest_dir="${OUTPUT_BASE}/${case_name}/${var_name}"
        local final_out="${dest_dir}/${member_name}.${var_name}.nc"
        local tmp_out="${TMP_BASE}/${case_name}.${member_name}.${var_name}.${$}.tmp.nc"

        if [[ -s "${final_out}" && "${OVERWRITE}" != "1" ]]; then
            continue
        fi

        local var_files=()
        local f
        for f in "${files[@]}"; do
            if ncks -m -v "${var_name}" "${f}" >/dev/null 2>&1; then
                var_files+=( "${f}" )
            fi
        done

        if [[ ${#var_files[@]} -eq 0 ]]; then
            printf '[MISSING] case=%s member=%s var=%s\n' "${case_name}" "${member_name}" "${var_name}" >> "${MISSING_LOG}"
            continue
        fi
        if [[ ${#var_files[@]} -lt ${#files[@]} ]]; then
            printf '[PARTIAL] case=%s member=%s var=%s segments_with_var=%s total_segments=%s\n' "${case_name}" "${member_name}" "${var_name}" "${#var_files[@]}" "${#files[@]}" >> "${MISSING_LOG}"
        fi

        mkdir -p "${dest_dir}"
        if [[ "${DRY_RUN}" == "1" ]]; then
            printf '[DRY] ncrcat %s -> %s\n' "${var_name}" "${final_out}"
            continue
        fi

        rm -f "${tmp_out}"
        if ncrcat -O -v "${var_name},${COORD_VARS}" "${var_files[@]}" "${tmp_out}" >/dev/null 2>&1; then
            ncks -4 -L 1 -O "${tmp_out}" "${final_out}"
            rm -f "${tmp_out}"
            printf '[OK] case=%s member=%s var=%s\n' "${case_name}" "${member_name}" "${var_name}"
        else
            rm -f "${tmp_out}"
            printf '[ERROR] case=%s member=%s var=%s\n' "${case_name}" "${member_name}" "${var_name}" >> "${MISSING_LOG}"
        fi
    done
}

export INPUT_BASE OUTPUT_BASE TMP_BASE DRY_RUN OVERWRITE COORD_VARS MISSING_LOG
export -f process_member

echo "INPUT_BASE=${INPUT_BASE}"
echo "OUTPUT_BASE=${OUTPUT_BASE}"
echo "MAX_JOBS=${MAX_JOBS}"
echo "DRY_RUN=${DRY_RUN}"
echo "OVERWRITE=${OVERWRITE}"
echo "EXTRACT_OMEGA=${EXTRACT_OMEGA}"
echo "Missing-variable log: ${MISSING_LOG}"

shopt -s nullglob
case_dirs=( "${INPUT_BASE}"/[0-9][0-9][0-9][0-9]-[0-9][0-9]* )
shopt -u nullglob

if [[ ${#case_dirs[@]} -eq 0 ]]; then
    echo "No Hindcast case directories found under ${INPUT_BASE}" >&2
    exit 1
fi

for case_dir in "${case_dirs[@]}"; do
    [[ -d "${case_dir}" ]] || continue
    case_name="$(basename "${case_dir}")"
    echo "======================================================="
    echo "Processing case: ${case_name}"

    mapfile -t prefixes < <(
        find "${case_dir}" -maxdepth 1 -type f -name "*.cam.h3.*.nc*" \
            | sed -E 's/\.[0-9]{4}-[0-9]{2}-[0-9]{2}-[0-9]{5}\.nc4?$//' \
            | sort -u
    )

    if [[ ${#prefixes[@]} -eq 0 ]]; then
        echo "[SKIP] no h3 files: ${case_name}"
        continue
    fi

    for prefix in "${prefixes[@]}"; do
        process_member "${prefix}" "${case_name}" &
        if [[ "$(jobs -r -p | wc -l)" -ge "${MAX_JOBS}" ]]; then
            wait -n
        fi
    done
    wait
done

rmdir "${TMP_BASE}" 2>/dev/null || true
echo "Done. Review missing-variable log: ${MISSING_LOG}"
