#!/bin/bash
#SBATCH --job-name=qec-pipeline
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# 통합 파이프라인: 게이트 -> 수거 -> QPU 제출 -> [프로파일] -> 데이터셋 -> 학습.
#
# QPU는 "루프당 코드별 1잡"으로 제출한다: 같은 ISA 회로를 qpu_pubs개의
# PUB(Primitive Unified Bloc — (회로, 파라미터, 샷)의 실행 단위)으로 담아
# 큐 엔트리 1개로 shots x qpu_pubs 를 확보한다 (큐 엔트리 = 루프당 백엔드
# 수, 현재 2개). 제출 직후 완료를 기다리지 않고 프로파일/데이터셋/학습을
# 진행하며(큐 대기와 겹침), 결과 "수거"는 다음 루프 시작부에서 한다:
# pending_jobs.json의 잡 중 DONE만 분석하고 나머지는 이월. 실패 잡은
# hardware/collect_failed.log에 사유를 남기고 제거된다 (루프 체인은 계속).
# 프로파일은 최근 profile_runs개 루프의 "실행 시점" 캘리브레이션 스냅샷
# (properties_run.json, 없으면 제출 시점 스냅샷) 평균이다.
#
# 고정 설정은 전부 config.json의 "pipeline" 섹션에서 읽는다:
#   conda_env / codes / backends / qpu_pubs(잡 내 PUB 수, 기본 5) /
#   profile_runs(프로파일 평균 창, 기본 5) / shots(PUB당 샷) / train_args
#
# sbatch 변수는 두 개뿐:
#   LOOPS=N   파이프라인 전체를 N회 반복 (기본 1; 루프 끝에 자기 재제출)
#   SMOKE=1   리허설: 데이터셋/학습 스모크 + QPU는 dry-run(1 PUB, 제출 없음)
#
# 사용 (repo 루트에서):
#   LOOPS=5 sbatch pipeline.sbatch      # 실전 5루프
#   sbatch pipeline.sbatch              # 1회
#   SMOKE=1 sbatch pipeline.sbatch      # 빠른 end-to-end 리허설
#
# 학습 인자에 --solution이 있으면 수거(analyze)에도 자동 전파된다.
# LOOPS 종료 후에도 pending이 남을 수 있다(마지막 루프의 잡은 다음 실행의
# 수거 단계 또는 `python hardware/run_hw.py collect --solution`으로 수거).

set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

pcfg() {  # config.json pipeline 섹션 읽기 (활성화 전이므로 시스템 python3)
    python3 - "$1" "$2" <<'PY'
import json, sys
v = json.load(open("config.json")).get("pipeline", {}).get(sys.argv[1])
if v is None:
    v = sys.argv[2]
print(" ".join(v) if isinstance(v, list) else v)
PY
}

CONDA_ENV="$(pcfg conda_env HHC)"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

set -u
export PYTHONUNBUFFERED=1

CODES="$(pcfg codes 'heavyhex surface')"
QPU_PUBS="$(pcfg qpu_pubs 5)"
PROFILE_RUNS="$(pcfg profile_runs 5)"
SHOTS="$(pcfg shots 50000)"
TRAIN_ARGS="$(pcfg train_args '')"
LOOPS="${LOOPS:-1}"
LOOP="${LOOP:-1}"

SMOKE_ARGS="" HW_EXTRA=""
if [[ "${SMOKE:-0}" == "1" ]]; then
    SMOKE_ARGS="--smoke"
    HW_EXTRA="--dry-run"
    QPU_PUBS=1
fi

backend_of() {  # 코드 -> QPU 백엔드 (config.json pipeline.backends)
    python3 - "$1" <<'PY'
import json, sys
b = json.load(open("config.json")).get("pipeline", {}).get("backends", {})
print(b.get(sys.argv[1],
            {"heavyhex": "ibm_boston", "surface": "ibm_miami"}[sys.argv[1]]))
PY
}

gate_of() {     # 코드 -> 규약 게이트 스크립트
    case "$1" in
        heavyhex) echo "verification/verify_equivalence.py" ;;
        surface)  echo "verification/verify_rotatedSurface3.py" ;;
    esac
}

# 이전 루프 잡이 락을 놓기 전에 다음 루프가 스케줄될 수 있으므로,
# 즉시 실패(-n) 대신 최대 2시간 대기 — 루프 체인이 레이스로 죽지 않게.
exec 9>/tmp/heavyhex_pipeline.lock
flock -w 7200 9 || { echo "2시간 내 락 획득 실패 — 중단"; exit 1; }

HW_SOLUTION=""
case " $TRAIN_ARGS $* " in *" --solution "*) HW_SOLUTION="--solution" ;; esac

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV"
echo "loop=$LOOP/$LOOPS qpu_pubs=$QPU_PUBS shots=$SHOTS profile_runs=$PROFILE_RUNS codes=$CODES smoke=${SMOKE:-0}"
echo "train_args=$TRAIN_ARGS $SMOKE_ARGS $*"

echo "=== [1/6] 규약 게이트 ==="
for code in $CODES; do
    echo "--- gate: $code ---"
    python "$(gate_of "$code")"
done

echo "=== [2/6] QPU 수거 (완료된 pending 잡 분석, 나머지 이월) ==="
if [[ -f keys.json ]]; then
    python hardware/run_hw.py collect $HW_SOLUTION || \
        echo "WARNING: collect 실패 — 이월하고 계속"
else
    echo "keys.json 없음 — 수거 건너뜀"
fi

echo "=== [3/6] QPU 제출 (코드별 1잡 x ${QPU_PUBS} PUB, 완료는 다음 루프 수거) ==="
if [[ -f keys.json ]]; then
    for code in $CODES; do
        backend="$(backend_of "$code")"
        echo "--- QPU submit: $code @ $backend ---"
        python hardware/run_hw.py submit --backend "$backend" \
            --code "$code" --shots "$SHOTS" --pubs "$QPU_PUBS" $HW_EXTRA
    done
else
    echo "keys.json 없음 — QPU 제출 건너뜀"
fi

echo "=== [4/6] QPU 캘리브레이션 프로파일 (최근 ${PROFILE_RUNS}개 루프 평균) ==="
declare -A PROFILE
for code in $CODES; do
    backend="$(backend_of "$code")"
    if out=$(python dataset_generation/make_qpu_avg_profile.py \
                 --code "$code" --backend "$backend" \
                 --n-runs "$PROFILE_RUNS" --suffix "loop$LOOP" 2>&1); then
        echo "$out"
        PROFILE[$code]=$(sed -n 's/^profile \(qpu\/[^:]*\):.*/\1/p' <<< "$out" | tail -1)
        echo "--- $code: 프로파일 ${PROFILE[$code]} 사용 ---"
    else
        echo "$out"
        echo "--- $code: QPU 이력 없음/생성 불가 — 이번 루프는 realistic만 ---"
        PROFILE[$code]=""
    fi
done

echo "=== [5/6] 데이터셋 생성 ==="
for code in $CODES; do
    echo "--- dataset: $code (realistic) ---"
    python dataset_generation/make_dataset.py --code "$code" $SMOKE_ARGS
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- dataset: $code (${PROFILE[$code]}) ---"
        python dataset_generation/make_dataset.py --code "$code" \
            -n "${PROFILE[$code]}" $SMOKE_ARGS
    fi
done

echo "=== [6/6] 학습 (epoch-resume 이어서) ==="
for code in $CODES; do
    echo "--- train: $code (realistic 그리드) ---"
    python train.py --code "$code" $TRAIN_ARGS $SMOKE_ARGS "$@"
    if [[ -n "${PROFILE[$code]}" ]]; then
        echo "--- train: $code (${PROFILE[$code]}) ---"
        python train.py --code "$code" $TRAIN_ARGS $SMOKE_ARGS "$@" \
            --config config.json -n "${PROFILE[$code]}"
    fi
done

PENDING_LEFT=$(python3 -c "
import json
try: print(len(json.load(open('hardware/pending_jobs.json'))))
except Exception: print(0)")
if (( LOOP < LOOPS )); then
    echo "=== 루프 $LOOP 완료 (pending ${PENDING_LEFT}건) — 다음 루프($((LOOP + 1))/$LOOPS) 재제출 ==="
    # --continue-stopped는 "이번 루프만" 유효 (레짐 변경 직후 1회 재수렴용)
    # — 다음 루프부터는 제거해서 skip 동작으로 복귀
    NEXT_ARGS=()
    for a in "$@"; do
        [[ "$a" == "--continue-stopped" ]] || NEXT_ARGS+=("$a")
    done
    if command -v sbatch >/dev/null 2>&1; then
        sbatch --export=ALL,LOOP=$((LOOP + 1)) pipeline.sbatch \
            ${NEXT_ARGS[@]+"${NEXT_ARGS[@]}"}
    else
        echo "WARNING: sbatch를 찾을 수 없어 재제출 불가 — LOOP=$((LOOP + 1))로 직접 제출할 것"
    fi
else
    echo "=== 전체 $LOOPS 루프 완료 — 미수거 pending ${PENDING_LEFT}건 ==="
    if (( PENDING_LEFT > 0 )); then
        echo "    (다음 sbatch 재제출 시 수거되거나, 수동으로:"
        echo "     python hardware/run_hw.py collect ${HW_SOLUTION})"
    fi
fi
