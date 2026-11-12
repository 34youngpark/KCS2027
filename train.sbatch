#!/bin/bash
#SBATCH --job-name=heavyhex-cnn
#SBATCH --partition=all
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Heavy-hex (3,3) CNN 학습 job (학습만 — 데이터셋은 미리 생성돼 있어야 함).
# repo 루트에서 제출할 것 (로그 경로가 제출 위치 기준이라서):
#   sbatch train.sbatch                       # config.json의 sweep 섹션 있으면 자동 스윕,
#                                             # 없으면 기본 단일 설정
#   sbatch train.sbatch --all --mwpm          # 전체 그리드 + 기준선
#   sbatch train.sbatch --config none         # 스윕 끄고 기본 단일 설정
# 추가 인자는 train.py로 그대로 전달돼 (-n/-p 등 선택 인자를 주면
# 자동 스윕 대신 그쪽이 돈다). 기본 하이퍼파라미터는 config.json의 train 섹션이
# 있으면 자동 적용돼.
# 데이터셋 생성부터 학습까지 한 번에 돌리려면 pipeline.sbatch 참고.

set -euo pipefail

CONDA_ENV="${CONDA_ENV:-HHC}"    # <-- EDIT: 본인 conda/mamba 환경 이름
                                 #     (또는 CONDA_ENV=이름 sbatch train.sbatch 로 덮어쓰기)
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

cd "${SLURM_SUBMIT_DIR:-$PWD}"

echo "host=$(hostname) gpu=${CUDA_VISIBLE_DEVICES:-unset} env=$CONDA_ENV args=$*"
python train.py "$@"
