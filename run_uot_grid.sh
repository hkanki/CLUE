#!/usr/bin/env bash

# ==============================
# UOT hyperparameter grid search
# ==============================

BASE_CFG="./config/domainnet/art_clipart_uot.yml"
TEMP_DIR="./config/domainnet/tmp_uot_grid"
LOG_DIR="./logs/uot_grid"

mkdir -p "$TEMP_DIR"
mkdir -p "$LOG_DIR"

# 探索したい値
ETA1_LIST=(0.0001 0.001 0.01)
ETA2_LIST=(0.001 0.01 0.1)
EPSILON_LIST=(0.02 0.05 0.1)
TAU_LIST=(0.2 0.5 1.0)

for ETA1 in "${ETA1_LIST[@]}"; do
  for ETA2 in "${ETA2_LIST[@]}"; do
    for EPSILON in "${EPSILON_LIST[@]}"; do
      for TAU in "${TAU_LIST[@]}"; do

        EXP_ID="art_clipart_uot_eta1_${ETA1}_eta2_${ETA2}_eps_${EPSILON}_tau_${TAU}"
        SAFE_EXP_ID=$(echo "$EXP_ID" | sed 's/\./p/g')

        TEMP_CFG="${TEMP_DIR}/${SAFE_EXP_ID}.yml"
        LOG_FILE="${LOG_DIR}/${SAFE_EXP_ID}.log"

        echo "======================================="
        echo "Running: ${SAFE_EXP_ID}"
        echo "uot_eta1=${ETA1}"
        echo "uot_eta2=${ETA2}"
        echo "uot_epsilon=${EPSILON}"
        echo "uot_tau=${TAU}"
        echo "======================================="

        # 一時yamlを作成
        python - <<EOF
from omegaconf import OmegaConf

base_cfg = "${BASE_CFG}"
temp_cfg = "${TEMP_CFG}"

cfg = OmegaConf.load(base_cfg)

cfg.id = "${SAFE_EXP_ID}"
cfg.uot_eta1 = ${ETA1}
cfg.uot_eta2 = ${ETA2}
cfg.uot_epsilon = ${EPSILON}
cfg.uot_tau = ${TAU}

OmegaConf.save(config=cfg, f=temp_cfg)
print(f"Saved temporary config: {temp_cfg}")
EOF

        # 既存checkpointがあるとハイパーパラメータを変えても再学習されない可能性があるため削除
        rm -f ./checkpoints/adapt/jumbot_*_ResNet34_net_art_clipart.pth

        # 古いUOT輸送計画が残っている場合も削除
        rm -f ./saved_pi_final.pt

        # 実行
        python train.py --load_from_cfg True --cfg_file "$TEMP_CFG" 2>&1 | tee "$LOG_FILE"

        echo "Finished: ${SAFE_EXP_ID}"
        echo ""

      done
    done
  done
done