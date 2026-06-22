# run_uot_grid.ps1

$PYTHON = "python"

Write-Host "Using Python:"
& $PYTHON -c "import sys; print(sys.executable)"

& $PYTHON -c "from omegaconf import OmegaConf; print('OmegaConf OK')"
if ($LASTEXITCODE -ne 0) {
    Write-Host "OmegaConfが現在のPython環境に入っていません。"
    Write-Host "conda環境を有効化した上で、以下を実行してください。"
    Write-Host "python -m pip install omegaconf"
    exit 1
}

$BASE_CFG = ".\config\domainnet\art_clipart_uot.yml"
$TEMP_DIR = ".\config\domainnet\tmp_uot_grid"
$LOG_DIR  = ".\logs\uot_grid"

New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

$ETA1_LIST    = @(0.00005, 0.0001, 0.0003)
$ETA2_LIST    = @(0.001, 0.003, 0.01)
$EPSILON_LIST = @(0.03, 0.05, 0.07)
$TAU_LIST     = @(0.2, 0.3, 0.5)

foreach ($ETA1 in $ETA1_LIST) {
    foreach ($ETA2 in $ETA2_LIST) {
        foreach ($EPSILON in $EPSILON_LIST) {
            foreach ($TAU in $TAU_LIST) {

                $EXP_ID = "art_clipart_uot_eta1_${ETA1}_eta2_${ETA2}_eps_${EPSILON}_tau_${TAU}"
                $SAFE_EXP_ID = $EXP_ID -replace "\.", "p"

                $TEMP_CFG = Join-Path $TEMP_DIR "${SAFE_EXP_ID}.yml"
                $LOG_FILE = Join-Path $LOG_DIR "${SAFE_EXP_ID}.log"

                if (Test-Path $LOG_FILE) {
                    Write-Host "[SKIP] already exists: $LOG_FILE"
                    continue
                }

                Write-Host "======================================="
                Write-Host "Running: $SAFE_EXP_ID"
                Write-Host "uot_eta1=$ETA1"
                Write-Host "uot_eta2=$ETA2"
                Write-Host "uot_epsilon=$EPSILON"
                Write-Host "uot_tau=$TAU"
                Write-Host "======================================="

                $pythonCode = @"
from omegaconf import OmegaConf

base_cfg = r"$BASE_CFG"
temp_cfg = r"$TEMP_CFG"

cfg = OmegaConf.load(base_cfg)

cfg.id = "$SAFE_EXP_ID"
cfg.uot_eta1 = $ETA1
cfg.uot_eta2 = $ETA2
cfg.uot_epsilon = $EPSILON
cfg.uot_tau = $TAU

# 比較実験として3回実行
cfg.runs = 1

OmegaConf.save(config=cfg, f=temp_cfg)
print(f"Saved temporary config: {temp_cfg}")
"@

                $pythonCode | & $PYTHON -

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[ERROR] failed to create yaml: $SAFE_EXP_ID"
                    continue
                }

                Remove-Item ".\checkpoints\adapt\jumbot_*_ResNet34_net_art_clipart.pth" -Force -ErrorAction SilentlyContinue
                Remove-Item ".\saved_pi_final.pt" -Force -ErrorAction SilentlyContinue

                & $PYTHON train.py --load_from_cfg True --cfg_file "$TEMP_CFG" 2>&1 |
                    Tee-Object -FilePath $LOG_FILE

                if ($LASTEXITCODE -ne 0) {
                    Write-Host "[ERROR] training failed: $SAFE_EXP_ID"
                } else {
                    Write-Host "Finished: $SAFE_EXP_ID"
                }

                Write-Host ""
            }
        }
    }
}