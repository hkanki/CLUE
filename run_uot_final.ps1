# run_uot_compare.ps1

$PYTHON = "python"

Write-Host "Using Python:"
& $PYTHON -c "import sys; print(sys.executable)"

# OmegaConfがインストールされているか確認
& $PYTHON -c "from omegaconf import OmegaConf; print('OmegaConf OK')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "OmegaConfが現在のPython環境に入っていません。"
    Write-Host "conda環境を有効化した上で、以下を実行してください。"
    Write-Host "python -m pip install omegaconf"
    exit 1
}

# 基準となる設定ファイル
$BASE_CFG = ".\config\domainnet\art_clipart_uot.yml"

# 一時設定ファイルとログの保存先
$TEMP_DIR = ".\config\domainnet\tmp_uot_compare"
$LOG_DIR  = ".\logs\uot_compare"

New-Item -ItemType Directory -Force -Path $TEMP_DIR | Out-Null
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null

# 比較するハイパーパラメータの組み合わせ
$PARAMETER_SETS = @(
    [PSCustomObject]@{
        ETA1    = 0.0001
        ETA2    = 0.001
        EPSILON = 0.1
        TAU     = 1.0
    },
    [PSCustomObject]@{
        ETA1    = 0.0001
        ETA2    = 0.001
        EPSILON = 0.07
        TAU     = 0.3
    },
    [PSCustomObject]@{
        ETA1    = 0.0001
        ETA2    = 0.003
        EPSILON = 0.05
        TAU     = 0.5
    },
    [PSCustomObject]@{
        ETA1    = 0.0003
        ETA2    = 0.001
        EPSILON = 0.03
        TAU     = 0.2
    },
    [PSCustomObject]@{
        ETA1    = 0.0003
        ETA2    = 0.001
        EPSILON = 0.05
        TAU     = 0.3
    },
    [PSCustomObject]@{
        ETA1    = 0.0003
        ETA2    = 0.001
        EPSILON = 0.07
        TAU     = 0.3
    }
)

$TOTAL_EXPERIMENTS = $PARAMETER_SETS.Count
$CURRENT_EXPERIMENT = 0

foreach ($PARAMS in $PARAMETER_SETS) {

    $CURRENT_EXPERIMENT++

    $ETA1    = $PARAMS.ETA1
    $ETA2    = $PARAMS.ETA2
    $EPSILON = $PARAMS.EPSILON
    $TAU     = $PARAMS.TAU

    $EXP_ID = "art_clipart_uot_eta1_${ETA1}_eta2_${ETA2}_eps_${EPSILON}_tau_${TAU}"

    # ファイル名で小数点を使用しないように変換
    $SAFE_EXP_ID = $EXP_ID -replace "\.", "p"

    $TEMP_CFG = Join-Path $TEMP_DIR "${SAFE_EXP_ID}.yml"
    $LOG_FILE = Join-Path $LOG_DIR "${SAFE_EXP_ID}.log"

    # 同名のログが存在する場合は実験済みとしてスキップ
    if (Test-Path $LOG_FILE) {
        Write-Host "[SKIP] already exists: $LOG_FILE"
        continue
    }

    Write-Host ""
    Write-Host "======================================="
    Write-Host "Experiment: $CURRENT_EXPERIMENT / $TOTAL_EXPERIMENTS"
    Write-Host "Running: $SAFE_EXP_ID"
    Write-Host "uot_eta1    = $ETA1"
    Write-Host "uot_eta2    = $ETA2"
    Write-Host "uot_epsilon = $EPSILON"
    Write-Host "uot_tau     = $TAU"
    Write-Host "======================================="

    # 基準YAMLを読み込み、ハイパーパラメータを書き換える
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

# 各ハイパーパラメータを3回ずつ実行
cfg.runs = 3

OmegaConf.save(config=cfg, f=temp_cfg)

print(f"Saved temporary config: {temp_cfg}")
print(f"eta1    : {cfg.uot_eta1}")
print(f"eta2    : {cfg.uot_eta2}")
print(f"epsilon : {cfg.uot_epsilon}")
print(f"tau     : {cfg.uot_tau}")
print(f"runs    : {cfg.runs}")
"@

    $pythonCode | & $PYTHON -

    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] failed to create yaml: $SAFE_EXP_ID"
        continue
    }

    # 前回の実験結果が次の実験に影響しないように削除
    Remove-Item `
        ".\checkpoints\adapt\jumbot_*_ResNet34_net_art_clipart.pth" `
        -Force `
        -ErrorAction SilentlyContinue

    Remove-Item `
        ".\saved_pi_final.pt" `
        -Force `
        -ErrorAction SilentlyContinue

    # 学習を実行し、標準出力とエラー出力をログに保存
    & $PYTHON train.py `
        --load_from_cfg True `
        --cfg_file "$TEMP_CFG" 2>&1 |
        Tee-Object -FilePath $LOG_FILE

    $TRAIN_EXIT_CODE = $LASTEXITCODE

    if ($TRAIN_EXIT_CODE -ne 0) {
        Write-Host "[ERROR] training failed: $SAFE_EXP_ID"
        Write-Host "Exit code: $TRAIN_EXIT_CODE"
    }
    else {
        Write-Host "[SUCCESS] Finished: $SAFE_EXP_ID"
        Write-Host "Log: $LOG_FILE"
    }
}

Write-Host ""
Write-Host "======================================="
Write-Host "All specified experiments have finished."
Write-Host "Logs are saved in: $LOG_DIR"
Write-Host "======================================="