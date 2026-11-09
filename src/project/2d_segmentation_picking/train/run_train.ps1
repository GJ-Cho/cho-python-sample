# 3단계 본 학습: RF-DETR-Seg Medium, [BOX] 5종 → 제품 7종([ONE]+[FEW]) 순서 (사용자 승인 2026-11-08)
#   최대 30 epoch, 검증 성능이 5 epoch 연속 개선 없으면 자동 정지. 센서 노이즈는 학습 split 에만 on-the-fly.
#   실행: powershell -ExecutionPolicy Bypass -File run_train.ps1
#   끊기면 다시 실행: last.ckpt 가 있으면 --resume 으로 이어서 학습한다. done.txt 가 있으면 건너뛴다.
#   진행 기록: runs\_logs\run_train.log (요약), runs\<job>\train.log (전체 출력)
$ErrorActionPreference = "Continue"
$py   = "C:\Zivid\3rdparty\venv_rfdetr\Scripts\python.exe"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$proj = Split-Path -Parent $here
$logd = Join-Path $proj "runs\_logs"
New-Item -ItemType Directory -Force $logd | Out-Null
$summary = Join-Path $logd "run_train.log"
$env:PYTHONIOENCODING = "utf-8"

$jobs = @(
    @{ name = "box_m";     dataset = "data/rfdetr/box" },
    @{ name = "product_m"; dataset = "data/rfdetr/product" }
)

Set-Location $proj
"$(Get-Date -Format s) start" | Add-Content $summary
foreach ($j in $jobs) {
    $out = Join-Path $proj "runs\$($j.name)"
    New-Item -ItemType Directory -Force $out | Out-Null
    if (Test-Path (Join-Path $out "done.txt")) { "$(Get-Date -Format s) $($j.name) skip (done)" | Add-Content $summary; continue }
    $args_ = @("train/train_rfdetr.py", "--dataset", $j.dataset, "--model", "medium", "--epochs", "30",
               "--batch", "2", "--grad-accum", "8", "--early-stop", "5", "--out", $out)
    $ckpt = Join-Path $out "last.ckpt"
    if (Test-Path $ckpt) { $args_ += @("--resume", $ckpt) }
    "$(Get-Date -Format s) $($j.name) begin resume=$(Test-Path $ckpt)" | Add-Content $summary
    & $py @args_ *>> (Join-Path $out "train.log")
    $code = $LASTEXITCODE
    "$(Get-Date -Format s) $($j.name) exit=$code" | Add-Content $summary
    if ($code -eq 0) { "ok" | Set-Content (Join-Path $out "done.txt") }
}
"$(Get-Date -Format s) all finished" | Add-Content $summary
