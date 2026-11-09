# 본 생성(3단계): [BOX] → [ONE] → [FEW] 순서로 차례로 만든다 (사용자 확정 2026-11-04)
#   수량: BOX 1,350 + 150 / ONE 1,260 + 140 / FEW 900 + 100 = 3,900장
#   이어서 만들기: 다시 실행하면 meta.json 이 있는 장면은 건너뛴다. 끊겨도 이 스크립트를 다시 실행하면 된다.
#   실행: powershell -ExecutionPolicy Bypass -File run_full.ps1
#   진행 기록: data\synthetic\_logs\run_full.log (요약), <job>.log (Blender 전체 출력)
$ErrorActionPreference = "Continue"
$bp  = "C:\Zivid\3rdparty\venv_blenderproc\Scripts\blenderproc.exe"
$bl  = "C:\Zivid\3rdparty\blender\blender-4.2.1"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$D   = Join-Path (Split-Path -Parent $here) "data\synthetic"
$log = Join-Path $D "_logs"
New-Item -ItemType Directory -Force $log | Out-Null
$summary = Join-Path $log "run_full.log"

# seed: 파일럿(0)과 겹치지 않게 학습 10, 테스트 20
$jobs = @(
    @{ name = "box_train"; script = "gen_pallet.py"; out = "box\train"; n = 1350; seed = 10; extra = @("--types", "cycle") },
    @{ name = "box_test";  script = "gen_pallet.py"; out = "box\test";  n = 150;  seed = 20; extra = @("--types", "cycle") },
    @{ name = "one_train"; script = "gen_bin.py";    out = "one\train"; n = 1260; seed = 10; extra = @("--mode", "one") },
    @{ name = "one_test";  script = "gen_bin.py";    out = "one\test";  n = 140;  seed = 20; extra = @("--mode", "one") },
    @{ name = "few_train"; script = "gen_bin.py";    out = "few\train"; n = 900;  seed = 10; extra = @("--mode", "few") },
    @{ name = "few_test";  script = "gen_bin.py";    out = "few\test";  n = 100;  seed = 20; extra = @("--mode", "few") }
)

function Count-Done($dir) {
    if (-not (Test-Path $dir)) { return 0 }
    return @(Get-ChildItem $dir -Directory | Where-Object { Test-Path (Join-Path $_.FullName "meta.json") }).Count
}

Set-Location $here
"$(Get-Date -Format s) start" | Add-Content $summary
# Blender 메모리가 장면마다 조금씩 쌓인다 (220장에서 1.6 → 4.4 GB, 장당 9 → 18 s).
# 그래서 $chunk 장마다 Blender 프로세스를 새로 띄운다. --n 을 100, 200, ... 으로 늘려 가며 부르면
# 이미 만든 장면은 건너뛰므로 매번 다음 100장만 만든다.
$chunk = 100
foreach ($j in $jobs) {
    $out = Join-Path $D $j.out
    for ($upto = $chunk; $upto -lt $j.n + $chunk; $upto += $chunk) {
        $n_ = [Math]::Min($upto, $j.n)
        # 실패하면(크래시 등) 최대 3번까지 다시 실행한다. 완성된 장면은 건너뛰므로 남은 것만 만든다.
        for ($try = 1; $try -le 3; $try++) {
            $done = Count-Done $out
            if ($done -ge $n_) { break }
            "$(Get-Date -Format s) $($j.name) upto=$n_ try=$try done=$done/$($j.n)" | Add-Content $summary
            $args_ = @("run", $j.script, "--custom-blender-path", $bl, "--out", $out, "--n", $n_, "--seed", $j.seed) + $j.extra
            & $bp @args_ *>> (Join-Path $log "$($j.name).log")
            "$(Get-Date -Format s) $($j.name) exit=$LASTEXITCODE" | Add-Content $summary
        }
    }
    "$(Get-Date -Format s) $($j.name) finished done=$(Count-Done $out)/$($j.n)" | Add-Content $summary
}
"$(Get-Date -Format s) all finished" | Add-Content $summary
