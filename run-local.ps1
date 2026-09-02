# Start CloudOptima locally: API + dashboard.
#
#   Right-click -> "Run with PowerShell", or from a terminal:
#       powershell -ExecutionPolicy Bypass -File run-local.ps1
#
# Everything binds to the explicit IPv4 address 127.0.0.1 rather than the name
# "localhost". On Windows, "localhost" resolves to the IPv6 address ::1 before
# 127.0.0.1, and a server bound to one stack is invisible on the other - which
# produces a browser that cannot reach an API that is demonstrably running.
# Using a literal IP removes name resolution from the picture entirely.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Definition
$backend = Join-Path $root "backend"
$frontend = Join-Path $root "frontend"

$API_HOST = "127.0.0.1"
$API_PORT = 8000
$UI_PORT  = 5173

function Write-Step($n, $msg) { Write-Host "[$n] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)       { Write-Host "    OK  $msg" -ForegroundColor Green }
function Write-Bad($msg)      { Write-Host "    !!  $msg" -ForegroundColor Red }

Write-Host ""
Write-Host "  CloudOptima - local development" -ForegroundColor White
Write-Host "  =============================="
Write-Host ""

# --- 0. Stop anything already holding the ports ---------------------------
Write-Step 0 "Clearing ports $API_PORT and $UI_PORT"
Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='uvicorn.exe' OR Name='node.exe'" |
    Where-Object { $_.CommandLine -like '*uvicorn*app.main*' -or $_.CommandLine -like '*vite*' } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 2
Write-Ok "ports free"

# --- 1. Check the trained artifacts exist ---------------------------------
Write-Step 1 "Checking model artifacts"
$needed = @("predictor_xgboost.json", "cpu_xgboost.joblib", "ram_xgboost.joblib",
            "anomaly_isolation_forest.joblib", "dqn_agent.json")
$missing = @($needed | Where-Object { -not (Test-Path (Join-Path $backend "artifacts\$_")) })
if ($missing.Count -gt 0) {
    Write-Bad "missing: $($missing -join ', ')"
    Write-Host "    Building them now (this takes a few minutes)..." -ForegroundColor Yellow
    Push-Location $backend
    python scripts/generate_data.py --days 30 --interval 5 --seed 42
    python scripts/train.py --no-tune --rl-episodes 12 --rl-ticks 288
    python scripts/evaluate.py --ticks 288 --repeats 2
    Pop-Location
} else {
    Write-Ok "all $($needed.Count) artifacts present"
}

# --- 2. Frontend environment ----------------------------------------------
Write-Step 2 "Writing frontend/.env"
"VITE_API_BASE_URL=http://${API_HOST}:${API_PORT}" |
    Set-Content -Path (Join-Path $frontend ".env") -Encoding utf8
Write-Ok "VITE_API_BASE_URL = http://${API_HOST}:${API_PORT}"

if (-not (Test-Path (Join-Path $frontend "node_modules"))) {
    Write-Step 2 "Installing frontend dependencies (first run only)"
    Push-Location $frontend; npm install; Pop-Location
}

# --- 3. Start the API ------------------------------------------------------
Write-Step 3 "Starting API on http://${API_HOST}:${API_PORT}"
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "cd /d `"$backend`" && uvicorn app.main:app --host $API_HOST --port $API_PORT" `
    -WindowStyle Minimized

$ready = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 2
    try {
        $r = Invoke-RestMethod -Uri "http://${API_HOST}:${API_PORT}/api/health" -TimeoutSec 5
        if ($r.status) {
            Write-Ok "API is up - status '$($r.status)', artifacts_ready=$($r.artifacts_ready)"
            if (-not $r.artifacts_ready) { Write-Bad "models missing; run: python scripts/train.py" }
            $ready = $true; break
        }
    } catch { }
}
if (-not $ready) { Write-Bad "API did not come up. Check the minimised API window for the error."; exit 1 }

# --- 4. Start the dashboard ------------------------------------------------
Write-Step 4 "Starting dashboard on http://${API_HOST}:${UI_PORT}"
Start-Process -FilePath "cmd.exe" `
    -ArgumentList "/c", "cd /d `"$frontend`" && npm run dev -- --host $API_HOST --port $UI_PORT" `
    -WindowStyle Minimized

$uiReady = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 2
    try {
        $null = Invoke-WebRequest -Uri "http://${API_HOST}:${UI_PORT}/" -TimeoutSec 5 -UseBasicParsing
        Write-Ok "dashboard is up"; $uiReady = $true; break
    } catch { }
}
if (-not $uiReady) { Write-Bad "dashboard did not come up. Check the minimised dashboard window."; exit 1 }

# --- 5. Open the browser ---------------------------------------------------
Write-Host ""
Write-Host "  Dashboard : http://${API_HOST}:${UI_PORT}" -ForegroundColor Green
Write-Host "  API docs  : http://${API_HOST}:${API_PORT}/docs" -ForegroundColor Green
Write-Host ""
Write-Host "  Use the address above, not 'localhost' - see the note at the top of" -ForegroundColor DarkGray
Write-Host "  this script for why. Close the two minimised windows to stop." -ForegroundColor DarkGray
Write-Host ""
Start-Process "http://${API_HOST}:${UI_PORT}"
