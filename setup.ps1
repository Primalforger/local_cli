Write-Host "=== Local AI CLI Setup ===" -ForegroundColor Green

$ollamaRunning = $true
try {
    Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Write-Host "Ollama is running" -ForegroundColor Green
}
catch {
    Write-Host "Ollama not running. Start it with: ollama serve" -ForegroundColor Red
    $ollamaRunning = $false
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

.venv\Scripts\Activate.ps1
pip install -q httpx rich prompt_toolkit pyyaml

$models = ollama list
if ($models -notmatch "qwen2.5-coder:14b") {
    Write-Host "Pulling qwen2.5-coder:14b..."
    ollama pull qwen2.5-coder:14b
}

$pythonPath = (Resolve-Path ".venv\Scripts\python.exe").Path
$cliPath = (Resolve-Path "cli.py").Path
$aliasLine = "function ai { & `"$pythonPath`" `"$cliPath`" @args }"

if (-not (Test-Path $PROFILE)) {
    New-Item -ItemType File -Path $PROFILE -Force | Out-Null
}

$existing = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
if ($existing -match "function ai") {
    Write-Host "ai alias already exists in profile" -ForegroundColor Green
}
else {
    Add-Content $PROFILE "`n$aliasLine"
    Write-Host "Added 'ai' command to PowerShell profile" -ForegroundColor Green
}

. $PROFILE

Write-Host ""
Write-Host "=== Setup Complete ===" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Cyan
Write-Host "  ai                              # interactive mode"
Write-Host "  ai 'explain quicksort'          # one-shot"
Write-Host "  ai -f main.py 'find bugs'      # with file context"
Write-Host "  type file.py | ai 'review'     # pipe mode"
Write-Host ""
Write-Host "New PowerShell windows will have 'ai' automatically."
