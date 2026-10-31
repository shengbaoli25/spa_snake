$ErrorActionPreference = 'Stop'
$appRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $appRoot
$pythonExe = Join-Path $appRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = Join-Path $projectRoot 'upper_computer\.venv\Scripts\python.exe'
}
if (-not (Test-Path -LiteralPath $pythonExe)) { throw 'Python runtime missing. Install Python and requirements.txt first.' }
Set-Location -LiteralPath $projectRoot
& $pythonExe -m spa_control.app @args
