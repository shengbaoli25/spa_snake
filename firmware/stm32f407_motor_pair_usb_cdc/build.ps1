param(
    [string]$ToolchainBin = '',
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$NinjaArgs = @()
)

$ErrorActionPreference = 'Stop'
$originalPath = $env:PATH
try {
    if ($ToolchainBin) {
        $toolchainPath = (Resolve-Path -LiteralPath $ToolchainBin).Path
        $env:PATH = $toolchainPath + [IO.Path]::PathSeparator + $env:PATH
    }
    foreach ($tool in @('ninja', 'arm-none-eabi-gcc', 'arm-none-eabi-objcopy')) {
        if (-not (Get-Command $tool -ErrorAction SilentlyContinue)) {
            throw "$tool was not found. Install Ninja and Arm GNU Toolchain; add them to PATH or pass -ToolchainBin."
        }
    }
    New-Item -ItemType Directory -Force -Path (Join-Path $PSScriptRoot 'build') | Out-Null
    Push-Location $PSScriptRoot
    try {
        & ninja @NinjaArgs
        if ($LASTEXITCODE -ne 0) { throw "Ninja failed with exit code $LASTEXITCODE" }
    } finally {
        Pop-Location
    }
} finally {
    $env:PATH = $originalPath
}
