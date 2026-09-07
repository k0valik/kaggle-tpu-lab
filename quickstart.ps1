<#
.SYNOPSIS
Serve Qwen3.8-27B — or one of its finetunes — on a free Kaggle TPU v5e-8.

.DESCRIPTION
    .\quickstart.ps1                # Qwen/Qwen3.8-27B (base)
    .\quickstart.ps1 serenity       # ReadyArt/Serenity-27B
    .\quickstart.ps1 uncensored     # orcarouter/Qwen3.8-27B-Uncensored (gated, needs HF_TOKEN)

Extra arguments go straight to `launch.py serve`:
    .\quickstart.ps1 serenity --max-model-len 131072 --max-num-seqs 16 --mtp 0

Environment:
    HF_TOKEN            Hugging Face token; required for a gated repo
    WEIGHTS_DATASET     Kaggle dataset mirroring the weights. Empty means the kernel
                        downloads them from Hugging Face (~10 min slower per launch).
#>
[CmdletBinding()]
param(
    [ValidateSet('base', 'serenity', 'uncensored')]
    [string]$Preset = 'base',

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Rest = @()
)

$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

switch ($Preset) {
    'base' {
        $hfModel = 'Qwen/Qwen3.8-27B'
        $servedName = 'qwen3.8-27b'
        $weightsDefault = 'rahim3/qwen3-8-27b-bf16'
    }
    'serenity' {
        $hfModel = 'ReadyArt/Serenity-27B'
        $servedName = 'serenity-27b'
        $weightsDefault = 'darkessid/serenity-27b-bf16'
    }
    'uncensored' {
        $hfModel = 'orcarouter/Qwen3.8-27B-Uncensored'
        $servedName = 'qwen3.8-27b-uncensored'
        $weightsDefault = ''
    }
}

$weights = if ($null -ne $env:WEIGHTS_DATASET) { $env:WEIGHTS_DATASET } else { $weightsDefault }

# launch.py checks the Kaggle CLI and its credentials itself, so we only need Python.
$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python) { $python = Get-Command python -ErrorAction SilentlyContinue }
if (-not $python) { throw 'Python is not on PATH.' }

# --- gated repos need a token ------------------------------------------------
$tokenArg = @()
if ($Preset -eq 'uncensored') {
    $hfToken = $env:HF_TOKEN
    $cached = Join-Path $HOME '.cache\huggingface\token'
    if (-not $hfToken -and (Test-Path -LiteralPath $cached)) {
        $hfToken = (Get-Content -LiteralPath $cached -Raw).Trim()
    }
    if (-not $hfToken) {
        throw @'
orcarouter/Qwen3.8-27B-Uncensored is gated. Request access on its model page, then
set a read-only token before launching:

    $env:HF_TOKEN = "hf_..."

The token is embedded in the source of the (private) Kaggle kernel, so use a
read-only one.
'@
    }
    $tokenArg = @('--hf-token', $hfToken)
}

# --- launch ------------------------------------------------------------------
$weightsLabel = if ($weights) { $weights } else { 'downloaded from Hugging Face inside the kernel' }
Write-Host "Model    : $hfModel  (served as '$servedName')"
Write-Host "Weights  : $weightsLabel"
Write-Host ''

& $python.Source launch.py serve `
    --hf-model $hfModel `
    --served-model-name $servedName `
    --weights-dataset $weights `
    @tokenArg @Rest
exit $LASTEXITCODE
