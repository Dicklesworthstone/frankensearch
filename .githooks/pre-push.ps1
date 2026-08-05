$ErrorActionPreference = 'Stop'
$hook = Join-Path $PSScriptRoot 'pre-push'
python $hook @args
exit $LASTEXITCODE
