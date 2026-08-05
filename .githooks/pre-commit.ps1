$ErrorActionPreference = 'Stop'
$hook = Join-Path $PSScriptRoot 'pre-commit'
python $hook @args
exit $LASTEXITCODE
