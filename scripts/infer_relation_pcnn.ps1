Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "未找到虚拟环境：$py"
}

Set-Location $root
& $py "scripts\infer_relation_pcnn.py" @args
