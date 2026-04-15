Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "未找到虚拟环境解释器：$py。请先在项目根目录创建 .venv。"
}

Set-Location $root
& $py "run.py" @args
