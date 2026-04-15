Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
  throw "未找到虚拟环境：$py 。请在项目根目录创建 .venv 并安装依赖。"
}

Set-Location $root
if ($args.Count -eq 0) {
  & $py "scripts\train_relation_pcnn.py" --seed-type Person
} else {
  & $py "scripts\train_relation_pcnn.py" @args
}
