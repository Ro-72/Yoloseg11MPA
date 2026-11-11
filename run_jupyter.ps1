$ErrorActionPreference = 'Stop'
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir = Join-Path $ProjectDir '.venv'
$PythonExe = Join-Path $VenvDir 'Scripts\python.exe'

if (-not (Test-Path -LiteralPath $PythonExe)) {
    python -m venv $VenvDir
}

& $PythonExe -m pip install --disable-pip-version-check -r (Join-Path $ProjectDir 'requirements.txt')
& $PythonExe -m jupyter lab --notebook-dir $ProjectDir

