# Run from any working directory. Uses project venv first, then this workspace runtime.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$bundlePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$workspacePackages = Join-Path $workspaceRoot 'work\python-packages'
if (Test-Path -LiteralPath $projectPython) {
    & $projectPython -X utf8 rep_counter.py @args
} elseif ((Test-Path -LiteralPath $bundlePython) -and (Test-Path -LiteralPath $workspacePackages)) {
    $previousPythonPath = $env:PYTHONPATH
    $previousMplConfig = $env:MPLCONFIGDIR
    try {
        $env:PYTHONPATH = $workspacePackages
        $env:MPLCONFIGDIR = Join-Path $workspaceRoot 'work'
        & $bundlePython -X utf8 rep_counter.py @args
    } finally {
        $env:PYTHONPATH = $previousPythonPath
        $env:MPLCONFIGDIR = $previousMplConfig
    }
} else {
    throw 'Python environment not found. Follow README.md installation steps.'
}
if ($LASTEXITCODE -ne 0) { throw "Counter exited with code $LASTEXITCODE" }
