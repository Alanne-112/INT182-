param(
    [Parameter(Mandatory=$true)]
    [ValidateSet('review_labels.py','train_model.py')]
    [string]$Script,
    [string[]]$ToolArgs
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$bundlePython = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$workspaceRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$workspacePackages = Join-Path $workspaceRoot 'work\python-packages'
$previousPythonPath=$env:PYTHONPATH
$previousMplConfig=$env:MPLCONFIGDIR
try {
    if (Test-Path -LiteralPath $projectPython) {
        & $projectPython -X utf8 $Script @ToolArgs
    } elseif ((Test-Path -LiteralPath $bundlePython) -and (Test-Path -LiteralPath $workspacePackages)) {
        $env:PYTHONPATH=$workspacePackages
        $env:MPLCONFIGDIR=Join-Path $workspaceRoot 'work'
        & $bundlePython -X utf8 $Script @ToolArgs
    } else { throw 'Python environment not found. See README.md.' }
    if ($LASTEXITCODE -ne 0) { throw "Tool exited with code $LASTEXITCODE" }
} finally {
    $env:PYTHONPATH=$previousPythonPath
    $env:MPLCONFIGDIR=$previousMplConfig
}
