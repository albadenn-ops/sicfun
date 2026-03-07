param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $CliArgs
)

$repoRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repoRoot ".venv-tools\\Scripts\\python.exe"
$script = Join-Path $PSScriptRoot "jcodemunch_query.py"

if (-not (Test-Path $python)) {
    Write-Error "Missing $python. Recreate the tool environment first."
    exit 1
}

& $python $script @CliArgs
exit $LASTEXITCODE
