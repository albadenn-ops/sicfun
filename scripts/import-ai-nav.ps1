$repoRoot = Split-Path -Parent $PSScriptRoot
$jCodeMunchScript = Join-Path $PSScriptRoot "jcodemunch.ps1"
$repoMapScript = Join-Path $PSScriptRoot "repomap.ps1"

function Invoke-RepoMap {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CliArgs
    )

    & $repoMapScript @CliArgs
}

function Invoke-JCodeMunch {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CliArgs
    )

    & $jCodeMunchScript @CliArgs
}

function Find-CodeSymbol {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $Query,
        [string] $Language = "scala",
        [int] $MaxResults = 10,
        [string] $Repo = "local/untitled",
        [string] $Kind,
        [string] $FilePattern
    )

    $args = @("search-symbols", $Query, "--repo", $Repo, "--max-results", $MaxResults)
    if ($Language) { $args += @("--language", $Language) }
    if ($Kind) { $args += @("--kind", $Kind) }
    if ($FilePattern) { $args += @("--file-pattern", $FilePattern) }
    & $jCodeMunchScript @args
}

function Get-CodeSymbol {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $SymbolId,
        [string] $Repo = "local/untitled",
        [int] $ContextLines = 0,
        [switch] $Verify
    )

    $args = @("get-symbol", $SymbolId, "--repo", $Repo, "--context-lines", $ContextLines)
    if ($Verify) { $args += "--verify" }
    & $jCodeMunchScript @args
}

function Find-CodeText {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $Query,
        [string] $Repo = "local/untitled",
        [string] $FilePattern = "*.scala",
        [int] $MaxResults = 20
    )

    & $jCodeMunchScript search-text $Query --repo $Repo --file-pattern $FilePattern --max-results $MaxResults
}

function Show-CodeTree {
    param(
        [string] $PathPrefix = "src/main/scala/sicfun/holdem",
        [string] $Repo = "local/untitled",
        [switch] $IncludeSummaries
    )

    $args = @("file-tree", "--repo", $Repo, "--path-prefix", $PathPrefix)
    if ($IncludeSummaries) { $args += "--include-summaries" }
    & $jCodeMunchScript @args
}

function Show-CodeOutline {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $FilePath,
        [string] $Repo = "local/untitled"
    )

    & $jCodeMunchScript file-outline $FilePath --repo $Repo
}

Set-Alias rmap Invoke-RepoMap
Set-Alias jcm Invoke-JCodeMunch
Set-Alias jfindsym Find-CodeSymbol
Set-Alias jgetsym Get-CodeSymbol
Set-Alias jfindtxt Find-CodeText
Set-Alias jtree Show-CodeTree
Set-Alias joutline Show-CodeOutline
