$repoRoot = Split-Path -Parent $PSScriptRoot
$defaultJCodeMunchRepo = "local/{0}" -f (Split-Path -Leaf $repoRoot)

function Resolve-NavigationCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string] $Name
    )

    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($null -eq $command) {
        return $null
    }

    return $command
}

function Invoke-NavigationCommand {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $Name,

        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CliArgs
    )

    $command = Resolve-NavigationCommand -Name $Name
    if ($null -eq $command) {
        throw "$Name was not found on PATH. This repo no longer vendors a local wrapper; install $Name or fall back to rg and targeted file reads."
    }

    & $command @CliArgs
}

function Invoke-RepoMap {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CliArgs
    )

    Invoke-NavigationCommand -Name "repomapper" @CliArgs
}

function Invoke-JCodeMunch {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CliArgs
    )

    Invoke-NavigationCommand -Name "jcodemunch" @CliArgs
}

function Find-CodeSymbol {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $Query,
        [string] $Language = "scala",
        [int] $MaxResults = 10,
        [string] $Repo = $defaultJCodeMunchRepo,
        [string] $Kind,
        [string] $FilePattern
    )

    $args = @("search-symbols", $Query, "--repo", $Repo, "--max-results", $MaxResults)
    if ($Language) { $args += @("--language", $Language) }
    if ($Kind) { $args += @("--kind", $Kind) }
    if ($FilePattern) { $args += @("--file-pattern", $FilePattern) }
    Invoke-NavigationCommand -Name "jcodemunch" @args
}

function Get-CodeSymbol {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $SymbolId,
        [string] $Repo = $defaultJCodeMunchRepo,
        [int] $ContextLines = 0,
        [switch] $Verify
    )

    $args = @("get-symbol", $SymbolId, "--repo", $Repo, "--context-lines", $ContextLines)
    if ($Verify) { $args += "--verify" }
    Invoke-NavigationCommand -Name "jcodemunch" @args
}

function Find-CodeText {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $Query,
        [string] $Repo = $defaultJCodeMunchRepo,
        [string] $FilePattern = "*.scala",
        [int] $MaxResults = 20
    )

    Invoke-NavigationCommand -Name "jcodemunch" "search-text" $Query "--repo" $Repo "--file-pattern" $FilePattern "--max-results" $MaxResults
}

function Show-CodeTree {
    param(
        [string] $PathPrefix = "",
        [string] $Repo = $defaultJCodeMunchRepo,
        [switch] $IncludeSummaries
    )

    $args = @("file-tree", "--repo", $Repo)
    if ($PathPrefix) { $args += @("--path-prefix", $PathPrefix) }
    if ($IncludeSummaries) { $args += "--include-summaries" }
    Invoke-NavigationCommand -Name "jcodemunch" @args
}

function Show-CodeOutline {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string] $FilePath,
        [string] $Repo = $defaultJCodeMunchRepo
    )

    Invoke-NavigationCommand -Name "jcodemunch" "file-outline" $FilePath "--repo" $Repo
}

Set-Alias rmap Invoke-RepoMap
Set-Alias jcm Invoke-JCodeMunch
Set-Alias jfindsym Find-CodeSymbol
Set-Alias jgetsym Get-CodeSymbol
Set-Alias jfindtxt Find-CodeText
Set-Alias jtree Show-CodeTree
Set-Alias joutline Show-CodeOutline
