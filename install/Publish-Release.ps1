<#
.SYNOPSIS
    Publish a BEN release to GitHub from the zips BuildAll.cmd produced.

.DESCRIPTION
    Picks up where BuildAll.cmd stops. Reads the version from _version.py,
    verifies the four release zips exist and match that version, tags the
    commit, then creates the GitHub release and uploads the assets.

    Creates a DRAFT by default. Review it on GitHub and hit publish, or pass
    -Publish to make it live immediately.

.PARAMETER Publish
    Publish the release instead of leaving it as a draft.

.PARAMETER SkipTag
    Do not create or push a git tag. Use when the tag already exists.

.PARAMETER Notes
    Release notes body. When omitted, GitHub generates notes from the merged
    pull requests since the previous tag.

.PARAMETER Force
    Continue even when the working tree is dirty or the tag already exists.

.PARAMETER DryRun
    Run every check and print what would happen, without tagging, pushing, or
    uploading anything. Worth doing before a multi-GB upload.

.EXAMPLE
    .\Publish-Release.ps1
    Draft release for the current version, notes generated from PRs.

.EXAMPLE
    .\Publish-Release.ps1 -Publish -Notes "Fixes the macOS PIMC DDS backend."
    Live release with hand-written notes.

.NOTES
    Uploads roughly 2.7 GB across four assets - expect this to take a while.
#>
[CmdletBinding()]
param(
    [switch]$Publish,
    [switch]$SkipTag,
    [string]$Notes,
    [switch]$Force,
    [switch]$DryRun,
    [string]$Version
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$PACKAGES = @('BBA', 'BEN', 'BENAll', 'MvsM')

function Fail($message) {
    Write-Host "[publish] ERROR: $message" -ForegroundColor Red
    exit 1
}

function Step($message) {
    Write-Host "[publish] $message" -ForegroundColor Cyan
}

# Native commands that write to stderr raise a terminating NativeCommandError
# under $ErrorActionPreference='Stop' in Windows PowerShell 5.1, even when they
# exit 0. Run them with 'Continue' and judge them by $LASTEXITCODE instead.
function Invoke-Native {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $exe = $Arguments[0]
        $rest = @($Arguments | Select-Object -Skip 1)
        & $exe @rest 2>$null
    }
    finally {
        $ErrorActionPreference = $previous
    }
}

# --- prerequisites ----------------------------------------------------------
if ($null -eq (Get-Command gh -ErrorAction SilentlyContinue)) {
    Fail "GitHub CLI ('gh') not found. Install it from https://cli.github.com/"
}
try { gh auth status 2>&1 | Out-Null } catch { }
if ($LASTEXITCODE -ne 0) {
    Fail "GitHub CLI is not authenticated. Run: gh auth login"
}

# --- version ----------------------------------------------------------------
if (-not [string]::IsNullOrWhiteSpace($Version)) {
    $version = $Version
    Step "using version override: $version"
}
else {
    $version = (& python "$PSScriptRoot\_version.py" | Select-Object -First 1)
}
if ([string]::IsNullOrWhiteSpace($version) -or $version -eq 'unknown') {
    Fail "Could not read the version from ..\src\game.py"
}
$version = $version.Trim()
$tag = "v$version"
Step "version $version  ->  tag $tag"

# --- the four zips ----------------------------------------------------------
$assets = @()
foreach ($name in $PACKAGES) {
    $zip = Join-Path $PSScriptRoot "$name-$version.zip"
    if (-not (Test-Path $zip)) {
        Fail "Missing $name-$version.zip. Run BuildAll.cmd first."
    }
    $item = Get-Item $zip
    if ($item.Length -lt 1MB) {
        Fail "$($item.Name) is only $($item.Length) bytes - the build looks incomplete."
    }
    $assets += $item
    Step ("  {0,-24} {1,8:N1} MB   {2}" -f $item.Name, ($item.Length / 1MB), $item.LastWriteTime)
}
$totalGb = ($assets | Measure-Object -Property Length -Sum).Sum / 1GB
Step ("total upload: {0:N2} GB" -f $totalGb)

# --- repository state -------------------------------------------------------
$dirty = Invoke-Native git status --porcelain
if ($LASTEXITCODE -ne 0) {
    Fail "Not a git repository (or git failed). Run this from the install\ folder of a BEN checkout."
}
if (-not [string]::IsNullOrWhiteSpace(($dirty -join "`n"))) {
    Write-Host "[publish] WARNING: working tree has uncommitted changes:" -ForegroundColor Yellow
    ($dirty -split "`n" | Select-Object -First 10) | ForEach-Object { Write-Host "           $_" }
    if (-not $Force) {
        Fail "Commit or stash first, or re-run with -Force. The release should match a real commit."
    }
}

$null = Invoke-Native gh release view $tag --json tagName
if ($LASTEXITCODE -eq 0) {
    Fail "Release $tag already exists. Delete it first (gh release delete $tag) or bump the version."
}

# --- tag --------------------------------------------------------------------
if (-not $SkipTag) {
    $tagExists = Invoke-Native git tag --list $tag
    if ($DryRun) {
        if ([string]::IsNullOrWhiteSpace($tagExists)) {
            Step "[dry-run] would create tag $tag and push it to origin"
        }
        else {
            Step "[dry-run] tag $tag already exists locally; would push it to origin"
        }
    }
    else {
        if ([string]::IsNullOrWhiteSpace($tagExists)) {
            Step "creating tag $tag"
            Invoke-Native git tag -a $tag -m $tag
            if ($LASTEXITCODE -ne 0) { Fail "git tag failed" }
        }
        else {
            Step "tag $tag already exists locally"
        }
        Step "pushing tag $tag"
        Invoke-Native git push origin $tag
        if ($LASTEXITCODE -ne 0) { Fail "git push failed - is the commit pushed to origin?" }
    }
}

# --- release ----------------------------------------------------------------
$ghArgs = @('release', 'create', $tag, '--title', $tag)
if ([string]::IsNullOrWhiteSpace($Notes)) {
    $ghArgs += '--generate-notes'
}
else {
    $ghArgs += @('--notes', $Notes)
}
if (-not $Publish) {
    $ghArgs += '--draft'
}
foreach ($item in $assets) { $ghArgs += $item.FullName }

if ($DryRun) {
    Step "[dry-run] would run:"
    Write-Host "          gh $($ghArgs -join ' ')"
    Write-Host ""
    Write-Host "[publish] dry run complete - nothing was tagged, pushed, or uploaded." -ForegroundColor Green
    exit 0
}

if ($Publish) {
    Step "creating LIVE release $tag and uploading $($assets.Count) assets ..."
}
else {
    Step "creating DRAFT release $tag and uploading $($assets.Count) assets ..."
}
Step "this uploads several GB - it will take a while"

& gh @ghArgs
if ($LASTEXITCODE -ne 0) { Fail "gh release create failed" }

Write-Host ""
if ($Publish) {
    Write-Host "[publish] done - release $tag is live" -ForegroundColor Green
}
else {
    Write-Host "[publish] done - DRAFT $tag created. Review it, then publish with:" -ForegroundColor Green
    Write-Host "          gh release edit $tag --draft=false"
}
gh release view $tag --json url --jq .url
