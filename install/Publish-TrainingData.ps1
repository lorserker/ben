<#
.SYNOPSIS
    Publish BEN's large training data files as GitHub release assets.

.DESCRIPTION
    The training data is far too large for the repository, so it is uploaded to
    a single fixed release tag and linked from the training documentation.

    Unlike Publish-Release.ps1 this is NOT tied to a version. The data changes
    rarely and independently of the code, so it lives on a stable tag - the
    documentation links keep working across every version release, and a large
    upload is not repeated for each one.

    The files to upload are listed in training-data.txt next to this script.
    Re-running the script adds to the existing release rather than creating a
    new one, so published links stay valid. Use -Force to replace an asset that
    is already there.

.PARAMETER Tag
    Release tag to publish to. Defaults to 'training-data'.

.PARAMETER Manifest
    Path to the manifest. Defaults to training-data.txt next to this script.

.PARAMETER Publish
    Publish the release instead of leaving it as a draft.

.PARAMETER Force
    Replace assets that already exist on the release (gh --clobber).

.PARAMETER DryRun
    Run every check and print what would happen, without creating the release
    or uploading anything. Worth doing before a large upload.

.EXAMPLE
    .\Publish-TrainingData.ps1 -DryRun
    Check the manifest resolves and show what would be uploaded.

.EXAMPLE
    .\Publish-TrainingData.ps1 -Publish
    Create or update the live training-data release.

.NOTES
    A GitHub release asset may be up to 2 GB. Anything larger has to be split,
    or hosted elsewhere.
#>
[CmdletBinding()]
param(
    [string]$Tag = 'training-data',
    [string]$Manifest,
    [switch]$Publish,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

$ASSET_LIMIT_BYTES = 2GB

function Fail($message) {
    Write-Host "[data] ERROR: $message" -ForegroundColor Red
    exit 1
}

function Step($message) {
    Write-Host "[data] $message" -ForegroundColor Cyan
}

# See the note in Publish-Release.ps1 - native commands that write to stderr
# raise a terminating NativeCommandError under $ErrorActionPreference='Stop' in
# Windows PowerShell 5.1, even when they exit 0. No param() block on purpose, so
# dash-prefixed tokens land in $args verbatim instead of binding to this
# function's own parameters.
function Invoke-Native {
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $exe = $args[0]
        $rest = @($args | Select-Object -Skip 1)
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

if ([string]::IsNullOrWhiteSpace($Manifest)) {
    $Manifest = Join-Path $PSScriptRoot 'training-data.txt'
}
if (-not (Test-Path $Manifest)) {
    Fail "Manifest not found: $Manifest"
}
Step "manifest: $Manifest"
Step "tag:      $Tag"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

# --- read the manifest ------------------------------------------------------
$entries = @()
$lineNo = 0
foreach ($raw in Get-Content -LiteralPath $Manifest) {
    $lineNo++
    $line = $raw.Trim()
    if ([string]::IsNullOrWhiteSpace($line)) { continue }
    if ($line.StartsWith('#')) { continue }

    $parts = $line -split '\|', 2
    if ($parts.Count -ne 2) {
        Fail "Line ${lineNo}: expected '<source path> | <asset name>', got: $line"
    }
    $source = $parts[0].Trim()
    $asset = $parts[1].Trim()
    if ([string]::IsNullOrWhiteSpace($source) -or [string]::IsNullOrWhiteSpace($asset)) {
        Fail "Line ${lineNo}: source path and asset name must both be given"
    }

    # Absolute paths are used as they are; anything else resolves from the repo root.
    if ([System.IO.Path]::IsPathRooted($source)) {
        $full = $source
    }
    else {
        $full = Join-Path $repoRoot $source
    }

    if (-not (Test-Path -LiteralPath $full)) {
        if ($source -match 'data-staging') {
            Fail "Line ${lineNo}: file not found: $full`n       Staged files are built by Compress-Boards.ps1 - run that first."
        }
        Fail "Line ${lineNo}: file not found: $full"
    }
    $item = Get-Item -LiteralPath $full
    if ($item.Length -gt $ASSET_LIMIT_BYTES) {
        Fail ("Line ${lineNo}: {0} is {1:N2} GB - over the 2 GB release asset limit" -f $item.Name, ($item.Length / 1GB))
    }

    $entries += [pscustomobject]@{
        Source = $item
        Asset  = $asset
    }
}

if ($entries.Count -eq 0) {
    Fail "Manifest lists no files. Nothing to upload."
}

# An asset name may only appear once, or the uploads collide on the release.
$dupes = $entries | Group-Object Asset | Where-Object { $_.Count -gt 1 }
if ($dupes) {
    Fail ("Duplicate asset name(s) in the manifest: " + (($dupes | ForEach-Object { $_.Name }) -join ', '))
}

foreach ($e in $entries) {
    Step ("  {0,-24} {1,8:N1} MB   <- {2}" -f $e.Asset, ($e.Source.Length / 1MB), $e.Source.FullName)
}
$totalBytes = 0
foreach ($e in $entries) { $totalBytes += $e.Source.Length }
Step ("total upload: {0:N1} MB across {1} file(s)" -f ($totalBytes / 1MB), $entries.Count)

# --- does the release already exist? ----------------------------------------
$existingAssets = @{}
$json = Invoke-Native gh release view $Tag --json assets
$releaseExists = ($LASTEXITCODE -eq 0)
if ($releaseExists) {
    Step "release $Tag already exists - assets will be added to it"
    try {
        $parsed = ($json -join '') | ConvertFrom-Json
        foreach ($a in $parsed.assets) { $existingAssets[$a.name] = [int64]$a.size }
    }
    catch {
        Step "  (could not read the current asset list - every file will be uploaded)"
    }
}
else {
    Step "release $Tag does not exist yet - it will be created"
}

# Skip anything already on the release at the same size. A 1 GB upload that
# fails part-way is then cheap to resume, instead of starting over. -Force
# re-uploads regardless.
$clobber = $false
if ($existingAssets.Count -gt 0 -and -not $Force) {
    $keep = @()
    foreach ($e in $entries) {
        if ($existingAssets.ContainsKey($e.Asset) -and $existingAssets[$e.Asset] -eq $e.Source.Length) {
            Step "  already uploaded, skipping $($e.Asset)"
        }
        else {
            if ($existingAssets.ContainsKey($e.Asset)) {
                Step "  size changed, will replace $($e.Asset)"
                $clobber = $true
            }
            $keep += $e
        }
    }
    $entries = $keep
    if ($entries.Count -eq 0) {
        Write-Host ""
        Write-Host "[data] nothing to do - every asset is already on $Tag at the same size." -ForegroundColor Green
        exit 0
    }
    $remaining = 0
    foreach ($e in $entries) { $remaining += $e.Source.Length }
    Step ("still to upload: {0:N1} MB across {1} file(s)" -f ($remaining / 1MB), $entries.Count)
}

# The files are staged under their manifest asset name before upload. gh names
# each asset after the file on disk, and several sources here do not match the
# name the documentation links to.
$staging = Join-Path ([System.IO.Path]::GetTempPath()) ("ben-training-data-" + [guid]::NewGuid().ToString('N'))

if ($DryRun) {
    Write-Host ""
    Step "[dry-run] would stage and upload:"
    foreach ($e in $entries) {
        Write-Host ("          {0}  ->  {1}" -f $e.Source.FullName, $e.Asset)
    }
    if ($releaseExists) {
        $clobber = ''
        if ($Force) { $clobber = ' --clobber' }
        Write-Host "          gh release upload $Tag <staged files>$clobber"
    }
    else {
        $draftFlag = ''
        if (-not $Publish) { $draftFlag = ' --draft' }
        Write-Host "          gh release create $Tag --title 'Training data' --notes ...$draftFlag <staged files>"
    }
    Write-Host ""
    Write-Host "[data] dry run complete - nothing was created or uploaded." -ForegroundColor Green
    exit 0
}

New-Item -ItemType Directory -Path $staging -Force | Out-Null
try {
    $staged = @()
    foreach ($e in $entries) {
        $target = Join-Path $staging $e.Asset
        Step "staging $($e.Asset) ..."
        Copy-Item -LiteralPath $e.Source.FullName -Destination $target -Force
        $staged += $target
    }

    Step "uploading - this will take a while"
    if ($releaseExists) {
        $ghArgs = @('release', 'upload', $Tag) + $staged
        if ($Force -or $clobber) { $ghArgs += '--clobber' }
    }
    else {
        $notes = "Training data for BEN. These files are too large for the repository, so they are published here and linked from the training documentation - see scripts/training/README2.md."
        $ghArgs = @('release', 'create', $Tag, '--title', 'Training data', '--notes', $notes)
        if (-not $Publish) { $ghArgs += '--draft' }
        $ghArgs += $staged
    }

    & gh @ghArgs
    if ($LASTEXITCODE -ne 0) { Fail "gh failed - the release was not updated" }
}
finally {
    if (Test-Path $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force -ErrorAction SilentlyContinue
    }
}

Write-Host ""
if ($releaseExists -or $Publish) {
    Write-Host "[data] done - assets uploaded to $Tag" -ForegroundColor Green
}
else {
    Write-Host "[data] done - DRAFT $Tag created. Review it, then publish with:" -ForegroundColor Green
    Write-Host "       gh release edit $Tag --draft=false"
}
gh release view $Tag --json url --jq .url
