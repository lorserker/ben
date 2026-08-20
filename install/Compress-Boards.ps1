<#
.SYNOPSIS
    Compress the current training boards ready for Publish-TrainingData.ps1.

.DESCRIPTION
    BBA\Boards holds tens of GB, most of which is not worth publishing: the
    source PBN files are an intermediate, the reject files are diagnostics, and
    the Version* folders are superseded. What matters is the -OK_boards.pbn set
    for the current BBA version - that is the complete input needed to retrain
    the bidding models.

    This zips each of those files into install\data-staging. The data is text
    and compresses about 8x, so roughly 6 GB becomes well under 1 GB.

    Only the top level of the source folder is scanned. Version* subfolders hold
    older BBA versions and are deliberately skipped.

    Run this before Publish-TrainingData.ps1, which uploads what lands here.

.PARAMETER Source
    Folder holding the board files. Defaults to BBA\Boards in the repository.

.PARAMETER Destination
    Where to write the zips. Defaults to install\data-staging.

.PARAMETER Force
    Rebuild zips even when they are already newer than their source file.

.PARAMETER DryRun
    List what would be compressed, without writing anything.

.EXAMPLE
    .\Compress-Boards.ps1 -DryRun
    Show which board files would be compressed and how big they are.

.EXAMPLE
    .\Compress-Boards.ps1
    Build the zips, skipping any that are already up to date.
#>
[CmdletBinding()]
param(
    [string]$Source,
    [string]$Destination,
    [switch]$Force,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

# Both are needed: ZipFile/ZipFileExtensions live in ...Compression.FileSystem,
# while ZipArchiveMode and CompressionLevel live in ...Compression itself.
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Fail($message) {
    Write-Host "[boards] ERROR: $message" -ForegroundColor Red
    exit 1
}

function Step($message) {
    Write-Host "[boards] $message" -ForegroundColor Cyan
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path

if ([string]::IsNullOrWhiteSpace($Source)) {
    $Source = Join-Path $repoRoot 'BBA\Boards'
}
if ([string]::IsNullOrWhiteSpace($Destination)) {
    $Destination = Join-Path $PSScriptRoot 'data-staging'
}

if (-not (Test-Path -LiteralPath $Source)) {
    Fail "Source folder not found: $Source"
}
Step "source:      $Source"
Step "destination: $Destination"

# Top level only - Version* subfolders are older BBA versions and are skipped.
$boards = @(Get-ChildItem -LiteralPath $Source -Filter '*-OK_boards.pbn' -File)
if ($boards.Count -eq 0) {
    Fail "No *-OK_boards.pbn files found in $Source"
}

$rawBytes = 0
foreach ($b in $boards) { $rawBytes += $b.Length }
Step ("found {0} board file(s), {1:N2} GB uncompressed" -f $boards.Count, ($rawBytes / 1GB))

if ($DryRun) {
    Write-Host ""
    foreach ($b in $boards) {
        $zipName = [System.IO.Path]::GetFileNameWithoutExtension($b.Name) + '.zip'
        Write-Host ("          {0,8:N0} MB  {1}  ->  {2}" -f ($b.Length / 1MB), $b.Name, $zipName)
    }
    Write-Host ""
    Write-Host "[boards] dry run complete - nothing was written." -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $Destination)) {
    New-Item -ItemType Directory -Path $Destination -Force | Out-Null
}

$built = 0
$skipped = 0
$zipBytes = 0
foreach ($b in $boards) {
    $zipName = [System.IO.Path]::GetFileNameWithoutExtension($b.Name) + '.zip'
    $zipPath = Join-Path $Destination $zipName

    if ((Test-Path -LiteralPath $zipPath) -and (-not $Force)) {
        $existing = Get-Item -LiteralPath $zipPath
        if ($existing.LastWriteTime -ge $b.LastWriteTime) {
            Step "  up to date, skipping $zipName"
            $zipBytes += $existing.Length
            $skipped++
            continue
        }
    }

    Step ("  compressing {0} ({1:N0} MB) ..." -f $b.Name, ($b.Length / 1MB))
    if (Test-Path -LiteralPath $zipPath) { Remove-Item -LiteralPath $zipPath -Force }

    # Compress-Archive is very slow on files this size in Windows PowerShell 5.1,
    # so go through the .NET ZipArchive API directly. The entry keeps the
    # original file name, so unzipping reproduces exactly what the pipeline reads.
    $archive = $null
    try {
        $archive = [System.IO.Compression.ZipFile]::Open($zipPath, [System.IO.Compression.ZipArchiveMode]::Create)
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive, $b.FullName, $b.Name, [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    finally {
        if ($archive) { $archive.Dispose() }
    }

    $made = Get-Item -LiteralPath $zipPath
    $zipBytes += $made.Length
    $built++
    Step ("    -> {0} ({1:N0} MB, {2:N1}x)" -f $zipName, ($made.Length / 1MB), ($b.Length / $made.Length))
}

Write-Host ""
Step ("built {0}, skipped {1} - {2:N2} GB total in {3}" -f $built, $skipped, ($zipBytes / 1GB), $Destination)
Write-Host "[boards] done - now run Publish-TrainingData.ps1" -ForegroundColor Green
