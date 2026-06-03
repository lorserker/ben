<#
  setup_wsl.ps1 - install the Ubuntu-24.04 WSL distro (if needed) and provision
  BEN inside it by running setup_linux.sh.

  Run from Windows, in the repo root:
      powershell -ExecutionPolicy Bypass -File setup_wsl.ps1

  First run installs the distro and opens its one-time username/password setup;
  complete that, then re-run this script to finish provisioning. If the distro
  is already installed it provisions straight away.
#>
param([string]$Distro = "Ubuntu-24.04")

$ErrorActionPreference = "Stop"
# Make wsl.exe emit UTF-8 so we can parse --list reliably on Windows PowerShell.
$env:WSL_UTF8 = "1"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

# Repo root = the folder containing this script. Map it to its /mnt/<drive> path.
$RepoWin = $PSScriptRoot
$drive   = $RepoWin.Substring(0,1).ToLower()
$RepoLinux = "/mnt/$drive" + ($RepoWin.Substring(2) -replace '\\','/')

Write-Host "[setup_wsl] distro:        $Distro"
Write-Host "[setup_wsl] repo (Windows): $RepoWin"
Write-Host "[setup_wsl] repo (WSL):     $RepoLinux"

# Is the distro already installed?
$installed = @(wsl --list --quiet 2>$null | ForEach-Object { $_.Trim() }) -contains $Distro

if (-not $installed) {
    Write-Host ""
    Write-Host "[setup_wsl] '$Distro' is not installed - installing it now ..."
    wsl --install -d $Distro
    Write-Host ""
    Write-Host "[setup_wsl] Complete the one-time Linux username/password setup for"
    Write-Host "            '$Distro', then re-run this script to provision BEN:"
    Write-Host "                powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    exit 0
}

Write-Host ""
Write-Host "[setup_wsl] '$Distro' is installed - provisioning BEN inside it ..."
Write-Host "            (you may be prompted for your Linux sudo password)"
wsl -d $Distro -- bash -lc "cd '$RepoLinux' && bash setup_linux.sh"
if ($LASTEXITCODE -ne 0) {
    Write-Host "[setup_wsl] setup_linux.sh failed (exit $LASTEXITCODE)."
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "[setup_wsl] done. To run BEN:"
Write-Host "    wsl -d $Distro"
Write-Host "    source ~/ben/bin/activate"
Write-Host "    cd $RepoLinux/src && python game.py --boards ../Challenges/martens_declarer_first10.pbn --auto true"
