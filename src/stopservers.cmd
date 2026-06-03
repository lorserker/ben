@echo off
rem stopservers.cmd - stop all BEN servers started by runservers.cmd
rem (appserver, appserverold, gameapi, gameserver) on Windows.
rem Kills the python processes whose command line runs those scripts;
rem leaves unrelated python processes alone.
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$pat='gameapi\.py|gameserver\.py|appserver\.py|appserverold\.py';" ^
  "$procs=Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match $pat };" ^
  "if (-not $procs) { Write-Host 'No BEN servers running.'; exit 0 }" ^
  "foreach ($p in $procs) { Write-Host ('Stopping PID ' + $p.ProcessId + ': ' + $p.CommandLine); Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }" ^
  "Write-Host 'All BEN servers stopped.'"
