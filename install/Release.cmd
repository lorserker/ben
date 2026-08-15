@echo off
rem ============================================================================
rem  Release.cmd  -  build and publish a BEN release in one step.
rem    1. BuildAll.cmd      freeze + assemble + zip the four packages
rem    2. Publish-Release.ps1  tag, create the GitHub release, upload the zips
rem
rem  Creates a DRAFT release by default - review it on GitHub, then publish.
rem
rem  Usage:
rem    Release.cmd                 build, then draft release
rem    Release.cmd -Publish        build, then publish live immediately
rem    Release.cmd -DryRun         build, then show what would be released
rem    Release.cmd -SkipBuild      skip the build, publish existing zips
rem
rem  Any other arguments are passed through to Publish-Release.ps1.
rem ============================================================================
setlocal

set "SKIPBUILD="
set "PSARGS="

:parse
if "%~1"=="" goto done_parse
if /I "%~1"=="-SkipBuild" (
    set "SKIPBUILD=1"
) else (
    set "PSARGS=%PSARGS% %1"
)
shift
goto parse
:done_parse

if defined SKIPBUILD (
    echo [Release] skipping build - using the existing zips
) else (
    call "%~dp0BuildAll.cmd"
    if errorlevel 1 (
        echo [Release] build failed - not publishing.
        exit /b 1
    )
)

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0Publish-Release.ps1"%PSARGS%
exit /b %errorlevel%
