@echo off
rem ============================================================================
rem  check_env.cmd  -  make release builds run in the conda 'TF2' environment
rem  ----------------------------------------------------------------------------
rem  PyInstaller bundles must be frozen against the packages pinned in
rem  ..\requirements.txt  (Python 3.12, numpy 2.x, tensorflow, PyInstaller).
rem  Freezing from a different interpreter gives a mixed numpy in the bundle and
rem  the app crashes at startup with "CPU dispatcher tracer already initlized".
rem
rem  Strategy:
rem    1. already in TF2            -> just verify
rem    2. else try 'conda activate' -> clean activation
rem    3. else fall back to using   %USERPROFILE%\.conda\envs\TF2 directly
rem       (covers a fresh / reconfigured VS Code terminal where 'conda' isn't on
rem        PATH because 'conda init cmd.exe' hasn't run for this shell)
rem    4. verify Python 3.12 + numpy 2.x, else bail out with errorlevel 1.
rem
rem  'call'ed by BuildAll.cmd / build.cmd before any PyInstaller step.
rem ============================================================================

if /I "%CONDA_DEFAULT_ENV%"=="TF2" goto :verify

echo [check_env] activating conda env 'TF2' ...
call conda activate TF2 2>nul
if not errorlevel 1 goto :verify

rem --- 'conda activate' not usable here: use the env directory directly --------
set "TF2_PREFIX=%USERPROFILE%\.conda\envs\TF2"
if not exist "%TF2_PREFIX%\python.exe" (
    echo.
    echo [check_env] ERROR: cannot use the 'TF2' environment.
    echo             'conda activate TF2' failed - conda is not on PATH for this
    echo             shell, e.g. a fresh VS Code terminal where 'conda init
    echo             cmd.exe' has not run - AND "%TF2_PREFIX%\python.exe"
    echo             does not exist either. Repair the TF2 env, or build from
    echo             an Anaconda Prompt.
    exit /b 1
)
echo [check_env] conda not usable in this shell - using "%TF2_PREFIX%" directly.
set "CONDA_PREFIX=%TF2_PREFIX%"
set "CONDA_DEFAULT_ENV=TF2"
set "PATH=%TF2_PREFIX%;%TF2_PREFIX%\Library\mingw-w64\bin;%TF2_PREFIX%\Library\usr\bin;%TF2_PREFIX%\Library\bin;%TF2_PREFIX%\Scripts;%TF2_PREFIX%\bin;%PATH%"

:verify
python -c "import numpy, PyInstaller, tensorflow" >nul 2>&1
if not errorlevel 1 goto :verify_ver
echo.
echo [check_env] ERROR: active Python is missing numpy / PyInstaller / tensorflow.
echo             In the TF2 env run:  pip install -r ..\requirements.txt
exit /b 1

:verify_ver
python -c "import sys, numpy; sys.exit(0 if sys.version_info[:2]==(3,12) and numpy.__version__.split('.')[0]=='2' else 1)"
if not errorlevel 1 goto :ok
echo.
echo [check_env] ERROR: wrong build environment - need Python 3.12 + numpy 2.x,
echo             per ..\requirements.txt - that is the conda 'TF2' env.
python -c "import sys, numpy; print('             have: Python '+sys.version.split()[0]+' + numpy '+numpy.__version__); print('             at:   '+sys.executable)"
exit /b 1

:ok
python -c "import sys, numpy; print('[check_env] OK - Python '+sys.version.split()[0]+' + numpy '+numpy.__version__+'  ['+sys.executable+']')"
exit /b 0
