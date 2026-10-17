# -*- mode: python ; coding: utf-8 -*-
import os
import importlib.util

# ml_dtypes >= 0.6 (TensorFlow 2.21) is a delvewheel wheel: its _ml_dtypes_ext.pyd needs a renamed
# msvcp140-<hash>.dll from the sibling folder site-packages\ml_dtypes.libs, which its __init__ adds
# with os.add_dll_directory(<package>\..\ml_dtypes.libs). PyInstaller 6.10 does not collect that
# folder, so the frozen exe died on `import tensorflow` with "DLL load failed while importing
# _ml_dtypes_ext". Ship it to _internal\ml_dtypes.libs. Empty for an older ml_dtypes without it.
_mld = importlib.util.find_spec('ml_dtypes')
_mld_libs = os.path.join(os.path.dirname(os.path.dirname(_mld.origin)), 'ml_dtypes.libs') if _mld else ''
ml_dtypes_libs = [(os.path.join(_mld_libs, f), 'ml_dtypes.libs') for f in os.listdir(_mld_libs)] if os.path.isdir(_mld_libs) else []



a = Analysis(
    ['..\\src\\BEN.py'],
    pathex=[],
    binaries=[] + ml_dtypes_libs,
    datas=[('..\\src\\logo.png', '.'), ('..\\src\\ben.ico', '.')],    
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Dev/notebook tooling + plotting/data libs that nothing in BEN imports at
    # runtime. tkinter is kept (this GUI uses it). Excluding these drops
    # thousands of files from Analysis -> much less Defender cold-scan.
    excludes=[
        'IPython', 'ipykernel', 'jedi', 'parso',
        'notebook', 'nbformat', 'nbconvert', 'nbclient',
        'jupyter', 'jupyter_client', 'jupyter_core', 'jupyterlab', 'qtconsole',
        'matplotlib', 'pandas', 'pytest', '_pytest',
        # requests imports simplejson only optionally and falls back to stdlib
        # json. A partially-collected simplejson (no __init__/_speedups) imports
        # as an empty namespace pkg, so `from simplejson import JSONDecodeError`
        # fails at runtime. Excluding it forces the stdlib-json path.
        'simplejson',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='BEN',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='..\\src\\ben.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='BEN',
)
