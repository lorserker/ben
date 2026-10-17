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
    ['..\\src\\pbn2bba.py'],
    pathex=['..\\src'],
    binaries=[] + ml_dtypes_libs,
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='pbn2bba',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
