# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\BENGUI.py'],
    pathex=[],
    binaries=[],
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
    name='BENGUI',
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
    name='BENGUI',
)
