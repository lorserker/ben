# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\game.py'],
    pathex=['..\\src', '..\\bin\\dds3-win'],
    binaries=[('..\\bin\\BGA\\windows\\x64\\BGADLL.dll', 'bin\\BGA\\windows\\x64'), ('..\\bin\\libbcalcdds.dll', 'bin'), ('..\\bin\\SuitCLib.dll', 'bin'), ('..\\bin\\BBA\\windows\\x64\\EPBot.dll', 'bin\\BBA\\windows\\x64')],
    datas=[(r'..\src\nn\*.py','nn')],
    # h5py 3.16 added the compiled submodule h5py._npystrings which the bundled
    # PyInstaller h5py hook does NOT auto-collect (every other h5py .pyd is).
    # Without it `import h5py` fails -> keras sets h5py=None -> model load dies
    # with "'NoneType' object has no attribute 'File'". Add just that one module
    # (collect_submodules('h5py') also works but drags in h5py.tests/pytest and
    # slows the build).
    hiddenimports=['dds3', 'h5py', 'h5py._npystrings'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Dev/notebook tooling + plotting/data libs that nothing in BEN imports at
    # runtime (only util.py has a notebook-only `from IPython...` inside a never-
    # called function; matplotlib/pandas have NO direct src import). Excluding
    # them drops thousands of files from Analysis -> much less Defender cold-scan
    # latency and a smaller dist. tkinter is intentionally NOT excluded (the GUI
    # specs need it). Verified the frozen exe still imports the full TF/NN stack.
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
    name='game',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='game',
)
