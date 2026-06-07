# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\TMCGUI.py'],
    pathex=['..\\src'],
    binaries=[],
    datas=[('..\\src\\tmcgui\\images\\bid', 'tmcgui\\images\\bid'),('..\\src\\tmcgui\\images', 'tmcgui\\images'),('..\\src\\tmcgui\\images\\deck\\width 100', 'tmcgui\\images\\deck\\width 100'),('..\\src\\logo.png', '.'), ('..\\src\\ben.ico', '.')],
    hiddenimports=[
        # pywin32 pure-Python submodules (win32/lib/*.py) are not traced by the
        # analyzer when only the compiled .pyd siblings (win32gui/win32process)
        # are imported, so they must be declared explicitly. Without this the
        # frozen exe fails at runtime with "No module named 'win32con'".
        'win32con', 'win32gui', 'win32process', 'win32api',
        'pywintypes', 'pythoncom', 'psutil',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Dev/notebook tooling + plotting/data libs that nothing in BEN imports at
    # runtime. tkinter/pygame are kept (this GUI uses them). Excluding these
    # drops thousands of files from Analysis -> much less Defender cold-scan.
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
    name='TMCGUI',
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
    name='TMCGUI',
)
