# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\TMCGUI.py'],
    pathex=['..\\src'],
    binaries=[],
    datas=[('..\\src\\tmcgui\\images\\bid', 'tmcgui\\images\\bid'),('..\\src\\tmcgui\\images', 'tmcgui\\images'),('..\\src\\tmcgui\\images\\deck\\width 100', 'tmcgui\\images\\deck\\width 100'),('..\\src\\logo.png', '.'), ('..\\src\\ben.ico', '.')],
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
