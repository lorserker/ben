# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\game.py'],
    pathex=['..\\src'],
    binaries=[('..\\bin\\dds.dll', 'bin'), ('..\\bin\\BGADLL.dll', 'bin'), ('..\\bin\\libbcalcdds.dll', 'bin'), ('..\\bin\\SuitCLib.dll', 'bin'), ('..\\bin\\EPBot86.dll', 'bin'), ('..\\bin\\EPBot64.dll', 'bin')],
    datas=[(r'..\src\nn\*.py','nn')],
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
