# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['..\\src\\gameserver.py'],
    pathex=['..\\src', '..\\bin\\dds3-win'],
    binaries=[('..\\bin\\BGA\\windows\\x64\\BGADLL.dll', 'bin\\BGA\\windows\\x64'), ('..\\bin\\libbcalcdds.dll', 'bin'), ('..\\bin\\SuitCLib.dll', 'bin'), ('..\\bin\\BBA\\windows\\x64\\EPBot.dll', 'bin\\BBA\\windows\\x64')],
    datas=[(r'..\src\nn\*.py','nn')],
    hiddenimports=['dds3'],
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
    name='gameserver',
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
    name='gameserver',
)
