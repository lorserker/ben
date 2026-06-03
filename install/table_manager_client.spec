# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['..\\src\\table_manager_client.py'],
    pathex=['..\\src', '..\\bin\\dds3-win'],
    binaries=[('..\\bin\\BGA\\windows\\x64\\BGADLL.dll', 'bin\\BGA\\windows\\x64'), ('..\\bin\\libbcalcdds.dll', 'bin'), ('..\\bin\\SuitCLib.dll', 'bin'), ('..\\bin\\BBA\\windows\\x64\\EPBot.dll', 'bin\\BBA\\windows\\x64')],
    datas=[(r'..\src\nn\*.py','nn')],
    hiddenimports=['dds3'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    exclude_binaries=True,
    name='table_manager_client',
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
    icon='..\\src\\ben.ico',
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='table_manager_client',
)
