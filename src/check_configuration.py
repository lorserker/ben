import os

os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T'
# Just disables the warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import sys
import platform
import datetime
from colorama import Fore
from bba.BBA import BBABotBid
from pimc.PIMC import BGADLL


verbose = True

import winreg
def load_dotnet_framework_assembly(assembly_path, verbose = False):
    """    
    Parameters:
        assembly_path (str): The path to the .NET assembly without the `.dll` extension.
    
    Returns:
        The loaded assembly reference or raises an exception on failure.
    """
    import clr
    print(f"Loading {assembly_path}")
    clr.AddReference(assembly_path)
    if verbose:
        print("Loaded .NET assembly using clr.AddReference")
    return None  # Assembly types can be imported directly in this mode

def load_dotnet_core_assembly(assembly_path, verbose = False):
    """  
    Parameters:
        assembly_path (str): The path to the .NET assembly without the `.dll` extension.
    
    Returns:
        The loaded assembly reference or raises an exception on failure.
    """
    import clr
    if verbose:
        print(f"Loading {assembly_path}")
    # Pythonnet 3.x
    from clr_loader import get_coreclr
    from pythonnet import set_runtime
    runtime = get_coreclr()
    set_runtime(runtime)

    import System
    load_context = System.Runtime.Loader.AssemblyLoadContext.Default
    loaded_assembly = load_context.LoadFromAssemblyPath(assembly_path)
    if verbose:
        print("Loaded .NET Core assembly using clr_loader")
    return loaded_assembly


def get_pythonnet_version():
    from importlib.metadata import version
    return version("pythonnet")

def setup_clr():
    import clr
    # Detect if AddReference is available
    if hasattr(clr, "AddReference"):
        print("Using AddReference for Pythonnet < 3.0.0 or Anaconda version")
        clr.AddReference("System")
    else:
        print("Using clr-loader for Pythonnet >= 3.0.0")
        from clr_loader import get_coreclr
        from pythonnet import set_runtime
        runtime = get_coreclr()
        set_runtime(runtime)

import winreg

def check_dotnet_version():
    # Open the registry key
    reg_path = r"SOFTWARE\Microsoft\NET Framework Setup\NDP\v4\Full"
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
    release, _ = winreg.QueryValueEx(key, "Release")
    
    # Mapping release numbers to versions
    version_map = {
        528040: "4.8",
        461808: "4.7.2",
        394802: "4.6.2",
    }
    for rel, ver in version_map.items():
        if release >= rel:
            return f".NET Framework {ver} is installed."
    return "A version of .NET Framework is installed but could not be determined."


def is_pyinstaller_executable():
    # Check for _MEIPASS attribute (specific to PyInstaller)
    if hasattr(sys, '_MEIPASS'):
        return True
    
    # Check if the current executable is the main PyInstaller executable
    if getattr(sys, 'frozen', False) and os.path.exists(sys.executable):
        return True

    return False

import winreg

def check_dotnet_versions():
    reg_path = r"SOFTWARE\Microsoft\NET Framework Setup\NDP"
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, reg_path)
    versions = []

    def parse_version(subkey_path):
        try:
            subkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey_path)
            # Check for "Version"
            try:
                version, _ = winreg.QueryValueEx(subkey, "Version")
                return version
            except FileNotFoundError:
                pass
            
            # Check for "Release"
            try:
                release, _ = winreg.QueryValueEx(subkey, "Release")
                # Map Release to a specific version
                release_map = {
                    528040: "4.8",
                    461808: "4.7.2",
                    394802: "4.6.2",
                    378675: "4.5.1"
                }
                for rel, ver in sorted(release_map.items(), reverse=True):
                    if release >= rel:
                        return f"Release {release} (v{ver})"
                return f"Release {release} (Unknown version)"
            except FileNotFoundError:
                pass
            
            return None
        except FileNotFoundError:
            return None

    i = 0
    while True:
        try:
            subkey_name = winreg.EnumKey(key, i)
            subkey_path = reg_path + "\\" + subkey_name
            
            # Check top-level version keys
            version = parse_version(subkey_path)
            if version:
                versions.append(f"{subkey_name}: {version}")
            
            # Check subkeys (e.g., Client, Full)
            subkey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, subkey_path)
            j = 0
            while True:
                try:
                    subsubkey_name = winreg.EnumKey(subkey, j)
                    subsubkey_path = subkey_path + "\\" + subsubkey_name
                    version = parse_version(subsubkey_path)
                    if version:
                        versions.append(f"{subkey_name}\\{subsubkey_name}: {version}")
                    j += 1
                except OSError:
                    break

            i += 1
        except OSError:
            break

    if versions:
        return "Installed .NET Framework versions:\n" + "\n".join(versions)
    else:
        return "No .NET Framework versions found."

print("BEN_HOME=",os.getenv('BEN_HOME'))

print(f"{Fore.CYAN}{datetime.datetime.now():%Y-%m-%d %H:%M:%S} check_configuration.py - Version 0.8.6.0")
if is_pyinstaller_executable():
    print(f"Running inside a PyInstaller-built executable. {platform.python_version()}")
else:
    print(f"Running in a standard Python environment: {platform.python_version()}")

print(f"Python version: {sys.version}{Fore.RESET}")

print(check_dotnet_versions())

if sys.platform == 'win32':
    # Print the PythonNet version
    sys.stderr.write(f"PythonNet: {get_pythonnet_version()}\n") 
    sys.stderr.write(f"{check_dotnet_version()}\n") 

bot = BBABotBid(None, None ,None, None, None, None, None, verbose)
print("BBA loaded")


pimc = BGADLL(None, None, None, None, None, None, verbose)
print("PIMC loaded")

