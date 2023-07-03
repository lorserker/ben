import os
import psutil
import signal
import time
import subprocess

def get_server_pid():
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['name'].lower() == 'python.exe' and 'python gameserver.py' in ' '.join(proc.info['cmdline']).lower():
            return proc.info['pid']
    return None

def start_server():
    # Replace this with your actual server startup command
    server_command = "python gameserver.py"  # Example command to start the server

    server_directory = r"D:\Github\ben\src"


 # Start the server within the activated Anaconda environment in a new shell
    subprocess.Popen(f"conda activate ben && {server_command}", shell=True, cwd=server_directory)


# Get the PID of the server process
server_pid = get_server_pid()

if server_pid:
    # Terminate the server process
    print(f"Stopping server with PID: {server_pid}")
    os.kill(server_pid, signal.SIGTERM)

    # Wait for the server process to exit
    while psutil.pid_exists(server_pid):
        time.sleep(0.1)

# Start the server again in a new shell
start_server()