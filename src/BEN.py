import sys
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
import subprocess
import webbrowser
import socket
import threading
import os
import json
import queue
import psutil
import signal

# Set up environment for UTF-8 encoding
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["TERM"] = "xterm-256color"  # Set terminal type to force color output
env["BEN_HOME"] = "."
env['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']

CONFIG_FILE = "BEN.settings.json"
update_lock = threading.Lock()

gameserver_name = "gameserver"
appserver_name = "appserver"

def is_process_running(process_name):
    """
    Check if there is any running process that contains the given name.
    """
    for proc in psutil.process_iter(['name', 'exe', 'cmdline']):
        try:
            # Check if process name matches
            if process_name in (proc.info['name'] or '') or \
               process_name in (proc.info['exe'] or '') or \
               process_name in ' '.join(proc.info['cmdline'] or []):
                return True

        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return False

class BridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.iconbitmap("ben.ico")
        self.root.title("Bridge with BEN. v0.8.7.0")
        self.root.geometry("1000x1000")

        # Center the window
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (1000 // 2)
        y = (screen_height // 2) - (1000 // 2)
        root.geometry(f"1000x1000+{x}+{y}")

        self.label = ttk.Label(root, text="Play bridge with BEN", font=("Arial", 14))
        self.label.pack(pady=50)

        # Menu Bar
        self.menu_bar = tk.Menu(root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Exit", command=self.on_exit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)

        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.on_about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)

        self.root.config(menu=self.menu_bar)

        # Instructions
        self.instructions = ttk.Label(root, text="Please start the BEN server and board manager first.", font=("Arial", 10))
        self.instructions.pack(pady=10)

        # Button states
        self.ben_server_running = False
        self.board_manager_running = False

        # Buttons
        self.ben_server_button = tk.Button(root, text="Start BEN Server and board manager", bg="green", command=self.toggle_ben_server)
        self.ben_server_button.pack(pady=5)

        self.play_button = tk.Button(root, text="Play", bg="red", state="disabled", command=self.on_play)
        self.play_button.pack(pady=20)

        output_label = ttk.Label(root, text="Application Output:")
        output_label.pack(anchor="w",padx=10)

        self.output_text = tk.Text(root, wrap="word", state="disabled", height=16, width=80, bg="black", fg="white")
        self.output_text.pack(side="left", fill="both", expand=True, padx=10, pady=10)

        # Define color tags for ANSI escape codes
        self.output_text.tag_configure("red", foreground="#FF7F50")
        self.output_text.tag_configure("green", foreground="#90EE90")
        self.output_text.tag_configure("blue", foreground="#ADD8E6")
        self.output_text.tag_configure("yellow", foreground="#FFD700")
        self.output_text.tag_configure("reset", foreground="black")
        self.output_text.configure(bg="#303030", fg="white")

        # Scrollbar for the output
        scrollbar = ttk.Scrollbar(root, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

        # Verbose checkbox and buttons (Start/Stop) in column 4
        self.detached_checkbox = tk.BooleanVar(value=False)

        # Check servers periodically
        self.check_ports_threaded()

        self.processes = []
        self.terminate_flag = False
        # Handle close event
        root.protocol("WM_DELETE_WINDOW", self.on_exit)

        # Load settings or set defaults
        self.settings = self.load_settings()
        self.gameport = int(self.settings.get("gameport", "4443"))
        self.boardport = int(self.settings.get("boardport", "8080"))
        self.settings["gameport"] = str(self.gameport)
        self.settings["boardport"] = str(self.boardport)
        self.save_settings()
        # Register signal handlers for termination
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)

    def load_settings(self):
        """Load settings from the configuration file."""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        return {}

    def save_settings(self):
        """Save settings to the configuration file."""
        with open(CONFIG_FILE, "w") as f:
            json.dump(self.settings, f)



    def show_closing_popup(self):
        popup = tk.Toplevel()
        popup.title("Closing")

        # Get dimensions of the main application window
        root_x = self.root.winfo_rootx()
        root_y = self.root.winfo_rooty()
        root_width = self.root.winfo_width()
        root_height = self.root.winfo_height()

        # Define the popup size
        popup_width = 300
        popup_height = 100

        # Calculate the center position
        center_x = root_x + (root_width // 2) - (popup_width // 2)
        center_y = root_y + (root_height // 2) - (popup_height // 2)

        # Set the geometry for the popup
        popup.geometry(f"{popup_width}x{popup_height}+{center_x}+{center_y}")
        tk.Label(popup, text="Closing application...", font=("Arial", 14)).pack(pady=20)
        popup.geometry("300x100")
        popup.update()  # Force display of the popup
        return popup

    def on_exit(self):
        popup = self.show_closing_popup()
        self.save_settings()
        self.stop_server()
        # Hide the closing popup
        popup.destroy()
        self.root.quit()

    def toggle_ben_server(self):
        if self.ben_server_running:
            self.stop_server()
            self.ben_server_running = False
            self.board_manager_running = False
        else:
            self.reset_output_text()
            self.start_server()
            self.ben_server_running = True
            self.board_manager_running = True

    def reset_output_text(self):
        """
        Clears all content from the output Text widget.
        """
        self.output_text.configure(state="normal")  # Temporarily enable editing
        self.output_text.delete("1.0", "end")  # Delete all text from line 1, character 0 to the end
        self.output_text.configure(state="disabled")  # Disable editing again

    def display_output(self, text, color=None):
        """
        Appends text to the output Text widget with an optional color.
        :param text: The text to display.
        :param color: The tag name (e.g., "red", "green", etc.) for the text color.
        """
        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        self.output_text.configure(state="normal")  # Enable editing temporarily
        if color and color in self.output_text.tag_names():
            self.output_text.insert("end", text, color)  # Apply the color tag
        else:
            self.output_text.insert("end", text)  # Default color
        self.output_text.configure(state="disabled")  # Disable editing
        self.output_text.see("end")  # Scroll to the end

    def start_server(self):

        def run_process(name, port, delay=None):
            output_queue = queue.Queue()
            try:
                time.sleep(delay)  # Wait for the specified delay if any
                self.display_output("\nStarting " + str(name) + "...\n", "green")

                exe_path = f"{name}.exe"
                cwd, cmd = None, []

                if os.path.exists(exe_path):
                    cmd = [exe_path]
                else:
                    # Fallback to running table_manager_client.py with python if exe is not found
                    if name == "appserver":
                        cwd = os.getcwd() + "\\frontend\\"
                        cmd = ["python", name+".py"]
                    else:
                        cwd = os.getcwd()
                        cmd = ["python", name+".py"]

                cmd.extend(["--port", str(port)])
                if name == "gameserver":
                    cmd.extend(["--seed", str(int(time.time()))])
                # Check the Detached checkbox
                if self.detached_checkbox.get():  # Assuming it's a Tkinter Checkbutton
                    cmd = ["cmd", "/k"] + cmd  # Prepend cmd /k to keep the console open
                    # Detached mode: Open in a new console, no output capture
                    creation_flags = subprocess.CREATE_NEW_CONSOLE
                    process = subprocess.Popen(
                        cmd,
                        stdout=None,  # Direct output to the new console
                        stderr=None,
                        env=env,
                        creationflags=creation_flags,
                    )
                    if port:  
                        cmd.extend(["--port", str(port)])

                    self.display_output("Running in detached mode (new console opened).\n", "green")
                    self.processes.append((process, subprocess.CREATE_NEW_CONSOLE, name))  # Add the process to the array
                            # Bring the main app back to focus
                    self.after(100, self.bring_to_focus)
                else:
                    # Hide the console window when launching the background process
                    creation_flags = subprocess.CREATE_NO_WINDOW
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env, creationflags=creation_flags, cwd=cwd)
                    def read_stream(stream, output_queue, color=None):
                        try:
                            for line in iter(stream.readline, ''):
                                if line.strip():  # Only put non-empty lines
                                    output_queue.put((line, color))
                        except Exception:
                            if process.poll() is not None:
                                if any(p[0] == process for p in self.processes):
                                    self.processes.remove((process, creation_flags, name))
                                    self.display_output(f"\nProcess {name} terminated. \n", "red")

                    # Start threads to read stdout and stderr
                    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, output_queue), daemon=True)
                    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, output_queue, "yellow"), daemon=True)
        
                    stdout_thread.start()
                    stderr_thread.start()
                    self.processes.append((process, creation_flags, name))  # Add the process to the array
                    # Process the output queue
                    while True:
                        try:
                            line,  color = output_queue.get(timeout=0.1)  # Timeout allows checking process status
                            if self.terminate_flag:
                                break  # Exit loop if termination flag is set
                            decoded_line = line.decode('utf-8', errors='replace')  # Decode bytes to string
                            self.display_output(decoded_line, color=color)
                        except queue.Empty:
                            if process is not None:
                                if process.poll() is not None:
                                    break  # Exit when the process has finished
                            time.sleep(0.1)

                    # Wait for threads to finish
                    stdout_thread.join(timeout=2)
                    stderr_thread.join(timeout=2)

                    # Ensure streams are closed
                    if process.stdout:
                        process.stdout.close()
                    if process.stderr:
                        process.stderr.close()

            except Exception as e:
                self.display_output(f"\nError starting the application. Error: {str(e)}\n", "red")
        
        self.terminate_flag = False
        threading.Thread(target=run_process, args=(gameserver_name,self.gameport,0), daemon=True).start()
        threading.Thread(target=run_process, args=(appserver_name,self.boardport, 3), daemon=True).start()

    def stop_server(self):
        self.terminate_flag = True
        self.thread_running = False
        self.display_output("Stopping applications...\n", "green")
        time.sleep(0.5)

        try:
            for process, flags, name in list(self.processes):
                if process is not None:
                    if flags == subprocess.CREATE_NEW_CONSOLE:
                        # Detached process
                        self.display_output(f"Detached process: PID={process.pid} Please close it manually.\n", "yellow")
                        # Optionally, notify the user to close it manually
                    else:
                        if is_process_running(name):
                            # Non-detached process
                            self.display_output(f"Terminating process: PID={process.pid}\n", "green")
                            try:
                                process.terminate()  # Terminate the process
                                process.wait(timeout=1)  # Ensure it exits
                            except subprocess.TimeoutExpired:
                                    self.display_output(f"Process PID={process.pid} did not terminate in time. Killing it...\n", "yellow")
                                    process.kill()  # Forcefully terminate if needed
                            except Exception as e:
                                self.display_output(f"Error terminating process PID={process.pid}: {e}\n", "red")
                            finally:
                                if process.stdout:
                                    process.stdout.close()  # Avoid hanging
                                if process.stderr:
                                    process.stderr.close()
                        else:
                            self.display_output(f"Process {name} is not running.\n", "yellow")
        finally:
            self.processes.clear()  # Clear the list of processes

        self.display_output("Applications stopped\n", "green")
        self.ben_server_button.config(text="Start BEN Server and board manager", bg="green", fg="white")
        self.ben_server_running = False
        self.play_button.config(state="disabled", bg="red", fg="black")

    def on_play(self):
        webbrowser.open(f"http://localhost:{self.boardport}/play")

    def is_port_open(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0

    def check_ports_threaded(self):
        """Run port checks in a separate thread and update the UI accordingly."""
        self.thread_running = True
        def threaded_task():
            while self.thread_running:
                gameserver_running = is_process_running(gameserver_name)
                appserver_running = is_process_running(appserver_name)
                ports_open = all([
                    self.is_port_open("localhost", self.gameport),
                    self.is_port_open("localhost", self.boardport)
                ])
                self.update_ui(gameserver_running, appserver_running, ports_open)
                if gameserver_running and appserver_running:
                    time.sleep(10)
                else:
                    time.sleep(1)

        # Run the task in a separate thread
        threading.Thread(target=threaded_task, daemon=True).start()

    def update_ui(self, gameserver_running, appserver_running, ports_open):
        """Update the UI based on the port and process checks."""
        if gameserver_running and appserver_running and ports_open:
            self.ben_server_button.config(text="Stop BEN Servers", bg="red", fg="black")
            self.ben_server_running = True
        else:
            self.ben_server_button.config(text="Start BEN Server and board manager", bg="green", fg="white")
            self.ben_server_running = False

        # Play button configuration
        if self.ben_server_running:
            self.play_button.config(state="normal", bg="green", fg="white")
        else:
            self.play_button.config(state="disabled", bg="red", fg="black")

    def on_about(self):
        messagebox.showinfo("About", "Play with BEN. Version 0.8.7.0")

    def terminate(self, signum, frame):
        """
        Terminate the application and clean up resources.
        """
        self.on_exit()        

def splash_screen():
    splash = tk.Toplevel()
    splash.title("Splash Screen")
    splash.geometry("600x600")
    splash.overrideredirect(True)
    splash.configure(bg="white")

    # Display splash content
    image = tk.PhotoImage(file="logo.png")
    image_label = tk.Label(splash, image=image, bg="white")
    image_label.pack(expand=True)

    splash.image = image  # Store the image in the splash object to avoid garbage collection

    screen_width = splash.winfo_screenwidth()
    screen_height = splash.winfo_screenheight()
    x = (screen_width // 2) - (600 // 2)
    y = (screen_height // 2) - (600 // 2)
    splash.geometry(f"600x600+{x}+{y}")

    splash.update()
    return splash  # Keep a reference to the image

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window initially

    # Show splash screen
    splash = splash_screen()

    # Setup the main application while splash is displayed
    BridgeApp(root)

    # Destroy splash and show the main window
    splash.destroy()
    root.deiconify()

    root.mainloop()


if __name__ == "__main__":
    main()
