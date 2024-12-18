import os
import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
import subprocess
import threading
import time
import queue
from colorama import Fore, Style, init
import sys
import signal

# Initialize colorama for ANSI color handling on Windows
init()

# Set up environment for UTF-8 encoding
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["TERM"] = "xterm-256color"  # Set terminal type to force color output
env["BEN_HOME"] = "."


class TableManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.iconbitmap("ben.ico")
        self.title("BEN server Interface. v0.8.4")
        self.geometry("880x750")  # Wider window size
        self.resizable(True, True)

        # Apply a modern ttk theme
        style = ttk.Style()
        style.theme_use("alt")

        # Change the background color of the root window
        self.configure(bg="green")  # Set to light gray or any desired color

        # ANSI color to Tkinter tag mapping
        self.ANSI_COLORS = {
            Fore.RED: {"foreground": "red"},
            Fore.GREEN: {"foreground": "green"},
            Fore.BLUE: {"foreground": "blue"},
            Fore.YELLOW: {"foreground": "gold"},
            Fore.CYAN: {"foreground": "cyan"},
            Fore.MAGENTA: {"foreground": "magenta"},
            Style.BRIGHT: {"font": ("TkDefaultFont", 10, "bold")},
            Style.RESET_ALL: {},  # Resets formatting
        }

        # Create UI elements
        self.create_widgets()

        # Configure color tags for the output Text widget
        for color, options in self.ANSI_COLORS.items():
            if options:
                self.output_text.tag_configure(color, **options)

        self.process = None
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        # Register signal handlers for termination
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)

        
    def create_widgets(self):
        # Frame for input fields
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=20)


        # Configure the columns (6 columns now)
        for i in range(6):
            input_frame.grid_columnconfigure(i, weight=1, minsize=20)

        # Create borders around each column for better visibility
        col_frames = []
        for i in range(6):
            col_frame = ttk.Frame(input_frame, relief="solid", borderwidth=1)
            col_frame.grid(row=0, column=i, rowspan=5, padx=2, pady=5, sticky="nswe")
            col_frames.append(col_frame)

        # Input fields (Host, Port, Name, Seat)
        fields = [
            ("Seed:", ""),
            ("BoardNo:", ""),
            ("Output PBN:", ""),
            ("Par Only:", ""),
            ("Port:", ""),
        ]
        self.inputs = {}

        for i, (label_text, default) in enumerate(fields):
            # Label for each input field (in column 0)
            label = ttk.Label(col_frames[0], text=label_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            entry = ttk.Entry(col_frames[1])
            entry.insert(0, default)

            entry.grid(row=i, column=0, padx=5, pady=5)
            self.inputs[label_text] = entry

        # Checkboxes (Bidding Only, Match Point) in column 3
        self.bidding_only = tk.BooleanVar(value=False)
        self.play_only = tk.BooleanVar(value=False)
        self.facit = tk.BooleanVar(value=False)
        self.matchpoint = tk.BooleanVar(value=False)

        ttk.Checkbutton(col_frames[2], text="Bidding Only", variable=self.bidding_only).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="Play Only", variable=self.play_only).grid(row=5, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="Facit", variable=self.facit).grid(row=6, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="Matchpoint", variable=self.matchpoint).grid(row=6, column=0, sticky="w", padx=5, pady=5)

        # Config field (Label and Entry in column 3)
        config_label = ttk.Label(col_frames[3], text="Config:")
        config_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.config_entry = ttk.Entry(col_frames[3], width=25)
        self.config_entry.insert(0, "config/default.conf")
        self.config_entry.grid(row=1, column=0, padx=5, pady=5)

        # Browse button for Config in column 3
        browse_button = ttk.Button(col_frames[3], text="Browse", command=self.browse_file)
        browse_button.grid(row=2, column=0, padx=5, pady=5)

        # Config field (Label and Entry in column 3)
        config_label = ttk.Label(col_frames[3], text="Boards:")
        config_label.grid(row=3, column=0, padx=5, pady=5, sticky="w")

        self.boards_entry = ttk.Entry(col_frames[3], width=25)
        self.boards_entry.insert(0, "")
        self.boards_entry.grid(row=4, column=0, padx=5, pady=5)

        # Browse button for Config in column 3
        browse_button = ttk.Button(col_frames[3], text="Browse", command=self.browse_boards)
        browse_button.grid(row=5, column=0, padx=5, pady=5)

        # Verbose checkbox and buttons (Start/Stop) in column 4
        self.verbose = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[4], text="Verbose", variable=self.verbose).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.auto = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[4], text="Auto", variable=self.auto).grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.api = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[4], text="API", variable=self.api).grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.server = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[4], text="Server", variable=self.server).grid(row=3, column=0, sticky="w", padx=5, pady=5)

        # Store the initial states of checkboxes
        self.previous_state = {
            'auto': False,
            'api': False,
            'server': False
        }

        # Add trace callbacks
        self.auto.trace_add("write", self.update_checkboxes)
        self.api.trace_add("write", self.update_checkboxes)
        self.server.trace_add("write", self.update_checkboxes)

        # Flag to avoid infinite loop
        self.updating = False

        # Move Start and Stop buttons to column 4
        self.start_button = ttk.Button(col_frames[4], text="Start Application", command=self.start_application)
        self.start_button.grid(row=4, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(col_frames[4], text="Stop Application", command=self.stop_application, state="disabled")
        self.stop_button.grid(row=5, column=0, padx=5, pady=5)

        self.next_button = ttk.Button(col_frames[5], text="Next", command=self.send_input, state="disabled")
        self.next_button.grid(row=1, column=0, padx=5, pady=5)

        # Output frame
        output_frame = ttk.Frame(self)
        output_frame.pack(fill="both", expand=True, pady=20, padx=20)

        output_label = ttk.Label(output_frame, text="Application Output:")
        output_label.pack(anchor="w")

        self.output_text = tk.Text(output_frame, wrap="word", state="disabled", height=16, width=80, bg="black", fg="white")
        self.output_text.pack(side="left", fill="both", expand=True)
        # Define color tags for ANSI escape codes
        self.output_text.tag_configure("red", foreground="#FF7F50")
        self.output_text.tag_configure("green", foreground="#90EE90")
        self.output_text.tag_configure("blue", foreground="#ADD8E6")
        self.output_text.tag_configure("yellow", foreground="#FFD700")
        self.output_text.tag_configure("reset", foreground="black")
        self.output_text.configure(bg="#303030", fg="white")

        # Scrollbar for the output
        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")

    def update_checkboxes(self, *args):
        # Check which checkbox triggered the update
        if self.auto.get() != self.previous_state['auto']:
            # If "Auto" has changed, uncheck the others
            if self.auto.get():
                self.api.set(False)
                self.server.set(False)
            self.previous_state['auto'] = self.auto.get()

        elif self.api.get() != self.previous_state['api']:
            # If "API" has changed, uncheck the others
            if self.api.get():
                self.auto.set(False)
                self.server.set(False)
            self.previous_state['api'] = self.api.get()

        elif self.server.get() != self.previous_state['server']:
            # If "Server" has changed, uncheck the others
            if self.server.get():
                self.auto.set(False)
                self.api.set(False)
            self.previous_state['server'] = self.server.get()

        self.updating = False  # Allow updates again
                    
    def browse_file(self):
        import os
        from tkinter import filedialog

        # Get the current path from the Config entry field
        current_path = self.config_entry.get()
        # Extract the directory
        initial_dir = os.path.dirname(current_path) if os.path.isdir(os.path.dirname(current_path)) else "."

        # Open file dialog with the initial directory
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir, 
            title="Select Configuration File", 
            filetypes=(("Configuration Files", "*.conf"), ("All Files", "*.*"))
        )
        if file_path:
            # Save relative path if possible
            rel_path = os.path.relpath(file_path, os.getcwd())
            self.config_entry.delete(0, tk.END)
            self.config_entry.insert(0, rel_path)

    def browse_boards(self):
        import os
        from tkinter import filedialog

        # Get the current path from the Config entry field
        current_path = self.config_entry.get()
        # Extract the directory
        initial_dir = os.path.dirname(current_path) if os.path.isdir(os.path.dirname(current_path)) else "."

        # Open file dialog with the initial directory
        file_path = filedialog.askopenfilename(
            initialdir=initial_dir, 
            title="Select Boards File", 
            filetypes=(("Configuration Files", "*.pbn"), ("All Files", "*.*"))
        )
        if file_path:
            # Save relative path if possible
            rel_path = os.path.relpath(file_path, os.getcwd())
            self.boards_entry.delete(0, tk.END)
            self.boards_entry.insert(0, rel_path)

    def start_application(self):
        # Disable start button, enable stop button
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.next_button.config(state="normal")

        boardno = self.inputs["BoardNo:"].get()
        seed = self.inputs["Seed:"].get()
        outputpbn = self.inputs["Output PBN:"].get()
        paronly = self.inputs["Par Only:"].get()
        port = self.inputs["Port:"].get()

        # Get command-line parameters
        boards = self.boards_entry.get()
        auto = self.auto.get()
        config = self.config_entry.get()
        playonly = self.play_only.get()
        bidding_only = self.bidding_only.get()
        facit = self.facit.get()
        matchpoint = self.matchpoint.get()
        verbose = self.verbose.get()
        api = self.api.get()
        server = self.server.get()

        self.reset_output_text()

        def run_process():
            try:
                output_queue = queue.Queue()
                self.display_output("\nStarting application...\n", "green")
                # Hide the console window when launching the background process
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = subprocess.CREATE_NO_WINDOW  # Prevent console window
                # Check if table_manager_client.exe exists
                if server:
                    exe_path = 'gameserver.exe'
                    script = "gameserver.py"
                else:
                    if api:
                        exe_path = 'gameapi.exe'
                        script = "gameapi.py"
                    else:
                        exe_path = 'game.exe'
                        script = "game.py"
                if os.path.exists(exe_path):
                    cmd = [exe_path] 
                else:
                    # Fallback to running table_manager_client.py with python if exe is not found
                    cmd = ["python", script]

                # Add arguments conditionally
                cmd.extend(["--matchpoint", str(matchpoint)])
                if boards:
                    cmd.extend(["--boards", boards])  
                if auto:
                    cmd.extend(["--auto", str(auto)])  # Include only if True
                if boardno:  # Only include if not zero or empty
                    cmd.extend(["--boardno", str(boardno)])
                if config:
                    cmd.extend(["--config", config])
                if playonly:
                    cmd.extend(["--playonly", str(playonly)])
                if bidding_only:
                    cmd.extend(["--biddingonly", str(bidding_only)])
                if outputpbn:  # Only include if not empty
                    cmd.extend(["--outputpbn", outputpbn])
                if paronly:  # Only include if not zero or empty
                    cmd.extend(["--paronly", str(paronly)])
                if facit:
                    cmd.extend(["--facit", str(facit)])
                if verbose:
                    cmd.extend(["--verbose", str(verbose)])
                if seed:  # Include seed only if it's not the default value
                    cmd.extend(["--seed", str(seed)])
                if port: 
                    cmd.extend(["--port", str(port)])

                self.process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env, creationflags=creation_flags)

                def read_stream(stream, output_queue, color=None):
                    try:
                        for line in iter(stream.readline, ''):
                            if line.strip():  # Only put non-empty lines
                                output_queue.put((line, color))
                    except Exception:
                        if self.process.poll() is not None:
                            self.process = None
                            self.display_output(f"\nProcess terminated. \n", "red")

                # Start threads to read stdout and stderr
                stdout_thread = threading.Thread(target=read_stream, args=(self.process.stdout, output_queue), daemon=True)
                stderr_thread = threading.Thread(target=read_stream, args=(self.process.stderr, output_queue, "yellow"), daemon=True)
    
                stdout_thread.start()
                stderr_thread.start()

                # Process the output queue
                while True:
                    try:
                        line, color = output_queue.get(timeout=0.5)  # Timeout allows checking process status
                        decoded_line = line.decode('utf-8', errors='replace')  # Decode bytes to string
                        self.display_output(decoded_line, color=color)
                    except queue.Empty:
                        if self.process is not None:
                            if self.process.poll() is not None:
                                break  # Exit when the process has finished
                        time.sleep(0.1)

                # Wait for threads to finish
                stdout_thread.join(timeout=2)
                stderr_thread.join(timeout=2)

                # Ensure streams are closed
                if self.process.stdout:
                    self.process.stdout.close()
                if self.process.stderr:
                    self.process.stderr.close()

                # Handle process exit status
                if self.process.returncode != 0:
                    self.display_output(f"\nApplication terminated with error code: {self.process.returncode}\n", "red")
            except Exception as e:
                self.display_output(f"\nError starting the application: {e}\n", "red")
            finally:
                self.process = None
                # Toggle buttons back after termination
                self.start_button.config(state="normal")
                self.stop_button.config(state="disabled")
                self.next_button.config(state="disabled")

        threading.Thread(target=run_process, daemon=True).start()

    def send_input(self):
        user_input = "X"
        if user_input:
            self.process.stdin.write(f"{user_input}\n".encode())
            self.process.stdin.flush()

    def stop_application(self):
        self.display_output("Stopping application...\n", "green")
        if self.process:
            self.process.terminate()
            #self.process.stdout.close()  # Close the stdout pipe to avoid hanging
            #self.process.stderr.close()  # Close the stderr pipe to avoid hanging
            self.process = None

        # Enable start button, disable stop button
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.display_output("Application stopped\n", "green")

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
        self.output_text.configure(state="normal")  # Enable editing temporarily
        if color and color in self.output_text.tag_names():
            self.output_text.insert("end", text, color)  # Apply the color tag
        else:
            self.output_text.insert("end", text)  # Default color
        self.output_text.configure(state="disabled")  # Disable editing
        self.output_text.see("end")  # Scroll to the end

    def terminate(self, signum, frame):
        """
        Terminate the application and clean up resources.
        """
        self.on_exit()        

    def on_exit(self):
        print("Closing application...")
        #popup = self.show_closing_popup()
        #self.save_settings()
        self.stop_application()
        # Hide the closing popup
        #popup.destroy()
        self.quit()

if __name__ == "__main__":
    app = TableManagerApp()
    app.mainloop()
