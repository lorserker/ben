import os
import sys
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import subprocess
import threading
import time
import queue
from colorama import Fore, Style, init

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
        self.title("Table Manager Interface v0.8")
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
        self.output_queue = queue.Queue()

    def validate_selection(self):
        seat = self.inputs["Seat:"].get()
        if seat == "":
            messagebox.showerror("Error", "Please select seat!")
            return False
        return True

    def create_widgets(self):
        # Frame for input fields
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=20)

        # Configure the columns 
        for i in range(4):
            input_frame.grid_columnconfigure(i, weight=1, minsize=20)

        # Create borders around each column for better visibility
        col_frames = []
        for i in range(4):
            col_frame = ttk.Frame(input_frame, relief="solid", borderwidth=1)
            col_frame.grid(row=0, column=i, rowspan=5, padx=2, pady=5, sticky="nswe")
            col_frames.append(col_frame)

        # Input fields (Host, Port, Name, Seat)
        fields = [
            ("Host:", ""),
            ("Port:", ""),
            ("Name:", "BEN"),
            ("Seat:", ""),
        ]
        self.inputs = {}

        for i, (label_text, default) in enumerate(fields):
            # Label for each input field (in column 0)
            label = ttk.Label(col_frames[0], text=label_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # Entry or Combobox for each input field (in column 1)
            if label_text == "Seat:":
                entry = ttk.Combobox(col_frames[1], values=["", "North", "East", "South", "West"])
                entry.set(default)
            else:
                entry = ttk.Entry(col_frames[1])
                entry.insert(0, default)

            entry.grid(row=i, column=0, padx=5, pady=5)
            self.inputs[label_text] = entry

        # Config field (Label and Entry in column 3)
        config_label = ttk.Label(col_frames[2], text="Config:")
        config_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        self.config_entry = ttk.Entry(col_frames[2], width=25)
        self.config_entry.insert(0, "config/default.conf")
        self.config_entry.grid(row=1, column=0, padx=5, pady=5)

        # Browse button for Config in column 3
        browse_button = ttk.Button(col_frames[2], text="Browse", command=self.browse_file)
        browse_button.grid(row=2, column=0, padx=5, pady=5)

        # Checkboxes (Bidding Only, Match Point) in column 3
        self.bidding_only = tk.BooleanVar(value=False)
        self.nosearch = tk.BooleanVar(value=False)
        self.matchpoint = tk.BooleanVar(value=False)

        ttk.Checkbutton(col_frames[2], text="Bidding Only", variable=self.bidding_only).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="No simulation", variable=self.nosearch).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="Match Point", variable=self.matchpoint).grid(row=5, column=0, sticky="w", padx=5, pady=5)

        # Verbose checkbox and buttons (Start/Stop) in column 4
        self.verbose = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[3], text="Verbose", variable=self.verbose).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Move Start and Stop buttons to column 4
        self.start_button = ttk.Button(col_frames[3], text="Start Application", command=self.start_application)
        self.start_button.grid(row=1, column=0, padx=5, pady=5)

        self.stop_button = ttk.Button(col_frames[3], text="Stop Application", command=self.stop_application, state="disabled")
        self.stop_button.grid(row=2, column=0, padx=5, pady=5)

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

    def browse_file(self):

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
            self.inputs["Name:"].delete(0, tk.END)
            self.inputs["Name:"].insert(0,  os.path.basename(rel_path).split(".")[0])

    def start_application(self):
        # Disable start button, enable stop button
        if not self.validate_selection():
            return
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")

        # Get command-line parameters
        host = self.inputs["Host:"].get()
        port = self.inputs["Port:"].get()
        name = self.inputs["Name:"].get()
        seat = self.inputs["Seat:"].get()
        config = self.config_entry.get()
        bidding_only = self.bidding_only.get()
        matchpoint = self.matchpoint.get()
        nosearch = self.nosearch.get()
        verbose = self.verbose.get()

        def read_output(stream, queue, color=None):
            while True:
                if stream and not stream.closed:
                    output = stream.readline()
                if not output:
                    break
                # Add the output to the queue with color
                queue.put((output, color))
            stream.close()


        def run_process():
            try:
                self.display_output("\nStarting application...\n", "green")
                # Check if table_manager_client.exe exists
                exe_path = 'table_manager_client.exe'
                # Hide the console window when launching the background process
                creation_flags = 0
                if sys.platform == "win32":
                    creation_flags = subprocess.CREATE_NO_WINDOW  # Prevent console window

                if os.path.exists(exe_path):
                    cmd = [exe_path]
                else:
                    # Fallback to running table_manager_client.py with python if exe is not found
                    cmd = [
                        "python", "table_manager_client.py"]

                # Add arguments conditionally
                if port:  
                    cmd.extend(["--host", str(host)])
                if port:  
                    cmd.extend(["--port", str(port)])
                if name:  
                    cmd.extend(["--name", str(name)])
                if seat:  
                    cmd.extend(["--seat", str(seat)])
                if config:
                    cmd.extend(["--config", config])
                if bidding_only:
                    cmd.extend(["--biddingonly", str(bidding_only)])
                if nosearch:
                    cmd.extend(["--nosearch", str(nosearch)])
                if matchpoint:
                    cmd.extend(["--matchpoint", str(matchpoint)])
                if verbose:
                    cmd.extend(["--verbose", str(verbose)])

                self.process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env, creationflags=creation_flags)
                
                # Start threads to read stdout and stderr
                stdout_thread = threading.Thread(target=read_output, args=(self.process.stdout, self.output_queue), daemon=True)
                stderr_thread = threading.Thread(target=read_output, args=(self.process.stderr, self.output_queue, "yellow"), daemon=True)
    
                stdout_thread.start()
                stderr_thread.start()

                # Process the output queue
                while True:
                    try:
                        line, color = self.output_queue.get(timeout=0.5)  # Timeout allows checking process status
                        decoded_line = line.decode('utf-8', errors='replace')  # Decode bytes to string
                        self.display_output(decoded_line, color=color)
                    except queue.Empty:
                        if self.process is not None:
                            if self.process.poll() is not None:
                                break  # Exit when the process has finished
                        time.sleep(0.1)

                # Wait for threads to finish
                stdout_thread.join()
                stderr_thread.join()

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

        threading.Thread(target=run_process, daemon=True).start()

    def stop_application(self):
        self.display_output("Stopping application...\n", "green")
        if self.process:
            self.process.terminate()
            self.process.stdout.close()  # Close the stdout pipe to avoid hanging
            self.process.stderr.close()  # Close the stderr pipe to avoid hanging
            self.process = None

        # Enable start button, disable stop button
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.display_output("Application stopped\n", "green")

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


if __name__ == "__main__":
    app = TableManagerApp()
    app.mainloop()
