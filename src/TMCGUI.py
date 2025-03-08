import os
import sys
import re
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from functools import partial
import subprocess
import threading
import time
import queue
from colorama import Fore, Style, init
import pygame
from pygame.locals import QUIT
from pathlib import Path
from tmcgui.table import Table
from tmcgui.drawing_func import redraw_bidding, redraw_playing
from tmcgui.bid import Bid
from tmcgui.card import Card
from tmcgui.player import Player
import json
import win32gui
import win32con
import win32process
import psutil
import signal

# Initialize colorama for ANSI color handling on Windows
init()

# Set up environment for UTF-8 encoding
env = os.environ.copy()
env["PYTHONIOENCODING"] = "utf-8"
env["TERM"] = "xterm-256color"  # Set terminal type to force color output
env["BEN_HOME"] = "."
env['PATH'] = os.path.dirname(__file__) + ';' + os.environ['PATH']
CONFIG_FILE = "TMCGUI.settings.json"
# Constants
WIDTH, HEIGHT = 1200, 1000
WHITE = (255, 255, 255)
update_lock = threading.Lock()
direction_map = {
    "South": 0,
    "West": 1,
    "North": 2,
    "East": 3,
}

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

def global_error_handler(exception_type, exception_value, traceback, widget):
    """Handle uncaught exceptions in tkinter callbacks."""
    messagebox.showerror("Unexpected Error", f"An error occurred:\n{exception_value}\n")

# Override the default tkinter exception handling
tk.Tk.report_callback_exception = global_error_handler

class TableManagerApp(tk.Tk):
    def __init__(self):
        super().__init__()

        # Window configuration
        self.iconbitmap("ben.ico")
        self.title("Table Manager Interface. v0.8.6.1")
        self.geometry("880x750")  # Wider window size
        self.resizable(True, True)

        # Apply a modern ttk theme
        style = ttk.Style()
        style.theme_use("alt")

        # Load settings or set defaults
        self.settings = self.load_settings()

        # Change the background color of the root window
        self.configure(bg="green")  
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

        self.processes = []
        # Bind the close event of the Tkinter window
        self.protocol("WM_DELETE_WINDOW", self.on_exit)
        # Register signal handlers for termination
        signal.signal(signal.SIGINT, self.terminate)
        signal.signal(signal.SIGTERM, self.terminate)
        self.terminate_flag = False
        self.seats = None

        # Get the main window handle
        self.hwnd = win32gui.GetForegroundWindow()
        self.pygame_visible = False

    def setup_pygame(self):
        pygame.init()
        try:
            base_directory = Path(__file__).parent
        except:
            base_directory = os.getcwd()

        # Images
        self.icon = pygame.image.load(os.path.join(base_directory, "tmcgui/images/icon.png"))


        self.font = pygame.font.SysFont("Arial", 32)
        self.pygame_visible = False
        top_right_x, top_right_y = self.get_top_right_corner()
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{top_right_x+100},{top_right_y+100}"

    def validate_seat(self):
        seat = self.inputs["Seat:"].get()
        if seat == "":
            messagebox.showerror("Error", "Please select seat!")
            return False
        return True

    def validate_name(self):
        name = self.inputs["Name:"].get()
        if name == "":
            messagebox.showerror("Error", "Please enter name!")
            return False
        return True

    def validate_config(self):
        config = self.config_entry.get()
        if config == "":
            messagebox.showerror("Error", "Please select a configuration file!")
            return False
        elif not os.path.isfile(config):
            messagebox.showerror("Error", f"The file '{config}' does not exist! Use the browse button to select a valid file.")
            return False
        return True

    def create_widgets(self):
        # Frame for input fields
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=20)

        # Configure the columns 
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
            ("Host:", self.settings.get("host", "")),
            ("Port:", self.settings.get("port", "")),
            ("Name:", self.settings.get("name", "")),
            ("Seat:", ""),
        ]
        self.inputs = {}

        for i, (label_text, default) in enumerate(fields):
            # Label for each input field (in column 0)
            label = ttk.Label(col_frames[0], text=label_text)
            label.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # Entry or Combobox for each input field (in column 1)
            if label_text == "Seat:":
                entry = ttk.Combobox(col_frames[1], values=["", "North", "East", "South", "West", "NS", "EW", "NESW"])
                entry.set(default)
            else:
                entry = ttk.Entry(col_frames[1])
                entry.insert(0, default)

            entry.grid(row=i, column=0, padx=5, pady=5)
            self.inputs[label_text] = entry

        self.table_button = ttk.Button(col_frames[1], text="View Table", command=self.toggleTable)
        self.table_button.grid(row=6, column=0, padx=5, pady=15)

        # Config field (Label and Entry in column 3)
        config_label = ttk.Label(col_frames[2], text="Config:")
        config_label.grid(row=0, column=0, padx=5, pady=5, sticky="w")

        config_file = self.settings.get("config", "")
        self.config_entry = ttk.Entry(col_frames[2], width=25)
        self.config_entry.insert(0, config_file)
        self.config_entry.grid(row=1, column=0, padx=5, pady=5)

        # Browse button for Config in column 3
        browse_button = ttk.Button(col_frames[2], text="Browse", command=self.browse_file)
        browse_button.grid(row=2, column=0, padx=5, pady=5)

        # Checkboxes (Bidding Only, Match Point) in column 3
        self.bidding_only = tk.BooleanVar(value=self.settings.get("bidding_only", False))
        self.nosearch = tk.BooleanVar(value=self.settings.get("nosearch", False))
        self.matchpoint = tk.BooleanVar(value=self.settings.get("matchpoint", False))

        ttk.Checkbutton(col_frames[2], text="Bidding Only", variable=self.bidding_only).grid(row=3, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="No simulation", variable=self.nosearch).grid(row=4, column=0, sticky="w", padx=5, pady=5)
        ttk.Checkbutton(col_frames[2], text="Match Point", variable=self.matchpoint).grid(row=5, column=0, sticky="w", padx=5, pady=5)

        # Verbose checkbox and buttons (Start/Stop) in column 4
        self.verbose = tk.BooleanVar(value=False)
        ttk.Checkbutton(col_frames[3], text="Verbose", variable=self.verbose).grid(row=0, column=0, sticky="w", padx=5, pady=5)

        # Verbose checkbox and buttons (Start/Stop) in column 4
        self.detached_checkbox = tk.BooleanVar(value=False)
        # Callback function to enable or disable the button
        def update_button_state():
            if self.detached_checkbox.get():
                self.start_button.config(state="normal")  # Enable button
            else:
                self.start_button.config(state="normal")  # Enable button
                for process, flags in self.processes:
                    if process is not None:
                        if flags != subprocess.CREATE_NEW_CONSOLE:
                            self.start_button.config(state="disabled")  # Disable button

        ttk.Checkbutton(col_frames[3], text="Detached", variable=self.detached_checkbox,command=update_button_state).grid(row=1, column=0, sticky="w", padx=5, pady=5)

        # Move Start and Stop buttons to column 4
        self.start_button = ttk.Button(col_frames[3], text="Start BEN client", command=self.start_ben_client)
        self.start_button.grid(row=2, column=0, padx=5, pady=5)

        # Define a common width for all buttons
        button_width = 21
        
        self.stop_button = ttk.Button(col_frames[3], text="Stop BEN client", command=self.stop_application, state="disabled")
        self.stop_button.grid(row=3, column=0, padx=5, pady=5)

        # Create and place the buttons without storing them
        ttk.Button(col_frames[5], text="Start Table Manager", command=self.start_tm_window, width=button_width).grid(row=0, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start RoboBridge client", command=self.start_robobridge_window, width=button_width).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start WBridge5 client", command=self.start_wbridge5_window, width=button_width).grid(row=2, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start Q-Plus client", command=self.start_qplus_window, width=button_width).grid(row=3, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start Powershark client", command=self.start_shark_window, width=button_width).grid(row=4, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start Lia client", command=self.start_lia_window, width=button_width).grid(row=5, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start Blue Chip client", command=self.start_bc_window, width=button_width).grid(row=6, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start TM Mediator", command=self.start_tmmediator_window, width=button_width).grid(row=7, column=0, padx=5, pady=5)
        ttk.Button(col_frames[5], text="Start GIB client", command=self.start_gib_window, width=button_width).grid(row=8, column=0, padx=5, pady=5)

        # Add "Save Log" button
        self.save_log_button = ttk.Button(col_frames[3], text="Save Log", command=self.save_log)
        self.save_log_button.grid(row=4, column=0, padx=5, pady=5)
        
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

    def browse_executable(self, exe_path, target_entry):
            # Update last directory and file
        exe_directory = os.path.dirname(exe_path)
        exe_file = os.path.basename(exe_path)

        # Open file dialog to ask the user to select an executable
        exe_path = filedialog.askopenfilename(
            title="Select Executable",
            filetypes=[
                ("Executable and Command Files", "*.exe;*.cmd"),  # Include .exe and .cmd
                ("Executable Files", "*.exe"),
                ("Command Files", "*.cmd"),
                ("All Files", "*.*")
            ],
            initialdir=exe_directory,
            initialfile=exe_file
        )

        if not exe_path:  # If no file is selected, exit the function
            self.display_output("No executable selected.\n", "red")
            return

        # Ensure the selected file is either an executable or a command script
        if not (exe_path.lower().endswith(".exe") or exe_path.lower().endswith(".cmd")):
            self.display_output("The selected file is not an executable or command file.\n", "red")
            return
        
        # Place the selected file path in the target Entry field
        target_entry.delete(0, tk.END)  # Clear existing content
        target_entry.insert(0, exe_path)  # Insert new file path
    
    def browse_file(self):

        # Get the current path from the Config entry field
        current_path = self.config_entry.get()
        # Extract the directory
        initial_dir = os.path.dirname(current_path) if os.path.isdir(os.path.dirname(current_path)) else "config"

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
            self.settings["config"] = rel_path
            self.settings["name"] = self.inputs["Name:"].get()
            self.save_settings()


    def start_tm_window(self):
        # Create a new window (Toplevel)
        exe_path = self.settings.get("TM_file", "")

        modal_window = tk.Toplevel(self)
        modal_window.title("Start Table Manager")

        # Make the window modal
        modal_window.transient(self)  # Make it appear above the main window
        modal_window.grab_set()  # Prevent interaction with other windows
        
        # Get cursor position
        cursor_x = self.winfo_pointerx()
        cursor_y = self.winfo_pointery()

        # Set window position at cursor
        modal_window.geometry(f"+{cursor_x}+{cursor_y}")        

        # Introduction text
        introduction_text = "Start Table Manager, select boards and start the table. Then start the clients."
        tk.Label(modal_window, text=introduction_text, anchor="w", padx=10, pady=10).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )

        # Label above the entry field
        tk.Label(modal_window, text="Table Manager Executable:", anchor="w").grid(
            row=1, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w"
        )

        # Wider entry field for better readability
        entry1 = ttk.Entry(modal_window, width=50)  # Adjust width as needed
        entry1.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        entry1.insert(0, exe_path)

        # Browse button beside the entry field
        browse_button = ttk.Button(modal_window, text="Browse", command=lambda: self.browse_executable(exe_path, entry1))
        browse_button.grid(row=2, column=1, padx=10, pady=5)
        # Add a submit button
        def on_submit():
            # Get values from the entry fields
            exe_path = entry1.get()
            self.settings["TM_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()  # Close the modal window
            
        # Add a submit button, centered below the input
        submit_button = ttk.Button(modal_window, text="Start", command=on_submit)
        submit_button.grid(row=3, column=0, columnspan=3, pady=20)

        # Adjust modal window size and center it
        modal_window.update_idletasks()
        modal_window.geometry(
            f"{modal_window.winfo_width()}x{modal_window.winfo_height()}+"
            f"{modal_window.winfo_pointerx() - modal_window.winfo_width() // 2}+"
            f"{modal_window.winfo_pointery() - modal_window.winfo_height() // 2}"
        )

        # Wait for the modal window to be closed
        self.wait_window(modal_window)

    def create_modal_window(self, title, exe_setting_key, label_text, on_submit_callback, additional_fields=None, introduction_text=None):
        """Helper method to create a modal window."""
        exe_path = self.settings.get(exe_setting_key, "")

        # Create modal window
        modal_window = tk.Toplevel(self)
        modal_window.title(title)
        modal_window.transient(self)
        modal_window.grab_set()

        # Position window at cursor
        cursor_x = self.winfo_pointerx()
        cursor_y = self.winfo_pointery()
        offset_x = 50  # Adjust horizontal offset as needed
        offset_y = 30  # Adjust vertical offset as needed
        modal_window.geometry(f"+{cursor_x + offset_x}+{cursor_y + offset_y}")
        
        # Introduction text
        tk.Label(modal_window, text=introduction_text, anchor="w", padx=10, pady=10).grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        # Executable label and entry field
        tk.Label(modal_window, text=label_text, anchor="w").grid(
            row=1, column=0, columnspan=2, padx=10, pady=(10, 5), sticky="w"
        )
        entry = ttk.Entry(modal_window, width=60)
        entry.grid(row=2, column=0, padx=10, pady=5, sticky="w")
        entry.insert(0, exe_path)

        # Browse button
        ttk.Button(modal_window, text="Browse", 
                command=lambda: self.browse_executable(exe_path, entry)).grid(
            row=2, column=1, padx=10, pady=5
        )

        # Additional fields
        if additional_fields:
            for i, (label, field_type) in enumerate(additional_fields):
                tk.Label(modal_window, text=label, anchor="w").grid(
                    row=3 + i, column=0, padx=10, pady=5, sticky="w"
                )
                if field_type == 'entry':
                    ttk.Entry(modal_window, width=50).grid(
                        row=3 + i, column=1, padx=10, pady=5
                    )
                elif field_type == 'checkbox':
                    ttk.Checkbutton(modal_window).grid(
                        row=3 + i, column=1, padx=10, pady=5
                    )

        # Submit button
        submit_button = ttk.Button(modal_window, text="Start", command=lambda: on_submit_callback(entry, modal_window))
        submit_button.grid(row=4 + len(additional_fields) if additional_fields else 3, column=0, columnspan=2, pady=20)

        # Adjust and center window
        modal_window.update_idletasks()
        window_width = modal_window.winfo_width()
        window_height = modal_window.winfo_height()
        center_x = cursor_x + offset_x - window_width // 2
        center_y = cursor_y + offset_y - window_height // 2
        modal_window.geometry(f"{window_width}x{window_height}+{center_x}+{center_y}")
        self.wait_window(modal_window)

    def start_robobridge_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["Robo_file"] = exe_path
            # Additional logic for RoboBridge-specific parameters
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "It is recommended to create a command file with the commands to start RoboBridge clients."
        self.create_modal_window(
            "Start Robobridge",
            "Robo_file",
            "Robobridge Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_wbridge5_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["wbridge5_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "WBridge5 will connect to any empty seat starting from South. Only works with Bridge Moniteur"
        self.create_modal_window(
            "Start WBridge5",
            "wbridge5_file",
            "WBridge5 Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_qplus_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["QPlus_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "You will have to manually connect the clients to the table manager"
        self.create_modal_window(
            "Start Q-Plus Bridge",
            "QPlus_file",
            "Q-Plus Bridge Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_shark_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["Shark_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "You will have to manually connect the clients to the table manager"
        self.create_modal_window(
            "Start Powershark",
            "Shark_file",
            "Powershark Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_lia_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["Lia_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "You will have to manually connect the clients to the table manager"
        self.create_modal_window(
            "Start Lia",
            "Lia_file",
            "Lia Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_bc_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["BC_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "You will have to manually connect the clients to the table manager"
        self.create_modal_window(
            "Start Blue Chip",
            "BC_file",
            "Blue Chip Executable:",
            on_submit,
            introduction_text=introduction_text
        )
    def start_tmmediator_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["TMMediator_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "TMMediator is used for Blue Chip, and maps the Blue Chip Ports to the default table manager ports."
        self.create_modal_window(
            "Start TMMediator",
            "TMMediator_file",
            "TMMediator Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_gib_window(self):
        def on_submit(entry, modal_window):
            exe_path = entry.get()
            self.settings["GIB_file"] = exe_path
            self.save_settings()
            self.start_appl(exe_path)
            modal_window.destroy()

        introduction_text = "It is recommended to create a command file with the commands to start GIB clients."
        self.create_modal_window(
            "Start GIB",
            "GIB_file",
            "GIB Executable:",
            on_submit,
            introduction_text=introduction_text
        )

    def start_appl(self, exe_path, paramlist=[]):

        try:           
            # Get the directory of the executable
            exe_dir = os.path.dirname(exe_path)
            
            # Run the executable with 'cmd.exe' to keep the console open
            # cmd = ["cmd", "/K", exe_path]  # /K keeps the console open after execution
            cmd = [exe_path]  # /K keeps the console open after execution
            for param in paramlist:
                cmd.extend([param])

            # Start the selected application in detached mode (no output capture)
            creation_flags = subprocess.CREATE_NEW_CONSOLE
            subprocess.Popen(
                cmd,
                stdout=None,  # No output capture
                stderr=None,
                creationflags=creation_flags,
                cwd=exe_dir
            )

            self.display_output(f"Started application: {exe_path}\n", "green")

        except Exception as e:
            self.display_output(f"Error starting application: {str(e)}\n", "red")

    def bring_to_focus(self):
        if self.hwnd:
            win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)  # Restore if minimized
            win32gui.SetForegroundWindow(self.hwnd)  # Bring to foreground

    def start_ben_client(self):
        # Disable start button, enable stop button
        if not self.validate_name():
            return
        if not self.validate_seat():
            return
        if not self.validate_config():
            return
        self.start_button.config(state="disabled")
        self.table_button.config(state="disabled")
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

        self.reset_output_text()
        
        def run_process(seat, delay=None):
            try:
                output_queue = queue.Queue()
                time.sleep(delay)  # Wait for the specified delay if any
                self.display_output("\nStarting table manager client for seat " + str(seat) + "...\n", "green")
                # Check if table_manager_client.exe exists
                exe_path = 'table_manager_client.exe'

                if os.path.exists(exe_path):
                    cmd = [exe_path]
                else:
                    # Fallback to running table_manager_client.py with python if exe is not found
                    cmd = [
                        "python", "table_manager_client.py"]

                # Add arguments conditionally
                cmd.extend(["--name", str(name)])
                cmd.extend(["--seat", str(seat)])
                cmd.extend(["--matchpoint", str(matchpoint)])
                if host:  
                    cmd.extend(["--host", str(host)])
                if port:  
                    cmd.extend(["--port", str(port)])
                if config:
                    cmd.extend(["--config", config])
                if bidding_only:
                    cmd.extend(["--biddingonly", str(bidding_only)])
                if nosearch:
                    cmd.extend(["--nosearch", str(nosearch)])
                if verbose:
                    cmd.extend(["--verbose", str(verbose)])

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
                    self.display_output("Running in detached mode (new console opened).\n", "green")
                    self.processes.append((process, subprocess.CREATE_NEW_CONSOLE))  # Add the process to the array
                            # Bring the main app back to focus
                    self.after(100, self.bring_to_focus)
                else:
                    # Hide the console window when launching the background process
                    creation_flags = subprocess.CREATE_NO_WINDOW
                    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=False, env=env, creationflags=creation_flags)

                    def read_stream(stream, output_queue, seat, color=None):
                        try:
                            for line in iter(stream.readline, ''):
                                if line.strip():  # Only put non-empty lines
                                    output_queue.put((line, seat, color))
                        except (ValueError, BufferError) as e:
                            # Handle known, acceptable exceptions as normal process termination
                            self.display_output(f"\nProcess finished normally.\n", "green")
                        except Exception as e:
                            if process.poll() is not None:
                                if any(p[0] == process for p in self.processes):
                                    self.processes.remove((process, creation_flags))
                                    self.display_output(f"\nProcess terminated. {e}\n", "red")


                    # Start threads to read stdout and stderr
                    stdout_thread = threading.Thread(target=read_stream, args=(process.stdout, output_queue, seat, "green"), daemon=True)
                    stderr_thread = threading.Thread(target=read_stream, args=(process.stderr, output_queue, seat, "yellow"), daemon=True)
        
                    stdout_thread.start()
                    stderr_thread.start()

                    self.processes.append((process, creation_flags))  # Add the process to the array
                    # Process the output queue
                    while True:
                        try:
                            line, seat, color = output_queue.get(timeout=0.1)  # Timeout allows checking process status
                            if self.terminate_flag:
                                break  # Exit loop if termination flag is set
                            decoded_line = line.decode('utf-8', errors='replace')  # Decode bytes to string
                            self.display_output(decoded_line, color=color)
                            if self.pygame_visible:
                                try:
                                    with update_lock:  # Ensure only one thread updates the window at a time
                                        self.update_window(decoded_line, seat)
                                except Exception as e:
                                    self.display_output(f"\nError updating window: {str(e)}\n", "red")
                                    print(decoded_line)
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
            finally:
                # Toggle buttons back after termination
                self.start_button.config(state="normal")
                self.table_button.config(state="normal")
                self.stop_button.config(state="disabled")
        
        # Mapping for combined seats
        combined_seats = {
            "North": ["North"],
            "East": ["East"],
            "South": ["South"],
            "West": ["West"],
            "NS": ["North", "South"],
            "EW": ["East", "West"],
            "NESW": ["North", "East", "South", "West"]
        }
        self.seats = combined_seats[seat]
        self.terminate_flag = False
        # Expand combined parameters into individual calls
        for index, single_seat in enumerate(combined_seats[seat]):
            # Introduce a slight delay to stagger the start of each thread
            delay = 2 * (index)  # Increasing delay as threads are started
            threading.Thread(target=run_process, args=(single_seat,delay), daemon=True).start()


    def parse_hand(self,s):
        # Translate hand from Tablemanager format to pygame format
            # Define the suits
        suits = ['S', 'H', 'D', 'C']
        try:
            hand = s[s.index(':') + 1 : s.rindex('.')] \
                .replace(' ', '').replace('-', '').replace('S', '').replace('H', '').replace('D', '').replace('C', '')
            # Split the input string by '.'
            suit_parts = hand.split('.')
            
            # Initialize the result list
            cards = []
            
            # Iterate over each suit and its corresponding cards
            for suit, cards_in_suit in zip(suits, suit_parts):
                for card in cards_in_suit:
                    # Prepend the suit to each card and add to the result
                    cards.append(Card(f'{suit}{card.replace("A","14").replace("K","13").replace("Q","12").replace("J","11").replace("T","10")}'))
            
            return cards

        except Exception as ex:
            print("Parse hand",s)
            print(ex)
            print(f"Protocol error. Received {s} - Expected a hand")

    def update_window(self, text, seat):
        if not self.pygame_visible:
            return
        pattern = r'^\d{2}:\d{2}:\d{2}$'
        tokens = text.split()
        if len(tokens) < 2:
            return
        if not re.match(pattern, tokens[0]):
            return
        if "ready for" in text:
            return
        if "Playing" in text:
            return
        if "trick" in text:
            # Count cards in the 4 hands
            # print(len(self.table.board.west), self.table.board.west)
            # print(len(self.table.board.north), self.table.board.north)
            # print(len(self.table.board.east), self.table.board.east)
            # print(len(self.table.board.south), self.table.board.south)
            return
        #print(tokens)
        if "seated" in text:
            #print("Updating seating")
            seat = direction_map[tokens[2]]
            self.user = Player(tokens[3])
            # We are at the bottoom
            self.user.position = 0
            self.table.set_player(seat,self.user.username)
            #self.table.set_player(self.user.position,self.user.username)
            #redraw_sitting(self.screen, self.font, self.table, self.user)
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "Teams" in text:
            #print("Updating teams", text)
            ns_team = tokens[6].replace('"','')
            if '/' in ns_team:
                north, south = ns_team.split('/')
            else:
                north = south = ns_team
            self.table.set_player(0,south)
            self.table.set_player(2,north)

            ew_team = tokens[9].replace('"','')
            if '/' in ew_team:
                east, west = ew_team.split('/')
            else:
                west = east = ew_team
            self.table.set_player(1,west)
            self.table.set_player(3,east)
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "cards" in text:
            #print("Updating cards")
            hand = ' '.join(tokens[2:-1])
            hand = self.parse_hand(hand)
            player = tokens[2].replace("'s","")
            if player == "Dummy":
                #print("Cards for dummy")
                self.table.board.dummy_cards = hand               
                self.table.board.dummy = self.dummy_i
                self.table.board.dummy_visible = True
                if self.dummy_i == 0:
                    self.table.board.south = hand
                if self.dummy_i == 1:
                    self.table.board.west = hand
                if self.dummy_i == 2:
                    self.table.board.north = hand
                if self.dummy_i == 3:
                    self.table.board.east = hand
                redraw_playing(self.screen, self.font, self.table, self.user)
            else:
                player_id = direction_map[player]
                if player_id == 0:
                    self.table.board.south = hand
                if player_id == 1:
                    self.table.board.west = hand
                if player_id == 2:
                    self.table.board.north = hand
                if player_id == 3:
                    self.table.board.east = hand
                redraw_bidding(self.screen, self.font, self.table, self.user)
        # Board number 1. Dealer North. Neither vulnerable.  
        if "number" in text:
            if seat != self.seats[0]:
                return
            #print("Updating board")
            number = int(tokens[4].replace('.',''))
            self.table.next_board(number)
            dealer = tokens[6].replace('.','')
            self.table.board.dealer = direction_map[dealer]
            vuln = tokens[7]
            self.table.board.vulnerable[0] = vuln == "Both" or vuln == "N/S"
            self.table.board.vulnerable[1] = vuln == "Both" or vuln == "E/W"
            self.table.board.vulnerable[2] = vuln == "Both" or vuln == "N/S"
            self.table.board.vulnerable[3] = vuln == "Both" or vuln == "E/W"
            self.table.board.auction = ['PAD_START'] * (self.user.position-1)
            self.dummy_i = -1
            self.declarer_i = -1
            #print(number, dealer, vuln, self.table.board.auction)
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "doubles" in text:
            if seat != self.seats[0]:
                return
            #print("Updating bidding X")
            bid = 'X'
            self.table.board.bidding.append(Bid(bid))
            self.table.board.auction.append(bid)
            # We must check for bidding ended so wew can update contract
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "redoubles" in text:
            if seat != self.seats[0]:
                return
            #print("Updating bidding XX")
            bid = 'XX'
            self.table.board.bidding.append(Bid(bid))
            self.table.board.auction.append(bid)
            # We must check for bidding ended so wew can update contract
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "asses" in text:
            if seat != self.seats[0]:
                return
            #print("Updating bidding P")
            bid = 'PASS'
            self.table.board.bidding.append(Bid(bid))
            self.table.board.auction.append(bid)
            # We must check for bidding ended so wew can update contract
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "bids" in text:
            if seat != self.seats[0]:
                return
            bid = tokens[4].replace("NT",'N')
            #print("Updating bidding", bid)
            self.table.board.bidding.append(Bid(bid))
            self.table.board.auction.append(bid)
            redraw_bidding(self.screen, self.font, self.table, self.user)
        if "Bidding:" in text:
            if seat != self.seats[0]:
                return
            #print(text)
            self.contract = tokens[3]
            self.declarer_i = 'SWNE'.index(self.contract[-1])
            self.dummy_i = (self.declarer_i + 2) % 4
            if self.declarer_i % 2 == 0:
                self.table.board.winning_side = [0, 2]
            if self.declarer_i % 2 == 1:
                self.table.board.winning_side = [1, 3]
            self.table.board.winning_bid = self.contract[:-1]
            self.table.board.declarer = (self.declarer_i, self.table.board.winning_side, None)
            self.table.board.dummy = self.dummy_i
            redraw_playing(self.screen, self.font, self.table, self.user)

        if "plays" in text:
            if seat != self.seats[0]:
                return
            player = tokens[2]
            player_id = direction_map[player]
            card = tokens[4]
            #print("Updating play", card, player_id, self.dummy_i,self.user.position)
            card = card[::-1]
            card = Card(card.replace("A","14").replace("K","13").replace("Q","12").replace("J","11").replace("T","10"))

            if player_id == 0 and player_id != self.dummy_i and "South" not in self.seats:
                if len(self.table.board.south) > 0:
                    self.table.board.south.pop(0)
                self.table.board.south.append(card)
                #print(player_id, self.table.board.south)
            if player_id == 1 and player_id != self.dummy_i and "West" not in self.seats:
                if len(self.table.board.west) > 0:
                    self.table.board.west.pop(0)
                self.table.board.west.append(card)
                #print(player_id, self.table.board.west)
            if player_id == 2 and player_id != self.dummy_i  and "North" not in self.seats:
                if len(self.table.board.north) > 0:
                    self.table.board.north.pop(0)
                self.table.board.north.append(card)
                #print(player_id, self.table.board.north)
            if player_id == 3 and player_id != self.dummy_i  and "East" not in self.seats:
                if len(self.table.board.east) > 0:
                    self.table.board.east.pop(0)
                self.table.board.east.append(card)
                #print(player_id, self.table.board.east)
            #print("Updating play", card)
            self.table.board.turn = player_id
            self.table.board.make_move(card.symbol)
            redraw_playing(self.screen, self.font, self.table, self.user)

    def stop_application(self):
        self.terminate_flag = True
        self.display_output("Stopping application...\n", "green")
        time.sleep(3)

        try:
            for process, flags in self.processes:
                if process is not None:
                    if flags == subprocess.CREATE_NEW_CONSOLE:
                        # Detached process
                        self.display_output(f"Detached process: PID={process.pid} Please close it manually.\n", "yellow")
                        # Optionally, notify the user to close it manually
                    else:
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
        finally:
            self.processes.clear()  # Clear the list of processes

            # Enable start button, disable stop button
            self.start_button.config(state="normal")
            self.table_button.config(state="normal")
            self.stop_button.config(state="disabled")
            self.display_output("Application stopped\n", "green")
            self.processes = []  # Clear the array after stopping all processes

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

    def show_closing_popup(self):
        popup = tk.Toplevel()
        popup.title("Closing")

        # Get dimensions of the main application window
        root_x = self.winfo_rootx()
        root_y = self.winfo_rooty()
        root_width = self.winfo_width()
        root_height = self.winfo_height()

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
        self.terminate_flag = True
        self.running = False
        popup = self.show_closing_popup()
        time.sleep(0.5)
        if self.pygame_visible: 
            self.pygame_visible = False  # Update the state once the thread finishes
            threading.Thread(target=self.waitForPygameQuit).start()
        self.stop_application()
        self.settings["host"] = self.inputs["Host:"].get()
        self.settings["port"] = self.inputs["Port:"].get()
        self.settings["config"] = self.config_entry.get()
        # Checkboxes (Bidding Only, Match Point) in column 3
        self.settings["bidding_only"] =self.bidding_only.get()
        self.settings["nosearch"] =self.nosearch.get()
        self.settings["matchpoint"] =self.matchpoint.get()

        self.save_settings()
        # Hide the closing popup
        popup.destroy()
        self.quit()


    def save_log(self):
        try:
            # Get the content of the Text widget
            log_content = self.output_text.get("1.0", tk.END).strip()
            
            # Remove extra blank lines
            cleaned_content = "\n".join(line for line in log_content.splitlines() if line.strip())
            
            # Write the cleaned content to log.txt
            with open("log.txt", "w", encoding="utf-8") as file:
                file.write(cleaned_content)            
            # Show a confirmation message
            messagebox.showinfo("Save Log", "Log saved in log.txt")
        except Exception as e:
            # Show an error message in case of failure
            messagebox.showerror("Error", f"Failed to save log: {e}")

    def embed_pygame_window(self):
        # Create a frame for the pygame display
        self.pygame_frame = tk.Frame(self.root, width=640, height=480, bg='yellow')
        self.pygame_frame.pack(fill=tk.BOTH, expand=True)
        # Set up the pygame display window inside the tkinter frame
        os.environ['SDL_WINDOWID'] = str(self.pygame_frame.winfo_id())
        pygame.display.set_mode((self.screen_width, self.screen_height))

        # Get the position of the tkinter window and set pygame window beside it
        x_position = self.root.winfo_x() + self.root.winfo_width()
        y_position = self.root.winfo_y()

        # Set the pygame window's position relative to the tkinter window
        os.environ['SDL_VIDEO_WINDOW_POS'] = f"{x_position},{y_position}"
        # Initialize a clock to control the frame rate
        self.clock = pygame.time.Clock()

    def stopTable(self):
        # Setting running to false will stop the pygame loop
        self.running = False  # Stop the Pygame loop
        threading.Thread(target=self.waitForPygameQuit).start()

    def waitForPygameQuit(self):
        self.pygame_visible = False  # Update the state once the thread finishes

    def toggleTable(self):
        self.table_button.config(state="disabled")
        if self.pygame_visible:
            self.stopTable()      # Initialize and start the table
            self.table_button.config(text="View Table")
        else:
            self.startTable()      # Initialize and start the table
            self.table_button.config(text="Close table")
        self.table_button.config(state="normal")


    def startTable(self):

        # Check if pygame is already running
        if self.pygame_visible:
            return  # Avoid multiple instances

        self.setup_pygame()
        self.pygame_visible = True

        # Create and start a thread for the pygame loop
        pygame_thread = threading.Thread(target=self.runPygameLoop)
        pygame_thread.daemon = True  # Ensure it exits when the main app exits
        pygame_thread.start()


    def get_top_right_corner(self):
        """
        Retrieves the top-right corner coordinates of the main Tkinter window.
        Returns:
            (int, int): (x, y) coordinates of the top-right corner.
        """
        # Ensure the window is updated to get accurate coordinates
        self.update_idletasks()

        x = self.winfo_x()
        y = self.winfo_y()
        width = self.winfo_width()
        # Calculate top-right corner
        top_right_x = x + width
        top_right_y = y

        return top_right_x, top_right_y

    def terminate(self, signum, frame):
        """
        Terminate the application and clean up resources.
        """
        self.on_exit()        

    # Initialize Pygame
    def runPygameLoop(self):
        # Create the Pygame window
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Bridge with BEN")
        pygame.display.set_icon(self.icon)

        self.table = Table(0)
        # South is zero, and then clockwise
        self.user = Player("")
        self.user.position = 0
        board_no = -1
        self.running = True
        self.table.next_board(board_no)
        self.table.board.dealer = -1
        self.table.board.available_bids = {}
        redraw_bidding(self.screen, self.font, self.table, self.user)
        while self.running:

            # Handle events
            for event in pygame.event.get():
                if event.type == QUIT:  # Close window
                    self.running = False
                    self.pygame_visible = False
                    self.table_button.config(text="View Table")
            # Update the display
            pygame.display.flip()

        pygame.display.quit()
        pygame.quit()

if __name__ == "__main__":
    # Create the main tkinter window
    app = TableManagerApp()
    # Start the tkinter event loop
    app.mainloop()