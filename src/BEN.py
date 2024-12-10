import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import time
import subprocess
import webbrowser
import socket
import threading
import os

processes = []

class BridgeApp:
    def __init__(self, root):
        self.root = root
        self.root.iconbitmap("ben.ico")
        self.root.title("Bridge with BEN. v0.8.3")
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
        self.ben_server_button = tk.Button(root, text="Start BEN Server", bg="green", command=self.toggle_ben_server)
        self.ben_server_button.pack(pady=5)

        self.board_manager_button = tk.Button(root, text="Start Board Manager", bg="green", command=self.toggle_board_manager)
        self.board_manager_button.pack(pady=5)

        self.play_button = tk.Button(root, text="Play", bg="red", state="disabled", command=self.on_play)
        self.play_button.pack(pady=20)

        # Check servers periodically
        self.root.after(2000, self.update_buttons)

        # Handle close event
        root.protocol("WM_DELETE_WINDOW", self.on_exit)

    def on_exit(self):
        print("Stopping BEN")
        self.on_app_stop()
        self.root.quit()

    def on_app_stop(self):
        for process, stop_event in processes:
            stop_event.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()

    def toggle_ben_server(self):
        if self.ben_server_running:
            self.stop_server("BEN Server", 8080)
            self.ben_server_running = False
        else:
            self.start_server("gameserver.exe", 8080)
            self.ben_server_running = True

    def toggle_board_manager(self):
        if self.board_manager_running:
            self.stop_server("Board Manager", 4443)
            self.board_manager_running = False
        else:
            self.start_server("appserver.exe", 4443)
            self.board_manager_running = True

    def start_server(self, exe_name, port):
        stop_event = threading.Event()

        def run_server():
            cmd = [exe_name] if os.path.exists(exe_name) else ["python", exe_name.replace(".exe", '.py')]
            try:
                process = subprocess.Popen(cmd)
                processes.append((process, stop_event))
                stop_event.wait()
                if process.poll() is None:
                    process.terminate()
                    process.wait()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to start {exe_name}: {e}")

        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()

    def stop_server(self, name, port):
        for process, stop_event in processes:
            stop_event.set()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        print(f"{name} on port {port} stopped.")

    def on_play(self):
        webbrowser.open("http://localhost:8080/play")

    def is_port_open(self, host, port):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0

    def update_buttons(self):
        # Update BEN Server button
        if self.is_port_open("localhost", 4443):
            self.ben_server_button.config(text="Stop BEN Server", bg="red", fg="black")
            self.ben_server_running = True
        else:
            self.ben_server_button.config(text="Start BEN Server", bg="green", fg="white")
            self.ben_server_running = False

        # Update Board Manager button
        if self.is_port_open("localhost", 8080):
            self.board_manager_button.config(text="Stop Board Manager", bg="red", fg="black")
            self.board_manager_running = True
        else:
            self.board_manager_button.config(text="Start Board Manager", bg="green", fg="white")
            self.board_manager_running = False

        # Play button configuration
        if self.ben_server_running and self.board_manager_running:
            self.play_button.config(state="normal", bg="green", fg="white")
        else:
            self.play_button.config(state="disabled", bg="red", fg="black")

        self.root.after(2000, self.update_buttons)

    def on_about(self):
        messagebox.showinfo("About", "Play with BEN. Version 0.8")


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
    app = BridgeApp(root)

    # Destroy splash and show the main window
    splash.destroy()
    root.deiconify()

    root.mainloop()


if __name__ == "__main__":
    main()
