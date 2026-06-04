"""Shared UI shell, theme, and dialogs for the LungInsight annotator."""
import json
import os
import tkinter as tk
from datetime import datetime
from tkinter import ttk, messagebox

APP_VERSION = "1.7"
CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".lunginjury", "config.json")

BRAND = {
    "bg": "#F8FAFC",
    "header": "#1B4965",
    "accent": "#2A9D8F",
    "text": "#1D3557",
    "muted": "#64748B",
}

FEATURE_DISPLAY = {
    "Neutrophils": "#00FF00",
    "Hyaline Membranes": "#FF0000",
    "Proteinaceous Debris": "#0000FF",
}

PLOT_COLORS = ["#1B4965", "#2A9D8F", "#E9C46A", "#E76F51", "#264653", "#457B9D"]


class LungInsightUIMixin:
    """UI layout, theme, status bar, loading overlay, and dialogs."""

    def setup_theme(self):
        self.style = ttk.Style(self.root)
        if "clam" in self.style.theme_names():
            self.style.theme_use("clam")
        font_family = "Segoe UI" if os.name == "nt" else "DejaVu Sans"
        self.ui_font = (font_family, 11)
        self.ui_font_bold = (font_family, 11, "bold")
        self.ui_font_title = (font_family, 14, "bold")
        self.style.configure(".", font=self.ui_font, background=BRAND["bg"])
        self.style.configure("TFrame", background=BRAND["bg"])
        self.style.configure("Header.TFrame", background=BRAND["header"])
        self.style.configure("Header.TLabel", background=BRAND["header"], foreground="white", font=self.ui_font_title)
        self.style.configure("HeaderSub.TLabel", background=BRAND["header"], foreground="#CAE9FF", font=self.ui_font)
        self.style.configure("Sidebar.TFrame", background="#E8EEF4")
        self.style.configure("SidebarTitle.TLabel", background="#E8EEF4", foreground=BRAND["text"], font=self.ui_font_bold)
        self.style.configure("Status.TLabel", background="#E2E8F0", foreground=BRAND["text"], font=(font_family, 10))
        self.style.configure("Accent.TButton", font=self.ui_font_bold)
        self.style.configure("Horizontal.TProgressbar", thickness=8)
        self.root.configure(bg=BRAND["bg"])

    def configure_root_window(self):
        self.root.title(f"LungInsight Annotator v{APP_VERSION}")
        self.root.geometry("1400x900")
        self.root.minsize(1100, 700)
        self._center_window(self.root, 1400, 900)

    def _center_window(self, window, width, height):
        window.update_idletasks()
        sw = window.winfo_screenwidth()
        sh = window.winfo_screenheight()
        x = max(0, (sw - width) // 2)
        y = max(0, (sh - height) // 2)
        window.geometry(f"{width}x{height}+{x}+{y}")

    def build_app_shell(self):
        self.shell = ttk.Frame(self.root)
        self.shell.pack(fill=tk.BOTH, expand=True)
        self.header_frame = ttk.Frame(self.shell, style="Header.TFrame")
        self.header_frame.pack(fill=tk.X)
        title_row = ttk.Frame(self.header_frame, style="Header.TFrame")
        title_row.pack(fill=tk.X, padx=16, pady=(10, 2))
        ttk.Label(title_row, text="LungInsight Annotator", style="Header.TLabel").pack(side=tk.LEFT)
        ttk.Label(
            title_row, text=f"v{APP_VERSION}", style="HeaderSub.TLabel"
        ).pack(side=tk.LEFT, padx=(8, 0))
        meta_row = ttk.Frame(self.header_frame, style="Header.TFrame")
        meta_row.pack(fill=tk.X, padx=16, pady=(0, 10))
        self.header_user_label = ttk.Label(meta_row, text="", style="HeaderSub.TLabel")
        self.header_user_label.pack(side=tk.LEFT)
        self.header_progress_label = ttk.Label(meta_row, text="", style="HeaderSub.TLabel")
        self.header_progress_label.pack(side=tk.LEFT, padx=(20, 0))
        self.header_mouse_label = ttk.Label(meta_row, text="", style="HeaderSub.TLabel")
        self.header_mouse_label.pack(side=tk.RIGHT)
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_bar = ttk.Progressbar(
            self.shell, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, padx=16, pady=(4, 0))
        body = ttk.Frame(self.shell)
        body.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.sidebar = ttk.Frame(body, style="Sidebar.TFrame", width=240)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 8))
        self.sidebar.pack_propagate(False)
        ttk.Label(self.sidebar, text="Controls", style="SidebarTitle.TLabel").pack(
            anchor=tk.W, padx=12, pady=(12, 6)
        )
        self.sidebar_inner = ttk.Frame(self.sidebar, style="Sidebar.TFrame")
        self.sidebar_inner.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.content_frame = ttk.Frame(body)
        self.content_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.legend_frame = ttk.Frame(body, width=160)
        self.legend_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(8, 0))
        self.legend_frame.pack_propagate(False)
        ttk.Label(self.legend_frame, text="Legend", style="SidebarTitle.TLabel").pack(
            anchor=tk.W, padx=8, pady=(12, 4)
        )
        self.legend_labels = {}
        for name, hex_color in FEATURE_DISPLAY.items():
            row = ttk.Frame(self.legend_frame)
            row.pack(fill=tk.X, padx=8, pady=2)
            swatch = tk.Canvas(row, width=14, height=14, highlightthickness=1, highlightbackground="#CBD5E1")
            swatch.pack(side=tk.LEFT)
            swatch.create_rectangle(1, 1, 13, 13, fill=hex_color, outline="")
            ttk.Label(row, text=name, font=(self.ui_font[0], 9)).pack(side=tk.LEFT, padx=6)
            self.legend_labels[name] = row
        self.coord_count_label = ttk.Label(self.legend_frame, text="Boxes: 0", font=(self.ui_font[0], 9))
        self.coord_count_label.pack(anchor=tk.W, padx=8, pady=(12, 0))
        status_frame = ttk.Frame(self.shell)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(status_frame, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.pack(side=tk.LEFT, fill=tk.X, padx=12, pady=6)
        self.feature_swatch = tk.Canvas(status_frame, width=16, height=16, highlightthickness=0)
        self.feature_swatch.pack(side=tk.RIGHT, padx=(0, 4), pady=6)
        self.status_feature_label = ttk.Label(status_frame, text="", style="Status.TLabel")
        self.status_feature_label.pack(side=tk.RIGHT, padx=(0, 12), pady=6)
        self.loading_frame = ttk.Frame(self.content_frame)
        self.loading_label = ttk.Label(self.loading_frame, text="Working…")
        self.loading_label.pack(pady=8)
        self.loading_bar = ttk.Progressbar(self.loading_frame, mode="indeterminate", length=320)
        self.loading_bar.pack(pady=4)
        self._update_feature_status()

    def set_status(self, message, notify=False):
        self.status_var.set(message)
        if notify:
            self.root.bell()

    def show_loading(self, message="Downloading from Google Drive…"):
        self.loading_label.config(text=message)
        self.loading_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)
        self.loading_bar.start(12)
        self.root.update_idletasks()

    def hide_loading(self):
        self.loading_bar.stop()
        self.loading_frame.place_forget()

    def update_chrome(self):
        total = len(self.image_list) or 1
        idx = min(self.image_index + 1, total) if self.image_list else 0
        pct = ((self.image_index) / total) * 100 if total else 0
        self.progress_var.set(pct)
        self.header_user_label.config(text=f"Observer: {self.user_name or '—'}")
        self.header_progress_label.config(text=f"Tile {idx} / {total}")
        gene = self.current_image_info.get("gene", "—") if self.current_image_info else "—"
        name = self.current_image_info.get("name", "") if self.current_image_info else ""
        short = (name[:36] + "…") if len(name) > 39 else name
        self.header_mouse_label.config(text=f"{gene} · {short}" if short else gene)
        self.coord_count_label.config(text=f"Boxes: {len(self.rectangles)}")
        self._update_feature_status()

    def _update_feature_status(self):
        feature = self.feature_type.get() if hasattr(self, "feature_type") else "Neutrophils"
        hex_color = FEATURE_DISPLAY.get(feature, "#888888")
        self.feature_swatch.delete("all")
        self.feature_swatch.create_rectangle(0, 0, 16, 16, fill=hex_color, outline=BRAND["muted"])
        self.status_feature_label.config(text=feature)

    def build_feature_chips(self, parent):
        ttk.Label(parent, text="Feature", style="SidebarTitle.TLabel").pack(anchor=tk.W, pady=(8, 4))
        chip_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        chip_frame.pack(fill=tk.X, pady=(0, 8))
        self.feature_chip_buttons = {}
        for name, color in FEATURE_DISPLAY.items():
            btn = tk.Button(
                chip_frame,
                text=name,
                fg="white",
                bg=color,
                activebackground=color,
                relief=tk.FLAT,
                padx=6,
                pady=4,
                command=lambda n=name: self._select_feature(n),
            )
            btn.pack(fill=tk.X, pady=2)
            self.feature_chip_buttons[name] = btn
        self.feature_type.trace_add("write", lambda *_: self._highlight_feature_chip())

    def _select_feature(self, name):
        self.feature_type.set(name)
        self._update_feature_status()

    def _highlight_feature_chip(self):
        current = self.feature_type.get()
        for name, btn in self.feature_chip_buttons.items():
            if name == current:
                btn.config(relief=tk.SUNKEN, bd=2)
            else:
                btn.config(relief=tk.FLAT, bd=0)

    def build_toolbar(self, parent):
        ttk.Label(parent, text="Actions", style="SidebarTitle.TLabel").pack(anchor=tk.W, pady=(8, 4))
        self.button_frame = ttk.Frame(parent, style="Sidebar.TFrame")
        self.button_frame.pack(fill=tk.X)
        self.continue_button = ttk.Button(
            self.button_frame, text="Process (P)", command=self.on_continue, style="Accent.TButton"
        )
        self.next_button = ttk.Button(self.button_frame, text="Next (N)", command=self.load_next_image)
        self.variability_button = ttk.Button(
            self.button_frame, text="Variability plots", command=self.generate_variability_plots
        )
        for btn in (self.continue_button, self.next_button, self.variability_button):
            btn.pack(fill=tk.X, pady=3)

    def bind_shortcuts(self):
        self.root.bind("<p>", lambda e: self.on_continue())
        self.root.bind("<P>", lambda e: self.on_continue())
        self.root.bind("<n>", lambda e: self.load_next_image())
        self.root.bind("<N>", lambda e: self.load_next_image())
        self.root.bind("<s>", lambda e: self._shortcut_save())
        self.root.bind("<S>", lambda e: self._shortcut_save())
        self.root.bind("<e>", lambda e: self._shortcut_edit())
        self.root.bind("<E>", lambda e: self._shortcut_edit())
        self.root.bind("<Key-1>", lambda e: self._select_feature("Neutrophils"))
        self.root.bind("<Key-2>", lambda e: self._select_feature("Hyaline Membranes"))
        self.root.bind("<Key-3>", lambda e: self._select_feature("Proteinaceous Debris"))

    def _shortcut_save(self):
        if hasattr(self, "save_button") and self.save_button.winfo_exists():
            self.on_save()

    def _shortcut_edit(self):
        if hasattr(self, "edit_button") and self.edit_button.winfo_exists():
            self.on_edit()

    def load_remembered_username(self):
        try:
            if os.path.exists(CONFIG_PATH):
                with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return data.get("user_name")
        except (OSError, json.JSONDecodeError):
            pass
        return None

    def save_remembered_username(self, username):
        try:
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w", encoding="utf-8") as f:
                json.dump({"user_name": username}, f)
        except OSError:
            pass

    def get_username(self):
        remembered = self.load_remembered_username()
        dialog = tk.Toplevel(self.root)
        dialog.title("Welcome to LungInsight")
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=BRAND["bg"])
        self._center_window(dialog, 480, 320)
        tk.Label(
            dialog,
            text="Lung Injury Tile Annotator",
            font=self.ui_font_title,
            fg=BRAND["text"],
            bg=BRAND["bg"],
        ).pack(pady=(20, 6))
        tk.Label(
            dialog,
            text="Enter your observer name (lowercase). Progress syncs to Google Drive.",
            wraplength=420,
            justify=tk.CENTER,
            fg=BRAND["muted"],
            bg=BRAND["bg"],
            font=self.ui_font,
        ).pack(pady=(0, 12))
        entry = ttk.Entry(dialog, width=36, font=self.ui_font)
        entry.pack(pady=4)
        if remembered:
            entry.insert(0, remembered)
        remember_var = tk.BooleanVar(value=bool(remembered))
        ttk.Checkbutton(dialog, text="Remember me on this computer", variable=remember_var).pack(pady=8)
        result = []

        def on_ok():
            username = entry.get().strip().lower()
            if not username:
                messagebox.showwarning("Name required", "Please enter your observer name.", parent=dialog)
                return
            if remember_var.get():
                self.save_remembered_username(username)
            result.append(username)
            dialog.destroy()

        btn_row = ttk.Frame(dialog)
        btn_row.pack(pady=12)
        ttk.Button(btn_row, text="Continue", command=on_ok, style="Accent.TButton").pack()
        entry.bind("<Return>", lambda _: on_ok())
        entry.focus_set()
        dialog.wait_window()
        return result[0] if result else None

    def confirm_dialog(self, title, message):
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.transient(self.root)
        dialog.grab_set()
        dialog.configure(bg=BRAND["bg"])
        self._center_window(dialog, 420, 180)
        ttk.Label(dialog, text=message, wraplength=380).pack(padx=20, pady=20)
        choice = {"ok": False}

        def yes():
            choice["ok"] = True
            dialog.destroy()

        row = ttk.Frame(dialog)
        row.pack(pady=8)
        ttk.Button(row, text="OK", command=yes, style="Accent.TButton").pack(side=tk.LEFT, padx=6)
        ttk.Button(row, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=6)
        dialog.wait_window()
        return choice["ok"]
