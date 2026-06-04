import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
import io
import webbrowser
from datetime import datetime
from io import BytesIO
import plotly.graph_objects as go
import plotly.io as pio
from concurrent.futures import ThreadPoolExecutor
from ui_shell import LungInsightUIMixin, BRAND, PLOT_COLORS
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from google.oauth2 import service_account
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.discovery import build
import json
import shutil
import uuid
import logging

class CloudImageApp(LungInsightUIMixin):
    def __init__(self, root):
        self.root = root
        self.configure_root_window()
        self.setup_theme()
        self.feature_type = tk.StringVar(value="Neutrophils")
        self.feature_type.trace_add("write", lambda *_: self._on_feature_changed())
        self.last_save_time = None
        self.edit_zoom = 1.0
        self.edit_pan_x = 0
        self.edit_pan_y = 0
        self._pan_start = None
        self.feature_colors = {
            "Neutrophils": (0, 255, 0),
            "Hyaline Membranes": (255, 0, 0),
            "Proteinaceous Debris": (0, 0, 255)
        }
        self.service_account_file = "lunginsightcloud-fa31002e7988.json"
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.input_folder_id = "1kTVr2h11XlnV3xntxjZbPNZebJ8vr5SX"
        self.output_folder_id = "1XrfiMR4nLvKb2kx7MiwwBfdZlpOmT9ub"
        self.coordinates_folder_id = "1XrfiMR4nLvKb2kx7MiwwBfdZlpOmT9ub"
        self.interobplt_thresh = 1
        self.current_image_info = {}
        self.rectangles = []
        self.image_index = 0
        self.image_list = []
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect_id = None
        self.mode = tk.StringVar(value="Add")
        self.current_image = None
        self.current_feature = "Neutrophils"
        self.image_processed = False
        self.upload_images = False
        self.user_name = self.get_username()
        if not self.user_name:
            self.root.quit()
            return
        self.drive_service = self.initialize_drive_service()
        self.pydrive = self.initialize_pydrive()
        self.folder_id_cache = {}  # Cache for folder/file IDs
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.final_dir = os.path.join(self.temp_dir, "final")
        self.state_dir = os.path.join(self.temp_dir, "state")
        self.coords_dir = os.path.join(self.temp_dir, "coordinates")
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.coords_dir, exist_ok=True)
        self.build_app_shell()
        self.bind_shortcuts()
        self.setup_initial_ui()
        self.show_loading("Loading image list from Google Drive…")
        self.root.update_idletasks()
        try:
            self.load_cloud_images()
        finally:
            self.hide_loading()
        self.set_status("Connected to Google Drive")
        if self.load_state():
            if self.image_index >= len(self.image_list):
                self.image_index = 0
        else:
            self.image_index = self.recover_last_index()
        if self.image_list:
            self.check_and_load_image()
        else:
            messagebox.showerror("Error", "No images found in cloud folder")
            self.root.quit()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_chrome()

    def _on_feature_changed(self):
        if hasattr(self, "status_var"):
            self._update_feature_status()

    def _event_to_image_coords(self, x, y):
        zoom = self.edit_zoom
        cx = (x - self.edit_pan_x) / zoom
        cy = (y - self.edit_pan_y) / zoom
        image = cv2.imread(os.path.join(self.processed_dir, self.current_image_info["name"]))
        original_height, original_width = image.shape[:2]
        scale_x = original_width / 1280
        scale_y = original_height / 512
        return int(cx * scale_x), int(cy * scale_y)

    def initialize_pydrive(self):
        gauth = GoogleAuth()
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            self.service_account_file, self.scopes)
        return GoogleDrive(gauth)

    def setup_initial_ui(self):
        for widget in self.sidebar_inner.winfo_children():
            widget.destroy()
        for widget in self.content_frame.winfo_children():
            if widget != self.loading_frame:
                widget.destroy()
        self.build_feature_chips(self.sidebar_inner)
        self.build_toolbar(self.sidebar_inner)
        self._highlight_feature_chip()
        self.image_label = ttk.Label(self.content_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.edit_mode = False

    def save_final_image(self):
        try:
            self.update_coordinates_file()
            final_path = os.path.join(self.final_dir, self.current_image_info['name'])
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            if not os.path.exists(processed_path):
                processed_path = os.path.join(self.temp_dir, self.current_image_info['name'])
            if os.path.exists(processed_path):
                image = cv2.imread(processed_path)
                for rect in self.rectangles:
                    x1, y1, x2, y2, class_name = rect
                    color = self.feature_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, class_name, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(final_path, image)
                self.image_processed = True
                self.clear_edit_widgets()
                self.setup_initial_ui()
                self.display_image(final_path)
                self.last_save_time = datetime.now()
                self.set_status(f"Edits saved at {self.last_save_time.strftime('%H:%M:%S')}")
                self.update_chrome()
            else:
                messagebox.showerror("Error", "No source image found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save edited image: {str(e)}")

    def setup_edit_ui(self):
        for widget in self.sidebar_inner.winfo_children():
            widget.destroy()
        for widget in self.content_frame.winfo_children():
            if widget != self.loading_frame:
                widget.destroy()
        self.edit_mode = True
        self.edit_zoom = 1.0
        self.edit_pan_x = 0
        self.edit_pan_y = 0
        ttk.Label(self.sidebar_inner, text="Edit mode", style="SidebarTitle.TLabel").pack(anchor=tk.W, pady=(8, 4))
        self.mode_label_name = ttk.Entry(self.sidebar_inner, width=28)
        self.mode_label_name.insert(0, f"{self.current_image_info['name']}")
        self.mode_label_name.config(state="readonly")
        self.mode_label_name.pack(fill=tk.X, pady=4)
        ttk.Label(self.sidebar_inner, text="Mode", style="SidebarTitle.TLabel").pack(anchor=tk.W, pady=(8, 2))
        self.add_radio = ttk.Radiobutton(
            self.sidebar_inner, text="Add (draw box)", variable=self.mode, value="Add", command=self.update_mode
        )
        self.remove_radio = ttk.Radiobutton(
            self.sidebar_inner, text="Remove (click box)", variable=self.mode, value="Remove", command=self.update_mode
        )
        self.add_radio.pack(anchor=tk.W)
        self.remove_radio.pack(anchor=tk.W, pady=(0, 8))
        self.build_feature_chips(self.sidebar_inner)
        ttk.Label(self.sidebar_inner, text="Scroll wheel: zoom · Middle-drag: pan", font=(self.ui_font[0], 9)).pack(
            anchor=tk.W, pady=4
        )
        self.finalize_button = ttk.Button(
            self.sidebar_inner, text="Save edits (S)", command=self.save_final_image, style="Accent.TButton"
        )
        self.finalize_button.pack(fill=tk.X, pady=8)
        canvas_frame = ttk.Frame(self.content_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        self.edit_canvas = tk.Canvas(
            canvas_frame, width=1280, height=512, bg="#F1F5F9",
            highlightthickness=1, highlightbackground="#94A3B8", cursor="crosshair"
        )
        self.edit_canvas.pack(padx=8, pady=8)
        self.edit_canvas.bind("<MouseWheel>", self._on_edit_wheel)
        self.edit_canvas.bind("<Button-2>", self._on_pan_start)
        self.edit_canvas.bind("<B2-Motion>", self._on_pan_move)
        self.edit_canvas.bind("<ButtonRelease-2>", self._on_pan_end)
        self.edit_canvas.bind("<Button-4>", lambda e: self._on_edit_wheel_delta(1))
        self.edit_canvas.bind("<Button-5>", lambda e: self._on_edit_wheel_delta(-1))

    def save_state(self):
        try:
            state = {
                'user_name': self.user_name,
                'image_index': self.image_index,
                'current_image_info': self.current_image_info
            }
            state_file_path = os.path.join(self.state_dir, 'app_state.json')
            with open(state_file_path, 'w') as f:
                json.dump(state, f)
            user_folder_id = self.create_or_get_user_folder()
            query = f"'{user_folder_id}' in parents and name='app_state.json' and trashed=false"
            existing_files = self.list_all_files(q=query, fields="files(id)")
            if existing_files:
                file_id = existing_files[0]['id']
                media = MediaIoBaseUpload(open(state_file_path, 'rb'), mimetype='application/json')
                self.drive_service.files().update(fileId=file_id, media_body=media).execute()
            else:
                file_metadata = {'name': 'app_state.json', 'parents': [user_folder_id]}
                media = MediaIoBaseUpload(open(state_file_path, 'rb'), mimetype='application/json')
                self.drive_service.files().create(body=file_metadata, media_body=media).execute()
        except Exception as e:
            logging.error(f"Failed to save state to cloud: {str(e)}")

    def initialize_drive_service(self):
        try:
            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(f"Service account file {self.service_account_file} not found")
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=self.scopes)
            if not credentials:
                raise ValueError("Failed to load credentials from service account file")
            return build('drive', 'v3', credentials=credentials)
        except Exception as e:
            messagebox.showerror("Authentication Error", f"Failed to initialize Google Drive service: {str(e)}")
            self.root.quit()
            raise

    def load_state(self):
        try:
            user_folder_id = self.create_or_get_user_folder()
            query = f"'{user_folder_id}' in parents and name='app_state.json' and trashed=false"
            state_files = self.list_all_files(q=query, fields="files(id)")
            state_file_path = os.path.join(self.state_dir, 'app_state.json')
            if state_files:
                state_file_id = state_files[0]['id']
                if self.download_from_drive(state_file_id, state_file_path):
                    with open(state_file_path, 'r') as f:
                        state = json.load(f)
                    saved_image_info = state.get('current_image_info', {})
                    if saved_image_info:
                        for img in self.image_list:
                            if img['id'] == saved_image_info.get('id'):
                                self.user_name = state.get('user_name', self.user_name)
                                self.image_index = state.get('image_index', 0)
                                if self.image_index >= len(self.image_list):
                                    self.image_index = 0
                                    return False
                                return True
            if os.path.exists(state_file_path):
                with open(state_file_path, 'r') as f:
                    state = json.load(f)
                saved_image_info = state.get('current_image_info', {})
                if saved_image_info:
                    for img in self.image_list:
                        if img['id'] == saved_image_info.get('id'):
                            self.user_name = state.get('user_name', self.user_name)
                            self.image_index = state.get('image_index', 0)
                            if self.image_index >= len(self.image_list):
                                self.image_index = 0
                                return False
                            return True
            self.image_index = 0
            return False
        except Exception as e:
            logging.error(f"Failed to load state: {str(e)}")
            self.image_index = 0
            return False

    def recover_last_index(self):
        try:
            user_folder_id = self.create_or_get_user_folder()
            mouse_folders = self.list_all_files(q=f"'{user_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false", fields="files(id, name)")
            last_index = 0
            for mouse_folder in mouse_folders:
                files = self.list_all_files(q=f"'{mouse_folder['id']}' in parents and trashed=false", fields="files(name)")
                for file in files:
                    if file['name'].endswith('_coords.txt'):
                        image_name = file['name'].replace('_coords.txt', '')
                        for i, img in enumerate(self.image_list):
                            if img['name'] == image_name and i > last_index:
                                last_index = i
            return last_index
        except Exception as e:
            logging.error(f"Failed to recover last index: {str(e)}")
            return 0

    def check_and_load_image(self):
        while self.image_index < len(self.image_list):
            self.current_image_info = self.image_list[self.image_index]
            if self.check_existing_annotations():
                self.image_index += 1
                self.image_processed = False
                self.rectangles = []
                if self.image_index >= len(self.image_list):
                    messagebox.showinfo("Complete", f"All {len(self.image_list)} images already annotated!")
                    self.root.quit()
                    return
                self.save_state()
            else:
                self.load_image()
                break
        if self.image_index >= len(self.image_list):
            messagebox.showinfo("Complete", f"No unannotated images found among {len(self.image_list)} tiles!")
            self.root.quit()

    def check_existing_annotations(self):
        try:
            user_folder_id = self.create_or_get_user_folder()
            mouse_name = self.current_image_info['gene']
            mouse_folder_id = self.create_or_get_mouse_folder(mouse_name, user_folder_id)
            image_name = self.current_image_info['name']
            coord_name = f"{os.path.splitext(image_name)[0]}_coords.txt"
            image_query = f"name='{image_name}' and '{mouse_folder_id}' in parents and trashed=false"
            image_files = self.list_all_files(q=image_query, fields="files(id)")
            coord_query = f"name='{coord_name}' and '{mouse_folder_id}' in parents and trashed=false"
            coord_files = self.list_all_files(q=coord_query, fields="files(id)")
            input_folders = self.list_all_files(q=f"'{self.input_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false", fields="files(id, name)")
            image_in_input = False
            for folder in input_folders:
                if folder['name'] == mouse_name:
                    image_query = f"name='{image_name}' and '{folder['id']}' in parents and trashed=false"
                    input_files = self.list_all_files(q=image_query, fields="files(id)")
                    if input_files:
                        image_in_input = True
                        break
            return len(image_files) > 0 and len(coord_files) > 0 and image_in_input
        except Exception as e:
            logging.error(f"Error checking annotations for {self.current_image_info['name']}: {str(e)}")
            return False

    def create_output_folder(self):
        try:
            user_query = f"name='{self.user_name}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            user_folders = self.list_all_files(q=user_query, fields="files(id,name)")
            if not user_folders:
                user_metadata = {
                    'name': self.user_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [self.output_folder_id]
                }
                user_folder = self.drive_service.files().create(
                    body=user_metadata, 
                    fields='id,name'
                ).execute()
                user_folder_id = user_folder['id']
            else:
                user_folder_id = user_folders[0]['id']
            original_folder = self.current_image_info['gene']
            folder_query = f"name='{original_folder}' and '{user_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            existing_folders = self.list_all_files(q=folder_query, fields="files(id,name)")
            if not existing_folders:
                folder_metadata = {
                    'name': original_folder,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [user_folder_id]
                }
                new_folder = self.drive_service.files().create(
                    body=folder_metadata, 
                    fields='id,name'
                ).execute()
                return new_folder['id']
            return existing_folders[0]['id']
        except Exception as e:
            logging.error(f"Failed to create output folder: {str(e)}")
            raise

    def create_or_get_folder(self, folder_name, parent_id):
        query = f"'{parent_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        existing = self.list_all_files(q=query, fields="files(id)")
        if existing:
            return existing[0]['id']
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
        return folder['id']

    def list_all_files(self, **kwargs):
        files = []
        page_token = None
        while True:
            response = self.drive_service.files().list(**kwargs, pageToken=page_token, pageSize=1000).execute()
            files.extend(response.get('files', []))
            page_token = response.get('nextPageToken')
            if not page_token:
                break
        return files

    def load_cloud_images(self):
        try:
            folders = self.list_all_files(q=f"'{self.input_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false", fields="files(id, name)")
            self.image_list = []
            for folder in folders:
                images = self.list_all_files(q=f"'{folder['id']}' in parents and trashed=false", fields="files(id, name, mimeType)")
                for img in images:
                    if img['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_list.append({
                            'id': img['id'],
                            'name': img['name'],
                            'gene': folder['name']
                        })
            if not self.image_list:
                raise FileNotFoundError("No images found in cloud folder")
        except Exception as e:
            messagebox.showerror("Cloud Error", f"Failed to load image list: {str(e)}")

    def download_from_drive(self, file_id, destination_path):
        try:
            file = self.pydrive.CreateFile({'id': file_id})
            file.GetContentFile(destination_path)
            return True
        except Exception as e:
            logging.error(f"Failed to download file {file_id} to {destination_path}: {str(e)}")
            return False

    def upload_or_update(self, file_path, file_name, parent_folder_id):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist")
            query = f"'{parent_folder_id}' in parents and name='{file_name}' and trashed=false"
            existing = self.list_all_files(q=query, fields="files(id)")
            if existing:
                file_id = existing[0]['id']
                file = self.pydrive.CreateFile({'id': file_id})
                file.SetContentFile(file_path)
                file.Upload()
            else:
                file = self.pydrive.CreateFile({
                    'title': file_name,
                    'parents': [{'id': parent_folder_id}]
                })
                file.SetContentFile(file_path)
                file.Upload()
            return True
        except Exception as e:
            logging.error(f"Upload error for {file_name}: {str(e)}")
            messagebox.showerror("Upload Error", f"Failed to upload {file_name}: {str(e)}")
            return False

    def load_image(self):
        self.current_image_info = self.image_list[self.image_index]
        temp_image_path = os.path.join(self.temp_dir, self.current_image_info['name'])
        self.feature_type.set("Neutrophils")
        self._highlight_feature_chip()
        self.show_loading("Downloading tile…")
        self.root.update_idletasks()
        try:
            ok = self.download_from_drive(self.current_image_info['id'], temp_image_path)
        finally:
            self.hide_loading()
        if ok:
            self.current_image_path = temp_image_path
            self.display_image(temp_image_path)
            coord_name = f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt"
            user_folder_id = self.create_or_get_user_folder()
            mouse_folder_id = self.create_or_get_mouse_folder(self.current_image_info['gene'], user_folder_id)
            query = f"name='{coord_name}' and '{mouse_folder_id}' in parents and trashed=false"
            files = self.list_all_files(q=query, fields="files(id)")
            if files:
                coord_path = os.path.join(self.coords_dir, coord_name)
                self.show_loading("Downloading coordinates…")
                self.root.update_idletasks()
                try:
                    if self.download_from_drive(files[0]['id'], coord_path):
                        self.rectangles = self.load_coordinates(self.current_image_info['name'])
                finally:
                    self.hide_loading()
            self.update_chrome()
            self.set_status(f"Loaded {self.current_image_info['name']}")
        else:
            messagebox.showerror("Error", f"Failed to download image: {self.current_image_info['name']}")

    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            max_w = max(self.content_frame.winfo_width() - 32, 900)
            max_h = max(self.content_frame.winfo_height() - 32, 400)
            image.thumbnail((max_w, max_h), Image.LANCZOS)
            self.image_tk = ImageTk.PhotoImage(image)
            if hasattr(self, "image_label") and self.image_label.winfo_exists():
                self.image_label.config(image=self.image_tk)
            self.update_chrome()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def on_continue(self):
        self.current_feature = self.feature_type.get()
        if self.current_feature == "Neutrophils":
            self.process_neutrophils()
        elif self.current_feature == "Hyaline Membranes":
            self.process_hyaline_membranes()
        elif self.current_feature == "Proteinaceous Debris":
            self.process_proteinaceous_debris()
        else:
            messagebox.showerror("Error", "Unknown feature type selected")
            return
        self.feature_type.set("Neutrophils")

    def process_neutrophils(self):
        try:
            tile = cv2.imread(self.current_image_path)
            image_intact = tile.copy()
            tile[tile > 220] = 255
            gray_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray_tile, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            mask = np.zeros_like(gray_tile)
            cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
            internal_mask = cv2.bitwise_not(mask)
            internal_only = cv2.bitwise_and(thresh, thresh, mask=internal_mask)
            internal_contours, _ = cv2.findContours(internal_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                if 300 < area < 900 and 0.5 < circularity < 1:
                    x, y, w, h = cv2.boundingRect(contour)
                    k = 1.05
                    padding = int(k * np.sqrt(area))
                    center_x = x + w // 2
                    center_y = y + h // 2
                    radius = padding
                    radius = min(radius, center_x, center_y, tile.shape[1] - center_x, tile.shape[0] - center_y)
                    neighborhood_mask = np.zeros_like(thresh, dtype=np.uint8)
                    cv2.circle(neighborhood_mask, (center_x, center_y), radius, 255, thickness=-1)
                    neighborhood = cv2.bitwise_and(thresh, thresh, mask=neighborhood_mask)
                    total_pixels = cv2.countNonZero(neighborhood_mask)
                    light_areas_mask = cv2.inRange(tile, (200, 200, 200), (255, 255, 255))
                    neighborhood_light = cv2.bitwise_and(light_areas_mask, light_areas_mask, mask=neighborhood_mask)
                    white_pixels = cv2.countNonZero(neighborhood_light)
                    white_percentage = white_pixels / total_pixels
                    score = self.calculate_score(area, circularity, white_percentage)
                    if score < 0.15:
                        continue
                    color = self.feature_colors["Neutrophils"]
                    cv2.rectangle(tile, (x, y), (x + w, y + h), color, 2)
                    score_text = f"{score * 100:.2f}%"
                    text_position = (x, y - 10)
                    cv2.putText(tile, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    self.rectangles.append((x, y, x + w, y + h, "Neutrophils"))
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            cv2.imwrite(processed_path, tile)
            self.update_coordinates_file()
            self.display_image(processed_path)
            self.show_post_processing_options()
        except Exception as e:
            messagebox.showerror("Processing Error", f"Neutrophil processing failed: {str(e)}")

    def process_hyaline_membranes(self):
        try:
            tile = cv2.imread(self.current_image_path)
            image_intact = tile.copy()
            hsv_tile = cv2.cvtColor(tile, cv2.COLOR_BGR2HSV)
            lower_pink = np.array([140, 50, 50])
            upper_pink = np.array([170, 255, 255])
            pink_mask = cv2.inRange(hsv_tile, lower_pink, upper_pink)
            kernel = np.ones((5, 5), np.uint8)
            pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_CLOSE, kernel)
            pink_mask = cv2.morphologyEx(pink_mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(pink_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                if area < 500 or perimeter == 0:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                elongation = max(w, h) / min(w, h) if min(w, h) > 0 else 0
                mask = np.zeros_like(pink_mask)
                cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
                mean_color = cv2.mean(hsv_tile, mask=mask)[:3]
                hue_score = 1.0 if 140 <= mean_color[0] <= 170 else 0.5
                score = self.calculate_hyaline_score(area, elongation, hue_score)
                if score < 0.3:
                    continue
                color = self.feature_colors.get("Hyaline Membranes", (0, 255, 255))
                cv2.rectangle(tile, (x, y), (x + w, y + h), color, 2)
                score_text = f"{score * 100:.2f}%"
                text_position = (x, y - 10)
                cv2.putText(tile, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                self.rectangles.append((x, y, x + w, y + h, "Hyaline Membranes"))
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            cv2.imwrite(processed_path, tile)
            self.update_coordinates_file()
            self.display_image(processed_path)
            self.show_post_processing_options()
        except Exception as e:
            messagebox.showerror("Processing Error", f"Hyaline membrane processing failed: {str(e)}")

    def process_proteinaceous_debris(self):
        try:
            original_path = os.path.join(self.temp_dir, self.current_image_info['name'])
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            if os.path.exists(original_path):
                shutil.copy2(original_path, processed_path)
                self.current_feature = "Proteinaceous Debris"
                self.display_image(processed_path)
                self.show_post_processing_options()
            else:
                messagebox.showerror("Error", "Original image not found")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to prepare for editing: {str(e)}")

    def calculate_score(self, area, circularity, white_percentage):
        area_score = min(max((area - 100) / (1000 - 100), 0), 1)
        circularity_score = min(max((circularity - 0.48) / (1 - 0.48), 0), 1)
        white_percentage_score = min(max((white_percentage - 0.05) / (1 - 0.05), 0), 1)
        score = 0.15 * area_score + 0.7 * circularity_score + 0.15 * white_percentage_score
        return score
    
    def calculate_hyaline_score(self, area, elongation, hue_score):
        area_score = min(max((area - 500) / (5000 - 500), 0), 1)
        elongation_score = min(max((elongation - 2) / (10 - 2), 0), 1)
        score = 0.4 * area_score + 0.4 * elongation_score + 0.2 * hue_score
        return score

    def update_coordinates_file(self):
        coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
        with open(coord_file, "w") as file:
            for rect in self.rectangles:
                x1, y1, x2, y2, class_name = rect
                file.write(f"{x1},{y1},{x2},{y2},{class_name}\n")

    def load_coordinates(self, image_name):
        coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(image_name)[0]}_coords.txt")
        rectangles = []
        if os.path.exists(coord_file):
            with open(coord_file, "r") as file:
                for line in file:
                    parts = line.strip().split(',')
                    if len(parts) == 5:
                        x1, y1, x2, y2 = map(int, parts[:4])
                        class_name = parts[4]
                        color = self.feature_colors.get(class_name, (0, 255, 0))
                        rectangles.append((x1, y1, x2, y2, class_name))
        return rectangles

    def show_post_processing_options(self):
        self.continue_button.pack_forget()
        self.next_button.pack_forget()
        self.variability_button.pack_forget()
        self.save_button = ttk.Button(self.button_frame, text="Save (S)", command=self.on_save, style="Accent.TButton")
        self.edit_button = ttk.Button(self.button_frame, text="Edit (E)", command=self.on_edit)
        self.save_button.pack(fill=tk.X, pady=3)
        self.edit_button.pack(fill=tk.X, pady=3)
        self.set_status("Processing complete — save or edit annotations")

    def on_save(self):
        try:
            final_path = os.path.join(self.final_dir, self.current_image_info['name'])
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            if os.path.exists(processed_path):
                image = cv2.imread(processed_path)
                for rect in self.rectangles:
                    x1, y1, x2, y2, class_name = rect
                    color = self.feature_colors.get(class_name, (0, 255, 0))
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, class_name, (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.imwrite(final_path, image)
                self.update_coordinates_file()
                self.image_processed = True
                self.last_save_time = datetime.now()
                self.setup_initial_ui()
                self.display_image(processed_path)
                self.set_status(
                    f"Saved locally at {self.last_save_time.strftime('%H:%M:%S')} — use Next to upload to Drive"
                )
            else:
                messagebox.showerror("Error", "No processed image found. Please process the image first.")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def finalize_and_upload(self):
        try:
            if not self.image_processed:
                messagebox.showwarning("Warning", "Image not processed. Please process and save before uploading.")
                return
            self.update_coordinates_file()
            final_path = os.path.join(self.final_dir, self.current_image_info['name'])
            coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
            if not os.path.exists(final_path):
                messagebox.showerror("Error", f"Final image not found: {self.current_image_info['name']}. Please save the image first.")
                return
            if not os.path.exists(coord_file):
                with open(coord_file, "w") as f:
                    pass
            user_folder_id = self.create_or_get_user_folder()
            mouse_name = self.current_image_info['gene']
            mouse_folder_id = self.create_or_get_mouse_folder(mouse_name, user_folder_id)
            coords_subfolder_id = self.create_or_get_folder('coords', mouse_folder_id)
            if self.upload_images:
                images_subfolder_id = self.create_or_get_folder('images', mouse_folder_id)
            if not self.upload_or_update(coord_file, os.path.basename(coord_file), coords_subfolder_id):
                return
            if self.upload_images:
                if not self.upload_or_update(final_path, self.current_image_info['name'], images_subfolder_id):
                    coord_query = f"name='{os.path.basename(coord_file)}' and '{coords_subfolder_id}' in parents and trashed=false"
                    existing_coord = self.list_all_files(q=coord_query, fields="files(id)")
                    if existing_coord:
                        self.drive_service.files().delete(fileId=existing_coord[0]['id']).execute()
                    messagebox.showerror("Upload Error", "Image upload failed; rolled back coordinate upload.")
                    return
            self.image_processed = False
            self.set_status(f"Uploaded {self.current_image_info.get('name', 'tile')} to Drive")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload to cloud: {str(e)}")

    def load_next_image(self):
        if self.current_image_info:
            self.finalize_and_upload()
        self.image_index += 1
        self.image_processed = False
        self.rectangles = []
        if self.image_index >= len(self.image_list):
            for i, image_info in enumerate(self.image_list):
                self.current_image_info = image_info
                if not self.check_existing_annotations():
                    self.image_index = i
                    self.save_state()
                    self.check_and_load_image()
                    return
            messagebox.showinfo("Complete", "All images processed!")
            self.root.quit()
            return
        self.save_state()
        self.check_and_load_image()

    def verify_folder_structure(self):
        try:
            user_query = f"'{self.output_folder_id}' in parents and name='{self.user_name}' and mimeType='application/vnd.google-apps.folder'"
            user_folders = self.list_all_files(q=user_query)
            if not user_folders:
                return False
            mouse_query = f"'{user_folders[0]['id']}' in parents and mimeType='application/vnd.google-apps.folder'"
            mouse_folders = self.list_all_files(q=mouse_query)
            return True
        except Exception as e:
            return False

    def create_or_get_user_folder(self):
        try:
            query = f"name='{self.user_name}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.list_all_files(q=query, fields="files(id)")
            if existing:
                return existing[0]['id']
            folder_metadata = {
                'name': self.user_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [self.output_folder_id]
            }
            folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
            return folder['id']
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create user folder: {str(e)}")
            raise

    def create_or_get_mouse_folder(self, mouse_name, parent_folder_id):
        try:
            mouse_query = f"name='{mouse_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing_mouse = self.list_all_files(q=mouse_query, fields="files(id)")
            if existing_mouse:
                mouse_folder_id = existing_mouse[0]['id']
            else:
                folder_metadata = {
                    'name': mouse_name,
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [parent_folder_id]
                }
                mouse_folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
                mouse_folder_id = mouse_folder['id']
            coords_query = f"name='coords' and '{mouse_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing_coords = self.list_all_files(q=coords_query, fields="files(id)")
            if not existing_coords:
                coords_metadata = {
                    'name': 'coords',
                    'mimeType': 'application/vnd.google-apps.folder',
                    'parents': [mouse_folder_id]
                }
                self.drive_service.files().create(body=coords_metadata, fields='id').execute()
            if self.upload_images:
                images_query = f"name='images' and '{mouse_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                existing_images = self.list_all_files(q=images_query, fields="files(id)")
                if not existing_images:
                    images_metadata = {
                        'name': 'images',
                        'mimeType': 'application/vnd.google-apps.folder',
                        'parents': [mouse_folder_id]
                    }
                    self.drive_service.files().create(body=images_metadata, fields='id').execute()
            return mouse_folder_id
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create mouse folder: {str(e)}")
            raise

    def on_edit(self):
        processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        if not os.path.exists(processed_path):
            original_path = os.path.join(self.temp_dir, self.current_image_info['name'])
            if os.path.exists(original_path):
                shutil.copy2(original_path, processed_path)
        self.enter_edit_mode()

    def enter_edit_mode(self):
        self.setup_edit_ui()
        processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        image_path = processed_path if os.path.exists(processed_path) else os.path.join(self.temp_dir, self.current_image_info['name'])
        if os.path.exists(image_path):
            if not os.path.exists(processed_path):
                shutil.copy2(image_path, processed_path)
            self.current_image = cv2.imread(processed_path)
            coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
            if os.path.exists(coord_file):
                self.rectangles = self.load_coordinates(self.current_image_info['name'])
            self.edit_canvas.bind("<ButtonPress-1>", self.on_drag_start)
            self.edit_canvas.bind("<B1-Motion>", self.on_drag_move)
            self.edit_canvas.bind("<ButtonRelease-1>", self.on_drag_end)
            self.edit_canvas.bind("<Delete>", lambda e: self._delete_last_rectangle())
            self._refresh_edit_canvas_view()
            self.update_mode()
            self.set_status("Edit mode — draw or remove annotation boxes")
        else:
            messagebox.showerror("Error", "Image not found")
            self.setup_initial_ui()

    def _on_edit_wheel(self, event):
        delta = 1 if event.delta > 0 else -1
        self._on_edit_wheel_delta(delta)

    def _on_edit_wheel_delta(self, direction):
        if direction > 0:
            self.edit_zoom = min(self.edit_zoom * 1.1, 3.0)
        else:
            self.edit_zoom = max(self.edit_zoom / 1.1, 0.5)
        self._refresh_edit_canvas_view()

    def _on_pan_start(self, event):
        self._pan_start = (event.x, event.y, self.edit_pan_x, self.edit_pan_y)

    def _on_pan_move(self, event):
        if self._pan_start:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self.edit_pan_x = self._pan_start[2] + dx
            self.edit_pan_y = self._pan_start[3] + dy
            self._refresh_edit_canvas_view()

    def _on_pan_end(self, _event):
        self._pan_start = None

    def _refresh_edit_canvas_view(self):
        if not hasattr(self, "edit_canvas") or not self.edit_canvas.winfo_exists():
            return
        base_path = os.path.join(self.processed_dir, self.current_image_info["name"])
        if not os.path.exists(base_path):
            return
        image = cv2.imread(base_path)
        for rect in self.rectangles:
            x1, y1, x2, y2, class_name = rect
            color = self.feature_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        disp = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(disp).resize((1280, 512), Image.LANCZOS)
        zw, zh = int(1280 * self.edit_zoom), int(512 * self.edit_zoom)
        pil = pil.resize((zw, zh), Image.LANCZOS)
        self.image_tk_edit = ImageTk.PhotoImage(pil)
        self.edit_canvas.delete("all")
        self.canvas_image = self.edit_canvas.create_image(
            self.edit_pan_x, self.edit_pan_y, anchor=tk.NW, image=self.image_tk_edit
        )
        self.update_chrome()

    def _delete_last_rectangle(self):
        if self.rectangles:
            self.rectangles.pop()
            self.update_coordinates_file()
            self._refresh_edit_canvas_view()

    def on_drag_start(self, event):
        if self.mode.get() == "Add":
            self.start_x, self.start_y = event.x, event.y
            self.rect_id = None
        elif self.mode.get() == "Remove":
            self.remove_rectangle(event.x, event.y)

    def on_drag_move(self, event):
        if self.mode.get() == "Add":
            if self.rect_id:
                self.edit_canvas.delete(self.rect_id)
            self.rect_id = self.edit_canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="red", width=2
            )

    def on_drag_end(self, event):
        if self.mode.get() == "Add":
            scaled_start_x, scaled_start_y = self._event_to_image_coords(self.start_x, self.start_y)
            scaled_end_x, scaled_end_y = self._event_to_image_coords(event.x, event.y)
            for rect in self.rectangles:
                x1, y1, x2, y2, _ = rect
                if x1 == scaled_start_x and y1 == scaled_start_y and x2 == scaled_end_x and y2 == scaled_end_y:
                    return
            self.save_rectangle(scaled_start_x, scaled_start_y, scaled_end_x, scaled_end_y)
            if self.rect_id:
                self.edit_canvas.delete(self.rect_id)
                self.rect_id = None

    def save_rectangle(self, x1, y1, x2, y2):
        class_name = self.feature_type.get()
        new_rect = (x1, y1, x2, y2, class_name)
        if new_rect not in self.rectangles:
            self.rectangles.append(new_rect)
        self.update_coordinates_file()
        self._refresh_edit_canvas_view()

    def remove_rectangle(self, x, y):
        scaled_x, scaled_y = self._event_to_image_coords(x, y)
        for rect in self.rectangles[:]:
            x1, y1, x2, y2, _ = rect
            if x1 <= scaled_x <= x2 and y1 <= scaled_y <= y2:
                self.rectangles.remove(rect)
                self.update_coordinates_file()
                self._refresh_edit_canvas_view()
                return

    def redraw_image(self):
        base_image_path = os.path.join(self.processed_dir, self.current_image_info["name"])
        image = cv2.imread(base_image_path)
        for rect in self.rectangles:
            x1, y1, x2, y2, class_name = rect
            color = self.feature_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, class_name, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.imwrite(base_image_path, image)
        self._refresh_edit_canvas_view()

    def update_mode(self):
        if not hasattr(self, "edit_canvas"):
            return
        if self.mode.get() == "Add":
            self.edit_canvas.config(cursor="crosshair")
        elif self.mode.get() == "Remove":
            self.edit_canvas.config(cursor="X_cursor")

    def clear_edit_widgets(self):
        self.edit_mode = False
        self.edit_zoom = 1.0
        self.edit_pan_x = 0
        self.edit_pan_y = 0

    def cleanup(self):
        try:
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception as e:
            logging.error(f"Cleanup failed: {str(e)}")

    def on_close(self):
        self.save_state()
        self.cleanup()
        self.root.quit()

    def generate_variability_plots(self):
        logger = logging.getLogger(__name__)
        temp_dir = None
        try:
            temp_dir = tempfile.mkdtemp()
            self.show_loading("Fetching observer list from Drive…")
            self.root.update_idletasks()
            observers = self.get_observers_from_drive()
            if len(observers) < 2:
                self.set_status(f"Need 2+ observers; found {len(observers)}")
                messagebox.showinfo(
                    "Info",
                    f"Need at least 2 observers for comparison. Found {len(observers)} observer(s).",
                )
                return
            self.loading_label.config(text="Building comparison cache (may take a minute)…")
            self.root.update_idletasks()
            common_images = self.find_common_images(observers)
            if len(common_images) < self.interobplt_thresh:
                self.set_status(f"Need {self.interobplt_thresh}+ common tiles; found {len(common_images)}")
                messagebox.showinfo(
                    "Info",
                    f"Need at least {self.interobplt_thresh} common images. Found {len(common_images)}.",
                )
                return
            self.loading_label.config(text="Downloading coordinate files…")
            self.root.update_idletasks()
            self.download_all_coords(observers, common_images, temp_dir)
            self.loading_label.config(text="Generating plots…")
            self.root.update_idletasks()
            image_files, html_files = self.generate_visualizations(temp_dir, observers, common_images)
            if not image_files:
                messagebox.showwarning(
                    "Warning",
                    "No variability plots generated. Check if coordinate files contain valid annotations.",
                )
                return
            self.display_plots_window(image_files)
            for html_path in html_files:
                try:
                    webbrowser.open(f"file://{html_path}")
                except Exception:
                    pass
            self.set_status(
                f"Plots ready — {len(observers)} observers, {len(common_images)} common tiles"
            )
        except Exception as e:
            logger.error(f"Failed to generate plots: {str(e)}")
            messagebox.showerror("Error", f"Failed to generate plots: {str(e)}")
        finally:
            self.hide_loading()
            if temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except Exception as e:
                    logger.error(f"Failed to clean up temp_dir: {str(e)}")


    def get_observers_from_drive(self):
        try:
            query = f"'{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            folders = self.list_all_files(q=query, fields="files(name,id)")
            return [folder['name'] for folder in folders]
        except Exception as e:
            logging.error(f"Failed to get observers: {str(e)}")
            return []

    def find_common_images(self, observers):
        try:
            logger = logging.getLogger(__name__)
            logger.info("Building file cache from Google Drive. This may take a moment...")
            self.folder_id_cache.clear()
            observer_folders_query = f"'{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            all_observer_folders = self.list_all_files(q=observer_folders_query, fields="files(id, name)")
            observer_folder_map = {f['name']: f['id'] for f in all_observer_folders if f['name'] in observers}
            
            for observer_name, observer_id in observer_folder_map.items():
                self.folder_id_cache[observer_name] = {}
                mouse_folders_query = f"'{observer_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                all_mouse_folders = self.list_all_files(q=mouse_folders_query, fields="files(id, name)")
                
                for mouse_folder in all_mouse_folders:
                    mouse_name = mouse_folder['name']
                    mouse_id = mouse_folder['id']
                    self.folder_id_cache[observer_name][mouse_name] = {'folder_id': mouse_id, 'files': {}}
                    coords_query = f"name='coords' and '{mouse_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                    coords_folders = self.list_all_files(q=coords_query, fields="files(id)")
                    if not coords_folders:
                        logger.warning(f"No 'coords' folder found in {observer_name}/{mouse_name}. Skipping.")
                        continue
                    coords_id = coords_folders[0]['id']
                    coord_files_query = f"'{coords_id}' in parents and title contains '_coords.txt' and trashed=false"
                    try:
                        all_coord_files = self.pydrive.ListFile({'q': coord_files_query}).GetList()
                        for coord_file in all_coord_files:
                            self.folder_id_cache[observer_name][mouse_name]['files'][coord_file['title']] = coord_file['id']
                    except Exception as e:
                        logger.error(f"Failed to list coordinate files in {observer_name}/{mouse_name}/coords: {str(e)}")
                        continue
            
            logger.info("Cache built successfully.")
            if not self.folder_id_cache:
                logger.warning("No data in folder_id_cache. Returning empty common images.")
                return []
            
            first_observer_mice = set(self.folder_id_cache[observers[0]].keys())
            common_mice = first_observer_mice.intersection(*(set(self.folder_id_cache[obs].keys()) for obs in observers[1:]))
            common_images = []
            seen_images = set()  # Track unique (mouse, image_name) pairs
            for mouse in common_mice:
                first_observer_files = set(self.folder_id_cache[observers[0]][mouse]['files'].keys())
                common_files = first_observer_files.intersection(*(set(self.folder_id_cache[obs][mouse]['files'].keys()) for obs in observers[1:]))
                for coord_file in common_files:
                    image_name = coord_file.replace('_coords.txt', '')
                    image_key = (mouse, image_name)
                    if image_key not in seen_images:
                        common_images.append(image_key)
                        seen_images.add(image_key)
            
            logger.info(f"Found {len(common_images)} common images across {len(observers)} observers.")
            return common_images
        except Exception as e:
            logger.error(f"Failed to find common images: {str(e)}")
            messagebox.showerror("Error", f"Failed to find common images: {str(e)}")
            return []

    def download_all_coords(self, observers, common_images, temp_dir):
        logger = logging.getLogger(__name__)
        def download_file(file, local_path):
            try:
                file.GetContentFile(local_path)
            except Exception as e:
                logger.error(f"Failed to download {file['title']} to {local_path}: {str(e)}")
                with open(local_path, 'w') as f:
                    pass
        
        download_tasks = []
        for observer in observers:
            observer_dir = os.path.join(temp_dir, observer)
            os.makedirs(observer_dir, exist_ok=True)
            for mouse, image_name in common_images:
                coord_name = f"{os.path.splitext(image_name)[0]}_coords.txt"
                local_coord_path = os.path.join(observer_dir, mouse, coord_name)
                os.makedirs(os.path.join(observer_dir, mouse), exist_ok=True)
                file_id = self.folder_id_cache.get(observer, {}).get(mouse, {}).get('files', {}).get(coord_name)
                if file_id:
                    file = self.pydrive.CreateFile({'id': file_id})
                    download_tasks.append((file, local_coord_path))
                else:
                    logger.warning(f"No coord file '{coord_name}' for {observer}/{mouse}. Creating empty file.")
                    with open(local_coord_path, 'w') as f:
                        pass
        
        with ThreadPoolExecutor(max_workers=100) as executor:
            executor.map(lambda task: download_file(task[0], task[1]), download_tasks)
        import gc
        gc.collect()

    def image_exists_for_observer(self, observer, mouse, image_name):
        try:
            observer_folder_id = self.folder_id_cache.get(observer, {}).get('folder_id')
            if not observer_folder_id:
                query = f"name='{observer}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                observer_folder = self.list_all_files(q=query, fields="files(id)")
                if not observer_folder:
                    return False
                observer_folder_id = observer_folder[0]['id']
            mouse_folder_id = self.folder_id_cache.get(observer, {}).get(mouse, {}).get('folder_id')
            if not mouse_folder_id:
                query = f"name='{mouse}' and '{observer_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                mouse_folder = self.list_all_files(q=query, fields="files(id)")
                if not mouse_folder:
                    return False
                mouse_folder_id = mouse_folder[0]['id']
            images_query = f"name='images' and '{mouse_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            images_folders = self.list_all_files(q=images_query, fields="files(id)")
            if not images_folders:
                return False
            images_folder_id = images_folders[0]['id']
            query = f"name='{image_name}' and '{images_folder_id}' in parents and trashed=false"
            images = self.list_all_files(q=query, fields="files(id)")
            return len(images) > 0
        except Exception as e:
            logging.error(f"Failed to check image existence: {str(e)}")
            return False

    def calculate_iou(self, box1, box2):
        x1, y1, x2, y2 = box1
        x1_b, y1_b, x2_b, y2_b = box2
        xi1 = max(x1, x1_b)
        yi1 = max(y1, y1_b)
        xi2 = min(x2, x2_b)
        yi2 = min(y2, y2_b)
        inter_width = max(0, xi2 - xi1)
        inter_height = max(0, yi2 - yi1)
        inter_area = inter_width * inter_height
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_b - x1_b) * (y2_b - y1_b)
        union_area = box1_area + box2_area - inter_area
        if union_area == 0:
            return 0.0
        return inter_area / union_area

    def generate_visualizations(self, temp_dir, observers, common_images):
        logger = logging.getLogger(__name__)
        try:
            viz_dir = os.path.join(temp_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            if len(observers) < 2:
                raise ValueError("Need at least 2 observers for comparison")
            class_mapping = {
                "Neutrophils": 0,
                "Hyaline Membranes": 1,
                "Proteinaceous Debris": 2
            }
            color_mapping = {"Common": "#E9C46A"}
            for i, observer in enumerate(observers):
                color_mapping[observer] = PLOT_COLORS[i % len(PLOT_COLORS)]
            image_files = []
            html_files = []
            for feature in class_mapping.keys():
                plot_data = []
                for mouse, image_name in common_images:
                    observer_boxes = {}
                    for observer in observers:
                        coord_path = os.path.join(temp_dir, observer, mouse, f"{os.path.splitext(image_name)[0]}_coords.txt")
                        boxes = []
                        if os.path.exists(coord_path):
                            try:
                                with open(coord_path, 'r') as f:
                                    lines = f.readlines()
                                    if not lines:
                                        logger.warning(f"Empty coordinate file: {coord_path}")
                                        continue
                                    for line in lines:
                                        parts = line.strip().split(',')
                                        if len(parts) >= 5 and parts[4] == feature:
                                            try:
                                                x1, y1, x2, y2 = map(int, parts[:4])
                                                boxes.append([x1, y1, x2, y2])
                                            except ValueError:
                                                logger.error(f"Invalid coordinate format in {coord_path}: {line}")
                                                continue
                            except Exception as e:
                                logger.error(f"Failed to read {coord_path}: {str(e)}")
                                continue
                        observer_boxes[observer] = boxes
                    if not all(observer_boxes.get(obs) for obs in observers):
                        common_count = 0
                    else:
                        common_boxes = observer_boxes[observers[0]].copy()
                        matched_indices = {obs: [False] * len(observer_boxes[obs]) for obs in observers}
                        for i in range(1, len(observers)):
                            current_observer = observers[i]
                            new_common_boxes = []
                            matched_indices[current_observer] = [False] * len(observer_boxes[current_observer])
                            for box1 in common_boxes:
                                best_iou = 0.1
                                best_match = None
                                best_idx = None
                                for j, box2 in enumerate(observer_boxes[current_observer]):
                                    if not matched_indices[current_observer][j]:
                                        iou = self.calculate_iou(box1, box2)
                                        if iou > best_iou:
                                            best_iou = iou
                                            best_match = box2
                                            best_idx = j
                                if best_match:
                                    avg_box = [
                                        (box1[0] + best_match[0]) / 2,
                                        (box1[1] + best_match[1]) / 2,
                                        (box1[2] + best_match[2]) / 2,
                                        (box1[3] + best_match[3]) / 2
                                    ]
                                    new_common_boxes.append(avg_box)
                                    matched_indices[current_observer][best_idx] = True
                            common_boxes = new_common_boxes
                        common_count = len(common_boxes)
                    counts = {observer: len(observer_boxes.get(observer, [])) for observer in observers}
                    plot_data.append({
                        'Image': f"{mouse}/{image_name}",
                        **counts,
                        'Common': common_count
                    })
                if not plot_data:
                    logger.warning(f"No plot data for {feature}. Skipping.")
                    continue
                df = pd.DataFrame(plot_data)
                if df.empty or df['Common'].sum() == 0:
                    logger.warning(f"No valid annotations for {feature}")
                    continue
                image_path, html_path = self.create_variability_plot(
                    df, "All Mice", feature, observers, color_mapping, viz_dir
                )
                image_files.append((feature, image_path))
                html_files.append(html_path)
                del df
                import gc
                gc.collect()
            return image_files, html_files
        except Exception as e:
            logger.error(f"Failed to generate visualizations: {str(e)}")
            raise

    def create_variability_plot(self, df, mouse, feature, observers, color_mapping, viz_dir):
        logger = logging.getLogger(__name__)
        try:
            df['Total'] = df[observers].sum(axis=1) + df['Common']
            for observer in observers:
                df[observer] = (df[observer] / df['Total']) * 100
            df['Common'] = (df['Common'] / df['Total']) * 100
            df = df.sort_values(by=observers[0], ascending=False)
            df['index'] = range(len(df))
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df['index'],
                y=df[observers[0]],
                name=f"<b>{observers[0]}</b>",
                marker_color=color_mapping[observers[0]],
                text=df[observers[0]].round(1).astype(str) + '%',
                textposition='inside'
            ))
            fig.add_trace(go.Bar(
                x=df['index'],
                y=df["Common"],
                name="<b>Common</b>",
                marker_color=color_mapping["Common"],
                base=df[observers[0]],
                text=df["Common"].round(1).astype(str) + '%',
                textposition='inside'
            ))
            for i, observer in enumerate(observers[1:], 1):
                base = df[[observers[0], "Common"]].sum(axis=1) if i == 1 else df[observers[:i] + ["Common"]].sum(axis=1)
                fig.add_trace(go.Bar(
                    x=df['index'],
                    y=df[observer],
                    name=f"<b>{observer}</b>",
                    marker_color=color_mapping.get(observer, f"hsl({i*60},50%,50%)"),
                    base=base,
                    text=df[observer].round(1).astype(str) + '%',
                    textposition='inside'
                ))
            title = f"Inter-Observer Variability — {feature}"
            tick_angle = -45 if len(df) > 12 else 0
            fig.update_layout(
                template="plotly_white",
                barmode="stack",
                title=dict(text=title, font=dict(size=16, color="#1B4965")),
                font=dict(family="DejaVu Sans, Arial, sans-serif", size=12, color="#1D3557"),
                xaxis_title="Tile index",
                yaxis_title="Percentage",
                xaxis=dict(tickmode="linear", tick0=0, dtick=1, tickangle=tick_angle),
                legend_title="Observers",
                yaxis=dict(range=[0, 100]),
                width=1200,
                height=600,
                margin=dict(t=80, b=80),
                plot_bgcolor="#F8FAFC",
                paper_bgcolor="#FFFFFF",
            )
            base = f"inter_observer_{feature.replace(' ', '_')}"
            viz_path = os.path.join(viz_dir, f"{base}.png")
            html_path = os.path.join(viz_dir, f"{base}.html")
            fig.write_image(viz_path, format="png", scale=2)
            pio.write_html(fig, file=html_path, auto_open=False, include_plotlyjs="cdn")
            del fig
            import gc
            gc.collect()
            return viz_path, html_path
        except Exception as e:
            logger.error(f"Failed to create variability plot: {str(e)}")
            raise

    def display_plots_window(self, image_files):
        try:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Inter-Observer Variability Plots")
            plot_window.geometry("1280x820")
            plot_window.configure(bg=BRAND["bg"])
            self._center_window(plot_window, 1280, 820)
            top = ttk.Frame(plot_window)
            top.pack(fill=tk.X, padx=12, pady=8)
            ttk.Label(top, text="Feature:", font=self.ui_font_bold).pack(side=tk.LEFT)
            feature_names = [f[0] for f in image_files]
            plot_paths = {f[0]: f[1] for f in image_files}
            selector = ttk.Combobox(top, values=feature_names, state="readonly", width=32)
            selector.pack(side=tk.LEFT, padx=8)
            if feature_names:
                selector.set(feature_names[0])
            canvas_holder = ttk.Frame(plot_window)
            canvas_holder.pack(fill=tk.BOTH, expand=True, padx=12, pady=4)
            image_label = ttk.Label(canvas_holder)
            image_label.pack()
            photos = {}

            def show_feature(*_):
                name = selector.get()
                path = plot_paths.get(name)
                if not path:
                    return
                image = Image.open(path)
                image.thumbnail((1200, 640), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                photos["current"] = photo
                image_label.config(image=photo)

            selector.bind("<<ComboboxSelected>>", show_feature)
            show_feature()
            notebook = ttk.Notebook(plot_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
            for feature, image_path in image_files:
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=feature[:12])
                thumb = Image.open(image_path)
                thumb.thumbnail((360, 200), Image.LANCZOS)
                photo = ImageTk.PhotoImage(thumb)
                photos[feature] = photo
                ttk.Label(tab_frame, image=photo).pack(padx=8, pady=8)
                ttk.Label(tab_frame, text=feature, font=self.ui_font).pack()
            plot_window.protocol("WM_DELETE_WINDOW", plot_window.destroy)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display plots: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CloudImageApp(root)
    root.mainloop()