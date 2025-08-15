import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
import io
from io import BytesIO
import plotly.graph_objects as go
from concurrent.futures import ProcessPoolExecutor
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload
from googleapiclient.discovery import build
import json
import shutil
import uuid

class CloudImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloud-Based Lung Injury Analysis")
        self.feature_type = tk.StringVar(value="Neutrophils")
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
        self.user_name = self.get_username()
        if not self.user_name:
            self.root.quit()
            return
        self.drive_service = self.initialize_drive_service()
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.final_dir = os.path.join(self.temp_dir, "final")
        self.state_dir = os.path.join(self.temp_dir, "state")
        self.coords_dir = os.path.join(self.temp_dir, "coordinates")
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.coords_dir, exist_ok=True)
        self.setup_initial_ui()
        self.load_cloud_images()
        if self.load_state():
            if self.image_index >= len(self.image_list):
                self.image_index = 0
        else:
            self.image_index = self.recover_last_index()  # New: Try to recover index
        if self.image_list:
            self.check_and_load_image()  # Modified: Check for duplicates before loading
        else:
            messagebox.showerror("Error", "No images found in cloud folder")
            self.root.quit()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def setup_initial_ui(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.image_label = ttk.Label(self.main_frame)
        self.image_label.pack(pady=10)
        self.feature_frame = ttk.Frame(self.main_frame)
        self.feature_frame.pack(pady=5)
        ttk.Label(self.feature_frame, text="Feature:").pack(side=tk.LEFT)
        ttk.Radiobutton(self.feature_frame, text="Neutrophils", variable=self.feature_type, 
                        value="Neutrophils").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.feature_frame, text="Hyaline Membranes", variable=self.feature_type,
                        value="Hyaline Membranes").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(self.feature_frame, text="Proteinaceous Debris", variable=self.feature_type,
                        value="Proteinaceous Debris").pack(side=tk.LEFT, padx=5)
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(pady=5)
        self.continue_button = ttk.Button(self.button_frame, text="Process", command=self.on_continue)
        self.next_button = ttk.Button(self.button_frame, text="Next Image", command=self.load_next_image)
        self.variability_button = ttk.Button(self.button_frame, text="Inter-Observer Variability Plot", 
                                             command=self.generate_variability_plots)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.next_button.pack(side=tk.LEFT, padx=5)
        self.variability_button.pack(side=tk.LEFT, padx=5)

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
            else:
                messagebox.showerror("Error", "No source image found.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save edited image: {str(e)}")

    def setup_edit_ui(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.mode_label_name = ttk.Entry(self.main_frame, width=50)
        self.mode_label_name.insert(0, f"{self.current_image_info['name']}")
        self.mode_label_name.config(state='readonly')
        self.mode_label_name.pack()
        self.mode_label = ttk.Label(self.main_frame, text="Mode:")
        self.mode_label.pack()
        self.add_radio = ttk.Radiobutton(self.main_frame, text="Add", variable=self.mode, value="Add", command=self.update_mode)
        self.remove_radio = ttk.Radiobutton(self.main_frame, text="Remove", variable=self.mode, value="Remove", command=self.update_mode)
        self.add_radio.pack()
        self.remove_radio.pack()
        self.edit_canvas = tk.Canvas(self.main_frame, width=1280, height=512, bg="gray")
        self.edit_canvas.pack()
        self.finalize_button = ttk.Button(self.main_frame, text="Save", command=self.save_final_image)
        self.finalize_button.pack(pady=5)

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
            existing_files = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            if existing_files:
                file_id = existing_files[0]['id']
                media = MediaIoBaseUpload(open(state_file_path, 'rb'), mimetype='application/json')
                self.drive_service.files().update(fileId=file_id, media_body=media).execute()
            else:
                file_metadata = {'name': 'app_state.json', 'parents': [user_folder_id]}
                media = MediaIoBaseUpload(open(state_file_path, 'rb'), mimetype='application/json')
                self.drive_service.files().create(body=file_metadata, media_body=media).execute()
        except Exception as e:
            pass  # Keep local state even if cloud upload fails

    def initialize_drive_service(self):
        try:
            if not os.path.exists(self.service_account_file):
                raise FileNotFoundError(f"Service account file {self.service_account_file} not found")
            creds = ServiceAccountCredentials.from_json_keyfile_name(
                self.service_account_file, self.scopes)
            if not creds:
                raise ValueError("Failed to load credentials from service account file")
            return build('drive', 'v3', credentials=creds)
        except Exception as e:
            messagebox.showerror("Authentication Error", f"Failed to initialize Google Drive service: {str(e)}")
            self.root.quit()
            raise

    def get_username(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("User Identification")
        dialog.geometry("300x150")
        tk.Label(dialog, text="Please enter your name:").pack(pady=10)
        entry = tk.Entry(dialog)
        entry.pack(pady=5)
        result = []
        def on_ok():
            result.append(entry.get())
            dialog.destroy()
        tk.Button(dialog, text="Submit", command=on_ok).pack(pady=10)
        dialog.wait_window()
        return result[0] if result else None

    def load_state(self):
        try:
            user_folder_id = self.create_or_get_user_folder()
            query = f"'{user_folder_id}' in parents and name='app_state.json' and trashed=false"
            state_files = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
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
                                return True
            # Fallback to local state if cloud state is missing
            if os.path.exists(state_file_path):
                with open(state_file_path, 'r') as f:
                    state = json.load(f)
                saved_image_info = state.get('current_image_info', {})
                if saved_image_info:
                    for img in self.image_list:
                        if img['id'] == saved_image_info.get('id'):
                            self.user_name = state.get('user_name', self.user_name)
                            self.image_index = state.get('image_index', 0)
                            return True
            return False
        except Exception as e:
            return False

    def recover_last_index(self):
        """Scan cloud for the last annotated image to estimate progress."""
        try:
            user_folder_id = self.create_or_get_user_folder()
            mouse_folders = self.drive_service.files().list(
                q=f"'{user_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)"
            ).execute().get('files', [])
            last_index = 0
            for mouse_folder in mouse_folders:
                files = self.drive_service.files().list(
                    q=f"'{mouse_folder['id']}' in parents and trashed=false",
                    fields="files(name)"
                ).execute().get('files', [])
                for file in files:
                    if file['name'].endswith('_coords.txt'):
                        image_name = file['name'].replace('_coords.txt', '')
                        for i, img in enumerate(self.image_list):
                            if img['name'] == image_name and i > last_index:
                                last_index = i
            return last_index
        except Exception as e:
            return 0

    def check_and_load_image(self):
            """Check if the current image has existing annotations and skip if found."""
            while self.image_index < len(self.image_list):
                self.current_image_info = self.image_list[self.image_index]
                if self.check_existing_annotations():
                    print(f"Skipping existing annotations for {self.current_image_info['name']}")
                    self.image_index += 1
                    self.image_processed = False
                    self.rectangles = []
                    if self.image_index >= len(self.image_list):
                        messagebox.showinfo("Complete", f"All {len(self.image_list)} images already annotated!")
                        self.root.quit()
                        return
                    self.save_state()
                else:
                    print(f"Loading unannotated image: {self.current_image_info['name']}")
                    self.load_image()
                    break
            if self.image_index >= len(self.image_list):
                messagebox.showinfo("Complete", f"No unannotated images found among {len(self.image_list)} tiles!")
                self.root.quit()

    def check_existing_annotations(self):
        """Check if the current image and its coordinates exist in the user's output folder."""
        try:
            user_folder_id = self.create_or_get_user_folder()
            mouse_name = self.current_image_info['gene']
            mouse_folder_id = self.create_or_get_mouse_folder(mouse_name, user_folder_id)
            image_name = self.current_image_info['name']
            coord_name = f"{os.path.splitext(image_name)[0]}_coords.txt"

            # Check if the image exists in the output folder
            image_query = f"name='{image_name}' and '{mouse_folder_id}' in parents and trashed=false"
            image_files = self.drive_service.files().list(q=image_query, fields="files(id)").execute().get('files', [])
            print(f"Checking output for image {image_name}: Found {len(image_files)} files")

            # Check if the coordinate file exists in the output folder
            coord_query = f"name='{coord_name}' and '{mouse_folder_id}' in parents and trashed=false"
            coord_files = self.drive_service.files().list(q=coord_query, fields="files(id)").execute().get('files', [])
            print(f"Checking output for coords {coord_name}: Found {len(coord_files)} files")

            # Verify the image exists in the input folder
            input_folders = self.drive_service.files().list(
                q=f"'{self.input_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)"
            ).execute().get('files', [])
            image_in_input = False
            for folder in input_folders:
                if folder['name'] == mouse_name:
                    image_query = f"name='{image_name}' and '{folder['id']}' in parents and trashed=false"
                    input_files = self.drive_service.files().list(q=image_query, fields="files(id)").execute().get('files', [])
                    print(f"Checking input for image {image_name} in folder {mouse_name}: Found {len(input_files)} files")
                    if input_files:
                        image_in_input = True
                        break

            # Log the result
            result = len(image_files) > 0 and len(coord_files) > 0 and image_in_input
            print(f"Duplicate check for {image_name}: image_in_output={len(image_files) > 0}, coords_in_output={len(coord_files) > 0}, image_in_input={image_in_input}, is_duplicate={result}")
            return result
        except Exception as e:
            print(f"Error checking annotations for {self.current_image_info['name']}: {str(e)}")
            return False

    def create_output_folder(self):
        try:
            user_query = f"name='{self.user_name}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder'"
            user_folders = self.drive_service.files().list(q=user_query, fields="files(id,name)").execute().get('files', [])
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
            existing_folders = self.drive_service.files().list(q=folder_query, fields="files(id,name)").execute().get('files', [])
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
            raise

    def create_or_get_folder(self, folder_name, parent_id):
        query = f"'{parent_id}' in parents and name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
        if existing:
            return existing[0]['id']
        folder_metadata = {
            'name': folder_name,
            'mimeType': 'application/vnd.google-apps.folder',
            'parents': [parent_id]
        }
        folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
        return folder['id']

    def load_cloud_images(self):
        try:
            folders = self.drive_service.files().list(
                q=f"'{self.input_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)").execute().get('files', [])
            self.image_list = []
            for folder in folders:
                images = self.drive_service.files().list(
                    q=f"'{folder['id']}' in parents and trashed=false",
                    fields="files(id, name, mimeType)").execute().get('files', [])
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
            request = self.drive_service.files().get_media(fileId=file_id)
            fh = io.FileIO(destination_path, 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
            return True
        except Exception as e:
            messagebox.showerror("Download Error", f"Failed to download file: {str(e)}")
            return False

    def upload_or_update(self, file_path, file_name, parent_folder_id):
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} does not exist")
            query = f"'{parent_folder_id}' in parents and name='{file_name}' and trashed=false"
            existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            media = MediaIoBaseUpload(open(file_path, 'rb'), mimetype='application/octet-stream', resumable=True)
            if existing:
                file_id = existing[0]['id']
                self.drive_service.files().update(fileId=file_id, media_body=media).execute()
                print(f"Updated file {file_name} in Google Drive (ID: {file_id})")
            else:
                file_metadata = {'name': file_name, 'parents': [parent_folder_id]}
                self.drive_service.files().create(body=file_metadata, media_body=media).execute()
                print(f"Uploaded new file {file_name} to Google Drive")
            return True
        except Exception as e:
            print(f"Upload error for {file_name}: {str(e)}")
            messagebox.showerror("Upload Error", f"Failed to upload {file_name}: {str(e)}")
            return False

    def load_image(self):
        self.current_image_info = self.image_list[self.image_index]
        temp_image_path = os.path.join(self.temp_dir, self.current_image_info['name'])
        self.feature_type.set("Neutrophils")
        if self.download_from_drive(self.current_image_info['id'], temp_image_path):
            self.current_image_path = temp_image_path
            self.display_image(temp_image_path)
            # Load existing annotations if any
            coord_name = f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt"
            user_folder_id = self.create_or_get_user_folder()
            mouse_folder_id = self.create_or_get_mouse_folder(self.current_image_info['gene'], user_folder_id)
            query = f"name='{coord_name}' and '{mouse_folder_id}' in parents and trashed=false"
            files = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            if files:
                coord_path = os.path.join(self.coords_dir, coord_name)
                if self.download_from_drive(files[0]['id'], coord_path):
                    self.rectangles = self.load_coordinates(self.current_image_info['name'])
        else:
            messagebox.showerror("Error", f"Failed to download image: {self.current_image_info['name']}")

    def display_image(self, image_path):
        try:
            image = Image.open(image_path)
            image = image.resize((1280, 512))
            self.image_tk = ImageTk.PhotoImage(image)
            if hasattr(self, 'image_label') and self.image_label.winfo_exists():
                self.image_label.config(image=self.image_tk)
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
        self.save_button = ttk.Button(self.button_frame, text="Save", command=self.on_save)
        self.edit_button = ttk.Button(self.button_frame, text="Edit", command=self.on_edit)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.edit_button.pack(side=tk.RIGHT, padx=5)

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
                messagebox.showinfo("Success", "Image saved locally. Click 'Next Image' to upload to cloud.")
                self.setup_initial_ui()
                self.display_image(processed_path)
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
            if not self.upload_or_update(final_path, self.current_image_info['name'], mouse_folder_id):
                return
            if not self.upload_or_update(coord_file, os.path.basename(coord_file), mouse_folder_id):
                return
            self.image_processed = False
        except Exception as e:
            messagebox.showerror("Error", f"Failed to upload to cloud: {str(e)}")

    def load_next_image(self):
        if self.current_image_info:
            self.finalize_and_upload()
        self.image_index += 1
        self.image_processed = False
        self.rectangles = []
        if self.image_index >= len(self.image_list):
            messagebox.showinfo("Complete", "All images processed!")
            self.root.quit()
            return
        self.save_state()
        self.check_and_load_image()

    def verify_folder_structure(self):
        try:
            user_query = f"'{self.output_folder_id}' in parents and name='{self.user_name}' and mimeType='application/vnd.google-apps.folder'"
            user_folders = self.drive_service.files().list(q=user_query).execute().get('files', [])
            if not user_folders:
                return False
            mouse_query = f"'{user_folders[0]['id']}' in parents and mimeType='application/vnd.google-apps.folder'"
            mouse_folders = self.drive_service.files().list(q=mouse_query).execute().get('files', [])
            return True
        except Exception as e:
            return False

    def create_or_get_user_folder(self):
        try:
            query = f"name='{self.user_name}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
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
            query = f"name='{mouse_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            if existing:
                return existing[0]['id']
            folder_metadata = {
                'name': mouse_name,
                'mimeType': 'application/vnd.google-apps.folder',
                'parents': [parent_folder_id]
            }
            folder = self.drive_service.files().create(body=folder_metadata, fields='id').execute()
            return folder['id']
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
            self.current_image = cv2.imread(image_path)
            image_pil = Image.open(image_path).resize((1280, 512))
            self.image_tk_edit = ImageTk.PhotoImage(image_pil)
            self.canvas_image = self.edit_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk_edit)
            coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
            if os.path.exists(coord_file):
                self.rectangles = self.load_coordinates(self.current_image_info['name'])
            self.edit_canvas.bind("<ButtonPress-1>", self.on_drag_start)
            self.edit_canvas.bind("<B1-Motion>", self.on_drag_move)
            self.edit_canvas.bind("<ButtonRelease-1>", self.on_drag_end)
        else:
            messagebox.showerror("Error", "Image not found")
            self.setup_initial_ui()

    def on_drag_start(self, event):
        if self.mode.get() == "Add":
            self.start_x, self.start_y = event.x, event.y
            self.rect_id = None
        elif self.mode.get() == "Remove":
            x, y = event.x, event.y
            self.remove_rectangle(x, y)

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
            canvas_width = 1280
            canvas_height = 512
            image = cv2.imread(os.path.join(self.processed_dir, self.current_image_info['name']))
            original_height, original_width = image.shape[:2]
            scale_x = original_width / canvas_width
            scale_y = original_height / canvas_height
            scaled_start_x = int(self.start_x * scale_x)
            scaled_start_y = int(self.start_y * scale_y)
            scaled_end_x = int(event.x * scale_x)
            scaled_end_y = int(event.y * scale_y)
            for rect in self.rectangles:
                x1, y1, x2, y2, _ = rect
                if x1 == scaled_start_x and y1 == scaled_start_y and x2 == scaled_end_x and y2 == scaled_end_y:
                    return
            self.save_rectangle(scaled_start_x, scaled_start_y, scaled_end_x, scaled_end_y)
            self.redraw_image()

    def save_rectangle(self, x1, y1, x2, y2):
        class_name = self.feature_type.get()
        new_rect = (x1, y1, x2, y2, class_name)
        if new_rect not in self.rectangles:
            self.rectangles.append(new_rect)
        self.update_coordinates_file()
        self.redraw_image()

    def remove_rectangle(self, x, y):
        canvas_width = 1280
        canvas_height = 512
        image = cv2.imread(os.path.join(self.processed_dir, self.current_image_info['name']))
        original_height, original_width = image.shape[:2]
        scale_x = original_width / canvas_width
        scale_y = original_height / canvas_height
        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)
        for rect in self.rectangles[:]:
            x1, y1, x2, y2, _ = rect
            if x1 <= scaled_x <= x2 and y1 <= scaled_y <= y2:
                self.rectangles.remove(rect)
                self.update_coordinates_file()
                self.redraw_image()
                line_thickness = 2
                line_length = 10
                color = (0, 0, 255)
                cv2.line(image, (scaled_x - line_length, scaled_y - line_length),
                         (scaled_x + line_length, scaled_y + line_length), color, thickness=line_thickness)
                cv2.line(image, (scaled_x - line_length, scaled_y + line_length),
                         (scaled_x + line_length, scaled_y - line_length), color, thickness=line_thickness)
                updated_image_path = os.path.join(self.processed_dir, self.current_image_info['name'])
                cv2.imwrite(updated_image_path, image)
                updated_image = Image.open(updated_image_path).resize((1280, 512))
                self.image_tk_edit = ImageTk.PhotoImage(updated_image)
                self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)
                return

    def redraw_image(self):
        base_image_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        image = cv2.imread(base_image_path)
        for rect in self.rectangles:
            x1, y1, x2, y2, class_name = rect
            color = self.feature_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f"{class_name}"
            cv2.putText(image, label, (x1, y1 - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        updated_image_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        cv2.imwrite(updated_image_path, image)
        updated_image = Image.open(updated_image_path).resize((1280, 512))
        self.image_tk_edit = ImageTk.PhotoImage(updated_image)
        self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)

    def update_mode(self):
        if self.mode.get() == "Add":
            self.edit_canvas.config(cursor="arrow")
        elif self.mode.get() == "Remove":
            self.edit_canvas.config(cursor="crosshair")

    def clear_edit_widgets(self):
        widgets_to_remove = [
            'mode_label_name', 'mode_label', 'add_radio', 
            'remove_radio', 'edit_canvas', 'finalize_button'
        ]
        for widget_name in widgets_to_remove:
            if hasattr(self, widget_name):
                widget = getattr(self, widget_name)
                try:
                    widget.pack_forget()
                    widget.destroy()
                except tk.TclError:
                    pass
                except Exception as e:
                    pass
                finally:
                    if hasattr(self, widget_name):
                        delattr(self, widget_name)

    def cleanup(self):
        try:
            for root_dir, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    if name != 'app_state.json':  # Preserve local app_state.json
                        os.remove(os.path.join(root_dir, name))
                for name in dirs:
                    os.rmdir(os.path.join(root_dir, name))
        except Exception as e:
            pass

    def on_close(self):
        self.save_state()
        self.cleanup()
        self.root.quit()

    def generate_variability_plots(self):
        try:
            temp_dir = tempfile.mkdtemp()
            observers = self.get_observers_from_drive()
            if len(observers) < 2:
                messagebox.showinfo(
                    "Info", 
                    f"Need at least 2 observers for comparison. Found {len(observers)} observer(s)."
                )
                return
            common_images = self.find_common_images(observers)
            if len(common_images) < self.interobplt_thresh:
                messagebox.showinfo(
                    "Info", 
                    f"Need at least {self.interobplt_thresh} common images for comparison. Found {len(common_images)} common images."
                )
                return
            self.process_observer_data(observers, common_images, temp_dir)
            image_files = self.generate_visualizations(temp_dir)
            self.display_plots_window(image_files)
            messagebox.showinfo(
                "Success", 
                f"Generated variability plots for {len(observers)} observers and {len(common_images)} common images."
            )
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate plots: {str(e)}")
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

    def get_observers_from_drive(self):
        try:
            query = f"'{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            folders = self.drive_service.files().list(
                q=query, 
                fields="files(name,id)"
            ).execute().get('files', [])
            return [folder['name'] for folder in folders]
        except Exception as e:
            return []

    def find_common_images(self, observers):
        try:
            observer_mice = {}
            for observer in observers:
                query = f"name='{observer}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                observer_folder = self.drive_service.files().list(
                    q=query, 
                    fields="files(id)"
                ).execute().get('files', [])
                if not observer_folder:
                    continue
                observer_folder_id = observer_folder[0]['id']
                query = f"'{observer_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                mouse_folders = self.drive_service.files().list(
                    q=query, 
                    fields="files(name)"
                ).execute().get('files', [])
                observer_mice[observer] = {mf['name'] for mf in mouse_folders}
            common_mice = set.intersection(*observer_mice.values())
            common_images = []
            for mouse in common_mice:
                first_observer = observers[0]
                query = f"name='{first_observer}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                first_observer_folder = self.drive_service.files().list(
                    q=query, 
                    fields="files(id)"
                ).execute().get('files', [])
                if not first_observer_folder:
                    continue
                first_observer_folder_id = first_observer_folder[0]['id']
                query = f"name='{mouse}' and '{first_observer_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
                mouse_folder = self.drive_service.files().list(
                    q=query, 
                    fields="files(id)"
                ).execute().get('files', [])
                if not mouse_folder:
                    continue
                mouse_folder_id = mouse_folder[0]['id']
                query = f"'{mouse_folder_id}' in parents and mimeType!='application/vnd.google-apps.folder' and trashed=false"
                images = self.drive_service.files().list(
                    q=query, 
                    fields="files(name)"
                ).execute().get('files', [])
                for image in images:
                    image_name = image['name']
                    if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        all_have = True
                        for observer in observers[1:]:
                            if not self.image_exists_for_observer(observer, mouse, image_name):
                                all_have = False
                                break
                        if all_have:
                            common_images.append((mouse, image_name))
            return common_images
        except Exception as e:
            return []

    def image_exists_for_observer(self, observer, mouse, image_name):
        try:
            query = f"name='{observer}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            observer_folder = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if not observer_folder:
                return False
            observer_folder_id = observer_folder[0]['id']
            query = f"name='{mouse}' and '{observer_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            mouse_folder = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if not mouse_folder:
                return False
            mouse_folder_id = mouse_folder[0]['id']
            query = f"name='{image_name}' and '{mouse_folder_id}' in parents and trashed=false"
            images = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            return len(images) > 0
        except Exception as e:
            return False

    def process_observer_data(self, observers, common_images, temp_dir):
        try:
            observer_dirs = {}
            for observer in observers:
                observer_dir = os.path.join(temp_dir, observer)
                os.makedirs(observer_dir, exist_ok=True)
                observer_dirs[observer] = observer_dir
            for mouse, image_name in common_images:
                for observer in observers:
                    image_path = os.path.join(observer_dirs[observer], image_name)
                    coord_file = os.path.splitext(image_name)[0] + "_coords.txt"
                    coord_path = os.path.join(observer_dirs[observer], coord_file)
                    self.download_observer_files(observer, mouse, image_name, image_path, coord_path)
        except Exception as e:
            raise

    def download_observer_files(self, observer, mouse, image_name, image_path, coord_path):
        try:
            query = f"name='{observer}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            observer_folder = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if not observer_folder:
                raise FileNotFoundError(f"Observer folder not found: {observer}")
            observer_folder_id = observer_folder[0]['id']
            query = f"name='{mouse}' and '{observer_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            mouse_folder = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if not mouse_folder:
                raise FileNotFoundError(f"Mouse folder not found: {mouse}")
            mouse_folder_id = mouse_folder[0]['id']
            query = f"name='{image_name}' and '{mouse_folder_id}' in parents and trashed=false"
            image_files = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if not image_files:
                raise FileNotFoundError(f"Image not found: {image_name}")
            self.download_from_drive(image_files[0]['id'], image_path)
            coord_name = os.path.splitext(image_name)[0] + "_coords.txt"
            query = f"name='{coord_name}' and '{mouse_folder_id}' in parents and trashed=false"
            coord_files = self.drive_service.files().list(
                q=query, 
                fields="files(id)"
            ).execute().get('files', [])
            if coord_files:
                self.download_from_drive(coord_files[0]['id'], coord_path)
        except Exception as e:
            raise

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

    def generate_visualizations(self, temp_dir):
        try:
            viz_dir = os.path.join(temp_dir, "visualizations")
            os.makedirs(viz_dir, exist_ok=True)
            observers = [
                d for d in os.listdir(temp_dir)
                if os.path.isdir(os.path.join(temp_dir, d)) and d != "visualizations"
            ]
            if len(observers) < 2:
                raise ValueError("Need at least 2 observers for comparison")
            class_mapping = {
                "Neutrophils": 0,
                "Hyaline Membranes": 1,
                "Proteinaceous Debris": 2
            }
            color_mapping = {
                observers[0]: "#2D6A4F",
                "Common": "#F4D35E",
                observers[1]: "#84C5A1"
            }
            image_files = []
            mouse_conditions = set()
            for observer in observers:
                observer_dir = os.path.join(temp_dir, observer)
                for file in os.listdir(observer_dir):
                    if file.endswith('_coords.txt'):
                        mouse = file.split('_')[0]
                        mouse_conditions.add(mouse)
            for mouse in mouse_conditions:
                for feature in class_mapping.keys():
                    plot_data = []
                    image_files_list = []
                    for observer in observers:
                        observer_dir = os.path.join(temp_dir, observer)
                        for file in os.listdir(observer_dir):
                            if file.startswith(mouse) and file.endswith('_coords.txt'):
                                image_files_list.append(file)
                    for coord_file in set(image_files_list):
                        image_name = coord_file.replace('_coords.txt', '')
                        observer_boxes = {}
                        for observer in observers:
                            coord_path = os.path.join(temp_dir, observer, coord_file)
                            boxes = []
                            if os.path.exists(coord_path):
                                with open(coord_path, 'r') as f:
                                    for line in f:
                                        parts = line.strip().split(',')
                                        if len(parts) >= 5 and parts[4] == feature:
                                            x1, y1, x2, y2 = map(int, parts[:4])
                                            boxes.append([x1, y1, x2, y2])
                            observer_boxes[observer] = boxes
                        if not all(observer_boxes[obs] for obs in observers):
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
                        counts = {observer: len(observer_boxes[observer]) for observer in observers}
                        print(f"Image: {image_name}, Feature: {feature}, Counts: {counts}, Common: {common_count}")
                        plot_data.append({
                            'Image': image_name,
                            **counts,
                            'Common': common_count
                        })
                    df = pd.DataFrame(plot_data)
                    if df['Common'].sum() == 0:
                        print(f"Warning: No common annotations for {feature} in {mouse}. Check coordinate files.")
                    image_path = self.create_variability_plot(
                        df, mouse, feature, observers, color_mapping, viz_dir
                    )
                    image_files.append((feature, image_path))
            return image_files
        except Exception as e:
            raise Exception(f"Failed to generate visualizations: {str(e)}")
    
    def create_variability_plot(self, df, mouse, feature, observers, color_mapping, viz_dir):
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
            title = f"<b>Inter-Observer Variability of {feature} Counts in {mouse} Tiles</b>"
            fig.update_layout(
                barmode='stack',
                title=title,
                xaxis_title="<b>Tile Index</b>",
                yaxis_title="<b>Percentage</b>",
                xaxis=dict(
                    tickmode='linear',
                    tick0=1,
                    dtick=2
                ),
                legend_title="<b>Observers</b>",
                yaxis=dict(range=[0, 100]),
                width=1200,
                height=600
            )
            viz_filename = f"inter_observer_{mouse}_{feature.replace(' ', '_')}.png"
            viz_path = os.path.join(viz_dir, viz_filename)
            fig.write_image(viz_path, format="png")
            return viz_path
        except Exception as e:
            raise Exception(f"Failed to create variability plot: {str(e)}")

    def display_plots_window(self, image_files):
        try:
            plot_window = tk.Toplevel(self.root)
            plot_window.title("Inter-Observer Variability Plots")
            plot_window.geometry("1280x800")
            notebook = ttk.Notebook(plot_window)
            notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            for feature, image_path in image_files:
                tab_frame = ttk.Frame(notebook)
                notebook.add(tab_frame, text=feature)
                image = Image.open(image_path)
                image = image.resize((1200, 600), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                image_label = ttk.Label(tab_frame, image=photo)
                image_label.image = photo
                image_label.pack(padx=10, pady=10)
            plot_window.protocol("WM_DELETE_WINDOW", plot_window.destroy)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display plots: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = CloudImageApp(root)
    root.mainloop()