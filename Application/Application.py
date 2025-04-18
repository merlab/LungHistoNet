import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np
import tempfile
import io
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
import json

class CloudImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cloud-Based Lung Injury Analysis")

        # Cloud configuration
        self.service_account_file = "lunginsightcloud-933a30e4085c.json"
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.input_folder_id = "1kTVr2h11XlnV3xntxjZbPNZebJ8vr5SX"  # Cloud input folder
        self.output_folder_id = "1XrfiMR4nLvKb2kx7MiwwBfdZlpOmT9ub"  # Cloud output folder
        self.coordinates_folder_id = "1XrfiMR4nLvKb2kx7MiwwBfdZlpOmT9ub"  # Coordinates folder

        self.current_image_info = {}  # To store current image metadata

        self.user_name = self.get_username()
        if not self.user_name:
            self.root.quit()
            return

        # Initialize cloud services
        self.drive_service = self.initialize_drive_service()
        self.drive = self.initialize_google_drive()

        # Application state
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.rect_id = None
        self.mode = tk.StringVar(value="Add")
        self.current_image = None
        self.rectangles = []
        self.image_index = 0
        self.image_list = []
        
        # Temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        self.final_dir = os.path.join(self.temp_dir, "final")
        self.state_dir = os.path.join(self.temp_dir, "state")
        self.coords_dir = os.path.join(self.temp_dir, "coordinates")
        
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        os.makedirs(self.coords_dir, exist_ok=True)

        # Load images from cloud
        self.load_cloud_images()

        # Load previous state if available
        if self.load_state():
            # Ensure image_index is valid
            if self.image_index >= len(self.image_list):
                self.image_index = 0
        else:
            self.image_index = 0

        # GUI Elements
        self.setup_ui()

        # Load the current image
        if self.image_list:
            self.load_image()
        else:
            messagebox.showerror("Error", "No images found in cloud folder")
            self.root.quit()

        # Save state on close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def save_state(self):
        """Save the current application state to a JSON file and overwrite in Google Drive by deleting existing file."""
        try:
            state = {
                'user_name': self.user_name,
                'image_index': self.image_index,
                'current_image_info': self.current_image_info
            }
            state_file_path = os.path.join(self.state_dir, 'app_state.json')
            
            # Save state to local file
            with open(state_file_path, 'w') as f:
                json.dump(state, f)
            
            # Get user folder ID
            user_folder_id = self.create_or_get_user_folder()
            
            # Check if app_state.json already exists and delete it
            query = f"'{user_folder_id}' in parents and name='app_state.json' and trashed=false"
            existing_files = self.drive_service.files().list(
                q=query, 
                fields="files(id, name)"
            ).execute().get('files', [])
            
            for file in existing_files:
                self.drive_service.files().delete(fileId=file['id']).execute()
                print(f"🗑️ Deleted existing app_state.json (ID: {file['id']})")
            
            # Upload new file
            self.upload_to_drive(state_file_path, 'app_state.json', user_folder_id)
            print(f"✅ Uploaded new app_state.json in folder: {self.user_name}")
                
        except Exception as e:
            print(f"❌ Failed to save state: {str(e)}")

    def initialize_drive_service(self):
        """Initialize Google Drive service."""
        creds = ServiceAccountCredentials.from_json_keyfile_name(
            self.service_account_file, self.scopes)
        return build('drive', 'v3', credentials=creds)
        
    def get_username(self):
        """Prompt for username before starting processing"""
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
        """Load the application state from Google Drive if it exists."""
        try:
            # Get user folder ID
            user_folder_id = self.create_or_get_user_folder()
            
            # Check if state file exists in user's folder
            query = f"'{user_folder_id}' in parents and name='app_state.json' and trashed=false"
            state_files = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            
            if state_files:
                state_file_id = state_files[0]['id']
                state_file_path = os.path.join(self.state_dir, 'app_state.json')
                
                # Download state file
                if self.download_from_drive(state_file_id, state_file_path):
                    with open(state_file_path, 'r') as f:
                        state = json.load(f)
                    
                    # Validate state
                    saved_image_info = state.get('current_image_info', {})
                    if saved_image_info:
                        # Check if the image still exists in the image_list
                        for img in self.image_list:
                            if img['id'] == saved_image_info.get('id'):
                                self.user_name = state.get('user_name', self.user_name)
                                self.image_index = state.get('image_index', 0)
                                self.current_image_info = saved_image_info
                                print(f"Loaded state: user={self.user_name}, image_index={self.image_index}")
                                return True
                    
                    print("ℹInvalid state (image not found), starting fresh.")
                    return False
                    
            print("ℹNo previous state found, starting fresh.")
            return False
        
        except Exception as e:
            print(f"Failed to load state: {str(e)}")
            return False
    
    def create_output_folder(self):
        """Create the complete output path: username/original_folder"""
        try:
            # 1. First create/verify user folder
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
                print(f"Created user folder: {self.user_name} (ID: {user_folder_id})")
            else:
                user_folder_id = user_folders[0]['id']
                print(f"ℹUsing existing user folder: {self.user_name} (ID: {user_folder_id})")

            # 2. Create the original folder (full name, no splitting)
            original_folder = self.current_image_info['gene']  # e.g. "Mouse1_GeneA"
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
                print(f"reated folder: {original_folder} (ID: {new_folder['id']})")
                return new_folder['id']
            
            print(f"ℹUsing existing folder: {original_folder} (ID: {existing_folders[0]['id']})")
            return existing_folders[0]['id']
            
        except Exception as e:
            print(f"Folder creation failed: {str(e)}")
            raise


    def create_or_get_folder(self, folder_name, parent_id):
        """Create folder or return existing one"""
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

    def initialize_google_drive(self):
        """Initialize PyDrive GoogleDrive instance."""
        gauth = GoogleAuth()
        gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
            self.service_account_file, self.scopes)
        return GoogleDrive(gauth)

    def load_cloud_images(self):
        """Load image list from Google Drive folder including subfolders."""
        try:
            # Query to get all folders in the input folder
            folders = self.drive_service.files().list(
                q=f"'{self.input_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false",
                fields="files(id, name)").execute().get('files', [])
            
            self.image_list = []
            
            # Process each gene folder
            for folder in folders:
                # Get all images in this gene folder
                images = self.drive_service.files().list(
                    q=f"'{folder['id']}' in parents and trashed=false",
                    fields="files(id, name, mimeType)").execute().get('files', [])
                
                for img in images:
                    if img['name'].lower().endswith(('.png', '.jpg', '.jpeg')):
                        # Store both image info and parent folder (gene) name
                        self.image_list.append({
                            'id': img['id'],
                            'name': img['name'],
                            'gene': folder['name']  # Add gene name to image info
                        })
            
            if not self.image_list:
                raise FileNotFoundError("No images found in cloud folder")
                
        except Exception as e:
            messagebox.showerror("Cloud Error", f"Failed to load image list: {str(e)}")

    def download_from_drive(self, file_id, destination_path):
        """Download a file from Google Drive."""
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

    def upload_to_drive(self, file_path, file_name, parent_folder_id):
        """Upload a file to Google Drive."""
        try:
            file_metadata = {
                'title': file_name,
                'parents': [{'id': parent_folder_id}]
            }
            file_to_upload = self.drive.CreateFile(file_metadata)
            file_to_upload.SetContentFile(file_path)
            file_to_upload.Upload()
            return True
        except Exception as e:
            messagebox.showerror("Upload Error", f"Failed to upload file: {str(e)}")
            return False

    def setup_ui(self):
        """Initialize the user interface."""
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)

        self.continue_button = tk.Button(self.root, text="Process", command=self.on_continue)
        self.next_button = tk.Button(self.root, text="Next Image", command=self.load_next_image)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.next_button.pack(side=tk.RIGHT, padx=5)

    def load_image(self):
        """Load and display the current image."""
        self.current_image_info = self.image_list[self.image_index]
        temp_image_path = os.path.join(self.temp_dir, self.current_image_info['name'])
        
        if self.download_from_drive(self.current_image_info['id'], temp_image_path):
            self.current_image_path = temp_image_path
            coord_file_name = f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt"
            self.rectangles = self.load_coordinates(coord_file_name)
            self.display_image(temp_image_path)

    def display_image(self, image_path):
        """Display the image in the GUI."""
        try:
            image = Image.open(image_path)
            image = image.resize((1280, 512))
            self.image_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=self.image_tk)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to display image: {str(e)}")

    def on_continue(self):
        """Process the current image."""
        self.process_image()

    def calculate_score(self, area, circularity, white_percentage):
        """Calculate a score based on area, circularity, and white percentage."""
        area_score = min(max((area - 100) / (1000 - 100), 0), 1)
        circularity_score = min(max((circularity - 0.48) / (1 - 0.48), 0), 1)
        white_percentage_score = min(max((white_percentage - 0.05) / (1 - 0.05), 0), 1)

        score = 0.15 * area_score + 0.7 * circularity_score + 0.15 * white_percentage_score
        return score

    def process_image(self):
        """Process the image to detect contours and calculate scores."""
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

            self.rectangles = []

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
                    red = int(255 * (1 - score))
                    green = int(255 * score)
                    color = (0, green, red)

                    cv2.rectangle(tile, (x, y), (x + w, y + h), color, 2)

                    score_text = f"{score * 100:.2f}%"
                    text_position = (x, y - 10)
                    cv2.putText(tile, score_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

                    self.rectangles.append((x, y, x + w, y + h, color))

            # Save processed image
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])

            cv2.imwrite(processed_path, tile)
            
            # Update coordinates file
            self.update_coordinates_file()
            
            # Display processed image
            self.display_image(processed_path)
            
            # Show post-processing options
            self.show_post_processing_options()

        except Exception as e:
            messagebox.showerror("Processing Error", f"Image processing failed: {str(e)}")

    def update_coordinates_file(self):
        """Update the coordinates file with the current rectangles."""
        coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
        
        with open(coord_file, "w") as file:
            for rect in self.rectangles:
                x1, y1, x2, y2, color = rect
                file.write(f"{x1},{y1},{x2},{y2},{color[0]},{color[1]},{color[2]}\n")

    def load_coordinates(self, image_name):
        """Load coordinates from the coordinates file."""
        coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(image_name)[0]}_coords.txt")
        rectangles = []
        
        if os.path.exists(coord_file):
            with open(coord_file, "r") as file:
                for line in file:
                    x1, y1, x2, y2, r, g, b = map(int, line.strip().split(","))
                    color = (r, g, b)
                    rectangles.append((x1, y1, x2, y2, color))
        
        return rectangles

    def show_post_processing_options(self):
        """Show options after processing the image."""
        for widget in self.root.pack_slaves():
            if isinstance(widget, tk.Button):
                widget.destroy()

        self.save_button = tk.Button(self.root, text="Save", command=self.on_save)
        self.edit_button = tk.Button(self.root, text="Edit", command=self.on_edit)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.edit_button.pack(side=tk.RIGHT, padx=5)

    def on_save(self):
        """Save the processed image with user/mouse folder structure"""
        try:
            # Create user folder first
            user_folder_id = self.create_or_get_user_folder()
            
            # Create mouse folder under user folder
            mouse_name = self.current_image_info['gene']
            mouse_folder_id = self.create_or_get_mouse_folder(mouse_name, user_folder_id)
            
            # Prepare files
            final_path = os.path.join(self.final_dir, self.current_image_info['name'])
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            
            if os.path.exists(processed_path):
                image = cv2.imread(processed_path)
                cv2.imwrite(final_path, image)
                
                # Upload to mouse-specific folder under user folder
                if self.upload_to_drive(final_path, self.current_image_info['name'], mouse_folder_id):
                    coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
                    if os.path.exists(coord_file):
                        self.upload_to_drive(coord_file, os.path.basename(coord_file), mouse_folder_id)
                    
                    messagebox.showinfo("Success", 
                        f"Saved to:\n{self.user_name}/{mouse_name}/\n"
                        f"File: {self.current_image_info['name']}")
                    self.load_next_image()
        
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")


    def verify_folder_structure(self):
        """Debug method to verify folder structure exists"""
        try:
            # Check user folder exists
            user_query = f"'{self.output_folder_id}' in parents and name='{self.user_name}' and mimeType='application/vnd.google-apps.folder'"
            user_folders = self.drive_service.files().list(q=user_query).execute().get('files', [])
            
            if not user_folders:
                print(f"User folder not found: {self.user_name}")
                return False
                
            # Check for at least one mouse folder
            mouse_query = f"'{user_folders[0]['id']}' in parents and mimeType='application/vnd.google-apps.folder'"
            mouse_folders = self.drive_service.files().list(q=mouse_query).execute().get('files', [])
            
            print(f"Found {len(mouse_folders)} mouse folders for user {self.user_name}")
            return True
            
        except Exception as e:
            print(f"Verification failed: {str(e)}")
            return False
    
    def create_or_get_user_folder(self):
        """Create or get user-specific folder in the root output directory"""
        try:
            # Query for existing user folder
            query = f"name='{self.user_name}' and '{self.output_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            
            if existing:
                return existing[0]['id']
            
            # Create new user folder
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
        """Create or get mouse-specific folder within user folder"""
        try:
            # Query for existing mouse folder
            query = f"name='{mouse_name}' and '{parent_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
            existing = self.drive_service.files().list(q=query, fields="files(id)").execute().get('files', [])
            
            if existing:
                return existing[0]['id']
            
            # Create new mouse folder
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
        """Enter edit mode."""
        self.enter_edit_mode()

    def enter_edit_mode(self):
        """Enter edit mode for the current image."""
        # First clear all widgets
        for widget in self.root.pack_slaves():
            widget.destroy()

        # Add filename display
        self.mode_label_name = tk.Entry(self.root, width=50)
        self.mode_label_name.insert(0, f"{self.current_image_info['name']}")
        self.mode_label_name.config(state='readonly')
        self.mode_label_name.pack()

        # Add mode selection
        self.mode_label = tk.Label(self.root, text="Mode:")
        self.mode_label.pack()
        self.add_radio = tk.Radiobutton(self.root, text="Add", variable=self.mode, value="Add", command=self.update_mode)
        self.remove_radio = tk.Radiobutton(self.root, text="Remove", variable=self.mode, value="Remove", command=self.update_mode)
        self.add_radio.pack()
        self.remove_radio.pack()

        # Create and pack the edit canvas
        self.edit_canvas = tk.Canvas(self.root, width=1280, height=512, bg="gray")
        self.edit_canvas.pack()

        # Load and display the processed image only in the canvas
        processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        self.current_image = cv2.imread(processed_path)
        
        # Convert and display the image
        image_pil = Image.open(processed_path).resize((1280, 512))
        self.image_tk_edit = ImageTk.PhotoImage(image_pil)
        
        # Clear any existing image on canvas and add new one
        self.edit_canvas.delete("all")
        self.canvas_image = self.edit_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk_edit)

        # Bind mouse events
        self.edit_canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.edit_canvas.bind("<B1-Motion>", self.on_drag_move)
        self.edit_canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        # Add save button
        self.finalize_button = tk.Button(self.root, text="Save", command=self.save_final_image)
        self.finalize_button.pack(side=tk.RIGHT, padx=5)

    def on_drag_start(self, event):
        """Handle the start of dragging."""
        if self.mode.get() == "Add":
            self.start_x, self.start_y = event.x, event.y
            self.rect_id = None
        elif self.mode.get() == "Remove":
            x, y = event.x, event.y
            self.remove_rectangle(x, y)

    def on_drag_move(self, event):
        """Handle dragging motion."""
        if self.mode.get() == "Add":
            if self.rect_id:
                self.edit_canvas.delete(self.rect_id)
            self.rect_id = self.edit_canvas.create_rectangle(
                self.start_x, self.start_y, event.x, event.y,
                outline="red", width=2
            )

    def on_drag_end(self, event):
        """Handle the end of dragging."""
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
        """Save a rectangle to the list."""
        color = (0, 255, 0)
        self.rectangles.append((x1, y1, x2, y2, color))
        self.update_coordinates_file()

    def remove_rectangle(self, x, y):
        """Remove a rectangle from the list."""
        canvas_width = 1280
        canvas_height = 512
        image = cv2.imread(os.path.join(self.processed_dir, self.current_image_info['name']))
        original_height, original_width = image.shape[:2]

        scale_x = original_width / canvas_width
        scale_y = original_height / canvas_height

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        for rect in self.rectangles:
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
        """Redraw the image with the current rectangles."""
        base_image_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        image = cv2.imread(base_image_path)

        for rect in self.rectangles:
            x1, y1, x2, y2, color = rect
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        updated_image_path = os.path.join(self.processed_dir, self.current_image_info['name'])
        cv2.imwrite(updated_image_path, image)
        updated_image = Image.open(updated_image_path).resize((1280, 512))
        self.image_tk_edit = ImageTk.PhotoImage(updated_image)
        self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)

    def update_mode(self):
        """Update the mode (Add/Remove)."""
        if self.mode.get() == "Add":
            self.edit_canvas.config(cursor="arrow")
        elif self.mode.get() == "Remove":
            self.edit_canvas.config(cursor="crosshair")

    def save_final_image(self):
        """Save the final image and reset the UI."""
        try:
            # Use current_image_info['name'] instead of current_file_name
            final_path = os.path.join(self.final_dir, self.current_image_info['name'])
            processed_path = os.path.join(self.processed_dir, self.current_image_info['name'])
            
            if os.path.exists(processed_path):
                image = cv2.imread(processed_path)
                cv2.imwrite(final_path, image)
                
                # Get user folder ID
                user_folder_id = self.create_or_get_user_folder()
                
                # Upload to mouse-specific folder
                mouse_name = self.current_image_info['gene']
                # Pass both mouse_name and user_folder_id
                mouse_folder_id = self.create_or_get_mouse_folder(mouse_name, user_folder_id)
                
                if self.upload_to_drive(final_path, self.current_image_info['name'], mouse_folder_id):
                    coord_file = os.path.join(self.coords_dir, f"{os.path.splitext(self.current_image_info['name'])[0]}_coords.txt")
                    if os.path.exists(coord_file):
                        self.upload_to_drive(coord_file, os.path.basename(coord_file), mouse_folder_id)
                    
                    messagebox.showinfo("Success", f"Results saved to {mouse_name} folder in cloud!")
                    self.reset_to_initial_page()
                else:
                    messagebox.showerror("Error", "Failed to upload to cloud")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save final image: {str(e)}")

    def reset_to_initial_page(self):
        """Reset the UI to the initial state."""
        for widget in self.root.pack_slaves():
            widget.destroy()
        
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        
        self.continue_button = tk.Button(self.root, text="Process", command=self.on_continue)
        self.next_button = tk.Button(self.root, text="Next Image", command=self.load_next_image)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.next_button.pack(side=tk.RIGHT, padx=5)
        
        self.load_next_image()

    def load_next_image(self):
        """Load the next image in the list."""
        self.image_index += 1
        
        if self.image_index >= len(self.image_list):
            messagebox.showinfo("Complete", "All images processed!")
            self.root.quit()
            return
        
        self.save_state()  # Save state after moving to next image
        self.load_image()

    def cleanup(self):
        """Clean up temporary files."""
        try:
            for root_dir, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root_dir, name))
                for name in dirs:
                    os.rmdir(os.path.join(root_dir, name))
            os.rmdir(self.temp_dir)
        except Exception as e:
            print(f"Cleanup error: {str(e)}")

    def on_close(self):
        """Handle application close."""
        self.save_state()  # Save state before closing
        self.cleanup()
        self.root.quit()


if __name__ == "__main__":
    root = tk.Tk()
    app = CloudImageApp(root)
    root.mainloop()