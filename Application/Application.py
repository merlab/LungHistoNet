import tkinter as tk
from tkinter import ttk , messagebox
from PIL import Image, ImageTk
import os
import cv2
import numpy as np


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Application")

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
        self.image_dirs = "data"
        self.neut_coords_dir = "coordinates"

        self.image_list = [img for img in os.listdir(self.image_dirs) if img.endswith(('.png', '.jpg', '.jpeg'))]

        if not self.image_list:
            raise FileNotFoundError(f"No images found in the image directory.")


        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

        self.continue_button = tk.Button(root, text="Continue", command=self.on_continue)
        self.regenerate_button = tk.Button(root, text="Next Image", command=self.load_next_image)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.regenerate_button.pack(side=tk.RIGHT, padx=5)

        self.load_image()

    def load_image(self):
        self.current_image_path = self.image_list[self.image_index]
        self.rectangles = self.load_coordinates(self.current_image_path)  
        self.display_image(f"{self.image_dirs}/{self.current_image_path}")

    def load_next_image(self):
        self.image_index += 1

        if self.image_index >= len(self.image_list):
            messagebox.showinfo("All Done", "All images have been processed. The application will now exit.")
            self.root.quit()  
            return

        self.load_image()


    def display_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((1280, 512)) 
        self.image_tk = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.image_tk)

    def on_continue(self):
        self.process_image()

    def calculate_score(self, area, circularity, white_percentage):
        area_score = min(max((area - 100) / (1000 - 100), 0), 1) 
        circularity_score = min(max((circularity - 0.48) / (1 - 0.48), 0), 1) 
        white_percentage_score = min(max((white_percentage - 0.05) / (1 - 0.05), 0), 1)  

        score = 0.15 * area_score + 0.7 * circularity_score + 0.15 * white_percentage_score
        return score
    
    def process_image(self):
        file_name = os.path.abspath(os.path.join(self.image_dirs, self.current_image_path))
        img_name = os.path.basename(self.current_image_path)
        out_dir = os.path.join("processed", img_name)

        tile = cv2.imread(file_name)
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
                # white_pixels = cv2.countNonZero(neighborhood)
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

        self.update_coordinates_file()

        os.makedirs("processed", exist_ok=True)
        cv2.imwrite(out_dir, tile)

        self.display_image(out_dir)
        self.show_post_processing_options()


    def update_coordinates_file(self):
        output_file = f"{self.neut_coords_dir}/{os.path.splitext(self.current_image_path)[0]}_coords.txt"
        os.makedirs(self.neut_coords_dir, exist_ok=True)

        with open(output_file, "w") as file:
            for rect in self.rectangles:
                x1, y1, x2, y2, color = rect
                file.write(f"{x1},{y1},{x2},{y2},{color[0]},{color[1]},{color[2]}\n")


    def save_rectangle(self, x1, y1, x2, y2):
        color = (0, 255, 0) 

        self.rectangles.append((x1, y1, x2, y2, color))

        self.update_coordinates_file()


  

    def remove_rectangle(self, x, y):
        canvas_width = 1280
        canvas_height = 512
        image = cv2.imread(f"processed/{self.current_image_path}")
        original_height, original_width = image.shape[:2]

        scale_x = original_width / canvas_width
        scale_y = original_height / canvas_height

        scaled_x = int(x * scale_x)
        scaled_y = int(y * scale_y)

        for rect in self.rectangles:
            x1, y1, x2, y2, score = rect
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

                updated_image_path = f"processed/{self.current_image_path}"
                cv2.imwrite(updated_image_path, image)

                updated_image = Image.open(updated_image_path).resize((1280, 512))
                self.image_tk_edit = ImageTk.PhotoImage(updated_image)
                self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)

                messagebox.showinfo("Rectangle Removed", f"Removed rectangle: {rect}")
                return



        updated_image_path = f"processed/{self.current_image_path}"
        cv2.imwrite(updated_image_path, image)

        updated_image = Image.open(updated_image_path).resize((1280, 512))
        self.image_tk_edit = ImageTk.PhotoImage(updated_image)
        self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)

        messagebox.showinfo("No Match", "No rectangle found at the clicked location.")






    def show_post_processing_options(self):
        for widget in self.root.pack_slaves():
            if isinstance(widget, tk.Button):
                widget.destroy()

        self.save_button = tk.Button(self.root, text="Save", command=self.on_save)
        self.edit_button = tk.Button(self.root, text="Edit", command=self.on_edit)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.edit_button.pack(side=tk.RIGHT, padx=5)

    def on_save(self):
        final_image_path = f"final/{self.current_image_path}"
        os.makedirs("final", exist_ok=True)

        processed_image_path = f"processed/{self.current_image_path}"
        image = cv2.imread(processed_image_path)
        cv2.imwrite(final_image_path, image)
        print(f"Image saved to final folder: {final_image_path}")

        with open("image_data.txt", "a") as file:
            file.write(f"Processed and saved image: {self.current_image_path}\n")

        self.reset_to_initial_page()

    def on_edit(self):
        self.enter_edit_mode()

    def enter_edit_mode(self):
        for widget in self.root.pack_slaves():
            widget.destroy()

        self.mode_label = tk.Label(self.root, text="Mode:")
        self.mode_label.pack()
        self.add_radio = tk.Radiobutton(self.root, text="Add", variable=self.mode, value="Add", command=self.update_mode)
        self.remove_radio = tk.Radiobutton(self.root, text="Remove", variable=self.mode, value="Remove", command=self.update_mode)
        self.add_radio.pack()
        self.remove_radio.pack()

        self.edit_canvas = tk.Canvas(self.root, width=1280, height=512, bg="gray")
        self.edit_canvas.pack()

        processed_image_path = f"processed/{self.current_image_path}"
        self.current_image = cv2.imread(processed_image_path) 
        self.image_tk_edit = ImageTk.PhotoImage(Image.open(processed_image_path).resize((1280, 512)))
        self.canvas_image = self.edit_canvas.create_image(0, 0, anchor=tk.NW, image=self.image_tk_edit)

        self.edit_canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.edit_canvas.bind("<B1-Motion>", self.on_drag_move)
        self.edit_canvas.bind("<ButtonRelease-1>", self.on_drag_end)

        self.finalize_button = tk.Button(self.root, text="Save", command=self.save_final_image)
        self.finalize_button.pack(side=tk.RIGHT, padx=5)

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
            image = cv2.imread(f"processed/{self.current_image_path}")
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



    def redraw_image(self):
        base_image_path = f"processed/{self.current_image_path}"
        image = cv2.imread(base_image_path)

        for rect in self.rectangles:
            x1, y1, x2, y2, color = rect
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        updated_image_path = f"processed/{self.current_image_path}"
        cv2.imwrite(updated_image_path, image)
        updated_image = Image.open(updated_image_path).resize((1280, 512))
        self.image_tk_edit = ImageTk.PhotoImage(updated_image)
        self.edit_canvas.itemconfig(self.canvas_image, image=self.image_tk_edit)



    def update_mode(self):
        if self.mode.get() == "Add":
            self.edit_canvas.config(cursor="arrow")
        elif self.mode.get() == "Remove":
            self.edit_canvas.config(cursor="crosshair")

    def save_final_image(self):
        final_image_path = f"final/{self.current_image_path}"
        os.makedirs("final", exist_ok=True)
        cv2.imwrite(final_image_path, self.current_image)
        print(f"Final image saved to: {final_image_path}")
        self.reset_to_initial_page()

    def reset_to_initial_page(self):
        for widget in self.root.pack_slaves():
            widget.destroy()
        self.image_label = tk.Label(self.root)
        self.image_label.pack(pady=10)
        self.load_image()
        self.continue_button = tk.Button(self.root, text="Continue", command=self.on_continue)
        self.regenerate_button = tk.Button(self.root, text="Next Image", command=self.load_next_image)
        self.continue_button.pack(side=tk.LEFT, padx=5)
        self.regenerate_button.pack(side=tk.RIGHT, padx=5)

    def load_coordinates(self, image_name):
        coord_file = f"{self.neut_coords_dir}/{os.path.splitext(image_name)[0]}_coords.txt"
        rectangles = []
        if os.path.exists(coord_file):
            with open(coord_file, "r") as file:
                for line in file:
                    x1, y1, x2, y2, r, g, b = map(int, line.strip().split(","))
                    color = (r, g, b)
                    rectangles.append((x1, y1, x2, y2, color)) 
        return rectangles



if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()