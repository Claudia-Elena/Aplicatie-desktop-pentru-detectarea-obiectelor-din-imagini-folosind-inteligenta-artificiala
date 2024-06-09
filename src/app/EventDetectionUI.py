import logging
import tkinter as tk
import webbrowser
from tkinter import filedialog
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
from src.LoadLoggingConfig import load_logging_config
from src.app.EventDetection import EventDetection
from src.db.DatabaseManager import DatabaseManager

class EventDetectionUI:
    def __init__(self, root):
        self.detected_objects_buttons_dict = {}
        try:
            load_logging_config()
            self.logger = logging.getLogger('staging')

            self.root = root
            self.root.title("Object Detection Application")
            self.root.resizable(False, False)

            self.backend = EventDetection()
            self.db_manager = DatabaseManager()

            self.image_path = ""
            self.image = None

            self.canvas = tk.Canvas(root, width=500, height=500)
            self.canvas.pack()

            self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
            self.select_button.pack()
            self.save_button = tk.Button(root, text="Save Detected Objects", command=self.save_detected_objects)
            self.save_button.pack()
            self.save_button.pack_forget()  # Hide the button

            self.label_text = tk.StringVar()
            self.label = tk.Label(root, textvariable=self.label_text)
            self.label.pack()

            self.detected_objects_buttons = []

            self.object_id_counter = 0  # Counter for assigning unique IDs to objects

        except Exception as e:
            self.logger.error("An error occurred during UI initialization: %s", str(e))

    def save_detected_objects(self):
        try:
            if not self.detected_objects_with_ids:
                self.logger.error("No objects detected to save.")
                return

            save_directory = filedialog.askdirectory(title="Select Directory to Save Objects")
            if not save_directory:
                self.logger.info("Saving canceled by user.")
                return

            for obj_id, obj_label, coords, confidence in self.detected_objects_with_ids:
                x, y, w, h = coords
                cropped_img = self.image[y:y + h, x:x + w]

                # Generate filename
                filename = f"{obj_label}_{obj_id}.jpg"  # Customize filename as needed

                # Save cropped image
                cv2.imwrite(os.path.join(save_directory, filename), cropped_img)

                # Save object details to database
                self.db_manager.insert_object(obj_label, confidence, x, y, w, h)

            self.logger.info("Detected objects saved successfully.")

        except Exception as e:
            self.logger.error("An error occurred while saving detected objects: %s", str(e))

    def show_saved_objects(self):
        try:
            saved_objects = self.db_manager.get_all_objects()
            if not saved_objects:
                self.label_text.set("No saved objects found.")
                return

            for obj in saved_objects:
                obj_id, label, confidence, x, y, w, h = obj
                self.logger.info(f"ID: {obj_id}, Label: {label}, Confidence: {confidence}, Coordinates: ({x}, {y}, {w}, {h})")
                # Here you can create buttons or labels to display these objects as needed
        except Exception as e:
            self.logger.error("An error occurred while retrieving saved objects: %s", str(e))

    def count_objects(self):
        try:
            object_counts = {}
            for obj_id, obj_label, _, _ in self.detected_objects_with_ids:
                object_counts[obj_label] = object_counts.get(obj_label, 0) + 1

            self.logger.info("Object Counts: " + str(object_counts))
            count_text = ", ".join([f"{label}: {count}" for label, count in object_counts.items()])
            if count_text:
                self.label_text.set("Detected Objects: {0}".format(count_text))
            else:
                self.label_text.set("Detected Objects: No objects detected")
        except Exception as e:
            self.logger.error("An error occurred while counting objects: %s", str(e))

    def display_cropped_image(self, selected_object_data):
        try:
            for obj_label, coords, _ in selected_object_data:
                x, y, w, h = coords
                cropped_img = self.image[y:y + h, x:x + w]
                cropped_img_pil = Image.fromarray(np.uint8(cropped_img))
                cropped_img_tk = ImageTk.PhotoImage(image=cropped_img_pil)

                cropped_window = tk.Toplevel(self.root)
                cropped_canvas = tk.Canvas(cropped_window, width=w, height=h)
                cropped_canvas.pack()
                cropped_canvas.create_image(0, 0, anchor=tk.NW, image=cropped_img_tk)
                cropped_canvas.image = cropped_img_tk

        except Exception as e:
            self.logger.error("An error occurred in display_cropped_image: %s", str(e))

    def display_image(self):
        try:
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            img_height, img_width, _ = img.shape
            detected_objects = self.backend.detect_objects(self.image_path)

            # Calculate the zoom factor based on the canvas size
            zoom_factor = min(self.canvas.winfo_width() / img_width, self.canvas.winfo_height() / img_height)

            # Resize the image for zooming
            img_resized = cv2.resize(img, None, fx=zoom_factor, fy=zoom_factor)

            for obj_label, coords, confidence in detected_objects:
                x, y, w, h = coords
                x, y, w, h = int(x * zoom_factor), int(y * zoom_factor), int(w * zoom_factor), int(h * zoom_factor)
                label_text = f"{obj_label} ({confidence * 100:.2f}%)"

                # Draw contours around the object
                cv2.drawContours(img_resized, [np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])], 0,
                                 (0, 255, 0), 2)

                # Calculate the center of the contour
                center_x = x + w // 2
                center_y = y + h // 2

                # Create clickable text label at the center of the contour
                clickable_text = f"Learn more about {obj_label}"
                text_id = f"clickable_{obj_label}"
                self.canvas.create_text(center_x, center_y, text=clickable_text, fill="blue",
                                        font=("Arial", 10), tags=text_id)
                self.canvas.tag_bind(text_id, "<Button-1>",
                                     lambda event, label=obj_label: self.open_wikipedia_page(label))

            img_pil = Image.fromarray(np.uint8(img_resized))
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk

            if self.image is None:
                raise RuntimeError("Image not loaded!")

        except Exception as e:
            self.logger.error("An error occurred during image display: %s", str(e))

    def open_wikipedia_page(self, obj_label):
        # Open web browser with Wikipedia page of the detected object
        url = f"https://en.wikipedia.org/wiki/{obj_label.replace(' ', '_')}"
        webbrowser.open(url)

    def select_image(self):
        try:
            self.image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png")])

            if self.image_path:
                self.logger.info("Selected image path: " + self.image_path)
                self.image = cv2.imread(self.image_path)

                if self.image is not None:
                    self.logger.info("Image loaded successfully.")
                    self.display_image()
                    self.detect_objects()
                    self.save_button.pack()  # Display the button
                else:
                    raise RuntimeError("Failed to load image.")
        except Exception as e:
            self.logger.error("An error occurred during image selection: %s", str(e))

    def detect_objects(self):
        try:
            if self.image is not None and self.image_path:
                detected_objects = self.backend.detect_objects(self.image_path)
                self.detected_objects_with_ids = []
                self.object_id_counter = 0

                for obj_label, coords, confidence in detected_objects:
                    self.object_id_counter += 1
                    obj_id = self.object_id_counter
                    self.detected_objects_with_ids.append((obj_id, obj_label, coords, confidence))

                self.logger.info("Image path: " + self.image_path)
                self.logger.info("Detected objects: " + str(self.detected_objects_with_ids))
                self.label_text.set("Detected Objects: Detecting...")

                if not self.detected_objects_with_ids:
                    self.label_text.set("Detected Objects: No objects detected")
                    raise RuntimeError("No objects detected")

                for button in self.detected_objects_buttons:
                    button.destroy()
                self.detected_objects_buttons.clear()

                for obj_id, obj_label, coords, confidence in self.detected_objects_with_ids:
                    button = tk.Button(self.root,
                                       text=f"Crop the object: {obj_label} ({confidence * 100:.2f}%) from the image",
                                       command=lambda id=obj_id: self.display_selected_object(id))

                    button.pack(pady=0)  # Adjust the padding here
                    self.detected_objects_buttons.append(button)
                    self.detected_objects_buttons_dict[obj_id] = button

                    # Save detected object details to database
                    x, y, w, h = coords
                    self.db_manager.insert_object(obj_label, confidence, x, y, w, h)

                self.count_objects()  # Update object counts

        except Exception as e:
            self.logger.error("An error occurred during object detection: %s", str(e))

    def display_selected_object(self, obj_id):
        try:
            selected_object = None
            for obj_id_detected, obj_label, coords, confidence in self.detected_objects_with_ids:
                if obj_id_detected == obj_id:
                    selected_object = (obj_label, coords, confidence)
                    break

            if selected_object is not None:
                obj_label, coords, confidence = selected_object
                x, y, w, h = coords
                h, w, x, y = self.adjust_object_coordinates(h, w, x, y)

                cropped_img = self.image[y:y + h, x:x + w]
                cropped_img_pil = Image.fromarray(np.uint8(cropped_img))
                cropped_img_tk = ImageTk.PhotoImage(image=cropped_img_pil)

                if hasattr(self, 'cropped_window') and self.cropped_window.winfo_exists():
                    self.cropped_window.destroy()  # Destroy the previous window if it exists
                self.cropped_window = tk.Toplevel(self.root)
                self.cropped_canvas = tk.Canvas(self.cropped_window, width=w, height=h)
                self.cropped_canvas.pack()
                self.cropped_canvas.create_image(0, 0, anchor=tk.NW, image=cropped_img_tk)
                self.cropped_canvas.image = cropped_img_tk

                # Open web browser with Wikipedia page of the detected object
                url = f"https://en.wikipedia.org/wiki/{obj_label.replace(' ', '_')}"
                clickable_text = f"link to {obj_label}"
                text_id = f"clickable_{obj_label}"
                self.cropped_canvas.create_text(w // 2, h // 2, text=clickable_text, fill="blue",
                                                font=("Arial", 10), tags=text_id)
                self.cropped_canvas.tag_bind(text_id, "<Button-1>",
                                             lambda event, label=obj_label: self.open_wikipedia_page(label))


        except Exception as e:
            self.logger.error("An error occurred during  display_selected_object: %s", str(e))

    def adjust_object_coordinates(self, h, w, x, y):
        h = max(h, 0)
        w = max(w, 0)
        x = max(x, 0)
        y = max(y, 0)
        return h, w, x, y







