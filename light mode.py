import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Real Time Digit Recognizer")
        self.root.geometry("400x600")
        self.root.resizable(False, False)

        # Light Mode Colors
        self.bg_color = "#F5F5F5"  # Light background
        self.text_color = "#333333"  # Dark text
        self.output_color = "#333333"  # Dark output text

        self.root.configure(bg=self.bg_color)

        # Title Label
        title = tk.Label(
            root,
            text="Draw a Digit",
            font=("SF Pro Display",40, "bold"),
            fg=self.text_color,
            bg=self.bg_color,
        )
        title.pack(pady=2)

        # Drawing Area (Canvas)
        self.canvas_frame = tk.Frame(root, bg=self.bg_color)
        self.canvas_frame.pack(pady=10)
        self.canvas_box = tk.Frame(self.canvas_frame, bg="white", bd=2, relief="solid")
        self.canvas_box.pack(padx=10, pady=5)

        # Canvas with white background and black ink
        self.canvas = tk.Canvas(
            self.canvas_box, width=280, height=290, bg="white", bd=0, highlightthickness=0
        )
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.pack()

        # Buttons Section (Increased Size)
        self.buttons_frame = tk.Frame(self.root, bg=self.bg_color)
        self.buttons_frame.pack(pady=20)

        # Clear Button (Larger Size)
        self.clear_icon = Image.open("erase-icon.png").resize((60, 60))
        self.clear_icon = ImageTk.PhotoImage(self.clear_icon)
        self.clear_button = tk.Button(
            self.buttons_frame,
            image=self.clear_icon,
            bg=self.bg_color,
            bd=0,
            activebackground="#333333",
            command=self.clear_canvas,
            width=70,
            height=70,
            highlightthickness=0,
            borderwidth=4,
        )
        self.clear_button.grid(row=0, column=0, padx=20)

        # Predict Button (Larger Size)
        self.predict_icon = Image.open("predict-icon.png").resize((60, 60))  
        self.predict_icon = ImageTk.PhotoImage(self.predict_icon)
        self.predict_button = tk.Button(
            self.buttons_frame,
            image=self.predict_icon,
            command=self.predict_digit,
            bg=self.bg_color,
            bd=0,
            relief="raised",  
            width=70,
            height=70,
            highlightthickness=0,
            borderwidth=4,
            activebackground="#333333",
        )
        self.predict_button.grid(row=0, column=1, padx=20)

        # Output Label (Below Predict and Clear Button)
        self.output_label = tk.Label(
            self.root,
            text="",
            font=("SF Pro Display",35, "bold"),  
            fg=self.output_color,
            bg=self.bg_color,
        )
        self.output_label.pack(pady=2)

        # PIL Image for canvas drawing
        self.image = Image.new("L", (280, 290), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        radius = 8  # Brush radius size 10
        self.canvas.create_oval(
            x - radius, y - radius, x + radius, y + radius, fill="black", width=10
        )
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 290], fill="white")
        self.output_label.config(text="")

    def predict_digit(self):
        # Resize and preprocess image
        image_resized = self.image.resize((28, 28))
        image_inverted = ImageOps.invert(image_resized)
        image_array = np.array(image_inverted).astype("float32") / 255
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict using the model
        prediction = model.predict(image_array).argmax()

        # Show predicted output
        self.output_label.config(text=f"Digit = {prediction}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
