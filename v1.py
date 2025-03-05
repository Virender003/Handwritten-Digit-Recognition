import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("mnist_model.h5")

# Create Tkinter GUI
class DigitRecognizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Handwritten Digit Recognizer")
        self.root.geometry("300x400")

        # Canvas for drawing
        self.canvas = tk.Canvas(root, width=280, height=280, bg="white")
        self.canvas.pack(pady=10)
        self.canvas.bind("<B1-Motion>", self.paint)

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Clear", command=self.clear_canvas, width=10).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Predict", command=self.predict_digit, width=10).grid(row=0, column=1, padx=5)

        # PIL Image for canvas drawing
        self.image = Image.new("L", (280, 280), "white")
        self.draw = ImageDraw.Draw(self.image)

    def paint(self, event):
        x, y = event.x, event.y
        radius = 8
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius, fill="black", width=0)
        self.draw.ellipse([x - radius, y - radius, x + radius, y + radius], fill="black")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 280, 280], fill="white")

    def predict_digit(self):
        # Resize and preprocess image
        image_resized = self.image.resize((28, 28))
        image_inverted = ImageOps.invert(image_resized)
        image_array = np.array(image_inverted).astype("float32") / 255
        image_array = image_array.reshape(1, 28, 28, 1)

        # Predict using the model
        prediction = model.predict(image_array).argmax()
        messagebox.showinfo("Prediction", f"The digit is: {prediction}")

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = DigitRecognizerApp(root)
    root.mainloop()
