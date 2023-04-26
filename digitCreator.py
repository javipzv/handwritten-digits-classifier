import tkinter as tk
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

class DrawingScreen:
    def __init__(self):
        self.model = tf.keras.models.load_model('model')
        self.im = None
        self.master = tk.Tk()
        self.frame = tk.Frame(self.master, width=400, height=400, highlightbackground="gray", highlightthickness=2, bg="lightblue")
        self.frame.pack(side='right', fill=tk.BOTH, expand=True)
        self.label = tk.Label(self.frame, text="", height=25, width=50, bg="lightblue")
        self.label.pack(expand=False)
        self.canvas = tk.Canvas(self.master, width=400, height=400, bg="white", highlightbackground="gray", highlightthickness=2)
        self.canvas.pack(side='left', fill=tk.BOTH, expand=True)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.save_button = tk.Button(self.canvas, text="Save", command=self.save_image, width=57)
        self.save_button.pack(side=tk.BOTTOM)
        self.delete_button = tk.Button(self.canvas, text="Delete", command=self.delete_image, width=57)
        self.delete_button.pack(side=tk.BOTTOM)

    def draw(self, event):
        x, y = event.x, event.y
        r = 5
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")

    def save_image(self):
        image = self.canvas.postscript(colormode="color")
        im = Image.open(BytesIO(image.encode("utf-8")))
        im = im.convert("L") # Black and white
        im = im.resize((28, 28)) # Resize to 28x28
        im = np.array(im) # Numpy array
        im = abs(im - 255) # Invert color
        im = im / 255 # Escale to 0-1
        self.im = im.reshape((1, 784))
        self.classify_digit()

    def delete_image(self):
        self.canvas.delete("all")

    def digit_drawer(self):
        self.master.title("Draw a digit")
        self.master.mainloop()
    
    def classify_digit(self):
        pred = self.model.predict(self.im, verbose = False)
        self.result_drawer(np.argmax(pred))

    def result_drawer(self, digit: int):
        self.label.config(text=f"Digit: {digit}")

drawing = DrawingScreen()
drawing.digit_drawer()