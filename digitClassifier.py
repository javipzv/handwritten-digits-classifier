import tkinter as tk
from PIL import Image
from io import BytesIO
import numpy as np
import tensorflow as tf

NN_model = tf.keras.models.load_model('model_NN')
CNN_model = tf.keras.models.load_model('model_CNN')

class Classifier_GUI:
    def __init__(self):
        self.model = NN_model
        self.model_name = "NN"
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
        self.model_button = tk.Button(self.canvas, text=f"Change model (using {self.model_name})", command=self.change_model, width=57)
        self.model_button.pack(side=tk.BOTTOM)

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
        
        if self.model_name == "CNN":
            im = im.reshape((1, 28, 28, 1)) # Three dimensions
        elif self.model_name == "NN":
            im = im.reshape((1, 28*28)) # One dimension
        self.im = im
        self.classify_digit()

    def delete_image(self):
        self.canvas.delete("all")

    def draw_digit(self):
        self.master.title("Digit Classifier")
        self.master.mainloop()
    
    def classify_digit(self):
        pred = self.model.predict(self.im, verbose = False)
        self.write_result(np.argmax(pred))

    def write_result(self, digit: int):
        self.label.config(text=f"Digit: {digit}")

    def change_model(self):
        if self.model_name == "NN":
            self.model = CNN_model
            self.model_name = "CNN"
        else:
            self.model = NN_model
            self.model_name = "NN"
        self.model_button.config(text=f"Change model (using {self.model_name})")

drawing = Classifier_GUI()
drawing.draw_digit()