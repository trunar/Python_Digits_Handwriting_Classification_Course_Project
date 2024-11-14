import tkinter as tk
from tkinter import messagebox

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.src.models import Sequential
from keras.src.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from keras.src.legacy.preprocessing.image import ImageDataGenerator

import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
import pickle
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from PIL import ImageGrab, ImageTk

dataset_path = 'dataset'
image_size = (28, 28)
batch_size = 32
epochs = 10

def check_files():
    cnn_file_exists = os.path.exists('cnnmodel.pkl')
    mlp_file_exists = os.path.exists('mlpmodel.pkl')

    cnn_button.config(state="normal" if cnn_file_exists else "disabled")
    mlp_button.config(state="normal" if mlp_file_exists else "disabled")

def train_models():
    train_cnn()
    train_mlp()
    check_files()
    messagebox.showinfo("Training", f"Models are successfully trained and saved.")

def train_cnn():
    print(f"CNN training started")
    start_time = time.time()
    train_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        dataset_path,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        color_mode='grayscale'
    )

    model = Sequential()
    model.add(Input(shape=(28, 28, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_generator, epochs=epochs, verbose=0)

    print(f"Training complete. Time: {time.time() - start_time:.2f} seconds")

    with open('cnnmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

def train_mlp():
    print(f"MLP training started")
    start_time = time.time()
    features = []
    labels = []

    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for i, image_name in enumerate(os.listdir(label_path)):
                image_path = os.path.join(label_path, image_name)

                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, image_size)

                image = image / 255.0

                assert image.shape == image_size, f"Expected image size {image_size}, got {image.shape}"

                image = image.flatten()

                features.append(image)
                labels.append(label)

    features = np.array(features)
    labels = np.array(labels)

    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    model = Sequential()
    model.add(Input(shape=(28 * 28,)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(features, labels, epochs=epochs, batch_size=batch_size, verbose=0)

    print(f"Training complete. Time: {time.time() - start_time:.2f} seconds")

    with open('mlpmodel.pkl', 'wb') as f:
        pickle.dump(model, f)

def run_model():
    x1, y1 = canvas.winfo_rootx(), canvas.winfo_rooty()
    x2, y2 = x1 + canvas.winfo_width(), y1 + canvas.winfo_height()

    img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
    img = img.resize(image_size)
    img = img.convert('L')

    img_array = np.array(img)

    img_array = img_array / 255.0

    assert img_array.shape == image_size, f"Expected image size {image_size}, got {img_array.shape}"

    img_array = img_array.flatten().reshape(1, -1)

    start_time = time.time()

    if selected_model.get() == "CNN":
        with open('cnnmodel.pkl', 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(img_array.reshape(1, 28, 28, 1))
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)
    elif selected_model.get() == "MLP":
        with open('mlpmodel.pkl', 'rb') as f:
            model = pickle.load(f)
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        confidence = np.max(predictions)

    elapsed_time = time.time() - start_time

    prediction_label.config(text=f"Prediction: {predicted_class}")
    confidence_label.config(text=f"Confidence: {confidence * 100:.2f}%")
    time_label.config(text=f"Time: {elapsed_time:.2f}s")

    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk

def clear_canvas():
    canvas.delete("all")
    prediction_label.config(text="Prediction: ")
    confidence_label.config(text="Confidence: ")
    time_label.config(text="Time: ")
    label_image.config(image='')

def check_run_button():
    if selected_model.get() == "None":
        run_button.config(state="disabled")
    else:
        run_button.config(state="normal")

def draw(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=15)

# Colors
main_color = "#3c3836"
secondary_color = "#ebdbb2"

# Window
root = tk.Tk()
root.title("Digit Handwriting Classification")
root.geometry("450x425")
root.resizable(False, False)
root.configure(bg=main_color)

# Frame Radio1 Radio2, Train
frame_radio_train = tk.Frame(root, bg=main_color)
frame_radio_train.pack(pady=10)

selected_model = tk.StringVar(value="None")
cnn_button = tk.Radiobutton(frame_radio_train, text="CNN Model", variable=selected_model, value="CNN", bg=secondary_color, fg=main_color, cursor="hand2", command=check_run_button)
cnn_button.pack(side="left", padx=10)

mlp_button = tk.Radiobutton(frame_radio_train, text="MLP Model", variable=selected_model, value="MLP", bg=secondary_color, fg=main_color, cursor="hand2", command=check_run_button)
mlp_button.pack(side="left", padx=10)

train_button = tk.Button(frame_radio_train, text="Train", command=train_models, bg=secondary_color, fg=main_color, bd=0, cursor="hand2")
train_button.pack(side="left", padx=10)

# Canvas
canvas = tk.Canvas(root, width=200, height=200, bg="white")
canvas.pack(pady=0)
canvas.bind("<B1-Motion>", draw)

label_image = tk.Label(root)
label_image.pack(pady=10)

# Frame Clear Run
frame_clear_run = tk.Frame(root, bg=main_color)
frame_clear_run.pack(pady=10)

clear_button = tk.Button(frame_clear_run, text="Clear", command=clear_canvas, bg=secondary_color, fg=main_color, cursor="hand2")
clear_button.pack(side="left", padx=10)

run_button = tk.Button(frame_clear_run, text="Run", command=run_model, bg=secondary_color, fg=main_color, cursor="hand2", state="disabled")
run_button.pack(side="left", padx=10)

# Frame Prediction і Confidence
frame_prediction_confidence = tk.Frame(root, bg=main_color)
frame_prediction_confidence.pack(pady=10)

prediction_label = tk.Label(frame_prediction_confidence, text="Prediction: ", bg=main_color, fg=secondary_color)
prediction_label.pack(side="left", padx=10)

confidence_label = tk.Label(frame_prediction_confidence, text="Confidence: ", bg=main_color, fg=secondary_color)
confidence_label.pack(side="left", padx=10)

# Time
time_label = tk.Label(root, text="Time: ", bg=main_color, fg=secondary_color)
time_label.pack(pady=2)

check_files()

root.mainloop()
