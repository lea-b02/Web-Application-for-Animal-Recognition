import speech_recognition as sr
import tkinter as tk
from tkinter import messagebox, filedialog, ttk
from PIL import Image, ImageTk
import threading
import cv2
import numpy as np
import torch
import sounddevice as sd
from scipy.io.wavfile import write
from ultralytics import YOLO
import librosa
import os
import sys

# Determine the path to the model file
if getattr(sys, 'frozen', False):
    model_path = os.path.join(sys._MEIPASS, 'best.pt')
else:
    model_path = 'best.pt'

# Initialize recognizer and YOLO model
recognizer = sr.Recognizer()
audio_data = None
target_class = None
model = YOLO('best.pt')
img_path = None
mic_index = None

# Load the trained sound model
sound_model = torch.load("sound_model1.pth", map_location=torch.device('cpu'))
sound_model.eval()

def select_image():
    global img_path
    img_path = filedialog.askopenfilename(
        title="Select Image",
        filetypes=(("Image files", "*.jpg *.jpeg *.png *.bmp *.gif"), ("All files", "*.*"))
    )
    if img_path:
        display_initial_image()

def display_initial_image():
    if img_path:
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        display_size = (500, 400)
        img_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    else:
        image_label.config(image='')
        image_label.image = None

def populate_microphone_list():
    """Populates the dropdown menu with available microphones."""
    devices = sd.query_devices()
    mic_options = [f"{index}: {device['name']}" for index, device in enumerate(devices) if device['max_input_channels'] > 0]
    microphone_dropdown['values'] = mic_options
    if mic_options:
        microphone_dropdown.current(0)  # Set the first microphone as default
        select_microphone(microphone_dropdown.get())  # Pass the full string from dropdown
    else:
        result_label.config(text="No microphones found.")

def select_microphone(selected_index):
    """Sets the selected microphone index from the dropdown."""
    global mic_index
    if isinstance(selected_index, str):  # If string from dropdown
        mic_index = int(selected_index.split(":")[0])  # Extract the index
        result_label.config(text=f"Selected microphone: {selected_index}")
    else:  # If direct integer index
        mic_index = selected_index
        result_label.config(text=f"Selected microphone: {mic_index}")

def record_audio():
    global target_class, mic_index
    if mic_index is None:
        result_label.config(text="No microphone selected. Please choose one.")
        return
    if img_path is None:
        result_label.config(text="Please select an image first.")
        return

    fs = 44100
    duration = 3
    result_label.config(text="Recording for 3 seconds... Speak now!")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16', device=mic_index)
    sd.wait()
    write('recorded_sound.wav', fs, audio)

    try:
        with sr.AudioFile('recorded_sound.wav') as source:
            audio_data = recognizer.record(source)
            target_class = recognizer.recognize_google(audio_data).lower()
        result_label1.config(text=f"Class: {target_class}")
        if target_class in ["cat", "cats"]:
            result_label.config(text=f"Recognized text: {target_class}")
            run_yolo("cat")  
        elif target_class in ["dog", "dogs"]:
            result_label.config(text=f"Recognized text: {target_class}")
            run_yolo("dog")
        else:
            result_label.config(text=f"Unrecognized text: {target_class}. Trying sound prediction.")
            sound_class, confidence = predict_sound('recorded_sound.wav')
            result_label1.config(text=f"Class is: {sound_class}")
            if sound_class in ["dog", "cat"]:
                run_yolo(sound_class)
                result_label.config(text=f"Sound prediction confidence: {confidence:.2%}")
            else:
                result_label.config(text=f"No 'Cat' or 'Dog' detected in the sound. Detected sound is: {target_class}")
    except sr.UnknownValueError:
        result_label.config(text="Could not understand the audio.")

def predict_sound(audio_file):
    audio, sr = librosa.load(audio_file, sr=16000)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=20)
    max_frames = 60
    if mfcc.shape[1] > max_frames:
        mfcc = mfcc[:, :max_frames]
    else:
        padding = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, padding)), mode="constant")
    mfcc_tensor = torch.tensor(mfcc, dtype=torch.float32).T.unsqueeze(0)
    with torch.no_grad():
        output = sound_model(mfcc_tensor)[0]
    confidence = torch.softmax(output, dim=0)
    pred_idx = torch.argmax(output).item()
    pred_class = 'cat' if pred_idx == 1 else 'dog'
    conf_value = confidence[pred_idx].item()
    return pred_class, conf_value

def start_recording():
    threading.Thread(target=record_audio).start()

def run_yolo(target_class):
    global img_path
    img = cv2.imread(img_path)
    results = model.predict(img_path)
    class_names = model.names
    found = False
    for result in results:
        for box in result.boxes:
            class_index = int(box.cls[0])
            label = class_names[class_index].lower()
            if label == target_class:
                found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                if confidence > 0.5:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f'{label} {confidence:.2f}'
                    cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    if not found:
        result_label.config(text=f"No {target_class} detected in the image.")
    else:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        display_size = (500, 400)
        img_pil.thumbnail(display_size, Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(image=img_pil)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    os.remove('recorded_sound.wav')

# Set up Tkinter GUI
window = tk.Tk()
window.title("Speech-to-Text and YOLO Detection")

# Add dropdown for microphone selection
microphone_dropdown = ttk.Combobox(window, state="readonly", width=50)
microphone_dropdown.pack(pady=10)
microphone_dropdown.bind("<<ComboboxSelected>>", lambda event: select_microphone(microphone_dropdown.get()))

select_image_button = tk.Button(window, text="Select Image", command=select_image)
select_image_button.pack(pady=5)

start_button = tk.Button(window, text="Start Recording", command=start_recording)
start_button.pack(pady=10)

result_label1 = tk.Label(window, text="", wraplength=400, justify="left")
result_label1.pack(pady=10)

result_label = tk.Label(window, text="Select an image and microphone.", wraplength=400, justify="left")
result_label.pack(pady=10)

image_label = tk.Label(window)
image_label.pack(pady=10)

window.geometry("600x800")
populate_microphone_list()  # Populate microphones on startup
window.mainloop()
