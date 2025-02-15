import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import requests
import sounddevice as sd
from scipy.io.wavfile import write
import os

SERVER_URL = "http://127.0.0.1:5000"
TEMP_AUDIO_FILE = "target_class_audio.wav"

def select_image():
    global img_path  # Chemin de l'image sélectionnée
    # Ouvre une boîte de dialogue pour sélectionner un fichier
    img_path = filedialog.askopenfilename(
        title="Sélectionnez une image",
        filetypes=(("Fichiers d'image", "*.jpg *.jpeg *.png *.bmp *.gif"), ("Tous les fichiers", "*.*"))
    )
    if not img_path:
        messagebox.showinfo("Info", "Aucun fichier sélectionné.")
        return

    # Afficher l'image sélectionnée dans l'interface
    display_image(img_path)

def display_image(image_path):
    """
    Charge et affiche l'image sélectionnée dans le widget Tkinter.
    """
    try:
        img = Image.open(image_path)
        img.thumbnail((400, 300))  # Redimensionner l'image pour qu'elle s'adapte à l'interface
        img_tk = ImageTk.PhotoImage(img)
        image_label.config(image=img_tk)
        image_label.image = img_tk
    except Exception as e:
        messagebox.showerror("Erreur", f"Impossible d'afficher l'image : {e}")

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

TEMP_AUDIO_FILE = "temp_audio.wav"

def record_audio():
    """
    Enregistre un audio de 3 secondes depuis le microphone et le sauvegarde dans un fichier temporaire.
    """
    global mic_index
    if mic_index is None:
        result_label.config(text="No microphone selected. Please choose one.")
        return
    duration = 3  # Durée de l'enregistrement (en secondes)
    fs = 44100  # Fréquence d'échantillonnage (Hz)

    try:
        messagebox.showinfo("Info", "Enregistrement vocal en cours... Parlez maintenant !")
        
        # Enregistrement audio
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        sd.wait()  # Attendre la fin de l'enregistrement

        # Sauvegarder dans un fichier temporaire
        write(TEMP_AUDIO_FILE, fs, audio_data)
        messagebox.showinfo("Info", "Enregistrement terminé. Fichier sauvegardé !")
        return TEMP_AUDIO_FILE  # Retourner le chemin du fichier temporaire
    except Exception as e:
        messagebox.showerror("Erreur", f"Erreur lors de l'enregistrement audio : {e}")
        return None
    
def send_to_server():
    """
    Envoie l'image et l'audio au serveur Flask et affiche les résultats.
    """
    if img_path is None:
        result_label.config(text="No image selected.")
        return

    if not os.path.exists(TEMP_AUDIO_FILE):
        result_label.config(text="No audio recorded. Please record your voice.")
        return

    try:
        with open(img_path, 'rb') as img_file, open(TEMP_AUDIO_FILE, 'rb') as audio_file:
            response = requests.post(
                f"{SERVER_URL}",
                files={'image': img_file, 'audio': audio_file}
                # timeout=60
                
            )

        if response.status_code == 200:
            result = response.json()
            detections = result.get('detections', [])
            if detections:
                # Dessiner les rectangles sur l'image
                draw_detections_on_image(detections)
                result_label.config(text=f"Detections: {detections}")
            else:
                result_label.config(text="No objects detected.")
        else:
            result_label.config(text=f"Error: {response.json().get('error')}")
    except Exception as e:
        result_label.config(text=f"Error: {e}")

def draw_detections_on_image(detections):
    """
    Dessine des rectangles sur l'image à partir des détections du serveur.
    """
    global img_path

    # Charger l'image avec OpenCV
    image = cv2.imread(img_path)
    for detection in detections:
        x1, y1, x2, y2 = detection['box']  # Coordonnées de la boîte
        label = detection['label']
        confidence = detection['confidence']

        # Dessiner le rectangle et l'étiquette
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label} ({confidence:.2f})",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # Convertir l'image OpenCV en un format PIL pour l'afficher dans Tkinter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convertir en RGB
    image_pil = Image.fromarray(image)
    image_tk = ImageTk.PhotoImage(image_pil)

    # Mettre à jour le widget d'affichage
    image_label.config(image=image_tk)
    image_label.image = image_tk

# GUI Setup
window = tk.Tk()
window.title("Client Application")

img_path = None

# Add dropdown for microphone selection
microphone_dropdown = ttk.Combobox(window, state="readonly", width=50)
microphone_dropdown.pack(pady=10)
microphone_dropdown.bind("<<ComboboxSelected>>", lambda event: select_microphone(microphone_dropdown.get()))

# Bouton pour sélectionner une image
select_image_button = tk.Button(window, text="Select Image", command=select_image)
select_image_button.pack(pady=5)

# Bouton pour enregistrer un audio
record_audio_button = tk.Button(window, text="Record Audio (3 sec)", command=record_audio)
record_audio_button.pack(pady=10)

# Bouton pour envoyer l'image et l'audio au serveur
send_button = tk.Button(window, text="Send to Server", command=send_to_server)
send_button.pack(pady=10)

# Label pour afficher les résultats
result_label = tk.Label(window, text="", wraplength=400)
result_label.pack(pady=10)

# Label pour afficher l'image sélectionnée
image_label = tk.Label(window)
image_label.pack(pady=10)

# Dimensions de la fenêtre
window.geometry("600x600")
populate_microphone_list()
window.mainloop()
