import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imageFile, setImageFile] = useState(null); // Fichier brut de l'image

  const [audioBlob, setAudioBlob] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  
  const [isRecording, setIsRecording] = useState(false);
  const [serverResponse, setServerResponse] = useState(null); // Pour afficher la réponse du serveur

  const SERVER_URL = "http://127.0.0.1:5000"; // Adresse de votre serveur Flask

  const handleSelectImage = (event) => {
    const file = event.target.files[0];
    if (file) {
      // Réinitialiser l'état et le canvas
      setImageFile(null); // Effacer le fichier brut précédent
      setSelectedImage(null); // Effacer l'image prévisualisée précédente
      
      // Effacer le canvas
      const canvas = document.getElementById('image-canvas');
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
  
      // Charger la nouvelle image
      setImageFile(file);
      setSelectedImage(URL.createObjectURL(file));
      console.log("Selected image:", file);
    }
  };  

  const handleRecording = async () => {
    try {
      // Check if MediaRecorder is available
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("MediaRecorder is not supported in this browser.");
        return;
      }

      setAudioBlob(null);
      setAudioUrl(null);
  
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  
      // Check if MediaRecorder is supported in the current browser
      if (typeof MediaRecorder === "undefined") {
        alert("MediaRecorder is not supported in your browser.");
        return;
      }
  
      const mediaRecorder = new MediaRecorder(stream);
      let audioChunks = [];
  
      // Start recording
      setIsRecording(true);
  
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
  
      mediaRecorder.onstop = () => {
        if (audioChunks.length > 0) {
          const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
          setAudioBlob(audioBlob);
          setAudioUrl(URL.createObjectURL(audioBlob));
          console.log("Audio recorded:", audioBlob);
        } else {
          console.error("No audio data captured.");
        }
        setIsRecording(false);
        stream.getTracks().forEach((track) => track.stop());
      };
  
      mediaRecorder.start();
      console.log("Recording started");
  
      // Stop recording after 3 seconds
      setTimeout(() => mediaRecorder.stop(), 3000);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      setIsRecording(false);
      alert("Failed to access the microphone. Check browser permissions.");
    }
  };

  const handleSendToServer = async () => {
    if (!imageFile || !audioBlob) {
      alert("Please select an image and record audio first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("audio", audioBlob, "temp_audio.wav"); // Ajouter un nom au fichier audio

    try {
      const response = await fetch(SERVER_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to process data on server");
      }

      const data = await response.json();
      console.log("Server Response:", data);
      setServerResponse(data);

      // Dessiner les détections et la prédiction audio sur l'image
      drawDetections(data.detections);
      drawAudioPrediction(data.audio_prediction);

    } catch (error) {
      console.error("Error:", error);
      alert("Failed to send data to the server.");
    }
  };

  const drawDetections = (detections) => {
    const canvas = document.getElementById('image-canvas');
    const ctx = canvas.getContext('2d');
    const img = new Image();
    img.src = selectedImage;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((detection) => {
        const [x_min, y_min, x_max, y_max] = detection.box;
        const confidence = detection.confidence;
        const label = detection.label;

        // Dessiner la boîte de détection
        ctx.beginPath();
        ctx.rect(x_min, y_min, x_max - x_min, y_max - y_min);
        ctx.lineWidth = 2;
        ctx.strokeStyle = 'red';
        ctx.stroke();

        // Ajouter le label et la confiance
        ctx.font = '16px Arial';
        ctx.fillStyle = 'blue';
        ctx.fillText(`${label}: ${confidence.toFixed(2)}`, x_min, y_min - 5);
      });
    };
  };

  const drawAudioPrediction = (audioPrediction) => {
    if (!audioPrediction) return;
  
    const { class: className, confidence } = audioPrediction;
    const canvas = document.getElementById('image-canvas');
    const ctx = canvas.getContext('2d');
  
    // Dessiner le label de la prédiction audio
    ctx.font = '20px Arial';
    ctx.fillStyle = 'green';
    
    // Positionner le texte en haut à gauche
    ctx.fillText(`Audio Prediction: ${className} (${confidence.toFixed(2)})`, 10, 30);
  };

  return (
    <div className="app-container">
      <h1>Client React</h1>
      <div className="button-container">
        <label htmlFor="image-upload" className="app-button">
          Select Image
        </label>
        <input
          type="file"
          id="image-upload"
          accept="image/*"
          onChange={handleSelectImage}
          style={{ display: 'none' }}
        />


        <button
          className="app-button"
          onClick={handleRecording}
          disabled={isRecording}
        >
          {isRecording ? "Recording..." : "Record Audio (3 sec)"}
        </button>

        {isRecording && <div className="progress-bar"></div>}
        {audioUrl && (
          <div className="audio-preview">
            <h3>Recorded Audio:</h3>
            <audio controls>
              <source src={audioUrl} type="audio/wav" />
              Your browser does not support the audio element.
            </audio>
          </div>
        )}

        <button className="app-button" onClick={handleSendToServer}>
          Send to Server
        </button>

        {selectedImage && (
          <div className="image-preview">
            <img src={selectedImage} alt="Selected" />
            <canvas id="image-canvas" style={{ position: 'absolute' }} />
          </div>
        )}
      </div>
    </div>
  );
}

export default App;