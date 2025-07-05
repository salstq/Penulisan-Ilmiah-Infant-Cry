import streamlit as st
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# === Load model dan encoder ===
st.title("Deteksi Jenis Tangisan Bayi 👶🔊")

@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_resource
def load_classifier_model():
    return load_model("best_model.h5")

@st.cache_resource
def load_label_encoder():
    encoder = LabelEncoder()
    encoder.classes_ = np.load("classes.npy", allow_pickle=True)
    return encoder

yamnet_model = load_yamnet()
best_model = load_classifier_model()
encoder = load_label_encoder()

# === Fungsi prediksi ===
def extract_mean_embedding(file):
    # Load audio dan resample ke 16kHz
    wav, sr = librosa.load(file, sr=16000)
    wav_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)
    
    # Dapatkan embedding dari YAMNet
    _, embeddings, _ = yamnet_model(wav_tensor)
    mean_embedding = tf.reduce_mean(embeddings, axis=0)
    return tf.expand_dims(mean_embedding, axis=0)  # Tambah batch dimensi

def predict_audio_class(audio_file):
    embedding = extract_mean_embedding(audio_file)
    prediction = best_model.predict(embedding)
    predicted_index = np.argmax(prediction)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction)
    return predicted_label, confidence

# === Upload audio ===
uploaded_file = st.file_uploader("Upload file audio .wav", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    with st.spinner("Mendeteksi..."):
        label, confidence = predict_audio_class(uploaded_file)
        st.success(f"✅ Prediksi: **{label}**")
        st.info(f"Akurasi keyakinan model: {confidence:.2%}")
