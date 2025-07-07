import streamlit as st
import numpy as np
import onnxruntime as ort
import librosa
from sklearn.preprocessing import LabelEncoder

# === Judul app ===
st.title("Deteksi Jenis Tangisan Bayi 👶🔊")

# === Load model dan encoder ===
@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("best_model.onnx")

@st.cache_resource
def load_label_encoder():
    encoder = LabelEncoder()
    encoder.classes_ = np.load("classes.npy", allow_pickle=True)
    return encoder

onnx_model = load_onnx_model()
encoder = load_label_encoder()

# === Fungsi Ekstraksi Fitur (MFCC) ===
def extract_mfcc_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)  # Sampling rate standar
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_scaled = np.mean(mfcc.T, axis=0)  # Ambil rata-rata setiap fitur
    return np.expand_dims(mfcc_scaled, axis=0)  # Bentuk: (1, 40)

# === Fungsi prediksi ===
def predict_audio_class(audio_file):
    features = extract_mfcc_features(audio_file).astype(np.float32)
    inputs = {onnx_model.get_inputs()[0].name: features}
    prediction = onnx_model.run(None, inputs)[0]
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
