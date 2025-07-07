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
def extract_mfcc_features(file):
    wav, sr = librosa.load(file, sr=16000)
    mfcc = librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc, axis=1)
    
    # Agar jadi 1024 dimensi, kamu bisa pad/extend
    padded = np.zeros(1024, dtype=np.float32)
    length = min(1024, mfcc_mean.shape[0])
    padded[:length] = mfcc_mean[:length]
    
    return padded.reshape(1, -1)

# === Fungsi prediksi ===
def predict_audio_class(audio_file):
    features = extract_mfcc_features(audio_file).astype(np.float32)
    inputs = {"input": features}  # sesuai nama input dari ONNX model kamu
    prediction = onnx_model.run(None, inputs)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = encoder.inverse_transform([predicted_index])[0]
    confidence = np.max(prediction)
    return predicted_label, confidence


# === Fungsi prediksi ===

# === Upload audio ===
uploaded_file = st.file_uploader("Upload file audio .wav", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with st.spinner("Mendeteksi..."):
        label, confidence = predict_audio_class(uploaded_file)
        st.success(f"✅ Prediksi: **{label}**")
        st.info(f"Akurasi keyakinan model: {confidence:.2%}")
