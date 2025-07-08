import streamlit as st
import numpy as np
import tensorflow_hub as hub
import onnxruntime as ort
import librosa
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# === Judul Aplikasi ===
st.title("👶 Deteksi Jenis Tangisan Bayi - ONNX")

# === Load Resources ===
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("best_model.onnx")

@st.cache_resource
def load_encoder():
    encoder = LabelEncoder()
    encoder.classes_ = np.load("classes.npy", allow_pickle=True)
    return encoder

yamnet_model = load_yamnet()
onnx_model = load_onnx_model()
encoder = load_encoder()

# === Fungsi Ekstraksi Fitur ===
def extract_mean_embedding(file):
    wav, sr = librosa.load(file, sr=16000)
    wav_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(wav_tensor)
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy().astype(np.float32)
    return np.expand_dims(mean_embedding, axis=0)  # shape: (1, 1024)

# === Fungsi Prediksi ===
def predict(file):
    embedding = extract_mean_embedding(file)
    input_name = onnx_model.get_inputs()[0].name
    output = onnx_model.run(None, {input_name: embedding})[0]
    pred_index = np.argmax(output)
    pred_label = encoder.inverse_transform([pred_index])[0]
    confidence = np.max(output)
    return pred_label, confidence

# === Upload file audio ===
uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    with st.spinner("Menganalisis audio..."):
        label, confidence = predict(uploaded_file)
        st.success(f"🎯 Prediksi: **{label}**")
        st.info(f"📊 Keyakinan Model: {confidence:.2%}")
