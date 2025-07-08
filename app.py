import streamlit as st
import numpy as np
import onnxruntime as ort
import librosa
from sklearn.preprocessing import LabelEncoder
import io

# === Judul Aplikasi ===
st.title("👶 Deteksi Jenis Tangisan Bayi - ONNX")

# === Load Resources ===
@st.cache_resource
def load_yamnet():
    return ort.InferenceSession("yamnet.onnx")

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
    # Load audio dari file uploader
    y, sr = librosa.load(io.BytesIO(file.read()), sr=16000)
    waveform = y.astype(np.float32)

    input_name = yamnet_model.get_inputs()[0].name
    print("Output count:", len(yamnet_model.get_outputs()))
    print("Output names:", [o.name for o in yamnet_model.get_outputs()])

    # Jalankan model ONNX
    outputs = yamnet_model.run(None, {input_name: waveform})

    # Gunakan output pertama (karena hanya ada 1 output dari model kamu)
    embeddings = outputs[0]  # biasanya bentuk (n_frames, 1024)

    # Rata-ratakan ke seluruh frame (jadi 1 vektor)
    mean_embedding = np.mean(embeddings, axis=0, keepdims=True).astype(np.float32)

    return mean_embedding  # bentuk akhir (1, 1024)

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
