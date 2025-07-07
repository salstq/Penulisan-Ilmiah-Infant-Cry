import streamlit as st
import numpy as np
import onnxruntime as ort
import tensorflow_hub as hub
import librosa
from sklearn.preprocessing import LabelEncoder

# === Judul app ===
st.title("Deteksi Jenis Tangisan Bayi 👶🔊")

# === Load model dan encoder ===
@st.cache_resource
def load_yamnet():
    return hub.load("https://tfhub.dev/google/yamnet/1")

@st.cache_resource
def load_onnx_model():
    return ort.InferenceSession("best_model.onnx")

@st.cache_resource
def load_label_encoder():
    encoder = LabelEncoder()
    encoder.classes_ = np.load("classes.npy", allow_pickle=True)
    return encoder

yamnet_model = load_yamnet()
onnx_model = load_onnx_model()
encoder = load_label_encoder()

# === Fungsi prediksi ===
def extract_mean_embedding(file):
    wav, sr = librosa.load(file, sr=16000)
    wav_tensor = tf.convert_to_tensor(wav, dtype=tf.float32)  # tetap pakai tf di sini
    _, embeddings, _ = yamnet_model(wav_tensor)
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    return np.expand_dims(mean_embedding, axis=0)

def predict_audio_class(audio_file):
    embedding = extract_mean_embedding(audio_file).astype(np.float32)
    inputs = {onnx_model.get_inputs()[0].name: embedding}
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
