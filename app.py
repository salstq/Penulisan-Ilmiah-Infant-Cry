import streamlit as st
import numpy as np
import tflite_runtime.interpreter as tflite
import librosa
from scipy.io import wavfile

# === Judul Aplikasi ===
st.title("👶 Deteksi Jenis Tangisan Bayi - TFLite")

# === Load model TFLite ===
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="best_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# === Label kelas ===
labels = ['belly pain', 'burping', 'discomfort', 'hungry', 'tired', 'other']

# === Fungsi Prediksi ===
def predict(file):
    y, sr = librosa.load(file, sr=22050, mono=True)
    
    # Ekstraksi fitur MFCC → kemudian dirata-rata
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)  # shape (40,)
    
    # Siapkan input model (shape: (1, 1024))
    input_data = np.zeros((1, 1024), dtype=np.float32)
    input_data[0, :len(mfcc_mean)] = mfcc_mean

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output_data)
    confidence = output_data[0][pred_index]
    return labels[pred_index], confidence

# === Upload file audio ===
uploaded_file = st.file_uploader("Upload file audio (.wav / .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner("Menganalisis audio..."):
        label, confidence = predict(uploaded_file)
        st.success(f"🎯 Prediksi: **{label}**")
        st.info(f"📊 Keyakinan Model: {confidence:.2%}")
