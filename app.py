import tflite_runtime.interpreter as tflite
import numpy as np
import librosa
import streamlit as st
from scipy.io import wavfile

st.title("👶 Deteksi Tangisan Bayi (TFLite - Tanpa TensorFlow)")

@st.cache_resource
def load_model():
    interpreter = tflite.Interpreter(model_path="best_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

labels = ['belly pain', 'burping', 'discomfort', 'hungry', 'tired', 'other']

def predict(file):
    y, sr = librosa.load(file, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    input_data = np.zeros((1, 1024), dtype=np.float32)
    input_data[0, :len(mfcc_mean)] = mfcc_mean

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    pred_index = np.argmax(output_data)
    confidence = output_data[0][pred_index]
    return labels[pred_index], confidence

uploaded_file = st.file_uploader("Upload file audio (.wav / .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    with st.spinner("Menganalisis audio..."):
        label, confidence = predict(uploaded_file)
        st.success(f"🎯 Prediksi: **{label}**")
        st.info(f"📊 Keyakinan Model: {confidence:.2%}")
