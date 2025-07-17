import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import io
import tflite_runtime.interpreter as tflite
from pydub import AudioSegment

# Konfigurasi halaman
st.set_page_config(
    page_title="Deteksi Tangisan Bayi",
    page_icon="👶",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Load models
@st.cache_resource
def load_models():
    yamnet = tflite.Interpreter(model_path="yamnet.tflite")
    yamnet.allocate_tensors()

    classifier = tflite.Interpreter(model_path="best_model.tflite")
    classifier.allocate_tensors()

    return yamnet, classifier

yamnet, classifier = load_models()

# I/O model
yamnet_input = yamnet.get_input_details()[0]
yamnet_output = yamnet.get_output_details()[1]  # embeddings

classifier_input = classifier.get_input_details()[0]
classifier_output = classifier.get_output_details()[0]

# Mapping label
label_map = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired']
label_display = ['Sakit Perut 🤕', 'Perlu Sendawa 💨', 'Tidak Nyaman 😣', 'Lapar/Haus 🍼', 'Tertidur 😴']
tips_dict = {
    "belly_pain": "Pijat perlahan perut bayi searah jarum jam. Jika berlanjut, konsultasikan ke dokter.",
    "burping": "Gendong bayi dan bantu sendawa dengan menepuk lembut punggungnya.",
    "discomfort": "Periksa popok, pakaian, atau suhu ruangan. Pastikan semua nyaman untuk bayi.",
    "hungry": "Coba susui bayi dengan ASI atau susu formula. Jika bayi sudah berusia 6 bulan ke atas, berikan juga MPASI sesuai usianya.",
    "tired": "Buat suasana tenang dan redup. Gendong atau ayun pelan-pelan.",
    "other": "Amati perilaku bayi lebih lanjut atau konsultasikan ke tenaga medis."
}

# Tampilan judul
st.markdown("<h1 style='text-align: center; color: #FF6F61;'>Deteksi Tangisan Bayi 👶🔊</h1>", unsafe_allow_html=True)

# Upload file
uploaded = st.file_uploader("🎵 Upload audio file (.wav atau .mp3)", type=["wav", "mp3"])

if uploaded:
    st.audio(uploaded)

    file_ext = uploaded.name.split('.')[-1].lower()

    # Konversi MP3 ke WAV jika perlu
    if file_ext == "wav":
        y, sr = sf.read(uploaded)
    elif file_ext == "mp3":
        try:
            audio_data = uploaded.read()
            audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
            audio = audio.set_channels(1).set_frame_rate(16000)
            samples = np.array(audio.get_array_of_samples()).astype(np.float32)
            y = samples / np.iinfo(samples.dtype).max
            sr = 16000
        except Exception as e:
            st.error("Gagal memproses file MP3. Pastikan ffmpeg sudah tersedia.")
            st.stop()
    else:
        st.error("Format tidak didukung.")
        st.stop()

    # Pranormalisasi audio
    if y.ndim > 1:
        y = y[:, 0]
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    y = y.astype(np.float32)

    # Ekstraksi embedding dari YAMNet
    yamnet.resize_tensor_input(yamnet_input['index'], [len(y)])
    yamnet.allocate_tensors()
    yamnet.set_tensor(yamnet_input['index'], y)
    yamnet.invoke()
    embeddings = yamnet.get_tensor(yamnet_output['index'])
    mean_embedding = np.mean(embeddings, axis=0).astype(np.float32).reshape(1, -1)

    # Klasifikasi DNN
    classifier.set_tensor(classifier_input['index'], mean_embedding)
    classifier.invoke()
    preds = classifier.get_tensor(classifier_output['index'])
    pred_index = np.argmax(preds)
    confidence = np.max(preds)

    # Threshold kelas tidak dikenali
    threshold = 0.6
    if confidence < threshold:
        pred_display = "Tidak Dikenali ❓"
        pred_label = "other"
    else:
        pred_label = label_map[pred_index]
        pred_display = label_display[pred_index]

    # Tampilan hasil prediksi
    st.markdown(f"""
        <div style='text-align: center; padding: 20px; background-color: #1f2937; border-radius: 10px; color: white;'>
            <h2>📢 Prediksi: <span style='color:#FF6F61;'>{pred_display}</span></h2>
            <h4>🎯 Akurasi: <span style='color:#FF6F61;'>{int(confidence * 100)}%</span></h4>
        </div>
    """, unsafe_allow_html=True)

    # Tampilkan saran
    saran = tips_dict.get(pred_label, "Tidak ada saran.")
    st.markdown(f"""
        <div style='padding: 1rem; margin-top: 20px; border-radius: 8px; background-color: #2563eb; color: white;'>
            <h4>💡 Saran:</h4>
            <p>{saran}</p>
        </div>
    """, unsafe_allow_html=True)
