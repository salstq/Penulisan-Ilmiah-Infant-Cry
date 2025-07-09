import streamlit as st
import numpy as np
import soundfile as sf
import librosa
import tflite_runtime.interpreter as tflite

# Load models
@st.cache_resource
def load_models():
    yamnet = tflite.Interpreter(model_path="yamnet.tflite")
    yamnet.allocate_tensors()

    classifier = tflite.Interpreter(model_path="model_cry.tflite")
    classifier.allocate_tensors()

    return yamnet, classifier

yamnet, classifier = load_models()

# Get I/O details
yamnet_input = yamnet.get_input_details()[0]
yamnet_output = yamnet.get_output_details()[1]  # embeddings

classifier_input = classifier.get_input_details()[0]
classifier_output = classifier.get_output_details()[0]

# Label mapping (ubah kalau perlu)
label_map = ['belly_pain', 'burping', 'discomfort', 'hungry', 'tired', 'other']

# UI
st.title("Infant Cry Classifier")
uploaded = st.file_uploader("Upload audio file (.wav)", type=["wav"])

if uploaded:
    st.audio(uploaded)

    # Load and resample audio
    y, sr = sf.read(uploaded)
    if y.ndim > 1:
        y = y[:, 0]  # ambil channel pertama (mono)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)

    y = y.astype(np.float32)

    # Run YAMNet TFLite
    yamnet.resize_tensor_input(yamnet_input['index'], [len(y)])
    yamnet.allocate_tensors()
    yamnet.set_tensor(yamnet_input['index'], y)
    yamnet.invoke()
    embeddings = yamnet.get_tensor(yamnet_output['index'])  # [N, 1024]

    mean_embedding = np.mean(embeddings, axis=0).astype(np.float32).reshape(1, -1)

    # Run classifier model
    classifier.set_tensor(classifier_input['index'], mean_embedding)
    classifier.invoke()
    preds = classifier.get_tensor(classifier_output['index'])  # [1, 6]

    pred_label = label_map[np.argmax(preds)]
    confidence = np.max(preds)

    st.markdown(f"### Prediksi: `{pred_label}`")
    st.markdown(f"**Confidence**: `{confidence:.4f}`")
