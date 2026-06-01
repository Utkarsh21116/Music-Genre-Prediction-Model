import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np
from preprocessing import preprocessing
import tempfile

genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

@st.cache_resource
def get_model():
    return load_model("music_genre_model2.h5", compile=False)

model = get_model()

st.title("🎵 Music Genre Classifier")

st.write(
    "Upload an audio file and the model will predict its genre."
)

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Predict Genre"):
        with st.spinner("Analyzing audio..."):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".wav"
            ) as tmp_file:

                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            try:
                X = preprocessing(temp_path)
                x = np.array(X)
                predictions = model.predict(x, verbose=0)
                final_probs = np.mean(predictions, axis=0)
                predicted_idx = np.argmax(final_probs)

                predicted_genre = genres[predicted_idx]

                confidence = final_probs[predicted_idx]*100

                st.success(
                    f"Predicted Genre: {predicted_genre.upper()}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence:.2f}%"
                )

                st.subheader(
                    "Top 3 Predictions"
                )

                top3 = np.argsort(
                    final_probs
                )[::-1][:3]

                for idx in top3:

                    st.write(
                        f"**{genres[idx]}** "
                        f"({final_probs[idx]*100:.2f}%)"
                    )

                    st.progress(
                        float(final_probs[idx])
                    )

            finally:
                os.remove(temp_path)