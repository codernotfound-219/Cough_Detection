import gradio as gr
import numpy as np
import librosa
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import signal
import pickle
import extractors
import base64

# Load your pre-trained models
with open('saved_models/cnn_lstm_mfcc', 'rb') as f:
    cnn_lstm_mfcc = pickle.load(f)
with open('saved_models/cnn_lstm_advanced', 'rb') as f:
    cnn_lstm_advanced = pickle.load(f)
with open('saved_models/cnn_spect', 'rb') as f:
    cnn_spect = pickle.load(f)

def create_mel_spectrogram_plot(mel_spec):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_spec, x_axis='time',
                             y_axis='mel', cmap='viridis')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    return plt.gcf()


def create_waveform_plot(y, sr):
    plt.figure(figsize=(10, 2))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    return plt.gcf()


def create_spectrogram_plot(y, sr):
    f, t, Sxx = signal.spectrogram(y, sr)
    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrogram')
    plt.colorbar(label='Intensity [dB]')
    plt.tight_layout()
    return plt.gcf()


def process_audio(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)

    # Create waveform plot
    waveform_plot = create_waveform_plot(y, sr)

    # Create spectrogram plot
    spectrogram_plot = create_spectrogram_plot(y, sr)

    # Extract mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Create mel spectrogram plot
    mel_spec_plot = create_mel_spectrogram_plot(mel_spec_db)

    mfcc_feature = extractors.mfcc_extract(audio_file)
    mfcc_feature = np.array([mfcc_feature])
    melSpect_feature = extractors.melSpect_extract(audio_file)
    melSpect_feature = melSpect_feature.reshape(1, 128, 1, 1)
    advanced_feature = extractors.advanced_extract(audio_file)
    advanced_feature = np.array([advanced_feature])

    # Make predictions
    prediction1 = round(cnn_lstm_mfcc.predict(mfcc_feature)[0][0]*100, 4)
    prediction2 = round(cnn_lstm_advanced.predict(advanced_feature)[0][0]*100, 4)
    prediction3 = round(cnn_spect.predict(melSpect_feature)[0][0]*100, 4)

    confidence = max(prediction2, prediction3)
    prediction_str = "Cough_Detected" if confidence > 37 else "Not_Detected"
    prediction_str = "Detection: " + prediction_str
    confidence_str = f"Cough Probability Level: {confidence:.2f}%"

    # Prepare results
    predictions = np.array([str(prediction1) + '%', str(prediction2) + '%', str(prediction3) + '%'])
    model_name = np.array(['mfcc_model', 'advn_model', 'spec_model'])
    results = {'MODEL' : model_name, 'PREDICTION' : predictions}
    results = pd.DataFrame(data=results)

    # Calculate and display some audio features
    zero_crossing_rate = round(librosa.feature.zero_crossing_rate(y)[0].mean(), 4)
    spectral_centroid = round(librosa.feature.spectral_centroid(y=y, sr=sr)[0].mean(), 2)
    spectral_rolloff = round(librosa.feature.spectral_rolloff(y=y, sr=sr)[0].mean(), 2)

    features = np.array([zero_crossing_rate, str(spectral_centroid) + ' Hz', str(spectral_rolloff) + ' Hz'])
    feature_name = np.array(['Zero Crossing Rate', 'Spectral Centroid', 'Spectral Rolloff'])
    audio_features = {'FEATURE' : feature_name, 'VALUE' : features}
    audio_features = pd.DataFrame(data=audio_features)

    return waveform_plot, spectrogram_plot, mel_spec_plot, results, audio_features, prediction_str, confidence_str

BACKGROUND_IMAGE_PATH = './assets/audiobg.jpg'

# Read the background image and encode it to base64
with open(BACKGROUND_IMAGE_PATH, "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode()

# Custom CSS for more attractive appearance with fixed background
custom_css = f"""
.gradio-container {{
    background-image: url(data:image/png;base64,{encoded_string});
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
.container {{
    margin: 0 auto;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    background-color: rgba(37, 46, 63, 0.2);
    backdrop-filter: blur(4px);
}}
.gr-button {{
    border-radius: 10px;
}}
.gr-input, .gr-box {{
    border-radius: 10px;
    border: 1px solid #ccc;
}}
.gr-panel {{
    border-radius: 15px;
    border: 1px solid #e0e0e0;
    padding: 15px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    background-color: rgba(37, 46, 63, 0.45);
}}
.gr-box > div {{
    border-radius: 10px;
}}
.gr-dataframe {{
    border-radius: 15px;
    overflow: hidden;
    background-color: rgba(37, 46, 63, 0.45);
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}}
.gr-dataframe table {{
    width: 100%;
    border-collapse: collapse;
}}
.gr-dataframe th, .gr-dataframe td {{
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #e0e0e0;
}}
.gr-dataframe th {{
    background-color: rgba(0, 0, 0, 0.05);
    font-weight: bold;
}}
.custom-label {{
    display: inline-block;
    padding: 0.25rem 0.75rem;
    margin-bottom: 0.5rem;
    border-radius: 0.5rem;
    background-color: #3B82F6;
    color: white;
    font-weight: bold;
}}
.final-prediction-panel {{
    background-color: rgba(0, 0, 0, 0.5);
    padding: 20px;
    border-radius: 10px;
}}
"""

# Create Gradio interface with a more attractive layout and fixed background
with gr.Blocks(theme=gr.themes.Base(), css=custom_css) as iface:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# 🎙️ Advanced Cough Detection System")
        gr.Markdown(
            "Upload an audio file to analyze and detect the presence of a cough using three different models.")

        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                analyze_btn = gr.Button("🔬 Analyze", variant="primary")
            waveform_output = gr.Plot(label="Audio Waveform", elem_classes="gr-panel")

        with gr.Row():
            with gr.Column():
                spectrogram_output = gr.Plot(
                    label="Spectrogram", elem_classes="gr-panel")
                mel_spec_output = gr.Plot(
                    label="Mel Spectrogram", elem_classes="gr-panel")
                with gr.Column(elem_classes="gr-panel final-prediction-panel"):
                    gr.Markdown('<span class="custom-label">Final Prediction</span>',
                            elem_classes="custom-label-wrapper")
                    prediction_output = gr.Markdown()
                    confidence_output = gr.Markdown()
            with gr.Column():
                with gr.Column(elem_classes="gr-panel"):
                    gr.Markdown('<span class="custom-label">Model Predictions</span>',
                                elem_classes="custom-label-wrapper")
                    results_output = gr.DataFrame(
                        headers=['MODEL', 'PREDICTION'],
                        col_count=(2, "fixed"),
                        row_count=(3, "fixed"),
                        elem_classes="gr-dataframe"
                    )
                with gr.Column(elem_classes="gr-panel"):
                    gr.Markdown('<span class="custom-label">Audio Features</span>',
                                elem_classes="custom-label-wrapper")
                    features_output = gr.DataFrame(
                        headers=['FEATURE', 'VALUE'],
                        col_count=(2, "fixed"),
                        row_count=(3, "fixed"),
                        elem_classes="gr-dataframe"
                    )

        analyze_btn.click(
            process_audio,
            inputs=audio_input,
            outputs=[waveform_output, spectrogram_output,
                     mel_spec_output, results_output, features_output, prediction_output, confidence_output]
        )

        gr.Markdown("## 🔬 Technical Details")
        gr.Markdown("""
        This system uses three different deep learning models to analyze the audio and detect the presence of coughs.
        The audio is processed using various signal processing techniques, including:
        - 📊 Waveform analysis
        - 🌈 Spectrogram generation
        - 🎵 Mel-frequency cepstral coefficient (MFCC) extraction
        
        The models are based on convolutional neural networks (CNNs) trained on a large dataset of cough and non-cough sounds.
        """)

# Launch the interface
iface.launch()
