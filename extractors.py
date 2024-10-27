import librosa
import numpy as np

def mfcc_extract(file_path):
    audio, sample_rate = librosa.load(file_path)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=50)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    return mfcc_scaled

def melSpect_extract(file_path):
    audio, sample_rate = librosa.load(file_path)
    mels = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    return mels

def advanced_extract(file_name):
    features = []
    audio, sample_rate = librosa.load(file_name)

    # mfcc
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # spectral_centroids
    spectral_centroids = librosa.feature.spectral_centroid(
        y=audio, sr=sample_rate)[0]
    features.extend([np.mean(spectral_centroids), np.std(spectral_centroids)])

    # zero crossin rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
    features.extend([np.mean(zero_crossing_rate), np.std(zero_crossing_rate)])

    # Rms Energy
    rms_energy = librosa.feature.rms(y=audio)[0]
    features.extend([np.mean(rms_energy), np.std(rms_energy)])

    # spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(
        y=audio, sr=sample_rate)[0]
    features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])

    # spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=sample_rate)
    features.extend(np.mean(spectral_contrast, axis=1))

    # Onset Strength
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    features.extend([np.mean(onset_env), np.std(onset_env)])

    return np.array(features)