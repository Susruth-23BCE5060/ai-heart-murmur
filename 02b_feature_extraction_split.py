import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfilt, resample, find_peaks
from sklearn.model_selection import GroupShuffleSplit
from tqdm import tqdm

MANIFEST_PATH = "./dataset_manifest.csv"
OUTPUT_DIR = "./data/processed_v2"

SAMPLE_RATE = 4000
WINDOW_SEC = 3.75
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
STRIDE_SAMPLES = int(SAMPLE_RATE * 1.875)

N_MELS = 64
TARGET_FRAMES = 150
VECTOR_LEN = 16

def apply_bandpass_filter(audio, sr=SAMPLE_RATE, lowcut=25.0, highcut=400.0, order=4):
    nyquist = 0.5 * sr
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', output='sos')
    return sosfilt(sos, audio)

def compute_shannon_envelope(signal):
    norm_sig = signal / (np.max(np.abs(signal)) + 1e-8)
    sq_sig = norm_sig ** 2
    return - sq_sig * np.log(sq_sig + 1e-8)

def extract_log_mel_spectrogram(window_audio):
    stft = librosa.stft(y=window_audio, n_fft=1024, hop_length=256)
    stft_mag = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=1024, n_mels=N_MELS, fmin=25, fmax=400)
    mel_spec = np.dot(mel_basis, stft_mag)
    log_mel = np.log(mel_spec + 1e-6)
    if log_mel.shape[1] != TARGET_FRAMES:
        log_mel = resample(log_mel, TARGET_FRAMES, axis=1)
    return np.expand_dims(log_mel, axis=-1)

def extract_peak_interval_vector(tsv_path, win_start_sec, win_end_sec, window_audio):
    shannon_env = compute_shannon_envelope(window_audio)
    intervals = []
    
    if os.path.exists(tsv_path):
        try:
            tsv_data = pd.read_csv(tsv_path, sep='\t', header=None, names=['start', 'end', 'state'])
            events = tsv_data[(tsv_data['start'] >= win_start_sec) & (tsv_data['start'] < win_end_sec)]
            s1_s2_events = events[events['state'].isin([1, 3])].sort_values(by='start')
            if len(s1_s2_events) >= 2:
                onsets = s1_s2_events['start'].values
                intervals = np.diff(onsets) * 1000.0
        except Exception:
            intervals = []

    if len(intervals) < 2:
        peaks, _ = find_peaks(shannon_env, distance=400, height=0.05)
        if len(peaks) >= 2:
            intervals = np.diff(peaks) / SAMPLE_RATE * 1000.0

    if len(intervals) >= 2:
        intervals_16 = resample(intervals, VECTOR_LEN)
    else:
        intervals_16 = np.zeros(VECTOR_LEN, dtype=np.float32)

    std_val = np.std(intervals_16)
    vec_zscore = (intervals_16 - np.mean(intervals_16)) / std_val if std_val > 1e-6 else intervals_16
    return np.expand_dims(vec_zscore, axis=-1)

def process_and_split_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH)
    
    X_spec, X_temp, Y, P_IDs = [], [], [], []

    print("Extracting features for V2 Dataset...")
    for _, row in tqdm(manifest.iterrows(), total=len(manifest)):
        try:
            audio, _ = librosa.load(row['Wav_Path'], sr=SAMPLE_RATE, mono=True)
        except:
            continue

        filtered_audio = apply_bandpass_filter(audio)
        if len(filtered_audio) < WINDOW_SAMPLES:
            filtered_audio = np.pad(filtered_audio, (0, WINDOW_SAMPLES - len(filtered_audio)))

        for start_idx in range(0, len(filtered_audio) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
            end_idx = start_idx + WINDOW_SAMPLES
            win_audio = filtered_audio[start_idx:end_idx]
            
            X_spec.append(extract_log_mel_spectrogram(win_audio))
            X_temp.append(extract_peak_interval_vector(row['Tsv_Path'], start_idx/SAMPLE_RATE, end_idx/SAMPLE_RATE, win_audio))
            Y.append(row['Label'])
            P_IDs.append(row['Subject_ID'])

    X_spec, X_temp, Y, P_IDs = np.array(X_spec), np.array(X_temp), np.array(Y), np.array(P_IDs)

    # Patient-Aware Train/Test Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(X_spec, Y, groups=P_IDs))

    np.save(os.path.join(OUTPUT_DIR, "X_train_spec.npy"), X_spec[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "X_train_temp.npy"), X_temp[train_idx])
    np.save(os.path.join(OUTPUT_DIR, "Y_train.npy"), Y[train_idx])
    
    np.save(os.path.join(OUTPUT_DIR, "X_test_spec.npy"), X_spec[test_idx])
    np.save(os.path.join(OUTPUT_DIR, "X_test_temp.npy"), X_temp[test_idx])
    np.save(os.path.join(OUTPUT_DIR, "Y_test.npy"), Y[test_idx])

if __name__ == "__main__":
    process_and_split_dataset()