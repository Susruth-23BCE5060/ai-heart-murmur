import os
import sys
import io
import tempfile
import traceback
import subprocess
import numpy as np
import pandas as pd
import librosa
from scipy.signal import butter, sosfilt, resample, find_peaks
import tensorflow as tf
import soundfile as sf
import streamlit as st

# Configuration constants matching model architecture
MODEL_PATH = "robust_pi_murmur_model.keras"
SAMPLE_RATE = 4000
WINDOW_SEC = 3.75
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SEC)
STRIDE_SAMPLES = int(SAMPLE_RATE * 1.875)
N_MELS = 64
TARGET_FRAMES = 150
VECTOR_LEN = 16

@st.cache_resource
def load_edge_model():
    """Loads and caches the compiled Keras model."""
    if os.path.exists(MODEL_PATH):
        try:
            return tf.keras.models.load_model(MODEL_PATH)
        except Exception as e:
            print(f"[ERROR] Model loading failed:\n{traceback.format_exc()}", file=sys.stderr)
            st.error(f"Failed to load model file '{MODEL_PATH}': {e}")
            return None
    return None

def apply_bandpass_filter(audio, sr=SAMPLE_RATE, lowcut=25.0, highcut=400.0, order=4):
    """4th-Order Butterworth Bandpass Filter (25Hz - 400Hz)."""
    nyquist = 0.5 * sr
    sos = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band', output='sos')
    return sosfilt(sos, audio)

def compute_shannon_envelope(signal):
    """Computes normalized Shannon envelope for heart peak detection."""
    norm_sig = signal / (np.max(np.abs(signal)) + 1e-8)
    sq_sig = norm_sig ** 2
    sq_sig = np.clip(sq_sig, 1e-12, None)
    return - sq_sig * np.log(sq_sig)

def extract_log_mel_spectrogram(window_audio):
    """Branch A: Log-Mel Spectrogram Extraction (64, 150, 1)."""
    stft = librosa.stft(y=window_audio, n_fft=1024, hop_length=256)
    stft_mag = np.abs(stft)
    mel_basis = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=1024, n_mels=N_MELS, fmin=25, fmax=400)
    mel_spec = np.dot(mel_basis, stft_mag)
    log_mel = np.log(mel_spec + 1e-6)
    if log_mel.shape[1] != TARGET_FRAMES:
        log_mel = resample(log_mel, TARGET_FRAMES, axis=1)
    return np.expand_dims(log_mel, axis=-1)

def extract_peak_interval_vector(window_audio):
    """Branch B: 1D Temporal Peak-Interval Vector Extraction (16, 1)."""
    shannon_env = compute_shannon_envelope(window_audio)
    peaks, _ = find_peaks(shannon_env, distance=400, height=0.05)
    intervals = np.diff(peaks) / SAMPLE_RATE * 1000.0 if len(peaks) >= 2 else []
    
    if len(intervals) >= 2:
        intervals_16 = resample(intervals, VECTOR_LEN)
    else:
        intervals_16 = np.zeros(VECTOR_LEN, dtype=np.float32)
        
    std_val = np.std(intervals_16)
    vec_zscore = (intervals_16 - np.mean(intervals_16)) / std_val if std_val > 1e-6 else intervals_16
    return np.expand_dims(vec_zscore, axis=-1)

def load_audio_to_numpy(uploaded_file, target_sr=SAMPLE_RATE):
    """
    Robust audio importer handling .wav, .mp3, .m4a.
    Converts compressed streams into standard uncompressed 1D PCM arrays.
    """
    file_bytes = uploaded_file.getvalue()
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    # 1. Attempt direct reading in-memory (ideal for standard WAV)
    try:
        audio_data, sr = sf.read(io.BytesIO(file_bytes))
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1) # Stereo to Mono
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        return audio_data.astype(np.float32)
    except Exception as direct_err:
        print(f"[INFO] Direct memory read skipped for format '{file_ext}': {direct_err}")

    # 2. Fallback: Convert .mp3 / .m4a using FFmpeg CLI to standard WAV PCM
    in_tmp_path = None
    out_tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as in_tmp:
            in_tmp.write(file_bytes)
            in_tmp_path = in_tmp.name

        out_tmp_path = in_tmp_path + "_converted.wav"

        cmd = [
            "ffmpeg", "-y",
            "-i", in_tmp_path,
            "-ar", str(target_sr),
            "-ac", "1",
            "-f", "wav",
            out_tmp_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        
        audio_data, sr = sf.read(out_tmp_path)
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        return audio_data.astype(np.float32)

    except FileNotFoundError:
        raise RuntimeError(
            "FFmpeg executable not found! Please run `brew install ffmpeg` in your terminal."
        )
    except subprocess.CalledProcessError as ffmpeg_err:
        stderr_msg = ffmpeg_err.stderr.decode('utf-8', errors='ignore')
        raise RuntimeError(
            f"FFmpeg failed to convert '{uploaded_file.name}'.\nLog details:\n{stderr_msg}"
        ) from ffmpeg_err
    except Exception as e:
        raise RuntimeError(f"Error processing audio stream for '{uploaded_file.name}': {e}") from e
    finally:
        for path in [in_tmp_path, out_tmp_path]:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

def process_audio_file(uploaded_file):
    """Processes uploaded file into dual-branch feature tensors."""
    audio = load_audio_to_numpy(uploaded_file, target_sr=SAMPLE_RATE)
    filtered_audio = apply_bandpass_filter(audio)
    
    if len(filtered_audio) < WINDOW_SAMPLES:
        filtered_audio = np.pad(filtered_audio, (0, WINDOW_SAMPLES - len(filtered_audio)))

    specs, temps = [], []
    for start_idx in range(0, len(filtered_audio) - WINDOW_SAMPLES + 1, STRIDE_SAMPLES):
        end_idx = start_idx + WINDOW_SAMPLES
        win_audio = filtered_audio[start_idx:end_idx]
        
        specs.append(extract_log_mel_spectrogram(win_audio))
        temps.append(extract_peak_interval_vector(win_audio))
        
    return np.array(specs), np.array(temps)

# --- UI Layout ---
st.title("Heart Murmur Detection Classifier")
st.write("Upload a phonocardiogram recording (`.wav`, `.mp3`, `.m4a`) to analyze for potential heart murmurs.")

model = load_edge_model()

uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file)
    
    if model is None:
        st.error(f"Model file '{MODEL_PATH}' was not found or could not be loaded!")
    else:
        with st.spinner("Extracting dual-branch features and calculating prediction..."):
            try:
                X_spec, X_temp = process_audio_file(uploaded_file)
                
                predictions = model.predict({"Input_A_Spectrogram": X_spec, "Input_B_PeakInterval": X_temp})
                mean_prob = float(np.mean(predictions))
                
                is_murmur = mean_prob >= 0.5
                confidence = mean_prob if is_murmur else (1.0 - mean_prob)
                
                st.markdown("---")
                st.subheader("Analysis Results")
                
                col1, col2 = st.columns(2)
                with col1:
                    if is_murmur:
                        st.error("### Result: Murmur Detected")
                    else:
                        st.success("### Result: Normal (No Murmur)")
                with col2:
                    st.metric(label="Confidence Score", value=f"{confidence * 100:.2f}%")
                    
                st.write(f"Raw Model Probability Score: `{mean_prob:.4f}`")

            except Exception as e:
                err_msg = traceback.format_exc()
                print(f"[ERROR] Processing exception:\n{err_msg}", file=sys.stderr)
                st.error("An error occurred while processing the audio file:")
                st.code(err_msg, language="python")