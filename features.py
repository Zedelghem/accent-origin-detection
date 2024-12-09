from glob import glob
from tqdm import tqdm
import numpy as np
import pandas as pd
import helpers as h
import librosa
import parselmouth
from parselmouth.praat import call
import torch
from torch import nn

def prepare_datasets(feature_extractor, feature_label: str, extractor_params: dict, extractor_expects_list=False):
    """
    Prepares training, testing (clean), and testing (noisy) datasets with feature extraction.

    Parameters:
        feature_extractor (callable): Function for extracting features from audio files.
        feature_label (str): Prefix for the feature column names.
        extractor_params (dict): Parameters to pass to the feature extraction function.
        extractor_expects_list (bool): If True, passes the entire list of files to the feature extractor.

    Returns:
        tuple: DataFrames and label arrays for training, clean test, and noisy test datasets.
    """

    print(f"Preparing datasets with feature extractor: {feature_extractor.__name__}")
    print(f"Parameter values: {extractor_params}")

    def extract_features_and_labels(files, desc):
        """
        Helper function to extract features and labels for a set of files.
        """
        if extractor_expects_list:
            features = feature_extractor(files, **extractor_params)
            labels = [h.get_label(file) for file in files]
        else:
            features = []
            labels = []
            for file in tqdm(files, desc=f"Processing {desc} Files"):
                features.append(feature_extractor(file, **extractor_params))
                labels.append(h.get_label(file))
        return features, labels

    # Prepare training data
    training_files = sorted(glob('project_data/train/*.wav'))
    training_features, training_labels = extract_features_and_labels(training_files, "Training")

    # Prepare clean test data
    clean_test_files = sorted(glob('project_data/test_clean/*.wav'))
    clean_test_features, clean_test_labels = extract_features_and_labels(clean_test_files, "Clean Test")

    # Prepare noisy test data
    noisy_test_files = sorted(glob('project_data/test_noisy/*.wav'))
    noisy_test_features, noisy_test_labels = extract_features_and_labels(noisy_test_files, "Noisy Test")

    # Create DataFrames for features
    feature_columns = [f'{feature_label}_{i}' for i in range(len(training_features[0]))]
    training_df = pd.DataFrame(data=np.stack(training_features), columns=feature_columns)
    clean_test_df = pd.DataFrame(data=np.stack(clean_test_features), columns=feature_columns)
    noisy_test_df = pd.DataFrame(data=np.stack(noisy_test_features), columns=feature_columns)

    # Labels as numpy arrays
    y_train = np.array(training_labels)
    y_clean_test = np.array(clean_test_labels)
    y_noisy_test = np.array(noisy_test_labels)

    return training_df, y_train, clean_test_df, y_clean_test, noisy_test_df, y_noisy_test

def dummy_feature_extractor(file_path, **params):
    return np.random.rand(40)

def extract_log_mel_filterbank(
    audio_file, n_mels=80, hop_length=256, n_fft=1024,
    mean=True, logpower=2, std=False, variance=False, delta=False, delta2=False, delta3=False
):
    """
    Extract an n-dimensional log-mel filterbank from an audio file, with options for mean, std, variance, and deltas.

    Parameters:
        audio_file (str): Path to the audio file.
        n_mels (int): Number of mel bands (default is 80).
        hop_length (int): Hop length for the STFT (default is 256).
        n_fft (int): Number of FFT components (default is 1024).
        mean (bool): Compute mean of the log-mel spectrogram (default: True).
        logpower (int or float): Power for the spectrogram computation (default: 2).
        std (bool): Compute standard deviation of the log-mel spectrogram (default: False).
        variance (bool): Compute variance of the log-mel spectrogram (default: False).
        delta (bool): Compute first derivative (delta) of the log-mel spectrogram (default: False).
        delta2 (bool): Compute second derivative (delta-delta) of the log-mel spectrogram (default: False).
        delta3 (bool): Compute third derivative (delta-delta-delta) of the log-mel spectrogram (default: False).

    Returns:
        np.ndarray: Concatenated feature array, including selected statistics and deltas.
    """

    audio, sr = librosa.load(audio_file, sr=None)  # sr=None ensures native sampling rate


    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length,
        power=logpower
    )

    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    features = []

    if mean:
        features.append(np.mean(log_mel_spectrogram, axis=1)) 
    if std:
        features.append(np.std(log_mel_spectrogram, axis=1)) 
    if variance:
        features.append(np.var(log_mel_spectrogram, axis=1)) 

    # derivatives
    if delta:
        delta_features = librosa.feature.delta(log_mel_spectrogram)  
        features.append(np.mean(delta_features, axis=1))  
    if delta2:
        delta2_features = librosa.feature.delta(log_mel_spectrogram, order=2) 
        features.append(np.mean(delta2_features, axis=1))  
    if delta3:
        delta3_features = librosa.feature.delta(log_mel_spectrogram, order=3)  
        features.append(np.mean(delta3_features, axis=1))

    if features:
        return np.concatenate(features, axis=0)  # Combined feature vector
    else:
        raise ValueError("At least one feature computation option (mean, std, variance, delta, delta2, delta3) must be True.")

def extract_f0_and_formants(audio_file, sr=16000, f0_min=50, f0_max=500, num_formants=5):
    """
    Extracts fundamental frequency (F0) and formants from an audio file.

    Parameters:
        audio_file (str): Path to the audio file.
        sr (int): Sampling rate for loading the audio (default: 16000 Hz).
        f0_min (float): Minimum F0 for estimation (default: 50 Hz).
        f0_max (float): Maximum F0 for estimation (default: 500 Hz).
        num_formants (int): Number of formants to extract (default: 5).

    Returns:
        dict: A dictionary containing:
            - 'f0_mean': Mean fundamental frequency.
            - 'f0_std': Standard deviation of F0.
            - 'formants': List of formant frequency arrays (one array per formant).
    """
    audio, _ = librosa.load(audio_file, sr=sr)

    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio, 
        fmin=f0_min, 
        fmax=f0_max, 
        sr=sr
    )

    f0 = f0[~np.isnan(f0)]
    f0_mean = np.mean(f0) if len(f0) > 0 else 0
    f0_std = np.std(f0) if len(f0) > 0 else 0

    sound = parselmouth.Sound(audio, sampling_frequency=sr)
    formant_data = call(sound, "To Formant (burg)", 0.025, 5, 5500, 0.05, 50)

    formants = []
    for formant_index in range(1, num_formants + 1):
        try:
            
            formant_freqs = [
                call(formant_data, "Get value at time", formant_index, t, "Hertz", "Linear")
                for t in np.arange(0, sound.duration, 0.01)  # Sample every 10ms
            ]
            # Remove NaNs and invalid frequencies
            formant_freqs = [f for f in formant_freqs if not np.isnan(f)]
            formants.append(formant_freqs)
        except Exception:
            formants.append([])

    results = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "formants": formants
    }

    return results

def extract_f0_cuda(audio_file, sr=16000, f0_min=50, f0_max=500, device="cuda"):
    """
    Extracts fundamental frequency (F0) from an audio file using CUDA acceleration where possible.

    Parameters:
        audio_file (str): Path to the audio file.
        sr (int): Sampling rate for loading the audio (default: 16000 Hz).
        f0_min (float): Minimum F0 for estimation (default: 50 Hz).
        f0_max (float): Maximum F0 for estimation (default: 500 Hz).
        device (str): Device to use for computation ("cuda" or "cpu").

    Returns:
        dict: A dictionary containing:
            - 'f0_mean': Mean fundamental frequency.
            - 'f0_std': Standard deviation of F0.
            - 'f0_values': Array of F0 values (valid F0 only, NaNs removed).
    """
    # Load audio
    audio, _ = librosa.load(audio_file, sr=sr)

    # Compute F0 using librosa's pyin
    f0_values, _, _ = librosa.pyin(audio, fmin=f0_min, fmax=f0_max, sr=sr)

    # Convert F0 values to PyTorch tensor for GPU processing (if required)
    f0_values_tensor = torch.tensor(f0_values, device=device)

    # Handle F0 values (remove NaNs)
    valid_f0 = f0_values_tensor[~torch.isnan(f0_values_tensor)]
    f0_mean = valid_f0.mean().item() if len(valid_f0) > 0 else 0
    f0_std = valid_f0.std().item() if len(valid_f0) > 0 else 0

    # Create output dictionary
    results = {
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "f0_values": valid_f0.cpu().numpy()  # Convert back to NumPy for easier handling
    }

    return results

class F0ExtractorModule(nn.Module):
    def __init__(self, sr=16000, f0_min=50, f0_max=500):
        super(F0ExtractorModule, self).__init__()
        self.sr = sr
        self.f0_min = f0_min
        self.f0_max = f0_max

    def forward(self, audio_batch):
        # Placeholder tensors for F0 means and stds
        f0_means = []
        f0_stds = []
        f0_values_list = []

        for audio in audio_batch:
            audio_numpy = audio.cpu().numpy()

            # Compute F0 using librosa's pyin
            f0_values, _, _ = librosa.pyin(audio_numpy, fmin=self.f0_min, fmax=self.f0_max, sr=self.sr)

            # Convert F0 values to tensor and remove NaNs
            f0_values_tensor = torch.tensor(f0_values, device=audio.device)
            valid_f0 = f0_values_tensor[~torch.isnan(f0_values_tensor)]

            # Compute mean and std
            f0_mean = valid_f0.mean().item() if len(valid_f0) > 0 else 0
            f0_std = valid_f0.std().item() if len(valid_f0) > 0 else 0

            # Append results
            f0_means.append(f0_mean)
            f0_stds.append(f0_std)
            f0_values_list.append(valid_f0)

        # Stack tensors for compatibility with DataParallel
        return (
            torch.tensor(f0_means, device=audio_batch.device),
            torch.tensor(f0_stds, device=audio_batch.device),
            f0_values_list,
        )


def process_audio_on_gpus(audio_files, sr=16000, f0_min=50, f0_max=500):
    """
    Distributes F0 extraction across all available GPUs for a batch of audio files.

    Parameters:
        audio_files (list of str): List of paths to audio files.
        sr (int): Sampling rate (default: 16000 Hz).
        f0_min (float): Minimum F0 for estimation (default: 50 Hz).
        f0_max (float): Maximum F0 for estimation (default: 500 Hz).

    Returns:
        list of dict: List of F0 extraction results for each audio file.
    """
    audio_tensors = []

    # Load audio files into tensors with a progress bar
    for audio_file in tqdm(audio_files, desc="Loading Audio Files", unit="file"):
        audio, _ = librosa.load(audio_file, sr=sr)
        audio_tensors.append(torch.tensor(audio, device="cpu"))  # Initially on CPU

    if not audio_tensors:
        raise ValueError("No valid audio files found for processing.")

    # Pad sequences for batch processing
    audio_batch = torch.nn.utils.rnn.pad_sequence(audio_tensors, batch_first=True)

    # Create the F0 extractor module
    f0_extractor = F0ExtractorModule(sr=sr, f0_min=f0_min, f0_max=f0_max)

    # Wrap the module with DataParallel
    model = torch.nn.DataParallel(f0_extractor)
    model.to("cuda")

    # Process batch with a progress bar
    results = []
    with tqdm(total=len(audio_files), desc="Extracting F0 on GPUs", unit="file") as pbar:
        f0_means, f0_stds, f0_values_list = model(audio_batch.to("cuda"))
        for i in range(len(audio_files)):
            results.append({
                "f0_mean": f0_means[i].item(),
                "f0_std": f0_stds[i].item(),
                "f0_values": f0_values_list[i].cpu().numpy(),
            })
            pbar.update(1)

    return results

def generate_mfcc(audio_file, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    """
    Generates Mel-Frequency Cepstral Coefficients (MFCCs) from an audio file.

    Parameters:
        audio_file (str): Path to the audio file.
        sr (int): Sampling rate (default: 16000 Hz).
        n_mfcc (int): Number of MFCC coefficients to generate (default: 13).
        n_fft (int): Length of the FFT window (default: 2048).
        hop_length (int): Number of samples between successive frames (default: 512).

    Returns:
        np.ndarray: Flattened 1D array of MFCCs (shape: [n_mfcc * n_time_frames]).
    """

    audio, _ = librosa.load(audio_file, sr=sr)


    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )


    return mfcc.flatten()


def compute_rms(audio, sr=16000, frame_length=2048, hop_length=512):
    """
    Computes the Root Mean Square (RMS) energy of an audio signal.

    Parameters:
        audio (np.ndarray): Audio signal as a 1D NumPy array.
        sr (int): Sampling rate of the audio signal (default: 16000).
        frame_length (int): Length of each frame for RMS computation (default: 2048).
        hop_length (int): Step size between frames (default: 512).

    Returns:
        np.ndarray: 1D array of RMS energy values for each frame.
    """

    audio, _ = librosa.load(audio, sr=sr)

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]  # librosa returns a 2D array; [0] gives the 1D array

    return rms

def compute_zcr(audio, sr=16000, frame_length=2048, hop_length=512):
    """
    Computes the Zero-Crossing Rate (ZCR) for an audio signal.

    Parameters:
        audio (np.ndarray): Audio signal as a 1D NumPy array.
        sr (int): Sampling rate of the audio signal (default: 16000).
        frame_length (int): Length of each frame for ZCR computation (default: 2048).
        hop_length (int): Step size between frames (default: 512).

    Returns:
        np.ndarray: 1D array of ZCR values for each frame.
    """

    audio, sr = librosa.load(audio, sr=16000)

    # Compute zero-crossing rate
    zcr = librosa.feature.zero_crossing_rate(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]  # librosa returns a 2D array; [0] gives the 1D array

    return zcr
