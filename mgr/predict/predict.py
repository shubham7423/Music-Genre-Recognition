import torch
import torchaudio
import torch.nn as nn

from mgr.configuration import load_configurations
import torchvision


def __resample(wavelet, sample_rate, target_sample_rate):
    """ Resample wavelet to target sample rate.

    Arguments
    ---------
    wavelet: torch.Tensor
        Wavelet to be resampled.
    sample_rate: int
        Sample rate of wavelet.
    target_sample_rate: int
        Target sample rate.

    Returns
    -------
    torch.Tensor
        Resampled wavelet.
    """
    if sample_rate != target_sample_rate:
        transformation = torchaudio.transforms.Resample(
            sample_rate, target_sample_rate)
        wavelet_RS = transformation(wavelet)
        return wavelet_RS
    else:
        return wavelet


def __cut_if(wavelet, num_samples):
    """ Cut wavelet to num_samples.

    Arguments
    ---------
    wavelet: torch.Tensor
        Wavelet to be cut.
    num_samples: int
        Number of samples to be cut.

    Returns
    -------
    torch.Tensor
        Cut wavelet.
    """
    if wavelet.shape[1] > num_samples:
        wavelet = wavelet[:, :num_samples]
    return wavelet


def __pad_if(wavelet, num_samples):
    if wavelet.shape[1] < num_samples:
        wavelet = nn.functional.pad(
            wavelet, (0, num_samples - wavelet.shape[1]))
    return wavelet


def get_random_samples(audio_path, num_samples):
    """ Get random samples from audio file.

    Arguments
    ---------
    audio_path: str
        Path to audio file.
    num_samples: int
        Number of samples to be extracted.

    Returns
    -------
    torch.Tensor
        Samples.
    """
    signal, sr = torchaudio.load(audio_path)
    signal = torch.mean(signal, dim=0, keepdims=True)

    if signal.shape[1] < sr * 3:
        print("Audio file is not long enough (should be at least 3 seconds)")
        return

    audio_samples = []
    for i in range(0, signal.shape[1], sr * 3):
        sample = signal[:, i: min(signal.shape[1], i + sr * 3)]
        audio_samples.append(sample)

    audio_samples = torch.stack(audio_samples[:-1])
    rand_samples_idx = torch.randperm(audio_samples.size(0))[
        :min(num_samples, audio_samples.size(0))]
    rand_samples = audio_samples[rand_samples_idx]
    return rand_samples, sr


def apply_preprocessing(signal_samples, sr, transform, sample_rate):
    """ Apply preprocessing to signal samples.

    Arguments
    ---------
    signal_samples: torch.Tensor
        Signal samples.
    sr: int
        Sample rate.
    transform: torchvision.transforms.Compose
        Transform to be applied.
    sample_rate: int
        Sample rate.

    Returns
    -------
    torch.Tensor
        Preprocessed signal samples.
    """
    transformed_samples = []
    for idx in range(signal_samples.size(0)):
        sample = signal_samples[idx]
        sample = __resample(sample, sr, sample_rate)
        sample = __cut_if(sample, sample_rate * 3)
        sample = __pad_if(sample, sample_rate * 3)
        transformed_samples.append(sample)
    transformed_samples = torch.stack(transformed_samples)
    return transform(transformed_samples)


def predict(
    model,
    audio_file_path,
    labels=[
        'Electronic',
        'Experimental',
        'Folk',
        'Hip-Hop',
        'Instrumental',
        'International',
        'Pop',
        'Rock']):
    """ Predict label of audio file.

    Arguments
    ---------
    model: torch.nn.Module
        Model to be used.
    audio_file_path: str
        Path to audio file.
    labels: list
        List of labels.

    Returns
    -------
    str
        Predicted label.
    """
    sample_rate = load_configurations()['sample_rate']
    device = load_configurations()['device']

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        normalized=True,
    )
    Amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)

    transform = torchvision.transforms.Compose([
        mel_spectrogram,
        Amp2db,
        torchvision.transforms.Normalize(0.5, 0.5)
    ])

    signal_samples, sr = get_random_samples(audio_file_path, 5)

    preprocessed_signals = apply_preprocessing(
        signal_samples, sr, transform, sample_rate)

    model.eval()
    with torch.no_grad():
        predicted_labels = model(preprocessed_signals.to(device))
    pred = torch.topk(torch.mean(predicted_labels, dim=0), 3).indices
    return [labels[x.item()] for x in pred]
