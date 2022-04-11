import os
import torch
import torchaudio
import pandas as pd
from sklearn.model_selection import train_test_split
import tqdm as tq

import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn

from mgr.configuration import load_configurations


class MREData(torch.utils.data.Dataset):
    """Load and preprocess the data

    Arguments:
    __________
    data_csv: str
        Path to the csv file containing the data
    audio_dir: str
        Path to the directory containing the audio files
    target_sample_rate: int
        Target sample rate for the audio files
    num_samples: int
        Number of samples to be extracted from the audio files
    mel_transformation: torch.nn.Module
        Mel-frequency transformation to be applied to the audio files
    Amp2db: torch.nn.Module
        Amplitude to decibel transformation to be applied to the audio files
    """

    def __init__(
            self,
            data_csv,
            audio_dir,
            target_sample_rate=44100,
            num_samples=480000,
            device="cpu",
            mel_transformation=None,
            Amp2db=None):
        self.data_csv = data_csv
        self.audio_dir = audio_dir
        self.device = device
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        if mel_transformation:
            self.mel_transformation = mel_transformation.to(self.device)
        else:
            self.mel_transformation = None

        if Amp2db:
            self.Amp2db = Amp2db.to(self.device)
        else:
            self.Amp2db = None

    def __len__(self):
        """Return the length of the dataset

        Returns:
        ________
        length: int
            Length of the dataset
        """
        return self.data_csv.shape[0]

    def __getitem__(self, idx):
        """Return the item at the given index

        Arguments:
        __________
        idx: int
            Index of the item to be returned

        Returns:
        ________
        wavelet: torch.Tensor
            Mel-frequency transformed audio file
        target: str
            Target label
        """
        audio_path = self.get_audio_path(
            self.audio_dir, self.data_csv.iloc[idx, 0])
        wavelet, sample_rate = torchaudio.load(audio_path)
        wavelet = wavelet.to(self.device)
        label = self.data_csv.iloc[idx, 1]
        wavelet = self.__resample(wavelet, sample_rate, idx)
        wavelet = self.__channel_down(wavelet)
        wavelet = self.__cut_if(wavelet)
        wavelet = self.__pad_if(wavelet)
        if self.mel_transformation:
            wavelet = self.mel_transformation(wavelet)
        if self.Amp2db:
            wavelet = self.Amp2db(wavelet)
        return wavelet, label

    def __channel_down(self, wavelet):
        """Downsample the audio file to mono

        Arguments:
        __________
        wavelet: torch.Tensor
            Audio file

        Returns:
        ________
        wavelet: torch.Tensor
            Downsampled audio file
        """
        if wavelet.shape[0] > 1:
            wavelet = torch.mean(wavelet, dim=0, keepdim=True)
        return wavelet

    def __resample(self, wavelet, sample_rate, idx):
        """Resample the audio file to the target sample rate

        Arguments:
        __________
        wavelet: torch.Tensor
            Audio file
        sample_rate: int
            Sample rate of the audio file
        idx: int
            Index of the audio file

        Returns:
        ________
        wavelet: torch.Tensor
            Resampled audio file
        """
        if sample_rate != self.target_sample_rate:
            try:
                transformation = torchaudio.transforms.Resample(
                    sample_rate, self.target_sample_rate).to(self.device)
                wavelet_RS = transformation(wavelet)
                return wavelet_RS
            except BaseException:
                print(self.data_csv.iloc[idx])
                return wavelet
        else:
            return wavelet

    def __cut_if(self, wavelet):
        """Cut the audio file to the target number of samples

        Arguments:
        __________
        wavelet: torch.Tensor
            Audio file

        Returns:
        ________
        wavelet: torch.Tensor
            Cut audio file
        """
        if wavelet.shape[1] > self.num_samples:
            wavelet = wavelet[:, :self.num_samples]
        return wavelet

    def __pad_if(self, wavelet):
        """Pad the audio file to the target number of samples

        Arguments:
        __________
        wavelet: torch.Tensor
            Audio file

        Returns:
        ________
        wavelet: torch.Tensor
            Padded audio file
        """
        if wavelet.shape[1] < self.num_samples:
            wavelet = nn.functional.pad(
                wavelet, (0, self.num_samples - wavelet.shape[1]))
        return wavelet

    def get_audio_path(self, audio_dir, track_id):
        """Return the path to the audio file

        Arguments:
        __________
        audio_dir: str
            Path to the directory containing the audio files
        track_id: str
            Track ID of the audio file

        Returns:
        ________
        audio_path: str
            Path to the audio file
        """
        return os.path.join(audio_dir, track_id + '.ogg')


def process():
    """Preprocess the data and save it to the disk"""
    CFG = load_configurations()

    data_csv = pd.read_csv(CFG['preprocessing']['csv_path'])
    labelEncoder = LabelEncoder()
    data_csv['label'] = labelEncoder.fit_transform(data_csv['label'])
    print("Classes: ", labelEncoder.classes_)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=CFG['sample_rate'],
        n_fft=2048,
        hop_length=512,
        n_mels=128,
        normalized=True,
    )

    Amp2db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80, )

    features = []
    labels = []

    dataset = MREData(
        data_csv,
        CFG['preprocessing']['audio_dir'],
        CFG['sample_rate'],
        CFG['sample_rate'] * 3,
        CFG['device'],
        mel_spectrogram,
        Amp2db)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    for x, y in tq.tqdm_notebook(loader, total=len(loader)):
        features.extend(x.cpu().numpy())
        labels.extend(y.cpu().numpy())

    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, shuffle=True, test_size=0.1)

    np.save(
        os.path.join(
            CFG['preprocessing']['train'],
            "features.npy"),
        train_features)
    np.save(
        os.path.join(
            CFG['preprocessing']['train'],
            "labels.npy"),
        train_labels)

    np.save(
        os.path.join(
            CFG['preprocessing']['test'],
            "features.npy"),
        test_features)
    np.save(
        os.path.join(
            CFG['preprocessing']['test'],
            "labels.npy"),
        test_labels)

    print("Preprocessed data stored in {} and {}: ".format(
        CFG['preprocessing']['train'], CFG['preprocessing']['test']))
