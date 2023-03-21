import numpy as np
from pathlib import Path
import torch

from vocoder.wave_rnn import vocoder_model


def load_model(path_to_weights: Path) -> None:
    """
    Loads the weights of the trained neural network into the vocoder model.

    :param path_to_weights: Path to the weights of the trained vocoder model.
    :return: None
    """
    if not vocoder_model.is_loaded():
        checkpoint = torch.load(path_to_weights)
        vocoder_model.load_state_dict(checkpoint['model_state'])
        vocoder_model.eval()


def is_loaded() -> bool:
    """
    Whether the weights of the trained neural network were loaded into the vocoder model.

    :return: True or False.
    """
    return vocoder_model.is_loaded()


def get_waveform(mel_spectrogram: np.ndarray) -> np.ndarray:
    """
    Converting the synthesized spectrogram to wav.

    :param mel_spectrogram: Spectrogram in mel-scale, written as a mathematical matrix.
    :return: wav as a one-dimensional array.
    """
    mel_spectrogram /= 4
    mel_spectrogram = torch.from_numpy(mel_spectrogram[None, ...])
    wav = vocoder_model.generate_wav(mel_spectrogram)
    return wav
