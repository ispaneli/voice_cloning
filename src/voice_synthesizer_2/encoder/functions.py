import librosa
import numpy as np
from pathlib import Path
from scipy.ndimage.morphology import binary_dilation
import struct
import torch
from typing import Optional, Union
import webrtcvad

from encoder.speaker_encoder import encoder_model


DEFAULT_SAMPLE_RATE = 16_000
MS_WINDOW_LENGTH = 25
MS_WINDOW_STEP = 10
MS_NUM_CHANNELS = 40
MS_NUM_FRAMES = 160

# VOICE ACTIVATION DETECTION params.
# Window size of the VAD. Can be 10, 20 or 30 msec only.
VAD_WINDOW_LENGTH = 30
# Maximum number of silent elements in the carriage.
VAD_MAX_LENGTH_OF_SILENCE = 6

VOICE_DETECTION_WINDOW_SIZE = (VAD_WINDOW_LENGTH * DEFAULT_SAMPLE_RATE) // 1000
INT16_MAX = (2 ** 15) - 1

REQUIRED_VOLUME = -30


def is_loaded() -> bool:
    return encoder_model.is_loaded()


def load_model(path_to_weights: Path) -> None:
    checkpoint = torch.load(path_to_weights)
    encoder_model.load_state_dict(checkpoint['model_state'])
    encoder_model.eval()


def add_frames_batch(frames_batch: np.ndarray) -> np.ndarray:
    """
    Generates embeddings for a batch of mel-spectrogram.

    :param frames_batch: A batch of mel-spectrogram.
    :return: The embeddings as an array.
    """
    frames = torch.from_numpy(frames_batch).to(encoder_model.device)
    partial_embeddings = encoder_model.get_partial_embeddings(frames).detach().cpu().numpy()
    return partial_embeddings


def normalize_volume(wav_array: np.ndarray, only_increase: bool = False, only_decrease: bool = False) -> np.ndarray:
    """
    Normalizes the recording volume, i.e., it leads to values acceptable for the neural network.

    The volume is measured in DBFs.
    :param wav_array: wav-file with a voice as an array.
    :param only_increase: Increase the volume of the audio track.
    :param only_decrease: Decrease the volume of the audio track.
    :return: wav-array with normal volume.
    """
    if only_increase and only_decrease:
        raise ValueError("You can't increase or decrease the volume at the same time.")

    volume_change = REQUIRED_VOLUME - 10 * np.log10(np.mean(wav_array ** 2))

    if (volume_change < 0 and only_increase) or (volume_change > 0 and only_decrease):
        return wav_array
    return wav_array * (10 ** (volume_change / 20))


def cut_long_silences(wav_array: np.ndarray) -> np.ndarray:
    """
    Recognizes and cuts long pauses from the audio track with the voice.
    I.e. removes moments when a person is silent from the audio track.

    :param wav_array: wav-file with a voice as an array.
    :return: wav-array without long silence moments.
    """
    wav_array = wav_array[:len(wav_array) - (len(wav_array) % VOICE_DETECTION_WINDOW_SIZE)]

    # Convert the float waveform to 16-bit mono PCM.
    pcm_wave = struct.pack(f'{len(wav_array)}h', *(np.round(wav_array * INT16_MAX)).astype(np.int16))

    # Perform voice activation detection.
    vad = webrtcvad.Vad(mode=3)

    voice_flags = []
    for window_start in range(0, len(wav_array), VOICE_DETECTION_WINDOW_SIZE):
        window_end = window_start + VOICE_DETECTION_WINDOW_SIZE
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2], sample_rate=DEFAULT_SAMPLE_RATE))
    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average.
    array_padded = np.concatenate((np.zeros(3), voice_flags, np.zeros(4)))
    sums = np.cumsum(array_padded, dtype=float)
    sums[8:] = sums[8:] - sums[:-8]
    audio_mask = sums[8 - 1:] / 8
    audio_mask = np.round(audio_mask).astype(np.bool)
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_LENGTH_OF_SILENCE + 1))
    audio_mask = np.repeat(audio_mask, VOICE_DETECTION_WINDOW_SIZE)

    return wav_array[audio_mask]


def wav_preprocess(*, path_to_voice: Union[Path, str, None] = None, wav_array: Optional[np.ndarray] = None,
                   sample_rate: Optional[int] = None) -> np.ndarray:
    """
    Converting a wav file to a format that is understandable for the neural network.

    :param path_to_voice: Path to wav-file with voice.
    :param wav_array: wav-file as an array.
    :param sample_rate: Sample rate of wav_array (default=16_000).
    :return: wav_array suitable for neural network operation.
    """
    if wav_array is None:
        wav_array, sample_rate = librosa.load(Path(path_to_voice), sr=None)

    if sample_rate is not None:
        if sample_rate != DEFAULT_SAMPLE_RATE:
            wav_array = librosa.resample(wav_array, sample_rate, DEFAULT_SAMPLE_RATE)

    wav_array = normalize_volume(wav_array, only_increase=True)
    wav_array = cut_long_silences(wav_array)

    return wav_array


def wav_to_mel_spectrogram(wav_array: np.ndarray) -> np.ndarray:
    """Converting a wav array to a spectrogram in Mel-scale."""
    frames = librosa.feature.melspectrogram(
        wav_array, DEFAULT_SAMPLE_RATE,
        n_fft=int(DEFAULT_SAMPLE_RATE * MS_WINDOW_LENGTH / 1000),
        hop_length=int(DEFAULT_SAMPLE_RATE * MS_WINDOW_STEP / 1000),
        n_mels=MS_NUM_CHANNELS
    )
    return frames.astype(np.float32).T


def get_partial_slices(wav_length: int) -> (list, list):
    """
    Computes where to split an utterance waveform and its
    corresponding mel spectrogram to obtain partial utterances.

    :param wav_length: length of wav_array.
    :return: Split waves and mel-spectrograms.
    """
    samples_per_frame = int((DEFAULT_SAMPLE_RATE * MS_WINDOW_STEP / 1000))
    n_frames = int(np.ceil((wav_length + 1) / samples_per_frame))
    frame_step = max(int(np.round(MS_NUM_FRAMES * 0.5)), 1)

    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - MS_NUM_FRAMES + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + MS_NUM_FRAMES])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    last_wav_range = wav_slices[-1]
    coverage = (wav_length - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < 0.75 and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def get_embedding(wav_array: np.ndarray) -> np.ndarray:
    """
    Computes an embedding for a single utterance.

    :param wav_array: wav-file as an array.
    :return: The embedding (result of the Speaker Encoder).
    """
    wave_slices, mel_slices = get_partial_slices(len(wav_array))
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav_array):
        wav_array = np.pad(wav_array, (0, max_wave_length - len(wav_array)), 'constant')

    frames = wav_to_mel_spectrogram(wav_array)
    frames_batch = np.array([frames[mel_slice] for mel_slice in mel_slices])
    partial_embeds = add_frames_batch(frames_batch)
    embedding_raw = np.mean(partial_embeds, axis=0)

    return embedding_raw / np.linalg.norm(embedding_raw, 2)
