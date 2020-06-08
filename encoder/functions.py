from pathlib import Path
import torch
import numpy as np
from typing import Optional, Union
import librosa
import struct
import webrtcvad
from scipy.ndimage.morphology import binary_dilation

from encoder.speaker_encoder import encoder_model

DEFAULT_SAMPLE_RATE = 16_000
MS_WINDOW_LENGTH = 25
MS_WINDOW_STEP = 10
MS_NUM_CHANNELS = 40
MS_NUM_FRAMES = 160

# VOICE ACTIVATION DETECTION params.
# Window size of the VAD. Can be 10, 20 or 30 msec only.
VAD_WINDOW_LENGTH = 30
VAD_SMOOTHING_CARRIAGE_LENGTH = 8
# Maximum number of silent elements in the carriage.
VAD_MAX_LENGTH_OF_SILENCE = 6

VOICE_DETECTION_WINDOW_SIZE = (VAD_WINDOW_LENGTH * DEFAULT_SAMPLE_RATE) // 1000
INT16_MAX = (2 ** 15) - 1


def is_loaded():
    return encoder_model.is_loaded()


def load_model(path_to_weights: Path):
    checkpoint = torch.load(path_to_weights)
    encoder_model.load_state_dict(checkpoint['model_state'])
    encoder_model.eval()


# The volume is measured in DBFs.
def normalize_volume(wav_array: np.ndarray, required_volume=-30, only_increase=False,
                     only_decrease=False) -> np.ndarray:
    if only_increase and only_decrease:
        raise ValueError("You can't increase or decrease the volume at the same time.")

    volume_change = required_volume - 10 * np.log10(np.mean(wav_array ** 2))

    if (volume_change < 0 and only_increase) or (volume_change > 0 and only_decrease):
        return wav_array
    return wav_array * (10 ** (volume_change / 20))


# Removes moments when a person is silent from the audio track.
def trim_long_silences(wav_array: np.ndarray) -> np.ndarray:
    wav_array = wav_array[:len(wav_array) - (len(wav_array) % VOICE_DETECTION_WINDOW_SIZE)]

    # Convert the float waveform to 16-bit mono PCM
    pcm_wave = struct.pack(f'{len(wav_array)}h', *(np.round(wav_array * INT16_MAX)).astype(np.int16))

    # Perform voice activation detection
    vad = webrtcvad.Vad(mode=3)

    voice_flags = []

    for window_start in range(0, len(wav_array), VOICE_DETECTION_WINDOW_SIZE):
        window_end = window_start + VOICE_DETECTION_WINDOW_SIZE
        voice_flags.append(vad.is_speech(pcm_wave[window_start * 2:window_end * 2],
                                         sample_rate=DEFAULT_SAMPLE_RATE))

    voice_flags = np.array(voice_flags)

    # Smooth the voice detection with a moving average.
    def moving_average(array, width):
        array_padded = np.concatenate((np.zeros((width - 1) // 2), array, np.zeros(width // 2)))
        ret = np.cumsum(array_padded, dtype=float)
        ret[width:] = ret[width:] - ret[:-width]
        return ret[width - 1:] / width

    audio_mask = moving_average(voice_flags, VAD_SMOOTHING_CARRIAGE_LENGTH)
    audio_mask = np.round(audio_mask).astype(np.bool)
    audio_mask = binary_dilation(audio_mask, np.ones(VAD_MAX_LENGTH_OF_SILENCE + 1))
    audio_mask = np.repeat(audio_mask, VOICE_DETECTION_WINDOW_SIZE)

    return wav_array[audio_mask]


def preprocess_wav(*, path_to_voice: Union[Path, str, None] = None, wav_array: Optional[np.ndarray] = None,
                   sample_rate: Optional[int] = None):
    if wav_array is None:
        wav_array, sample_rate = librosa.load(Path(path_to_voice), sr=None)

    if sample_rate is not None:
        if sample_rate != DEFAULT_SAMPLE_RATE:
            wav_array = librosa.resample(wav_array, sample_rate, DEFAULT_SAMPLE_RATE)

    wav_array = normalize_volume(wav_array, only_increase=True)
    wav_array = trim_long_silences(wav_array)

    return wav_array


def wav_to_mel_spectrogram(wav):
    frames = librosa.feature.melspectrogram(
        wav,
        DEFAULT_SAMPLE_RATE,
        n_fft=int(DEFAULT_SAMPLE_RATE * MS_WINDOW_LENGTH / 1000),
        hop_length=int(DEFAULT_SAMPLE_RATE * MS_WINDOW_STEP / 1000),
        n_mels=MS_NUM_CHANNELS
    )
    return frames.astype(np.float32).T


def embed_frames_batch(frames_batch):
    frames = torch.from_numpy(frames_batch).to(encoder_model.device)
    embed = encoder_model.forward(frames).detach().cpu().numpy()
    return embed


def compute_partial_slices(wav_length):
    min_pad_coverage = 0.75
    overlap = 0.5

    samples_per_frame = int((DEFAULT_SAMPLE_RATE * MS_WINDOW_STEP / 1000))
    n_frames = int(np.ceil((wav_length + 1) / samples_per_frame))
    frame_step = max(int(np.round(MS_NUM_FRAMES * (1 - overlap))), 1)

    # Compute the slices
    wav_slices, mel_slices = [], []
    steps = max(1, n_frames - MS_NUM_FRAMES + frame_step + 1)
    for i in range(0, steps, frame_step):
        mel_range = np.array([i, i + MS_NUM_FRAMES])
        wav_range = mel_range * samples_per_frame
        mel_slices.append(slice(*mel_range))
        wav_slices.append(slice(*wav_range))

    # Evaluate whether extra padding is warranted or not
    last_wav_range = wav_slices[-1]
    coverage = (wav_length - last_wav_range.start) / (last_wav_range.stop - last_wav_range.start)
    if coverage < min_pad_coverage and len(mel_slices) > 1:
        mel_slices = mel_slices[:-1]
        wav_slices = wav_slices[:-1]

    return wav_slices, mel_slices


def embed_utterance(wav_array):
    # Compute where to split the utterance into partials and pad if necessary
    wave_slices, mel_slices = compute_partial_slices(len(wav_array))
    max_wave_length = wave_slices[-1].stop
    if max_wave_length >= len(wav_array):
        wav_array = np.pad(wav_array, (0, max_wave_length - len(wav_array)), 'constant')

    # Split the utterance into partials
    frames = wav_to_mel_spectrogram(wav_array)
    frames_batch = np.array([frames[mel_slice] for mel_slice in mel_slices])
    partial_embeds = embed_frames_batch(frames_batch)

    # Compute the utterance embedding from the partial embeddings
    raw_embed = np.mean(partial_embeds, axis=0)
    embed = raw_embed / np.linalg.norm(raw_embed, 2)

    return embed, partial_embeds, wave_slices
