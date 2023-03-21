import librosa
import librosa.filters
import numpy as np
import tensorflow as tf
from scipy import signal
from scipy.io import wavfile
from abc import ABC, abstractmethod


class FileManagement:
    """
        Отвечает за сохранения файлов
    """
    def load_wav(path, sr):
        return librosa.core.load(path, sr=sr)[0]

    def save_wav(wav, path, sr):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        wavfile.write(path, sr, wav.astype(np.int16))

    def save_wavenet_wav(wav, path, sr):
        librosa.output.write_wav(path, wav, sr=sr)


class Filter(ABC):
    """
        Абстрактный класс для применния фильтров к
        аудиозаписи.
    """
    @abstractmethod
    def apply(self, wav, k, flag):
        pass

    @abstractmethod
    def inverse(self, wav, k, flag):
        pass


class PreEmphasis(Filter):
    """
        Класс преусиления
    """
    @staticmethod
    def apply(wav, k, preemphasize=True):
        if preemphasize:
            return signal.lfilter([1, -k], [1], wav)
        return wav

    @staticmethod
    def inverse(wav, k, inv_preemphasize=True):
        if inv_preemphasize:
            return signal.lfilter([1], [1, -k], wav)
        return wav


class AudioPreprocess:
    """
        Класс для технической обработки записей.
    """
    @staticmethod
    def start_and_end_indices(quantized, silence_threshold=2):
        for start in range(quantized.size):
            if abs(quantized[start] - 127) > silence_threshold:
                break
        for end in range(quantized.size - 1, 1, -1):
            if abs(quantized[end] - 127) > silence_threshold:
                break

        assert abs(quantized[start] - 127) > silence_threshold
        assert abs(quantized[end] - 127) > silence_threshold

        return start, end

    @staticmethod
    def get_hop_size(hparams):
        hop_size = hparams.hop_size
        if hop_size is None:
            assert hparams.frame_shift_ms is not None
            hop_size = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
        return hop_size


class Transform(ABC):
    """
        Класс для перехода в новые единицы измерения.
    """
    @abstractmethod
    def apply(self, data, params):
        pass

    @abstractmethod
    def inverse(self, data, params):
        pass


class MelTransform(Transform):
    basis = None
    inv_basis = None
    """
        Переход в Мелы.
    """
    def __init__(self, hparams):
        assert hparams.fmax <= hparams.sample_rate // 2
        self.basis = librosa.filters.mel(hparams.sample_rate, hparams.n_fft,
                                         n_mels=hparams.num_mels,
                                         fmin=hparams.fmin,
                                         fmax=hparams.fmax)

    @classmethod
    def apply(cls, spectrogram, hparams):
        if cls.basis is None:
            cls.basis = MelTransform(hparams).basis
        return np.dot(cls.basis, spectrogram)

    @classmethod
    def inverse(cls, mel_spectrogram, hparams):
        if cls.inv_basis is None:
            cls.inv_basis = np.linalg.pinv(MelTransform(hparams).inv_basis)
        return np.maximum(1e-10, np.dot(cls.inv_basis, mel_spectrogram))


class LinearTransform(Transform):
    """
        Удобно записать так.
    """
    @staticmethod
    def apply(spectrogram, hparams):
        return spectrogram

    @staticmethod
    def inverse(spectrogram, hparams):
        return spectrogram


class AmpDbTransform(Transform):
    """
        Переход в ДБ.
    """
    @staticmethod
    def apply(x, hparams):
        min_level = np.exp(hparams.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def inverse(x):
        return np.power(10.0, x * 0.05)


class STFourierTransform(Transform):
    """
        Применения преобразования Фурье.
    """
    @staticmethod
    def apply(y, hparams):
        return librosa.stft(y=y, n_fft=hparams.n_fft,
                            hop_length=AudioPreprocess.get_hop_size(hparams),
                            win_length=hparams.win_size)

    @staticmethod
    def inverse(y, hparams):
        return librosa.istft(y, hop_length=AudioPreprocess.get_hop_size(hparams),
                             win_length=hparams.win_size)


class Normalization(Transform):
    """
        Нормализация
    """
    @staticmethod
    def apply(S, hparams):
        if hparams.allow_clipping_in_normalization:
            if hparams.symmetric_mels:
                return np.clip((2 * hparams.max_abs_value) * (
                        (S - hparams.min_level_db) / (-hparams.min_level_db))
                               - hparams.max_abs_value, -hparams.max_abs_value,
                               hparams.max_abs_value)
            else:
                return np.clip(hparams.max_abs_value *
                               ((S - hparams.min_level_db) / (-hparams.min_level_db)), 0,
                               hparams.max_abs_value)

        assert S.max() <= 0 and S.min() - hparams.min_level_db >= 0

        if hparams.symmetric_mels:
            return (2 * hparams.max_abs_value) * (
                    (S - hparams.min_level_db) / (-hparams.min_level_db)) \
                   - hparams.max_abs_value
        else:
            return hparams.max_abs_value * ((S - hparams.min_level_db) /
                                            (-hparams.min_level_db))

    @staticmethod
    def inverse(D, hparams):
        if hparams.allow_clipping_in_normalization:
            if hparams.symmetric_mels:
                return (((np.clip(D, -hparams.max_abs_value,
                                  hparams.max_abs_value) + hparams.max_abs_value) *
                         -hparams.min_level_db / (2 * hparams.max_abs_value))
                        + hparams.min_level_db)
            else:
                return ((np.clip(D, 0, hparams.max_abs_value) *
                         - hparams.min_level_db / hparams.max_abs_value)
                        + hparams.min_level_db)

        if hparams.symmetric_mels:
            return (((D + hparams.max_abs_value) * -hparams.min_level_db / (
                    2 * hparams.max_abs_value)) + hparams.min_level_db)
        else:
            return ((D * - hparams.min_level_db / hparams.max_abs_value)
                    + hparams.min_level_db)


class Spectrogram:
    method_dict = {'linear': LinearTransform,
                   'mel': MelTransform}
    """
        Класс для работ с спектрограммой
    """
    @classmethod
    def make_spectrogram(cls, wav, hparams, parameter):
        method = cls.method_dict[parameter]
        D = STFourierTransform.apply(PreEmphasis.apply(wav, hparams.preemphasis,
                                                       hparams.preemphasize), hparams)
        S = AmpDbTransform.apply(method.apply(np.abs(D), hparams), hparams) \
            - hparams.ref_level_db
        if hparams.signal_normalization:
            return Normalization.apply(S, hparams)
        return S

    @classmethod
    def inverse_spectrogram(cls, spectrogram, hparams, parameter):
        method = cls.method_dict['parameter']
        if hparams.signal_normalization:
            D = Normalization.inverse(spectrogram, hparams)
        else:
            D = spectrogram

        S = method.inverse(AmpDbTransform.inverse(D + hparams.ref_level_db), hparams)  # Convert back to linear

        return PreEmphasis.inverse(_griffin_lim(S ** hparams.power, hparams),
                                   hparams.preemphasis, hparams.preemphasize)


def _griffin_lim(S, hparams):
    """librosa implementation of Griffin-Lim
    Based on https://github.com/librosa/librosa/issues/434
    """
    angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
    S_complex = np.abs(S).astype(np.complex)
    y = STFourierTransform.inverse(S_complex * angles, hparams)
    for i in range(hparams.griffin_lim_iters):
        angles = np.exp(1j * np.angle(STFourierTransform.apply(y, hparams)))
        y = STFourierTransform.inverse(S_complex * angles, hparams)
    return y


##########################################################
# Librosa correct padding
def librosa_pad_lr(x, fsize, fshift):
    return 0, (x.shape[0] // fshift + 1) * fshift - x.shape[0]
