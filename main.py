from synthesizer.inference import Synthesizer
from inspect import getsourcefile
import encoder
import vocoder
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import os
import re

MAIN_PATH = os.path.abspath(getsourcefile(lambda: None))[0:-7]
PATH_TO_ENCODER_WEIGHTS = Path(MAIN_PATH + 'saved_models\\pretrained_encoder.pt')
PATH_TO_VOCODER_WEIGHTS = Path(MAIN_PATH + 'saved_models\\pretrained_vocoder.pt')
PATH_TO_SYNTHESIZER_WEIGHTS = Path(MAIN_PATH + '\\synthesizer\\saved_models\\logs-pretrained\\taco_pretrained')

SPECIAL_SYMBOLS = [' ', '.', ',', '!', '?', ':', ';']
SYNTHESIZER_MODEL = None


def play(wav_array):
    sd.stop()
    sd.play(wav_array, Synthesizer.sample_rate, blocking=True)


def check_and_get_path(path_to_file: str, exists: bool) -> Path:
    result = Path(path_to_file)

    if result.exists() is exists:
        if result.is_file() is exists:
            if result.match('*.wav'):
                return result
            else:
                raise FileExistsError
        else:
            raise FileExistsError
    else:
        raise FileExistsError(f"This file does not exist.\nparams: {result}, exists={exists}")


def check_and_get_message(message: str) -> str:
    message = message.replace('\n', ' ')

    for letter in message:
        if not (re.search(r'[a-zA-Z]', letter) or letter in SPECIAL_SYMBOLS):
            raise ValueError

    return message


def process_of_encoder(wav_array):
    if not encoder.is_loaded():
        encoder.load_model(PATH_TO_ENCODER_WEIGHTS)

    encoder_wav = encoder.wav_preprocess(wav_array=wav_array)
    embedding = encoder.get_embedding(encoder_wav)

    return embedding


def process_of_synthesizer(embedding, message):
    global SYNTHESIZER_MODEL

    if SYNTHESIZER_MODEL is None:
        SYNTHESIZER_MODEL = Synthesizer(PATH_TO_SYNTHESIZER_WEIGHTS, low_mem=False)

    texts = [message, '']

    embeds = np.stack([embedding] * len(texts))
    specs = SYNTHESIZER_MODEL.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spectrogram = np.concatenate(specs, axis=1)

    return spectrogram, breaks


def process_of_vocode(spectrogram, breaks):
    if not vocoder.is_loaded():
        vocoder.load_model(PATH_TO_VOCODER_WEIGHTS)

    wav = vocoder.get_waveform(spectrogram)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    wav = wav / np.abs(wav).max() * 0.97

    return wav


def clone_voice(path_to_voice: str, message: str, path_to_result: str, play_result: bool = False) -> None:
    path_to_voice = check_and_get_path(path_to_voice, exists=True)
    message = check_and_get_message(message)
    path_to_result = check_and_get_path(path_to_result, exists=False)

    wav_array = Synthesizer.load_preprocess_wav(path_to_voice)
    embedding = process_of_encoder(wav_array)
    spectrogram, breaks = process_of_synthesizer(embedding, message)
    wav = process_of_vocode(spectrogram, breaks)

    if play_result:
        play(wav)

    sf.write(path_to_result, wav, 16_000)


if __name__ == '__main__':
    print('module main was started!')
    MESSAGE = 'Hello!\nThis is a test message.\nI want to check how the main function works.\nThanks for your attention.\nBye!'
    clone_voice('voice_for_test.wav', MESSAGE, 'result.wav', play_result=True)
    print('done!')

    size = os.path.getsize(Path('result.wav'))

    if 135_000 < size < 180_000:
        print(f'Файл result.wav нормальный по весу.')
        os.remove(Path('result.wav'))
        print('Файл был успешно удален!')
    else:
        raise ValueError('Несоответсвующий вес:', size)