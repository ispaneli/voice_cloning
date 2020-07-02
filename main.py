from inspect import getsourcefile
import numpy as np
import os
from pathlib import Path
import re
import sounddevice as sd
import soundfile as sf

import encoder
import vocoder
from synthesizer.inference import Synthesizer


MAIN_PATH = os.path.abspath(getsourcefile(lambda: None))[0:-7]
PATH_TO_ENCODER_WEIGHTS = Path(MAIN_PATH + 'saved_models/pretrained_encoder.pt')
PATH_TO_VOCODER_WEIGHTS = Path(MAIN_PATH + 'saved_models/pretrained_vocoder.pt')
PATH_TO_SYNTHESIZER_WEIGHTS = Path(MAIN_PATH + 'synthesizer/saved_models/logs-pretrained/taco_pretrained')

SPECIAL_SYMBOLS = [' ', '.', ',', '!', '?', ':', ';']
SYNTHESIZER_MODEL = None


def play(wav_array):
    """
    Play this wav-file on your audio device.

    :param wav_array: wav-file with a voice as an array.
    :return: None
    """
    sd.stop()
    sd.play(wav_array, Synthesizer.sample_rate, blocking=True)


def check_and_get_path(path_to_file: str, exists: bool) -> Path:
    """
    Checks whether the file path entered in the function is correct.

    :param path_to_file: Path to the file being checked.
    :param exists: If the file must exist, it is True; if not, it is False.
    :return: A proven path to the file (type: pathlib.Path).
    """
    result = Path(path_to_file)

    if result.exists() is exists:
        if result.is_file() is exists:
            if result.match('*.wav'):
                return result
            else:
                raise FileExistsError("It's not a wav-file.")
        else:
            raise FileExistsError("It's not a file.")
    else:
        raise FileExistsError("This file does not exist.")


def check_and_get_message(message: str) -> str:
    """
    Checks the correctness of the message received in the function
    (the message must consist ONLY of English letters and punctuation marks).

    :param message: The message that was passed to the function.
    :return: The correct message.
    """
    if not isinstance(message, str):
        raise TypeError("The message type must be a string.")

    message = message.replace('\n', ' ')

    for letter in message:
        if not (re.search(r'[a-zA-z]', letter) or letter in SPECIAL_SYMBOLS):
            raise ValueError("The message can only contain English letters and punctuation marks.")

    return message


def process_of_encoder(wav_array):
    """
    The process of converting a wav array to parameters that characterize this voice.\

    :param wav_array: wav-file with a voice as an array.
    :return: The characteristics of the voice.
    """
    if not encoder.is_loaded():
        encoder.load_model(PATH_TO_ENCODER_WEIGHTS)

    encoder_wav = encoder.wav_preprocess(wav_array=wav_array)
    embedding = encoder.get_embedding(encoder_wav)

    return embedding


def process_of_synthesizer(embedding, message):
    """
    The process of converting voice and message characteristics into a speech spectrogram.

    :param embedding: The characteristics of the voice.
    :param message: The text to be voiced.
    :return: A speech spectrogram and breaks of this utterance.
    """
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
    """
    The process of "voicing" the spectrogram.

    :param spectrogram: A speech spectrogram.
    :param breaks: Breaks of needed utterance.
    :return: Wav-file as an array.
    """
    if not vocoder.is_loaded():
        vocoder.load_model(PATH_TO_VOCODER_WEIGHTS)

    wav_array = vocoder.get_waveform(spectrogram, breaks)

    return wav_array


def clone_voice(path_to_voice: str, message: str, path_to_result: str, play_result: bool = False) -> None:
    """
    Converting a fragment of a person's voice and a message into a voice message voiced by this voice.

    :param path_to_voice: Path to the file with the example of a human voice.
    :param message: The text that the neural network should voice.
    :param path_to_result: The path where the result of the program should be saved.
    :param play_result: Voice the result of the program execution or not.
    :return: None
    """
    path_to_voice = check_and_get_path(path_to_voice, exists=True)
    message = check_and_get_message(message)
    path_to_result = check_and_get_path(path_to_result, exists=False)

    wav_array = Synthesizer.load_preprocess_wav(path_to_voice)
    embedding = process_of_encoder(wav_array)
    spectrogram, breaks = process_of_synthesizer(embedding, message)
    result = process_of_vocode(spectrogram, breaks)

    if play_result:
        play(result)

    sf.write(path_to_result, result, 16_000)


if __name__ == '__main__':
    # EXAMPLE (How to this work):
    MESSAGE = 'Hello!\nThis is a test message.\n' \
              'I want to check how the main function works.' \
              '\nThanks for your attention.\nBye!'

    clone_voice('voice_for_test.wav', MESSAGE, 'result.wav', play_result=True)
