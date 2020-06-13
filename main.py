from synthesizer.inference import Synthesizer
import encoder
import vocoder
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import os


def init_encoder():
    path_to_weights = Path('./saved_models/pretrained_encoder.pt')
    encoder.load_model(path_to_weights)


def init_vocoder():
    path_to_weights = Path('./saved_models/pretrained_vocoder.pt')
    vocoder.load_model(path_to_weights)


def add_real_utterance(wav):
    if not encoder.is_loaded():
        init_encoder()

    encoder_wav = encoder.wav_preprocess(wav_array=wav)
    embed = encoder.get_embedding(encoder_wav)

    return embed


def synthesize(synthesizer, embed):
    # Synthesize the spectrogram
    if synthesizer is None:
        checkpoints_dir = Path('./synthesizer/saved_models/logs-pretrained/taco_pretrained')
        synthesizer = Synthesizer(checkpoints_dir, low_mem=False)
    if not synthesizer.is_loaded():
        print(synthesizer.checkpoint_fpath)

    MESSAGE = 'Hello!\nThis is a test message.\nI want to check how the main function works.\nThanks for your attention.\nBye!'

    texts = [MESSAGE.replace('\n', ' '), '']

    embeds = np.stack([embed] * len(texts))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    return spec, breaks


def play(wav, sample_rate):
    sd.stop()
    sd.play(wav, sample_rate, blocking=True)


def vocode(spec, breaks):
    assert spec is not None

    # Synthesize the waveform
    if not vocoder.is_loaded():
        init_vocoder()

    wav = vocoder.get_waveform(spec)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    play(wav, Synthesizer.sample_rate)

    sf.write('result.wav', wav, 16_000)


def voice_create(synthesizer, embed):
    spec, breaks = synthesize(synthesizer, embed)
    vocode(spec, breaks)


def main(path_to_voice_sample: Path):
    wav = Synthesizer.load_preprocess_wav(path_to_voice_sample)
    embed = add_real_utterance(wav)

    # synthesizer будет типа Synthesizer.
    synthesizer = None
    voice_create(synthesizer, embed)


if __name__ == '__main__':
    print('module main was started!')
    main(Path('voice_for_test.wav'))
    print('done!')

    path_to_result = '/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/result.wav'
    size = os.path.getsize(path_to_result)

    if 450_000 < size < 500_000:
        print(f'Файл result.wav нормальный по весу.')
        os.remove(path_to_result)
        print('Файл был успешно удален!')
    else:
        raise ValueError('Несоответсвующий вес:', size)
