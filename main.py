from synthesizer.inference import Synthesizer
import encoder
from vocoder import inference as vocoder
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
import os

PATH_TO_VOICE = Path('/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/voice_for_test.wav')
FILENAME = 'voice_for_test.wav'
SPEAKER_NAME = 'Ordinary men'


def init_encoder():
    model_fpath = Path('/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/saved_models/pretrained_encoder.pt')
    encoder.load_model(model_fpath)


def init_vocoder():
    model_fpath = Path('/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt')
    vocoder.load_model(model_fpath)


def add_real_utterance(wav, filename, speaker_name):
        # Compute the mel spectrogram
        spec = Synthesizer.make_spectrogram(wav)

        # Compute the embedding
        if not encoder.is_loaded():
            init_encoder()
        encoder_wav = encoder.preprocess_wav(wav_array=wav)
        embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav)

        print('done')
        return embed
        # # Add the utterance
        # utterance = Utterance(filename, speaker_name, wav, spec, embed, partial_embeds, False)
        # self.utterances.add(utterance)
        # self.ui.register_utterance(utterance)
        #
        # # Plot it
        # self.ui.draw_embed(embed, name, "current")
        # self.ui.draw_umap_projections(self.utterances)


def synthesize(synthesizer, embed):
    # Synthesize the spectrogram
    if synthesizer is None:
        checkpoints_dir = Path('/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/synthesizer/saved_models/logs-pretrained/taco_pretrained')
        synthesizer = Synthesizer(checkpoints_dir, low_mem=False)
    if not synthesizer.is_loaded():
        print(synthesizer.checkpoint_fpath)

    MESSAGE = 'Hello!\nMy name is John.\nIt is test node!\n'
    MESSAGE = 'Hello!\nThis is a test message.\nI want to check how the main function works.\nThanks for your attention.\nBye!'

    # Первый вариант.
    #texts = MESSAGE.split("\n")
    # Второй вариант (работает куда лучше и четче).
    texts = [MESSAGE.replace('\n', ' '), '']

    # embed = self.ui.selected_utterance.embed
    embeds = np.stack([embed] * len(texts))
    specs = synthesizer.synthesize_spectrograms(texts, embeds)
    breaks = [spec.shape[1] for spec in specs]
    spec = np.concatenate(specs, axis=1)

    return spec, breaks

    # self.ui.draw_spec(spec, "generated")
    # self.current_generated = (self.ui.selected_utterance.speaker_name, spec, breaks, None)
    # self.ui.set_loading(0)

def play(wav, sample_rate):
    sd.stop()
    sd.play(wav, sample_rate, blocking=True)


def vocode(spec, breaks):
    assert spec is not None

    # Synthesize the waveform
    if not vocoder.is_loaded():
        init_vocoder()

    def vocoder_progress(i, seq_len, b_size, gen_rate):
        real_time_factor = (gen_rate / Synthesizer.sample_rate) * 1000
        line = "Waveform generation: %d/%d (batch size: %d, rate: %.1fkHz - %.2fx real time)" \
               % (i * b_size, seq_len * b_size, b_size, gen_rate, real_time_factor)
        print(line)

    current_vocoder_fpath = Path('/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/vocoder/saved_models/pretrained/pretrained.pt')

    if current_vocoder_fpath is not None:
        wav = vocoder.infer_waveform(spec, progress_callback=vocoder_progress)
    else:
        wav = Synthesizer.griffin_lim(spec)

    # Add breaks
    b_ends = np.cumsum(np.array(breaks) * Synthesizer.hparams.hop_size)
    b_starts = np.concatenate(([0], b_ends[:-1]))
    wavs = [wav[start:end] for start, end, in zip(b_starts, b_ends)]
    breaks = [np.zeros(int(0.15 * Synthesizer.sample_rate))] * len(breaks)
    wav = np.concatenate([i for w, b in zip(wavs, breaks) for i in (w, b)])

    # Play it
    wav = wav / np.abs(wav).max() * 0.97
    play(wav, Synthesizer.sample_rate)

    # Compute the embedding
    # TODO: this is problematic with different sampling rates, gotta fix it
    if not encoder.is_loaded():
        init_encoder()
    encoder_wav = encoder.preprocess_wav(wav_array=wav)
    embed, partial_embeds, _ = encoder.embed_utterance(encoder_wav)

    sampling_frequency = 16000
    #mydata = sd.rec(wav, frames=sampling_frequency, dtype='float64', channels=2, blocking=True)
    sf.write('result.wav', wav, sampling_frequency)


def voice_create(synthesizer, embed):
    spec, breaks = synthesize(synthesizer, embed)
    vocode(spec, breaks)


def main(path_to_voice=PATH_TO_VOICE, filename=FILENAME, speaker_name=SPEAKER_NAME):
    wav = Synthesizer.load_preprocess_wav(path_to_voice)
    EMBED = add_real_utterance(wav, filename, speaker_name)

    # synthesizer будет типа Synthesizer.
    synthesizer = None
    voice_create(synthesizer, EMBED)


if __name__ == '__main__':
    print('module main was started!')
    main()
    print('done!')

    path_to_result = '/home/ilya/PycharmProjects/Real-Time-Voice-Cloning/result.wav'
    size = os.path.getsize(path_to_result)

    if 450_000 < size < 500_000:
        print(f'Файл result.wav нормальный по весу.')
        os.remove(path_to_result)
        print('Файл был успешно удален!')
    else:
        raise ValueError('Несоответсвующий вес:', size)



