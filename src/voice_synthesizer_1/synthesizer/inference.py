from synthesizer.tacotron2 import Tacotron2
from synthesizer.hparams import hparams
from multiprocess.pool import Pool
from pathlib import Path
from typing import Union, List
import tensorflow as tf
import numpy as np
import librosa


class Synthesizer:
    sample_rate = hparams.sample_rate
    hparams = hparams

    def __init__(self, checkpoints_dir: Path, verbose=True, low_mem=False):
        """
        Creates a synthesizer ready for inference. The actual model isn't loaded in
        memory until needed or until load() is called.
        
        :param checkpoints_dir: path to the directory containing the checkpoint file
        as well as the weight files (.data, .index and .meta files)
        :param verbose: if False, only tensorflow's output will be printed
        :param low_mem: if True, the model will be loaded in a separate process and its resources 
        will be released after each usage. Adds a large overhead, only recommended if your GPU 
        memory is low (<= 2gb)
        """
        self.verbose = verbose
        self._low_mem = low_mem

        # Prepare the model
        self._model = None  # type: Tacotron2
        checkpoint_state = tf.train.get_checkpoint_state(checkpoints_dir)
        if checkpoint_state is None:
            raise Exception("Could not find any synthesizer weights under %s" % checkpoints_dir)
        self.checkpoint_fpath = checkpoint_state.model_checkpoint_path
        if verbose:
            model_name = checkpoints_dir.parent.name.replace("logs-", "")
            step = int(self.checkpoint_fpath[self.checkpoint_fpath.rfind('-') + 1:])
            print("Found synthesizer \"%s\" trained to step %d" % (model_name, step))

    def is_loaded(self):
        """
        Whether the model is loaded in GPU memory.
        """
        return self._model is not None

    def load(self):
        """
        Effectively loads the model to GPU memory given the weights file that was passed in the
        constructor.
        """
        if self._low_mem:
            raise Exception("Cannot load the synthesizer permanently in low mem mode")
        tf.reset_default_graph()
        self._model = Tacotron2(self.checkpoint_fpath, hparams)

    def synthesize_spectrograms(self, texts: List[str],
                                embeddings: Union[np.ndarray, List[np.ndarray]],
                                return_alignments=False):
        """
        Synthesizes mel spectrograms from texts and speaker embeddings.
        :param texts: a list of N text prompts to be synthesized
        :param embeddings: a numpy array or list of speaker embeddings of shape (N, 256)
        :param return_alignments: if True, a matrix representing the alignments between the
        characters
        and each decoder output step will be returned for each spectrogram
        :return: a list of N melspectrograms as numpy arrays of shape (80, Mi), where Mi is the
        sequence length of spectrogram i, and possibly the alignments.
        """
        if not self._low_mem:
            # Usual inference mode: load the model on the first request and keep it loaded.
            if not self.is_loaded():
                self.load()
            specs, alignments = self._model.my_synthesize(embeddings, texts)
        else:
            # Low memory inference mode: load the model upon every request. The model has to be
            # loaded in a separate process to be able to release GPU memory (a simple workaround
            # to tensorflow's intricacies)
            specs, alignments = Pool(1).starmap(Synthesizer._one_shot_synthesize_spectrograms,
                                                [(self.checkpoint_fpath, embeddings, texts)])[0]

        return (specs, alignments) if return_alignments else specs

    @staticmethod
    def load_preprocess_wav(fpath):
        """
        Loads and preprocesses an audio file under the same conditions the audio files were used to
        train the synthesizer.
        """
        wav = librosa.load(fpath, hparams.sample_rate)[0]
        if hparams.rescale:
            wav = wav / np.abs(wav).max() * hparams.rescaling_max
        return wav