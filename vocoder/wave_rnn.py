import numpy as np
from scipy.signal import lfilter
import torch


TARGET = 8000
OVERLAP = 800


class ResBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.conv2 = torch.nn.Conv1d(128, 128, kernel_size=1, bias=False)
        self.batch_norm1 = torch.nn.BatchNorm1d(128)
        self.batch_norm2 = torch.nn.BatchNorm1d(128)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = torch.nn.functional.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        k_size = 5
        self.conv_in = torch.nn.Conv1d(80, 128, kernel_size=k_size, bias=False)
        self.batch_norm = torch.nn.BatchNorm1d(128)
        self.layers = torch.nn.ModuleList()
        for i in range(10):
            self.layers.append(ResBlock())
        self.conv_out = torch.nn.Conv1d(128, 128, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = torch.nn.functional.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(torch.nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class StepUpSamplingNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()

        step_up_scales = (5, 5, 8)
        total_scale = np.cumproduct(step_up_scales)[-1]
        self.indent = 2 * total_scale
        self.resnet = MelResNet()
        self.resnet_stretch = Stretch2d(total_scale, 1)
        self.up_layers = torch.nn.ModuleList()
        for scale in step_up_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = torch.nn.Conv2d(1, 1, kernel_size=k_size, padding=padding, bias=False)
            conv.weight.data.fill_(1. / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        aux = self.resnet(m).unsqueeze(1)
        aux = self.resnet_stretch(aux)
        aux = aux.squeeze(1)
        m = m.unsqueeze(1)
        for f in self.up_layers: m = f(m)
        m = m.squeeze(1)[:, :, self.indent:-self.indent]
        return m.transpose(1, 2), aux.transpose(1, 2)


def decode_mu_law(data):
    """Returns the value according to Mu law."""
    mu = 2 ** 9 - 1
    return np.sign(data) / mu * ((mu + 1) ** np.abs(data) - 1)


def get_gru_cell(gru: torch.nn.GRU) -> torch.nn.GRUCell:
    """
    Gives a ready-made cell of GRU.

    'ih' is input-hidden; 'hh' is hidden-hidden.
    :param gru: Gated Recurrent Unit.
    :return: ready-made cell of GRU.
    """
    gru_cell = torch.nn.GRUCell(input_size=gru.input_size, hidden_size=gru.hidden_size)

    gru_cell.weight_ih.data = gru.weight_ih_l0.data
    gru_cell.weight_hh.data = gru.weight_hh_l0.data

    gru_cell.bias_ih.data = gru.bias_ih_l0.data
    gru_cell.bias_hh.data = gru.bias_hh_l0.data

    return gru_cell


def add_tensor_pad(input_data: torch.Tensor, *, pad_length: int = 2, side: str) -> torch.Tensor:
    """
    Adds a tensor pad before, after, or before and after the input data.

    :param input_data: Data without tensor pad.
    :param pad_length: Length of tensor pad, type: int.
    :param side: Where to add a tensor pad. Only 'before', 'after' or 'both'.
    :return: Data with tensor pad.
    """
    batch_size, freq_size, volume_size = input_data.size()

    if side == 'both':
        total = freq_size + 2 * pad_length
    else:
        total = freq_size + pad_length

    output_data = torch.zeros(batch_size, total, volume_size).cuda()

    if side == 'before' or side == 'both':
        output_data[:, pad_length:pad_length + freq_size, :] = input_data
    elif side == 'after':
        output_data[:, :freq_size, :] = input_data

    return output_data


def add_overlap(input_data: torch.Tensor) -> torch.Tensor:
    """Fold the tensor with overlap for quick batched inference."""

    # size() gives (batch_size, freq_size, volume_size)
    _, freq_size, volume_size = input_data.size()

    # Calculate variables needed.
    num_folds = (freq_size - OVERLAP) // (TARGET + OVERLAP)
    remaining = freq_size - (num_folds * (OVERLAP + TARGET) + OVERLAP)

    # Pad if some time steps poking out
    if remaining != 0:
        num_folds += 1
        pad_length = TARGET + 2 * OVERLAP - remaining
        input_data = add_tensor_pad(input_data, pad_length=pad_length, side='after')

    output_data = torch.zeros(num_folds, TARGET + 2 * OVERLAP, volume_size).cuda()

    # Get the values for the folded tensor
    for i in range(num_folds):
        start = i * (TARGET + OVERLAP)
        end = start + TARGET + 2 * OVERLAP
        output_data[i] = input_data[:, start:end, :]

    return output_data


def add_crossfade(input_data: np.ndarray) -> np.ndarray:
    """
    Applies a crossfade and unfolds into a 1d array.

    :param input_data: Loop before applying the crossfade.
    :return: Magnifier after applying the crossfade.
    """
    num_folds, freq_size = input_data.shape
    target = freq_size - 2 * OVERLAP
    total_length = num_folds * (target + OVERLAP) + OVERLAP

    # Generating silence at the beginning.
    silence_length = OVERLAP // 2
    fade_length = OVERLAP - silence_length
    silence = np.zeros(silence_length, dtype=np.float64)

    # Process of crossfading.
    equal_power = np.linspace(-1, 1, fade_length, dtype=np.float64)
    fade_in = np.sqrt(0.5 * (1 + equal_power))
    fade_out = np.sqrt(0.5 * (1 - equal_power))
    fade_in = np.concatenate([silence, fade_in])
    fade_out = np.concatenate([fade_out, silence])

    # Apply gain factor.
    input_data[:, :OVERLAP] *= fade_in
    input_data[:, -OVERLAP:] *= fade_out

    # Generating the result by summing the samples in a loop.
    output_data = np.zeros(total_length, dtype=np.float64)
    for i in range(num_folds):
        start = i * (target + OVERLAP)
        end = start + target + 2 * OVERLAP
        output_data[start:end] += input_data[i]

    return output_data


class WaveRNN(torch.nn.Module):
    _loaded = False

    def __init__(self):
        super().__init__()

        self.upsample = StepUpSamplingNetwork()

        # Input linear transformation.
        self.I = torch.nn.Linear(in_features=113, out_features=512)

        # 'GRU' is Gated Recurrent Units, popular type of RNN (Recurrent Neural Network).
        self.rnn1 = torch.nn.GRU(input_size=512, hidden_size=512, batch_first=True)
        self.rnn2 = torch.nn.GRU(input_size=544, hidden_size=512, batch_first=True)

        # Other linear transformation.
        self.fc1 = torch.nn.Linear(in_features=544, out_features=512)
        self.fc2 = torch.nn.Linear(in_features=544, out_features=512)
        self.fc3 = torch.nn.Linear(in_features=512, out_features=2**9)

        self.step = torch.nn.Parameter(torch.zeros(1).long(), requires_grad=False)

    @staticmethod
    def is_loaded():
        return WaveRNN._loaded

    def load_state_dict(self, *args, **kwargs):
        WaveRNN._loaded = True
        return super().load_state_dict(*args, **kwargs)

    def generate_wav(self, mel_spectrogram: torch.Tensor) -> np.ndarray:
        """
        The model "voices" the Mel spectrogram, thereby generating a wav audio track.

        :param mel_spectrogram: Spectrogram in mel-scale, written as a tensor.
        :return: Wav file written as an array.
        """
        self.eval()

        first_gru_cell = get_gru_cell(self.rnn1)
        second_gpu_cell = get_gru_cell(self.rnn2)

        with torch.no_grad():
            mel_spectrogram = mel_spectrogram.cuda()
            wave_length = (mel_spectrogram.size(-1) - 1) * 200

            mel_spectrogram = add_tensor_pad(mel_spectrogram.transpose(1, 2), side='both')
            mel_spectrogram, aux = self.upsample(mel_spectrogram.transpose(1, 2))
            mel_spectrogram = add_overlap(input_data=mel_spectrogram)
            aux = add_overlap(input_data=aux)

            # size() gives (batch_size, freq_size, volume_size)
            batch_size, freq_size, _ = mel_spectrogram.size()

            first_hidden_tensor = torch.zeros(batch_size, 512).cuda()
            second_hidden_tensor = torch.zeros(batch_size, 512).cuda()
            main_tensor = torch.zeros(batch_size, 1).cuda()

            aux_split = [aux[:, :, 32 * i:32 * (i + 1)] for i in range(4)]
            output_data = []

            for i in range(freq_size):
                mel_tensor = mel_spectrogram[:, i, :]
                aux_tensors = [aux_tensor[:, i, :] for aux_tensor in aux_split]

                main_tensor = torch.cat([main_tensor, mel_tensor, aux_tensors[0]], dim=1)
                main_tensor = self.I(main_tensor)
                first_hidden_tensor = first_gru_cell(main_tensor, first_hidden_tensor)

                main_tensor += first_hidden_tensor
                input_data = torch.cat([main_tensor, aux_tensors[1]], dim=1)
                second_hidden_tensor = second_gpu_cell(input_data, second_hidden_tensor)
                main_tensor += second_hidden_tensor
                main_tensor = torch.cat([main_tensor, aux_tensors[2]], dim=1)
                main_tensor = torch.nn.functional.relu(self.fc1(main_tensor))
                main_tensor = torch.cat([main_tensor, aux_tensors[3]], dim=1)
                main_tensor = torch.nn.functional.relu(self.fc2(main_tensor))

                probability_of_sampling = torch.nn.functional.softmax(self.fc3(main_tensor), dim=1)
                categorical_distribution = torch.distributions.Categorical(probability_of_sampling)

                sample = 2 * categorical_distribution.sample().float() / (2**9 - 1) - 1
                output_data.append(sample)
                main_tensor = sample.unsqueeze(-1)

        output_data = torch.stack(output_data).transpose(0, 1)
        output_data = output_data.cpu().numpy()
        output_data = output_data.astype(np.float64)
        output_data = add_crossfade(output_data)
        output_data = decode_mu_law(output_data)
        output_data = lfilter([1], [1, -0.97], output_data)

        # Fade-out at the end to avoid signal cutting out suddenly.
        fade_out = np.linspace(1, 0, 4_000)
        output_data = output_data[:wave_length]
        output_data[-4_000:] *= fade_out

        self.train()

        return output_data


vocoder_model = WaveRNN().cuda()
