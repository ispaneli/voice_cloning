import torch


class SpeakerEncoder(torch.nn.Module):
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _loss_device = torch.device('cpu')
    _loaded = False

    def __init__(self):
        super().__init__()

        self.lstm = torch.nn.LSTM(input_size=40, hidden_size=256, num_layers=3,
                                  batch_first=True).to(SpeakerEncoder._device)
        self.linear = torch.nn.Linear(in_features=256, out_features=256).to(SpeakerEncoder._device)
        self.relu = torch.nn.ReLU().to(SpeakerEncoder._device)
        self.similarity_weight = torch.nn.Parameter(torch.tensor([10.])).to(SpeakerEncoder._loss_device)
        self.similarity_bias = torch.nn.Parameter(torch.tensor([-5.])).to(SpeakerEncoder._loss_device)
        self.loss_fn = torch.nn.CrossEntropyLoss().to(SpeakerEncoder._loss_device)

    @property
    def device(self):
        return SpeakerEncoder._device

    @staticmethod
    def is_loaded():
        return SpeakerEncoder._loaded

    def load_state_dict(self, *args, **kwargs):
        SpeakerEncoder._loaded = True
        return super().load_state_dict(*args, **kwargs)

    def forward(self, utterances, hidden_init=None):
        out, (hidden, cell) = self.lstm(utterances, hidden_init)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        embeds = embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

        return embeds


encoder_model = SpeakerEncoder()
