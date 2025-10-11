from torch.nn import functional as F
from torch import nn

class SequenceModel(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim):
        super().__init__()
        self.embedder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embedding_dim),
        )

        self.prompt_projection = nn.Linear(embedding_dim, hidden_dim)

        self.key_projection = nn.Linear(embedding_dim, hidden_dim)

        self.value_projection = nn.Linear(embedding_dim, hidden_dim)

        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, players, prompts):
        embedded_player = self.embedder(players)
        prompts = embedded_player[:, prompts,:]
        prompts = self.prompt_projection(prompts)
        keys = self.key_projection(players)
        values = self.value_projection(players)
