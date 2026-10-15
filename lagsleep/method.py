import torch
from torch import nn


def entropy(probability):
    return -(probability * probability.clamp_min(1e-8).log()).sum(-1)


def disagreement(probability, weights):
    mixture = (probability * weights[..., None]).sum(1)
    return (entropy(mixture) - (weights * entropy(probability)).sum(1)).clamp_min(0)


class Encoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, width, 25, stride=5, padding=12), nn.GELU(),
            nn.Conv1d(width, width, 9, stride=4, padding=4), nn.GELU(),
            nn.Conv1d(width, width, 7, stride=2, padding=3), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
        )

    def forward(self, x):
        x = (x - x.mean(-1, keepdim=True)) / x.std(-1, keepdim=True).clamp_min(1e-5)
        return self.net(x)


class LagSleep(nn.Module):
    def __init__(self, width=48, temperature=1):
        super().__init__()
        self.eeg = Encoder(width)
        self.eog = Encoder(width)
        self.relation = nn.Sequential(nn.Linear(3 * width, width), nn.GELU(), nn.Linear(width, 1))
        self.fusion = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(), nn.Linear(width, 5))
        self.unimodal = nn.Linear(width, 5)
        self.temperature = temperature

    def forward(self, eeg, candidates):
        batch, count, length = candidates.shape
        e = self.eeg(eeg[:, None])
        o = self.eog(candidates.reshape(batch * count, 1, length)).reshape(batch, count, -1)
        repeated = e[:, None].expand(-1, count, -1)
        pair = torch.cat([repeated, o], -1)
        scores = self.relation(torch.cat([pair, (repeated - o).abs()], -1)).squeeze(-1)
        weights = (scores / self.temperature).softmax(-1)
        probability = self.fusion(pair).softmax(-1)
        mixture = (probability * weights[..., None]).sum(1)
        return {"mixture": mixture, "candidates": probability, "weights": weights,
                "eeg": self.unimodal(e).softmax(-1), "js": disagreement(probability, weights),
                "offset_entropy": entropy(weights), "feature": self.fusion(torch.cat([e, (o * weights[..., None]).sum(1)], -1)).softmax(-1)}
