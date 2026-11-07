"""Torch architecture; forward returns per-position hidden states for efficient projection."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from data import CONTEXT, VOCAB_SIZE, WIDTH, char_row


class Residual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv1d(channels, channels, 3, padding=1)
        self.mix = nn.Conv1d(channels, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        h = x + self.mix(F.gelu(self.local(x)))
        return self.norm(h.transpose(1, 2)).transpose(1, 2)


class Stage(nn.Module):
    def __init__(self, previous, channels):
        super().__init__()
        self.project = nn.Conv1d(previous, channels, 1)
        self.norm = nn.LayerNorm(channels)
        self.blocks = nn.Sequential(Residual(channels), Residual(channels))

    def forward(self, x):
        x = F.gelu(self.project(x)).transpose(1, 2)
        return self.blocks(self.norm(x).transpose(1, 2))


class CharacterEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(70, WIDTH, padding_idx=0)
        self.stages = nn.Sequential(Stage(128, 64), Stage(64, 32), Stage(32, 16))
        self.readout = nn.Conv1d(16, 1, kernel_size=1)

    def forward(self, codes):
        h = self.embed(codes.long()).transpose(1, 2)
        return self.readout(self.stages(h)).squeeze(1)


class CausalLM(nn.Module):
    def __init__(self, branch, dropout):
        super().__init__()
        self.branch = branch
        # Initialize every shared layer before either alternative input branch.
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(WIDTH, 4, 512, dropout=dropout,
                                    activation='gelu', batch_first=True, norm_first=True) for _ in range(8)])
        self.out = nn.Linear(WIDTH, VOCAB_SIZE)
        self.input = nn.Embedding(VOCAB_SIZE, WIDTH, padding_idx=0) if branch == 'lookup' else CharacterEncoder()
        angle = torch.arange(CONTEXT).float()[:, None] * torch.exp(
            torch.arange(0, WIDTH, 2).float() * (-math.log(10000.) / WIDTH))[None, :]
        position = torch.zeros(CONTEXT, WIDTH)
        position[:, 0::2], position[:, 1::2] = angle.sin(), angle.cos()
        self.register_buffer('position', position)
        self.register_buffer('forbid', torch.ones(CONTEXT, CONTEXT, dtype=torch.bool).triu(1))

    def forward(self, keys, ids, chars):
        if self.branch == 'lookup':
            h = self.input(ids[keys])
        else:
            # No dropout/batch statistics in the character encoder: repeated spellings
            # can share a forward pass within this batch, while gradients still sum.
            unique, inv = torch.unique(keys, return_inverse=True)
            h = self.input(chars[unique])[inv.reshape_as(keys)]
        length = keys.shape[1]; mask = self.forbid[:length, :length]; pad = keys.eq(0)
        h = h + self.position[:length]
        for layer in self.layers:
            h = layer(h, src_mask=mask, src_key_padding_mask=pad)
        return h


def smoke(device):
    torch.set_num_threads(1)
    keys = torch.tensor([[1, 2, 3], [2, 3, 0]], device=device)
    ids = torch.arange(5, device=device)
    chars = torch.tensor([[0]*128, *map(char_row,['sun','star','mars','jupiter'])], device=device)
    for branch, expected in [('lookup', 6_726_176), ('character', 4_230_145)]:
        torch.manual_seed(21); net = CausalLM(branch, 0.0).to(device); net.eval()
        assert sum(p.numel() for p in net.parameters()) == expected
        changed = keys.clone(); changed[0, 2] = 4
        with torch.no_grad():
            a, b = net(keys,ids,chars), net(changed,ids,chars)
            torch.testing.assert_close(a[0,:2], b[0,:2], rtol=2e-5, atol=2e-5)
        net.train(); opt = torch.optim.AdamW(net.parameters(),lr=2e-4)
        h = net(keys,ids,chars); loss = F.cross_entropy(net.out(h[keys != 0]), torch.tensor([2,3,1,3,1],device=device))
        loss.backward(); assert torch.isfinite(loss); opt.step()
        print(f'{branch}: params={expected}, prefix invariant, optimizer step passed.')
