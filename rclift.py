"""RCLiFT: shared RC neuronal dynamics inside Transformer attention and FFNs.

Input: [batch, 168, 22] historical load, available weather and calendar features.
Output: [batch, 24] electrical-load forecasts in the input load's scale.
The implementation requires PyTorch only. Experiment checkpoints load directly.
"""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F


def _inverse_softplus(x):
    return x + torch.log(-torch.expm1(-x))


def _relax(e, rate, elapsed=1, lag=1):
    batch, length, width = e.shape
    if lag > 1:
        e = e.reshape(batch, length // lag, lag, width).transpose(1, 2).reshape(batch * lag, length // lag, width)
        rate = rate.reshape(batch, length // lag, lag, width).transpose(1, 2).reshape(batch * lag, length // lag, width)
    retention = torch.exp(-elapsed * rate)
    a = torch.cat((torch.zeros_like(retention[:, :1]), retention[:, 1:]), 1)
    state = torch.cat((e[:, :1], ((1 - retention) * e)[:, 1:]), 1)
    offset = 1
    while offset < a.shape[1]:
        state = torch.cat((state[:, :offset], state[:, offset:] + a[:, offset:] * state[:, :-offset]), 1)
        a = torch.cat((a[:, :offset], a[:, offset:] * a[:, :-offset]), 1)
        offset *= 2
    if lag > 1:
        state = state.reshape(batch, lag, length // lag, width).transpose(1, 2).reshape(batch, length, width)
    return state


def _context(x):
    changes = []
    for lag in (1, 24):
        previous = torch.cat((x[:, :1, :2].expand(-1, lag, -1), x[:, :-lag, :2]), 1)
        changes.append(x[:, :, :2] - previous)
    return torch.cat((x, *changes), -1)


def _periodic_relax(e, rate):
    a = torch.exp(-rate)
    prefix = (1-a)*e
    offset = 1
    while offset < a.shape[1]:
        prefix = torch.cat((prefix[:, :offset], prefix[:, offset:] + a[:, offset:]*prefix[:, :-offset]), 1)
        a = torch.cat((a[:, :offset], a[:, offset:]*a[:, :-offset]), 1)
        offset *= 2
    initial = prefix[:, -1:]/(-torch.expm1(-rate.sum(1, keepdim=True)))
    return prefix + torch.exp(-rate.cumsum(1))*initial


class LiquidTokenProjection(nn.Module):
    def __init__(self, mixed_time=False):
        super().__init__()
        self.mixed_time = mixed_time
        self.equilibrium = nn.Linear(26, 64)
        self.conductance = nn.Linear(26, 64)
        self.capacity = nn.Parameter(_inverse_softplus(torch.logspace(0, math.log10(168), 64)))
        self.gain = nn.Parameter(torch.full((64,), math.log(math.expm1(1.5))))
        self.gate = nn.Linear(26, 64)
        self.output = nn.Linear(64, 64)
        nn.init.normal_(self.conductance.weight, std=.02)
        nn.init.constant_(self.conductance.bias, math.log(math.expm1(1)))

    def forward(self, x):
        equilibrium = torch.tanh(self.equilibrium(x))
        rate = (F.softplus(self.conductance(x)) + 1e-5) / (F.softplus(self.capacity) + 1e-5)
        state = torch.cat((_relax(equilibrium[:, :, :32], rate[:, :, :32]),
                          _relax(equilibrium[:, :, 32:], rate[:, :, 32:], 24, 24)), -1)
        activation = state + F.softplus(self.gain) * (equilibrium - state)
        hourly = self.output(activation * torch.sigmoid(self.gate(x)))
        if self.mixed_time:
            daily = hourly[:, :144].reshape(x.shape[0], 6, 24, 64).mean(2)
            return torch.cat((daily, hourly[:, -24:]), 1)
        return hourly[:, -24:]


class LiquidFFN(nn.Module):
    def __init__(self, dropout=.1, periodic=False):
        super().__init__()
        self.content = nn.Linear(64, 128)
        self.equilibrium = nn.Linear(64, 128)
        self.conductance = nn.Linear(64, 128)
        self.log_capacity = nn.Parameter((torch.logspace(math.log10(.5), math.log10(24), 128) + 1e-5).log())
        self.gain = nn.Parameter(torch.full((128,), .5))
        self.out = nn.Linear(128, 64)
        self.dropout = nn.Dropout(dropout)
        self.periodic = periodic
        nn.init.normal_(self.conductance.weight, std=.02)
        nn.init.constant_(self.conductance.bias, math.log(math.expm1(1)))

    def fields(self, x):
        equilibrium = torch.tanh(self.equilibrium(x))
        rate = (F.softplus(self.conductance(x)) + 1e-5) / self.log_capacity.exp()
        state = _periodic_relax(equilibrium, rate) if self.periodic else _relax(equilibrium, rate)
        content = F.gelu(self.content(x))
        activation = content * (2 * torch.sigmoid(equilibrium + self.gain * state))
        return activation, content, equilibrium, state, rate

    def forward(self, x):
        activation, *_ = self.fields(x)
        return self.out(self.dropout(activation))


class SharedRCBlock(nn.Module):
    def __init__(self, strength=.25, routing='rate', dropout=.1, periodic=False, circular=False,
                 mixed_time=False):
        super().__init__()
        if routing not in ('rate', 'operator', 'current_query', 'state_value'):
            raise ValueError(routing)
        self.strength, self.routing = strength, routing
        self.circular = circular
        self.mixed_time = mixed_time
        self.norm1, self.norm2 = nn.LayerNorm(64), nn.LayerNorm(64)
        self.qkv = nn.Linear(64, 192)
        self.attention_out = nn.Linear(64, 64)
        self.dropout = nn.Dropout(dropout)
        self.ffn = LiquidFFN(dropout, periodic)

    def forward(self, z):
        x = self.norm1(z)
        batch, length, width = x.shape
        if self.routing == 'rate':
            query, key, value = self.qkv(x).chunk(3, -1)
            rate = (F.softplus(self.ffn.conductance(x)) + 1e-5) / self.ffn.log_capacity.exp()
        else:
            activation, content, equilibrium, state, rate = self.ffn.fields(x)
            retained = x + self.ffn.out(activation)
            if self.routing == 'current_query':
                change = content * (2 * torch.sigmoid(equilibrium + self.ffn.gain * (equilibrium - state)))
                query = F.linear(x + self.ffn.out(change), self.qkv.weight[:width], self.qkv.bias[:width])
                key, value = F.linear(retained, self.qkv.weight[width:], self.qkv.bias[width:]).chunk(2, -1)
            elif self.routing == 'state_value':
                query, key = F.linear(x, self.qkv.weight[:2*width], self.qkv.bias[:2*width]).chunk(2, -1)
                value = F.linear(retained, self.qkv.weight[2*width:], self.qkv.bias[2*width:])
            else:
                query, key, value = self.qkv(retained).chunk(3, -1)
        q, k, v = (a.reshape(batch, length, 4, 16).transpose(1, 2) for a in (query, key, value))
        dt = x.new_tensor([24.] * 6 + [12.5] + [1.] * 23)[None, :, None] if self.mixed_time else 1.
        head_rate = (rate * dt).reshape(batch, length, 4, 32).mean(-1).transpose(1, 2)
        clock = head_rate.cumsum(2)
        distance = (clock[:, :, :, None] - clock[:, :, None, :]).abs()
        if self.circular:
            distance = torch.minimum(distance, head_rate.sum(2)[:, :, None, None]-distance)
        bias = -self.strength * distance
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=bias,
            dropout_p=self.dropout.p if self.training else 0.)
        z = z + self.dropout(self.attention_out(y.transpose(1, 2).reshape(batch, length, width)))
        if self.mixed_time:
            x = self.norm2(z)
            equilibrium = torch.tanh(self.ffn.equilibrium(x))
            rate = (F.softplus(self.ffn.conductance(x)) + 1e-5) / self.ffn.log_capacity.exp()
            state = _relax(equilibrium, rate * dt)
            activation = F.gelu(self.ffn.content(x)) * (2 * torch.sigmoid(
                equilibrium + self.ffn.gain * state))
            return z + self.dropout(self.ffn.out(self.ffn.dropout(activation)))
        return z + self.dropout(self.ffn(self.norm2(z)))


class ContinuedRCBlock(SharedRCBlock):
    """The state forming attention projections continues inside the FFN."""
    def __init__(self, strength=.25, dropout=.1, periodic=False, activation='gated', continue_state=True):
        super().__init__(strength=strength, routing='current_query',
                         dropout=dropout, periodic=periodic)
        self.activation = activation
        self.continue_state = continue_state
        if activation == 'voltage':
            nn.init.constant_(self.ffn.gain, math.log(math.expm1(1.5)))

    def forward(self, z, carried_state=None):
        x = self.norm1(z)
        batch, length, width = x.shape
        neuron = self.ffn
        equilibrium = torch.tanh(neuron.equilibrium(x))
        rate = (F.softplus(neuron.conductance(x)) + 1e-5) / neuron.log_capacity.exp()
        if carried_state is None:
            state = (_periodic_relax(equilibrium, rate) if neuron.periodic
                     else _relax(equilibrium, rate))
        else:
            retention = torch.exp(-rate)
            state = retention * carried_state + (1 - retention) * equilibrium
        if self.activation == 'voltage':
            content = torch.sigmoid(neuron.content(x))
            retained = neuron.out(content * state)
            changing = neuron.out(content * F.softplus(neuron.gain) * (equilibrium - state))
        else:
            content = F.gelu(neuron.content(x))
            retained = x + neuron.out(content * (2 * torch.sigmoid(equilibrium + neuron.gain * state)))
            changing = x + neuron.out(content * (2 * torch.sigmoid(
                equilibrium + neuron.gain * (equilibrium - state))))
        query = F.linear(changing, self.qkv.weight[:width], self.qkv.bias[:width])
        key, value = F.linear(retained, self.qkv.weight[width:], self.qkv.bias[width:]).chunk(2, -1)
        q, k, v = (a.reshape(batch, length, 4, width // 4).transpose(1, 2)
                   for a in (query, key, value))
        clock = rate.reshape(batch, length, 4, -1).mean(-1).cumsum(1).transpose(1, 2)
        distance = (clock[:, :, :, None] - clock[:, :, None, :]).abs()
        attended = F.scaled_dot_product_attention(q, k, v,
            attn_mask=-self.strength * distance,
            dropout_p=self.dropout.p if self.training else 0.)
        z = z + self.dropout(self.attention_out(
            attended.transpose(1, 2).reshape(batch, length, width)))
        x = self.norm2(z)
        equilibrium_after = torch.tanh(neuron.equilibrium(x))
        rate_after = (F.softplus(neuron.conductance(x)) + 1e-5) / neuron.log_capacity.exp()
        if self.continue_state:
            retention = torch.exp(-rate_after)
            state_after = retention * state + (1 - retention) * equilibrium_after
        else:
            state_after = (_periodic_relax(equilibrium_after,rate_after) if neuron.periodic
                           else _relax(equilibrium_after,rate_after))
        if self.activation == 'voltage':
            activation = torch.sigmoid(neuron.content(x)) * (
                state_after + F.softplus(neuron.gain) * (equilibrium_after - state_after))
        else:
            activation = F.gelu(neuron.content(x)) * (2 * torch.sigmoid(
                equilibrium_after + neuron.gain * state_after))
        z = z + self.dropout(neuron.out(neuron.dropout(activation)))
        return z, state_after


class RCLiFT(nn.Module):
    """Resistance–Capacitance-inspired Liquid-state Fusion Transformer.

    ``routing='rate'`` shares the neuronal response parameters with attention.
    Other routing settings additionally reuse neuronal activations in Q/K/V.
    State continuation carries the attention state into the FFN. Neuronal
    sharing ties the bank; depth carry retains its state across encoder blocks.
    States are reset for each historical window; one head predicts all 24 hours.
    """
    def __init__(self, strength=.25, routing='rate', dropout=.1, periodic=False, circular=False,
                 state_bridge=False, share_neurons=False, carry_depth=False, activation='gated',
                 mixed_time=False, continue_state=True):
        super().__init__()
        self.state_bridge, self.carry_depth = state_bridge, carry_depth
        self.embedding = LiquidTokenProjection(mixed_time)
        tokens = 30 if mixed_time else 24
        self.position = nn.Parameter(torch.randn(1, tokens, 64) * .02)
        self.blocks = nn.ModuleList([
            ContinuedRCBlock(strength, dropout, periodic, activation, continue_state) if state_bridge else
            SharedRCBlock(strength, routing, dropout, periodic, circular, mixed_time) for _ in range(2)])
        if share_neurons:
            for block in self.blocks[1:]:
                block.ffn = self.blocks[0].ffn
        self.norm = nn.LayerNorm(64)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(tokens * 64, 24))

    def forward(self, x, cal=None):
        observed = x[:, -24:, :1]
        mean = observed.mean(1, keepdim=True).detach()
        std = (observed.var(1, keepdim=True, unbiased=False) + 1e-4).sqrt().detach()
        normal = torch.cat(((x[:, :, :1] - mean) / std, x[:, :, 1:]), -1)
        z = self.embedding(_context(normal)) + self.position
        if self.state_bridge:
            state = None
            for block in self.blocks:
                z, state = block(z, state if self.carry_depth else None)
        else:
            for block in self.blocks:
                z = block(z)
        return self.head(self.norm(z).flatten(1)) * std[:, 0] + mean[:, 0]
