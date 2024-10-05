import torch
from torch.functional import F


def g(hidden_states):
    return torch.where(hidden_states >= 0, hidden_states + 0.5, hidden_states.sigmoid())


def log_g(hidden_states):
    return torch.where(hidden_states >= 0, (F.relu(hidden_states) + 0.5).log(), -F.softplus(-hidden_states))


def heinsen_associative_scan_log(log_coefficients, log_values):
    a_star = log_coefficients.cumsum(dim=1)
    log_h0_plus_b_star = (log_values - a_star).logcumsumexp(dim=1)
    log_h = a_star + log_h0_plus_b_star
    return log_h.exp()
