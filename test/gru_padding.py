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


seq_len = 10
hidden_states = torch.randn(size=(4, seq_len, 768))
gate = torch.randn(size=(4, seq_len, 768))


# usual forward path
log_coefficients = -F.softplus(gate)

log_z = -F.softplus(-gate)
log_tilde_hidden_states = log_g(hidden_states)
log_values = log_z + log_tilde_hidden_states

out2 = heinsen_associative_scan_log(log_coefficients, log_values)[:, :seq_len]

# padding path
# add empty initial states by adding -inf == 0 log space
dtype = gate.dtype
gate = F.pad(gate, (0, 0, 1, 0), value=torch.finfo(dtype).min)
hidden_states = F.pad(hidden_states, (0, 0, 1, 0))

log_coefficients = -F.softplus(gate)

log_z = -F.softplus(-gate)
log_tilde_hidden_states = log_g(hidden_states)
log_values = log_z + log_tilde_hidden_states

out_tmp = heinsen_associative_scan_log(log_coefficients, log_values)
out = out_tmp[:, -seq_len:]

# check if we produced the same :)
assert torch.allclose(out2, out, atol=1e-5)
