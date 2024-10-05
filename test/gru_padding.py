import torch
from torch.functional import F

from test.gru_ops import g, log_g, heinsen_associative_scan_log


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
assert torch.allclose(out2, out)
