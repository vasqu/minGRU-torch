import torch
from torch.functional import F

from test.gru_ops import g, log_g, heinsen_associative_scan_log


seq_len = 10
hidden_states = torch.randn(size=(4, seq_len, 768))
gate = torch.randn(size=(4, seq_len, 768))

hidden_states_partial = hidden_states[:, :(seq_len-1), :]
hidden_states_last = hidden_states[:, -1, :].unsqueeze(1)
gate_partial = gate[:, :(seq_len-1), :]
gate_last = gate[:, -1, :].unsqueeze(1)


# total path
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


# caching path
dtype = gate.dtype
gate_partial = F.pad(gate_partial, (0, 0, 1, 0), value=torch.finfo(dtype).min)
hidden_states_partial = F.pad(hidden_states_partial, (0, 0, 1, 0))

log_coefficients = -F.softplus(gate_partial)

log_z = -F.softplus(-gate_partial)
log_tilde_hidden_states = log_g(hidden_states_partial)
log_values = log_z + log_tilde_hidden_states

out_tmp = heinsen_associative_scan_log(log_coefficients, log_values)

last_state = out_tmp[:, -1, :].unsqueeze(1)
out2 = out_tmp[:, -(seq_len-1):]

assert torch.allclose(out[:, :(seq_len-1), :], out2)

# sequential forward
hidden_states_last = g(hidden_states_last)
gate_last = gate_last.sigmoid()

out3 = torch.lerp(last_state, hidden_states_last, gate_last)

assert torch.allclose(out, torch.cat((out2, out3), dim=1))
