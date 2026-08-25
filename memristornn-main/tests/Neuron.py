import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- Parameters (Kept identical to your file) ---
batch_size = 1
num_steps = 100
dis_steps = 20
dtype = torch.float
device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# Simulation specific
scale1 = 128
in_features = 1
out_features = 1


class MIF(nn.Linear):
    def __init__(self, in_features, out_features, layer_index, lr=0.01, batch_size=1, bias=False,
                # R_on=1000,
                # R_off=100000,
                # v_on=110 * 10**(-1),
                # v_off=5 * 10**(-1),
                # tau=1 * 10**(-3),
                # E1=0,
                # E2=50 * 10**(-3),
                # C=100 * 10**(-6),
                # k_th=0.6 * 25):
                 R_on=1000,
                 R_off=100000,
                 v_on=110,
                 v_off=5,
                 tau=100,
                 E1=0,
                 E2=50,
                 C=100 * 10**(-6),
                 k_th=0.6 * 25,):
        super().__init__(in_features, out_features, bias=bias)
        self.in_features, self.out_features, self.layer_index, self.batch_size = in_features, out_features, layer_index, batch_size
        self.threshold, self.R_on, self.R_off, self.v_on, self.v_off, self.tau = 1, R_on, R_off, v_on, v_off, tau
        self.E1, self.E2, self.C, self.k_th = E1, E2, C, k_th

        # State initialization
        self.x1_pos = torch.ones(
            (self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.x2_pos = torch.ones(
            (self.batch_size, self.out_features), device=device, dtype=dtype) * 0.0238
        self.G1_pos = self.x1_pos / self.R_on + (1-self.x1_pos)/self.R_off
        self.G2_pos = self.x2_pos / self.R_on + (1-self.x2_pos)/self.R_off
        self.v_pos = torch.zeros(
            (self.batch_size, self.out_features), device=device, dtype=dtype)

        # Compiled step
        self.compiled_mem_step = torch.compile(self.mem_step, fullgraph=True)

    def mem_step(self, _input, w, b, x1, x2, G1, G2, v):
        _input_mm = F.linear(_input, w, b)
        v = (v + (_input_mm + G1 * self.E1 + G2 * self.E2) /
             self.C) / (1 + ((G1 + G2) / self.C))
        v.clamp_(0.0, 1000.0)
        inv_tau = 1.0 / self.tau
        x1 = inv_tau * ((1 - x1) * torch.sigmoid(((v-self.E1)-self.v_on)/self.k_th) -
                        x1 * torch.sigmoid((self.v_off-(v-self.E1))/self.k_th)) + x1
        x2 = inv_tau * ((1 - x2) * torch.sigmoid(((v-self.E2)-self.v_on)/self.k_th) -
                        x2 * torch.sigmoid((self.v_off-(v-self.E2))/self.k_th)) + x2
        x1.clamp_(0.0, 1.0)
        x2.clamp_(0.0, 1.0)
        G1 = (x1/self.R_on + (1 - x1)/self.R_off)
        G2 = (x2/self.R_on + (1 - x2)/self.R_off)
        return x1, x2, G1, G2, v


class Alpha(nn.Module):
    def __init__(self, feature_size, batch_size, num_steps, device, tau_alpha=4.0):
        super(Alpha, self).__init__()
        self.feature_size, self.batch_size, self.num_steps, self.device, self.tau_alpha = feature_size, batch_size, num_steps, device, tau_alpha

    def forward_alpha(self, _input, a, I):
        a = -a/self.tau_alpha + _input
        I = (a-I)/self.tau_alpha + I
        return a, I

    def forward(self, _input_x_pos, dis_steps):
        inp0_pos = torch.zeros(
            (self.batch_size, self.feature_size), device=device)
        a0_pos, I0_pos = torch.zeros_like(inp0_pos), torch.zeros_like(inp0_pos)
        I0_pos_history = torch.empty(
            self.num_steps, self.batch_size, self.feature_size, device=device)

        for step in range(self.num_steps):
            if step % dis_steps == 0:
                a0_pos, I0_pos = self.forward_alpha(_input_x_pos, a0_pos, I0_pos)
            else:
                a0_pos, I0_pos = self.forward_alpha(inp0_pos, a0_pos, I0_pos)
            I0_pos_history[step] = I0_pos
        return I0_pos_history

# --- Simulation Execution ---


# 1. Initialize Components
alpha_filter = Alpha(in_features, batch_size, num_steps, device)
neuron = MIF(in_features, out_features, 0, batch_size=batch_size).to(device)
neuron2 = MIF(in_features, out_features, 0, batch_size=batch_size).to(device)

# 2. Generate Input (Constant stimulus)
_input = torch.ones((batch_size, in_features), device=device) * 1
I_history = alpha_filter(_input, dis_steps) / scale1

# 3. Run Neuron Simulation and record states
v_trace, x1_trace, x2_trace = [], [], []

# Detached state variables for manual loop
x1, x2 = neuron.x1_pos.clone(), neuron.x2_pos.clone()
G1, G2 = neuron.G1_pos.clone(), neuron.G2_pos.clone()
v = neuron.v_pos.clone()

print("Simulating single neuron dynamics...")
for step in range(num_steps):
    x1, x2, G1, G2, v = neuron.mem_step(
        I_history[step], torch.tensor([[0.1]]).to(device), neuron.bias, x1, x2, G1, G2, v) #neuron.weight

    # Store results (squeezing to get scalar values for the single neuron)
    v_trace.append(v.item())
    x1_trace.append(x1.item())
    x2_trace.append(x2.item())
print(sum(v_trace))

# v_trace2, x1_trace2, x2_trace2 = [], [], []

# # Detached state variables for manual loop
# x1, x2 = neuron2.x1_pos.clone(), neuron2.x2_pos.clone()
# G1, G2 = neuron2.G1_pos.clone(), neuron2.G2_pos.clone()
# v = neuron2.v_pos.clone()

# print("Simulating single neuron dynamics...")
# for step in range(num_steps):
#     x1, x2, G1, G2, v = neuron.mem_step(
#         v_trace[step], neuron.weight, neuron.bias, x1, x2, G1, G2, v)

#     # Store results (squeezing to get scalar values for the single neuron)
#     v_trace2.append(v.item())
#     x1_trace2.append(x1.item())
#     x2_trace2.append(x2.item())

# --- Plotting ---
plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(12, 6))
plt.plot(I_history[:, 0, 0].cpu(), color='green', label='Alpha Output (Feature 0)')
plt.title('Alpha Filter Output (Temporal Signal)')
plt.xlabel('Time Step')
plt.ylabel('Signal Intensity')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

fig, ax1 = plt.subplots(figsize=(12, 6))
# Plot Voltage
ax1.set_xlabel('Time Step')
ax1.set_ylabel('Membrane Potential (V)', color='tab:red')
ax1.ticklabel_format(useOffset=False, style='plain', axis='y')
ax1.plot(v_trace, color='tab:red', linewidth=2, label='Voltage (V)')
ax1.tick_params(axis='y', labelcolor='tab:red')
ax1.grid(True, alpha=0.3)

# Plot internal states x1, x2 on second axis
ax2 = ax1.twinx()
ax2.set_ylabel('State Variables (x1, x2)', color='tab:blue')
ax2.plot(x1_trace, color='tab:blue', linestyle='--', label='State x1')
ax2.plot(x2_trace, color='tab:cyan', linestyle=':', label='State x2')
ax2.tick_params(axis='y', labelcolor='tab:blue')

plt.title('MIF Neuron Dynamics: Single Neuron Simulation')
fig.tight_layout()
plt.show()
