import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Parameters consistent with the original code ---
alpha = 128         # tau_alpha
num_steps = 1000    # Total simulation steps
dis_steps = 100     # Input display/sparseness steps
# Steps where input is applied (0, 100, 200, ...)
input_steps = range(0, num_steps, dis_steps)

batch_size = 1      # Simulating one neuron
num_inputs = 1
dtype = torch.float
device = torch.device("cpu")

# --- 2. Isolated Alpha Class ---

class Alpha(nn.Module):

    def __init__(
        self,
        tau_alpha=alpha,
    ):
        super(Alpha, self).__init__()

        self.tau_alpha = tau_alpha

    def forward(self, _input, a, I):
        a = -a/self.tau_alpha + _input
        I = (a-I)/self.tau_alpha + I
        return a, I

    def init_Alpha(self, batch_size, *args):
        # Initializes the states to zero
        I = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        a = torch.zeros((batch_size, *args), device=device, dtype=dtype)
        return a, I,


# --- 3. Instantiate and Initialize ---
alpha_neuron = Alpha().to(device)
a, I = alpha_neuron.init_Alpha(batch_size, num_inputs)

# Create the full input signal tensor
input_signal = torch.zeros(num_steps, batch_size,
                           num_inputs, device=device, dtype=dtype) 

# Define the *image data* (e.g., a pixel value) that gets injected at the input steps.
# We'll use a constant input value of 1.0 to simulate a single active pixel.
input_data_pulse = torch.ones(
    batch_size, num_inputs, device=device, dtype=dtype) * 1

# Define the zero input used when the input is not applied
inp0 = torch.zeros(batch_size, num_inputs, device=device, dtype=dtype)

# --- 4. Run Simulation with Sparse Input ---
a_rec = []
I_rec = []
input_rec = []  # To record the actual signal being fed into the alpha neuron

for step in range(num_steps):
    if step in input_steps and (step < 4000 or step > 8000):
        # Input 'x' is applied
        current_input = input_data_pulse
    else:
        # Zero input 'inp0' is applied
        current_input = inp0

    # Run the forward pass
    a, I = alpha_neuron(current_input, a, I)

    # Record the states and the input applied
    a_rec.append(a.squeeze().cpu().numpy())
    I_rec.append(I.squeeze().cpu().numpy())
    input_rec.append(current_input.squeeze().cpu().numpy())

# Convert lists to NumPy arrays for plotting
a_rec = np.array(a_rec)
I_rec = np.array(I_rec)
input_rec = np.array(input_rec)

# --- 5. Plotting ---
# plt.figure(figsize=(12, 6))

# # Plot the raw input
# plt.plot(input_rec,
#          label=f'Sparse Input $I_{{in}}$ (Applied every {dis_steps} steps)', color='gray', linestyle='--')

# # Plot the intermediate state 'a'
# plt.plot(a_rec, label='Intermediate State $a$', color='orange')

# # Plot the filtered output current 'I'
# plt.plot(I_rec, label='Filtered Output Current $I$', color='blue', linewidth=2)

# plt.title(
#     f'Alpha Neuron Simulation with Sparse Input ($\u03C4_\u03B1$ = {alpha}, $T_{{total}}$ = {num_steps})')
# plt.xlabel('Time Step')
# plt.ylabel('Value')
# plt.legend()
# plt.grid(True)
# plt.show()

   # --- Plotting ---
fig, (ax1, ax2) = plt.subplots(
    2, 1, figsize=(8, 10), sharex=True)

steps = np.linspace(0, num_steps, num_steps)
ax1.plot(steps, I_rec, label='Current I')
ax1.set_ylabel('Current (uA)')
ax1.legend()
ax1.set_title('Fig 4a: Synaptic Current Dynamics')
ax1.grid(True, alpha=0.3)

ax2.plot(steps, a_rec, label='Alpha state a', alpha=0.6)
ax2.set_ylabel('State (0-1)')
ax2.legend()
ax2.set_title('Fig 4b: Internal Memristor States')
ax2.grid(True, alpha=0.3)

# ax3.plot(steps, v_history, label='Membrane v')
# ax3.set_ylabel('Voltage (mV)')
# ax3.set_xlabel('Time (ms)')
# ax3.set_title('Fig 4c: Membrane Potential Response')
# ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
