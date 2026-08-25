import torch
import matplotlib.pyplot as plt

# --- 1. Define LIF Parameters ---
beta = 0.8        # Membrane potential decay rate (from LeakyLayer initialization)
threshold = 1.0   # Spike threshold (default for snn.Leaky)
num_steps = 100    # Simulation time steps

# --- 2. Define Input Current ---
# Static current I_in = 1.0 for all time steps
static_current_in = .25
input_current = torch.ones(num_steps) * static_current_in

# --- 3. Initialize Variables ---
# U_post is the membrane potential AFTER the spike-reset at the previous step
U_post = torch.zeros(1)
U_pre_history = []  # Stores potential BEFORE reset (for plotting)
Spike_history = []  # Stores output spikes

# --- 4. Simulation Loop ---
for t in range(num_steps):
    # Step 1: Integration (Decay + Input)
    # The term U_pre[t] represents the membrane potential just before checking the threshold
    U_pre = beta * U_post + input_current[t]

    # Step 2: Spike Generation (Heaviside/Step Function)
    Spike = U_pre >= threshold

    # Step 3: Reset (Reset-by-Subtraction)
    # The term U_post[t] is the potential used for the NEXT step's calculation
    U_post = U_pre - Spike * threshold

    # Record results for plotting
    U_pre_history.append(U_pre.item())
    Spike_history.append(Spike.item())

# Convert history to tensors for plotting
U_mem_pre = torch.tensor(U_pre_history)
Spikes = torch.tensor(Spike_history)

# --- 5. Generate Graph ---
time = torch.arange(num_steps)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# Plot Membrane Potential
ax1.plot(time, U_mem_pre, color='blue', marker='o')
ax1.axhline(threshold, linestyle='--', color='red', label=f'Threshold ({threshold})')
ax1.set_ylabel('Membrane Potential ($U$)')
ax1.set_title(f'LIF Neuron Response to Static Current $I_{{in}} = {static_current_in}$ ($\\beta={beta}$)')
ax1.set_ylim(-0.1, 1.1)
ax1.legend()
ax1.grid(True, axis='y')

# Plot Spikes (Raster Plot)
ax2.plot(time, Spikes, '|', color='black', markersize=10)
ax2.set_xlabel('Time Step ($t$)')
ax2.set_ylabel('Spike ($S$)')
ax2.set_yticks([0, 1])
ax2.grid(True, axis='y')

plt.tight_layout()
plt.show()