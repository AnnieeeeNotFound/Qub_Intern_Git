"""
Step 5: AHAH training loop for the 16 memristors.
"""

import os
import time
import csv
import random

# =========================================================
# USER CONTROL PANEL
# =========================================================

DATASET = "ortho8"
# Options: ortho2, ortho4, ortho8, antiortho8

ROUTINE = "combo"
# Options: combo, always, mistakes

SYNAPSE_PAIRING = "swap-2-11"
# "default"   : synapse i = M(i+1) + M(i+9)   (Ch.11 convention)
# "swap-2-11" : sacrifice M2+M11 into ONE synapse (both are dead),
#               pair the healthy M3+M10 together -> 7/8 usable synapses

FAMP = 1.0
# SET pulse amplitude in volts

RAMP = 1.0
# RESET pulse amplitude in volts

PULSE_WIDTH_US = 50
# Pulse width in microseconds

EPOCHS = 30
# Number of training epochs

RANDOM_SEED = 0
# 0 = no randomization
# Any other number = repeatable randomized order

READ_VOLTAGE = -0.08
# Small read voltage

SHOW_PLOT_WINDOW = True


# =========================================================
# Validate settings
# =========================================================

if DATASET not in (
    "ortho2",
    "ortho4",
    "ortho8",
    "antiortho8",
):
    raise ValueError("Invalid DATASET.")

if ROUTINE not in (
    "combo",
    "always",
    "mistakes",
):
    raise ValueError("Invalid ROUTINE.")

if FAMP <= 0 or RAMP <= 0:
    raise ValueError("FAMP and RAMP must be positive.")

if PULSE_WIDTH_US <= 0:
    raise ValueError("PULSE_WIDTH_US must be positive.")

if EPOCHS <= 0:
    raise ValueError("EPOCHS must be positive.")


# =========================================================
# Synapse pairing maps (A-branch DIO bit, B-branch DIO bit)
# =========================================================
# The V2 board has one shared node + one R_s per branch, so a
# "synapse" is purely a software choice of which two DIO
# switches to open at the same time. Any A-bit + B-bit pair
# is electrically valid.

SYNAPSE_MAPS = {
    # Ch.11 convention: synapse i = M(i+1) + M(i+9)
    "default": [
        (0, 8),
        (1, 9),
        (2, 10),
        (3, 11),
        (4, 12),
        (5, 13),
        (6, 14),
        (7, 15),
    ],

    # Sacrificial pairing: M2 (stuck HRS) and M11 (erratic LRS)
    # locked into ONE synapse; healthy M3 + M10 form a new pair.
    # Expected ceiling: 7/8 trainable synapses.
    "swap-2-11": [
        (0, 8),    # syn 1: M1 + M9
        (1, 10),   # syn 2: M2 + M11   <- sacrificial (both dead)
        (2, 9),    # syn 3: M3 + M10   <- both healthy
        (3, 11),   # syn 4: M4 + M12
        (4, 12),   # syn 5: M5 + M13
        (5, 13),   # syn 6: M6 + M14
        (6, 14),   # syn 7: M7 + M15
        (7, 15),   # syn 8: M8 + M16
    ],
}

if SYNAPSE_PAIRING not in SYNAPSE_MAPS:
    raise ValueError("Invalid SYNAPSE_PAIRING.")

# Sanity check: every pairing must use each of the 16 DIO bits
# exactly once (bijective map, no double-driven memristor).
for _name, _map in SYNAPSE_MAPS.items():
    _bits = [b for pair in _map for b in pair]
    if sorted(_bits) != list(range(16)):
        raise ValueError(
            f"Synapse map '{_name}' is not a bijection "
            f"onto the 16 DIO bits."
        )

ACTIVE_MAP = SYNAPSE_MAPS[SYNAPSE_PAIRING]


# =========================================================
# WaveForms setup
# =========================================================

DWF_LIB_DIR = r"D:\Digilent\WaveForms3"

os.environ["PATH"] = (
    DWF_LIB_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DWF_LIB_DIR)


import numpy as np
import matplotlib.pyplot as plt

from pydwf import DwfLibrary
from pydwf import (
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAnalogOutIdle,
    DwfAcquisitionMode,
    DwfTriggerSource,
    DwfState,
)

from pydwf.utilities.open_dwf_device import openDwfDevice


# =========================================================
# Dataset definitions
# =========================================================

DATASETS = {
    "ortho2": [
        (True, [0, 1, 2, 3]),
        (False, [4, 5, 6, 7]),
    ],

    "ortho4": [
        (True, [0, 1]),
        (True, [2, 3]),
        (False, [4, 5]),
        (False, [6, 7]),
    ],

    "ortho8": [
        (True, [0]),
        (True, [1]),
        (True, [2]),
        (True, [3]),
        (False, [4]),
        (False, [5]),
        (False, [6]),
        (False, [7]),
    ],

    "antiortho8": [
        (False, [0]),
        (False, [1]),
        (False, [2]),
        (False, [3]),
        (True, [4]),
        (True, [5]),
        (True, [6]),
        (True, [7]),
    ],
}

PATTERNS = DATASETS[DATASET]


# =========================================================
# Fixed hardware settings
# =========================================================

RS = 20_000.0
N_SAMPLES = 8192
SAMPLE_RATE = 500_000
SETTLE_TIME_S = 0.05
PULSE_WIDTH_S = PULSE_WIDTH_US * 1e-6
K_EMA = 0.05


# =========================================================
# Hardware variables
# =========================================================

dev = None
aio = None
awg = None
scope = None
dio = None


# =========================================================
# Hardware helper functions
# =========================================================

def set_w1_voltage(voltage):
    awg.nodeFunctionSet(
        0,
        DwfAnalogOutNode.Carrier,
        DwfAnalogOutFunction.DC,
    )

    awg.nodeAmplitudeSet(
        0,
        DwfAnalogOutNode.Carrier,
        0.0,
    )

    awg.nodeOffsetSet(
        0,
        DwfAnalogOutNode.Carrier,
        voltage,
    )

    awg.runSet(0, 100.0)

    awg.idleSet(
        0,
        DwfAnalogOutIdle.Offset,
    )

    awg.triggerSourceSet(
        0,
        DwfTriggerSource.None_,
    )

    awg.nodeEnableSet(
        0,
        DwfAnalogOutNode.Carrier,
        True,
    )

    awg.configure(0, True)


def set_w1_idle():
    set_w1_voltage(0.0)


def fire_pulse(voltage):
    set_w1_voltage(voltage)

    start_time = time.perf_counter()

    while time.perf_counter() - start_time < PULSE_WIDTH_S:
        pass

    set_w1_idle()


def set_dio(value):
    dio.outputSet(value)
    dio.configure()


def pattern_bits(spikes, branch="both"):
    """
    Convert synapse indices into one-hot DIO bits using the
    ACTIVE_MAP pairing.

    A branch:
        synapse 0 -> DIO bit ACTIVE_MAP[0][0]
        synapse 7 -> DIO bit ACTIVE_MAP[7][0]

    B branch:
        synapse 0 -> DIO bit ACTIVE_MAP[0][1]
        synapse 7 -> DIO bit ACTIVE_MAP[7][1]
    """

    value = 0

    for synapse in spikes:
        a_bit, b_bit = ACTIVE_MAP[synapse]

        if branch in ("a", "both"):
            value |= 1 << a_bit

        if branch in ("b", "both"):
            value |= 1 << b_bit

    return value


def capture_means():
    scope.configure(False, True)

    while True:
        status = scope.status(True)

        if status == DwfState.Done:
            break

        time.sleep(0.001)

    number_samples = scope.statusSamplesValid()

    if number_samples <= 0:
        raise RuntimeError("No samples captured.")

    voltage_1 = float(
        np.mean(
            scope.statusData(
                0,
                number_samples,
            )
        )
    )

    voltage_2 = float(
        np.mean(
            scope.statusData(
                1,
                number_samples,
            )
        )
    )

    return voltage_1, voltage_2


# =========================================================
# Measurement instructions
# =========================================================

def instruction_flv(spikes):
    """
    Forward low-voltage read.

    Returns:
        synapse output Vy
        A-branch conductance
        B-branch conductance
    """

    set_dio(pattern_bits(spikes, "both"))
    set_w1_voltage(READ_VOLTAGE)

    time.sleep(SETTLE_TIME_S)

    voltage_1, voltage_2 = capture_means()

    set_w1_idle()

    vy = -(
        voltage_1 - voltage_2
    ) / READ_VOLTAGE

    if abs(voltage_1) > 0.002:
        conductance_a = (
            (READ_VOLTAGE - voltage_1)
            / (RS * voltage_1)
            * 1e6
        )
    else:
        conductance_a = float("nan")

    if abs(voltage_2) > 0.002:
        conductance_b = (
            (READ_VOLTAGE - voltage_2)
            / (RS * voltage_2)
            * 1e6
        )
    else:
        conductance_b = float("nan")

    return vy, conductance_a, conductance_b


def instruction_pulse(spikes, branch, voltage):
    set_dio(pattern_bits(spikes, branch))
    fire_pulse(voltage)


def execute_learning(state, vy):
    """
    Decide which SET/RESET operations to apply.
    """

    if ROUTINE == "combo":
        if state:
            return ["fa", "rb"]

        if vy > 0:
            return ["ra", "fb"]

        return []

    if ROUTINE == "always":
        if state:
            return ["fa", "rb"]

        return ["ra", "fb"]

    if ROUTINE == "mistakes":
        if vy < 0 and state:
            return ["fa", "rb"]

        if vy > 0 and not state:
            return ["ra", "fb"]

        return []

    return []


def apply_learning(spikes, instructions):
    for instruction in instructions:

        if instruction == "fa":
            instruction_pulse(
                spikes,
                "a",
                -FAMP,
            )

        elif instruction == "fb":
            instruction_pulse(
                spikes,
                "b",
                -FAMP,
            )

        elif instruction == "ra":
            instruction_pulse(
                spikes,
                "a",
                +RAMP,
            )

        elif instruction == "rb":
            instruction_pulse(
                spikes,
                "b",
                +RAMP,
            )


def read_all_synapses():
    weights = []

    for synapse in range(8):
        vy, _, _ = instruction_flv([synapse])
        weights.append(vy)

    return weights


def read_all_conductances():
    conductances = []

    for dio_bit in range(16):
        set_dio(1 << dio_bit)
        set_w1_voltage(READ_VOLTAGE)

        time.sleep(SETTLE_TIME_S)

        voltage_1, voltage_2 = capture_means()

        set_w1_idle()

        if dio_bit < 8:
            node_voltage = voltage_1
        else:
            node_voltage = voltage_2

        if abs(node_voltage) > 0.002:
            conductance = (
                (READ_VOLTAGE - node_voltage)
                / (RS * node_voltage)
                * 1e6
            )
        else:
            conductance = float("nan")

        conductances.append(conductance)

    set_dio(0x0000)

    return conductances


# =========================================================
# Main experiment
# =========================================================

try:
    dwf = DwfLibrary()
    dev = openDwfDevice(dwf)

    print("Device opened.")
    print(f"Dataset: {DATASET}")
    print(f"Routine: {ROUTINE}")
    print(f"Epochs: {EPOCHS}")
    print(f"FAMP: {FAMP} V")
    print(f"RAMP: {RAMP} V")
    print(f"Pulse width: {PULSE_WIDTH_US} us")
    print(f"Synapse pairing: {SYNAPSE_PAIRING}")

    print("Pairing map:")
    for synapse, (a_bit, b_bit) in enumerate(ACTIVE_MAP):
        print(
            f"  Synapse {synapse + 1}: "
            f"M{a_bit + 1} + M{b_bit + 1}"
        )

    # -----------------------------------------------------
    # Power supplies
    # -----------------------------------------------------

    aio = dev.analogIO
    aio.reset()

    aio.channelNodeSet(0, 1, +5.0)
    aio.channelNodeSet(0, 0, 1.0)

    aio.channelNodeSet(1, 1, -5.0)
    aio.channelNodeSet(1, 0, 1.0)

    aio.enableSet(True)
    aio.configure()

    time.sleep(0.5)
    aio.status()

    print(
        f"Supplies: "
        f"V+ = {aio.channelNodeStatus(0, 1):+.3f} V, "
        f"V- = {aio.channelNodeStatus(1, 1):+.3f} V"
    )

    # -----------------------------------------------------
    # AWG
    # -----------------------------------------------------

    awg = dev.analogOut
    awg.reset(0)
    set_w1_idle()

    # -----------------------------------------------------
    # Oscilloscope
    # -----------------------------------------------------

    scope = dev.analogIn
    scope.reset()
    scope.frequencySet(SAMPLE_RATE)

    scope.channelEnableSet(0, True)
    scope.channelEnableSet(1, True)

    try:
        scope.channelRangeSet(0, 1.0)
        scope.channelRangeSet(1, 1.0)
    except Exception:
        scope.channelRangeSet(0, 5.0)
        scope.channelRangeSet(1, 5.0)

    scope.bufferSizeSet(N_SAMPLES)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)
    scope.triggerSourceSet(DwfTriggerSource.None_)

    # -----------------------------------------------------
    # Digital I/O
    # -----------------------------------------------------

    dio = dev.digitalIO
    dio.reset()
    dio.outputEnableSet(0xFFFF)
    set_dio(0x0000)

    # -----------------------------------------------------
    # Initial conductance
    # -----------------------------------------------------

    print("\nInitial conductances:")

    initial_conductances = read_all_conductances()

    for index, conductance in enumerate(initial_conductances):
        print(
            f"M{index + 1:02d}: "
            f"{conductance:.3f} uS"
        )

    # -----------------------------------------------------
    # Initial synapse weights
    # -----------------------------------------------------

    initial_weights = read_all_synapses()
    weight_history = [initial_weights]

    accuracy_ema = 0.0
    accuracy_history = []

    rng = None

    if RANDOM_SEED != 0:
        rng = random.Random(RANDOM_SEED)

    # -----------------------------------------------------
    # Training loop
    # -----------------------------------------------------

    print("\nTraining started.")

    start_time = time.time()

    for epoch in range(EPOCHS):

        order = list(range(len(PATTERNS)))

        if rng is not None:
            rng.shuffle(order)

        for pattern_index in order:

            target_state, spikes = PATTERNS[pattern_index]

            vy, _, _ = instruction_flv(spikes)

            predicted_state = vy > 0
            correct = predicted_state == target_state

            if correct:
                accuracy_ema = (
                    (1.0 - K_EMA)
                    * accuracy_ema
                    + K_EMA
                )
            else:
                accuracy_ema = (
                    (1.0 - K_EMA)
                    * accuracy_ema
                )

            accuracy_history.append(accuracy_ema)

            instructions = execute_learning(
                target_state,
                vy,
            )

            apply_learning(
                spikes,
                instructions,
            )

        current_weights = read_all_synapses()
        weight_history.append(current_weights)

        print(
            f"Epoch {epoch + 1:03d}/{EPOCHS} "
            f"accuracy EMA = {accuracy_ema:.3f}"
        )

    elapsed = time.time() - start_time

    print(
        f"\nTraining complete in "
        f"{elapsed:.1f} seconds."
    )
    # =========================================================
    # Final accuracy
    # =========================================================

    final_accuracy = accuracy_ema * 100.0

    print(
        f"\nFINAL TRAINING ACCURACY: "
        f"{final_accuracy:.2f}%"
    )
    # -----------------------------------------------------
    # Final conductance
    # -----------------------------------------------------

    final_conductances = read_all_conductances()

    print("\nFinal conductances:")

    for index, conductance in enumerate(final_conductances):
        initial = initial_conductances[index]

        print(
            f"M{index + 1:02d}: "
            f"{conductance:.3f} uS "
            f"(initial {initial:.3f} uS)"
        )

    # -----------------------------------------------------
    # Plot results
    # -----------------------------------------------------

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(11, 11),
    )

    # Accuracy
    axes[0].plot(
        np.arange(len(accuracy_history)),
        accuracy_history,
        color="#2471A3",
    )

    axes[0].set_xlabel(
        "Pattern presentation"
    )

    axes[0].set_ylabel(
        "Accuracy EMA"
    )

    axes[0].set_ylim(
        -0.05,
        1.05,
    )

    axes[0].set_title(
        f"AHAH training: {DATASET}, {ROUTINE}, "
        f"pairing={SYNAPSE_PAIRING}"
    )

    axes[0].grid(alpha=0.3)

    # Synapse weights
    epoch_axis = np.arange(
        len(weight_history)
    )

    for synapse in range(8):
        a_bit, b_bit = ACTIVE_MAP[synapse]

        axes[1].plot(
            epoch_axis,
            [
                weights[synapse]
                for weights in weight_history
            ],
            "o-",
            markersize=3,
            label=(
                f"Syn{synapse + 1} "
                f"(M{a_bit + 1}+M{b_bit + 1})"
            ),
        )

    axes[1].axhline(
        0,
        color="black",
        linewidth=0.8,
    )

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Synapse output Vy")
    axes[1].set_title("Synapse weight evolution")
    axes[1].legend(ncol=4, fontsize=8)
    axes[1].grid(alpha=0.3)

    # Conductance comparison
    memristor_labels = [
        f"M{i + 1}"
        for i in range(16)
    ]

    x = np.arange(16)
    width = 0.38

    axes[2].bar(
        x - width / 2,
        initial_conductances,
        width,
        label="Initial",
        color="#95A5A6",
    )

    axes[2].bar(
        x + width / 2,
        final_conductances,
        width,
        label="Final",
        color="#C0392B",
    )

    axes[2].set_xticks(x)
    axes[2].set_xticklabels(memristor_labels)
    axes[2].set_xlabel("Memristor")
    axes[2].set_ylabel("Conductance (uS)")
    axes[2].set_title("Initial and final conductance")
    axes[2].legend()
    axes[2].grid(alpha=0.3, axis="y")

    figure.tight_layout()

    output_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    plot_path = os.path.join(
        output_directory,
        f"ahah_{DATASET}_{ROUTINE}_"
        f"{SYNAPSE_PAIRING}.png",
    )

    figure.savefig(
        plot_path,
        dpi=120,
    )

    print("\nPlot saved to:")
    print(plot_path)

    if SHOW_PLOT_WINDOW:
        plt.show()
    else:
        plt.close(figure)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = os.path.join(
        output_directory,
        time.strftime(
            f"ahah_{DATASET}_{ROUTINE}_"
            "%Y%m%d_%H%M%S.csv"
        ),
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "memristor",
                "initial_conductance_uS",
                "final_conductance_uS",
            ]
        )

        for index in range(16):
            writer.writerow(
                [
                    f"M{index + 1}",
                    initial_conductances[index],
                    final_conductances[index],
                ]
            )

    print("CSV saved to:")
    print(csv_path)

finally:
    # Safety shutdown

    try:
        if awg is not None:
            set_w1_idle()
            awg.configure(0, False)
    except Exception:
        pass

    try:
        if dio is not None:
            dio.outputSet(0x0000)
            dio.outputEnableSet(0x0000)
            dio.configure()
    except Exception:
        pass

    try:
        if aio is not None:
            aio.channelNodeSet(0, 0, 0.0)
            aio.channelNodeSet(1, 0, 0.0)
            aio.enableSet(False)
            aio.configure()
    except Exception:
        pass

    try:
        if dev is not None:
            dev.close()
    except Exception:
        pass

    print("\nDevice closed.")