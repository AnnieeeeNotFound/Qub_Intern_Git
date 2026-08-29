"""
Step 8: OR sanity-check (linearly separable) on the memristor chip.

Downscaled hardware replication of the tutor's MIF + Forward-Forward
network (Main.ipynb): greedy layer-wise LOCAL learning, no backprop.

Architecture (7 synapses = 14 healthy memristors, M2/M11 quarantined):

    inputs (spikes)          hidden layer             output
    x1 x2 x3 -->  h1 = theta(w11*x1 + w12*x2 + w13*x3)
    x1 x2     -->  h2 = theta(w21*x1 + w22*x2)
                                |
                    y  = theta(v1*h1 + v2*h2)

Each "weight" is a differential memristor synapse (A-B conductance
difference), exactly like 05. Neurons are time-multiplexed: only one
neuron's synapse switches are closed at any moment, so each branch
node sums only that neuron's active inputs.

Training is two greedy phases, mirroring the tutor's train_next_layer:
    Phase 1: hidden neurons trained with the AHAH combo rule on
             software-assigned local targets (pattern detectors).
    Phase 2: hidden layer FROZEN; output trained on the hidden
             activations with the same combo rule on the true label.

A "reset-all" weight initialization parks every used memristor low
before training, so all synapse weights start near zero.
"""

import os
import sys as _sys
import time
import csv
import argparse as _argparse

# =========================================================
# USER CONTROL PANEL
# =========================================================

DATASET = "or2"
# OR of two inputs (linearly separable sanity check). Default for this
# dedicated script; override with `--dataset xor2|parity3` if needed.
# Options:
#   xor2    : XOR of two inputs. NOT linearly separable:
#             single-layer ceiling 75%, two-layer target 100%.
#   parity3 : parity of three inputs. Not fully representable by a
#             2-hidden-unit net: output is a 2-input linear threshold and
#             3-parity is not a 2-feature linear function, so the 2-2-1
#             ceiling is 50% (4/8). Hidden layer still learns its detectors;
#             this run ALSO exercises h1's x2/x3 synapses (M5+M13, M8+M16)
#             to validate those previously-idle devices.
#   or2     : OR of two inputs (linearly separable sanity check).

MODE = "two-layer"
# "two-layer"   : hidden (h1, h2) -> output, greedy local training
# "single-layer": h1 alone as a direct classifier (baseline)

TARGET_MODE = "auto"
# "auto" : h1 / h2 learn single-pattern detectors picked from the
#          class-1 patterns (locally separable sub-problems)
# "label": both hidden neurons trained directly on the label

# =========================================================
# Command-line override + auto-logging
# =========================================================
# Lets you run e.g. `python 07_ff_two_layer.py --dataset xor2`
# without editing the control panel above.
_cli = _argparse.ArgumentParser(
    description="Two-layer feed-forward memristor network",
    allow_abbrev=False,
)
_cli.add_argument(
    "--dataset",
    choices=("xor2", "parity3", "or2"),
    default=None,
    help="Override the DATASET control-panel setting.",
)
_cli_args, _ = _cli.parse_known_args()

if _cli_args.dataset is not None:
    DATASET = _cli_args.dataset

# Auto-logging: mirror all stdout into run_logs/<name>_<timestamp>.txt so
# every hardware run leaves a self-contained, timestamped log next to the
# code (works whether run from VS Code F5 or the command line).
_LOG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "run_logs",
)
os.makedirs(_LOG_DIR, exist_ok=True)

_log_name = (
    f"or_{DATASET}_{MODE}_{TARGET_MODE}_"
    f"{time.strftime('%Y%m%d_%H%M%S')}.txt"
)
_log_path = os.path.join(_LOG_DIR, _log_name)
_log_file = open(_log_path, "w", buffering=1)


class _Tee:
    """Write every line to both the real console and the run log."""

    def write(self, data):
        _sys.__stdout__.write(data)
        _log_file.write(data)

    def flush(self):
        _sys.__stdout__.flush()
        _log_file.flush()


_sys.stdout = _Tee()
print(f"[log] writing run log to: {_log_path}")

INIT_RESET_PULSES = 3
# Weight initialization: RESET every USED memristor this many times
# before training, so all synapse weights start near zero.
# (M2 and M11 are never touched - they are not in the topology.)

DETECTOR_BOOST = 2
# Hidden-layer local targets are SINGLE-PATTERN detectors (e.g. h2 fires
# only on the pattern (0,1)). That detector pattern shares its active input
# with a class-0 pattern ((1,1) also activates input2), so every epoch the
# detector gets one excitation (on its own pattern) AND one suppression (on
# the conflicting class-0 pattern) -> the net conductance move washes out and
# the neuron can land marginally on either side of zero. Boosting the
# excitation pulse on the detector pattern breaks the symmetry so the
# detector input converges decisively positive. This is still pure local
# learning (no global error, no backprop) - just extra weight on the target.

PHASE1_EPOCHS = 30
# Hidden-layer training epochs

PHASE2_EPOCHS = 30
# Output-layer training epochs (hidden frozen)

FAMP = 1.0
# SET pulse amplitude in volts

RAMP = 1.0
# RESET pulse amplitude in volts

PULSE_WIDTH_US = 50
# Pulse width in microseconds

READ_VOLTAGE = -0.08
# Small read voltage

SHOW_PLOT_WINDOW = True


# =========================================================
# Validate settings
# =========================================================

if DATASET not in ("xor2", "parity3", "or2"):
    raise ValueError("Invalid DATASET.")

if MODE not in ("two-layer", "single-layer"):
    raise ValueError("Invalid MODE.")

if TARGET_MODE not in ("auto", "label"):
    raise ValueError("Invalid TARGET_MODE.")

if FAMP <= 0 or RAMP <= 0:
    raise ValueError("FAMP and RAMP must be positive.")

if PHASE1_EPOCHS <= 0 or PHASE2_EPOCHS <= 0:
    raise ValueError("Epoch counts must be positive.")

if INIT_RESET_PULSES < 0:
    raise ValueError("INIT_RESET_PULSES must be >= 0.")

if DETECTOR_BOOST < 1:
    raise ValueError("DETECTOR_BOOST must be >= 1.")


# =========================================================
# Topology: (neuron, input_index, a_bit, b_bit)
# DIO bit m drives memristor M(m+1).
# M1-M8 = A branch (scope 1+), M9-M16 = B branch (scope 2+).
# M2 (bit 1) and M11 (bit 10) are dead and deliberately unused.
# =========================================================

SYNAPSES = [
    ("h1", 1, 0, 8),    # S1: M1  + M9
    ("h1", 2, 4, 12),   # S2: M5  + M13  (unused in xor2)
    ("h1", 3, 7, 15),   # S3: M8  + M16  (unused in xor2; M16 stuck-high)
    ("h2", 1, 5, 13),   # S4: M6  + M14  (reliable positive pair)
    ("h2", 2, 2, 9),    # S5: M3  + M10  (strong positive pair)
    ("out", 1, 6, 14),  # S6: M7  + M15
    ("out", 2, 3, 11),  # S7: M4  + M12  (clean positive pair)
]

INPUT_BITS = {
    "h1": (1, 2, 3),
    "h2": (1, 2),
    "out": (1, 2),
}

HIDDEN_NEURONS = ["h1", "h2"] if MODE == "two-layer" else ["h1"]

# Neuron -> list of (input_index, a_bit, b_bit)
SYNAPSES_OF = {
    neuron: [
        (input_index, a_bit, b_bit)
        for (n, input_index, a_bit, b_bit) in SYNAPSES
        if n == neuron
    ]
    for neuron in ("h1", "h2", "out")
}


# =========================================================
# Datasets: list of (label, spikes) where spikes are the ACTIVE
# input indices (1-based). An empty tuple = all inputs silent.
# =========================================================

DATASETS = {
    # x1 XOR x2
    "xor2": [
        (False, ()),
        (True, (1,)),
        (True, (2,)),
        (False, (1, 2)),
    ],

    # x1 XOR x2 XOR x3 (label = odd popcount)
    # Order chosen so the auto detectors pick:
    #   h1 -> "001" (input3 only)  -> exercises/validates M8+M16 (h1 x3)
    #   h2 -> "010" (input2 only)  -> exercises/validates M3+M10 (h2 x2)
    # h1's x1 (M1+M9) and x2 (M5+M13) synapses still get pulses on the
    # other patterns, so all three h1 inputs are exercised.
    "parity3": [
        (False, ()),
        (True, (3,)),
        (True, (2,)),
        (True, (1,)),
        (False, (1, 2)),
        (False, (1, 3)),
        (False, (2, 3)),
        (True, (1, 2, 3)),
    ],

    # x1 OR x2 (sanity: linearly separable)
    "or2": [
        (False, ()),
        (True, (1,)),
        (True, (2,)),
        (True, (1, 2)),
    ],
}

PATTERNS = DATASETS[DATASET]

N_INPUTS = max(
    len(spikes) if spikes else 0
    for _, spikes in PATTERNS
)
N_INPUTS = min(N_INPUTS, 3)

DATASET_NOTES = {
    "xor2": (
        "XOR is not linearly separable: single-layer ceiling 75%, "
        "two-layer target 100%."
    ),
    "parity3": (
        "3-input parity. NOT fully representable by a 2-hidden-unit net: "
        "the output is a 2-input linear threshold and 3-parity is not a "
        "2-feature linear function, so the 2-2-1 ceiling is 50% (4/8). "
        "The hidden layer still learns its detectors; this run also "
        "exercises h1's x2/x3 synapses (M5+M13, M8+M16) to validate them."
    ),
    "or2": (
        "OR is linearly separable: both modes should reach 100%."
    ),
}


# =========================================================
# Hidden-layer local targets
# =========================================================
# "auto" assigns each hidden neuron a single-pattern detector
# target: fire on exactly one class-1 pattern. Such targets are
# always separable through the origin, so the combo rule can
# realize them. The output layer then learns which detectors
# imply class 1.

positive_indices = [
    index
    for index, (label, _) in enumerate(PATTERNS)
    if label
]

DETECTORS = {}

if MODE == "two-layer" and TARGET_MODE == "auto":
    DETECTORS = {
        "h1": positive_indices[0] if len(positive_indices) >= 1 else None,
        "h2": positive_indices[1] if len(positive_indices) >= 2 else None,
    }


def hidden_target(neuron, pattern_index):
    """Local training target for a hidden neuron."""

    if MODE == "single-layer" or TARGET_MODE == "label":
        return PATTERNS[pattern_index][0]

    detector = DETECTORS.get(neuron)

    return detector is not None and pattern_index == detector


def pattern_string(spikes):
    """Render a pattern like '1,0' or '1,0,1'."""

    return ",".join(
        "1" if index in spikes else "0"
        for index in range(1, N_INPUTS + 1)
    )


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
# Fixed hardware settings
# =========================================================

RS = 20_000.0
N_SAMPLES = 8192
SAMPLE_RATE = 500_000
SETTLE_TIME_S = 0.05
PULSE_WIDTH_S = PULSE_WIDTH_US * 1e-6
K_EMA = 0.10


# =========================================================
# Hardware variables
# =========================================================

dev = None
aio = None
awg = None
scope = None
dio = None


# =========================================================
# Hardware helper functions (same proven pattern as 04/05)
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
            scope.statusData(0, number_samples)
        )
    )

    voltage_2 = float(
        np.mean(
            scope.statusData(1, number_samples)
        )
    )

    return voltage_1, voltage_2


# =========================================================
# Neuron-level instructions
# =========================================================

def dio_word(neuron, active_inputs, branch):
    """
    One-hot DIO word for one neuron's ACTIVE input synapses.

    branch "a": A-side memristor of each active synapse.
    branch "b": B-side memristor.
    branch "both": both sides (used for reads).
    """

    value = 0

    for input_index, a_bit, b_bit in SYNAPSES_OF[neuron]:

        if input_index in active_inputs:
            if branch in ("a", "both"):
                value |= 1 << a_bit

            if branch in ("b", "both"):
                value |= 1 << b_bit

    return value


def read_neuron(neuron, active_inputs):
    """
    Differential read of one neuron's weighted sum.

    Only this neuron's active-input synapse switches are closed,
    so the branch nodes see exactly the active input synapses.
    Returns Vy = -(V1 - V2) / V_READ (same convention as 05).
    An all-silent input gives h = 0 by definition (no switches).
    """

    if not active_inputs:
        return 0.0

    set_dio(dio_word(neuron, active_inputs, "both"))
    set_w1_voltage(READ_VOLTAGE)

    time.sleep(SETTLE_TIME_S)

    voltage_1, voltage_2 = capture_means()

    set_w1_idle()
    set_dio(0x0000)

    return -(voltage_1 - voltage_2) / READ_VOLTAGE


def pulse_neuron(neuron, active_inputs, branch, voltage):
    """Apply one SET/RESET pulse to a neuron's active synapses."""

    if not active_inputs:
        return

    set_dio(dio_word(neuron, active_inputs, branch))
    fire_pulse(voltage)
    set_dio(0x0000)


def apply_neuron_learning(neuron, active_inputs, target, h):
    """
    Combo (Ideal Supervised) rule, scoped to one neuron:
        target True            -> FA + RB  (raise the response)
        target False and h > 0 -> RA + FB  (suppress the response)
    """

    if not active_inputs:
        return

    if target:
        pulse_neuron(neuron, active_inputs, "a", -FAMP)
        pulse_neuron(neuron, active_inputs, "b", +RAMP)

    elif h > 0:
        pulse_neuron(neuron, active_inputs, "a", +RAMP)
        pulse_neuron(neuron, active_inputs, "b", -FAMP)


def read_all_conductances():
    """Read all 16 memristors one by one (as in 03/05)."""

    conductances = []

    for dio_bit in range(16):
        set_dio(1 << dio_bit)
        set_w1_voltage(READ_VOLTAGE)

        time.sleep(SETTLE_TIME_S)

        voltage_1, voltage_2 = capture_means()

        set_w1_idle()

        node_voltage = (
            voltage_1 if dio_bit < 8 else voltage_2
        )

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


def reset_all_weights():
    """
    Weight initialization: RESET (+V) every USED memristor a few
    times so every synapse weight (G_A - G_B) starts near zero.
    Bits 1 (M2) and 10 (M11) are never driven.
    """

    used_bits = [
        bit
        for _, _, a_bit, b_bit in SYNAPSES
        for bit in (a_bit, b_bit)
    ]

    for _ in range(INIT_RESET_PULSES):
        for bit in used_bits:
            set_dio(1 << bit)
            fire_pulse(+RAMP)

    set_dio(0x0000)


# =========================================================
# Forward pass (hardware)
# =========================================================

def active_inputs_of(neuron, spikes):
    return [
        index
        for index in INPUT_BITS[neuron]
        if index in spikes
    ]


def hardware_forward(spikes):
    """
    Full forward pass on hardware.

    Returns (h1, h2, y, predicted_label).
    h2 / y are None in single-layer mode.
    """

    active_1 = active_inputs_of("h1", spikes)
    h1 = read_neuron("h1", active_1)

    if MODE == "single-layer":
        return h1, None, None, h1 > 0

    spike_1 = h1 > 0

    active_2 = active_inputs_of("h2", spikes)
    h2 = read_neuron("h2", active_2)
    spike_2 = h2 > 0

    output_active = [
        index + 1
        for index, fired in enumerate((spike_1, spike_2))
        if fired
    ]

    y = read_neuron("out", output_active)

    predicted = (y > 0) if output_active else False

    return h1, h2, y, predicted


# =========================================================
# Main experiment
# =========================================================

try:
    dwf = DwfLibrary()
    dev = openDwfDevice(dwf)

    print("Device opened.")
    print(f"Dataset: {DATASET}  ({DATASET_NOTES[DATASET]})")
    print(f"Mode: {MODE}")
    print(f"Target mode: {TARGET_MODE}")
    print(f"Phase 1 epochs: {PHASE1_EPOCHS}")
    print(f"Phase 2 epochs: {PHASE2_EPOCHS}")
    print(f"FAMP: {FAMP} V, RAMP: {RAMP} V")
    print(f"Init RESET pulses: {INIT_RESET_PULSES}")

    print("\nTopology:")
    for neuron in ("h1", "h2", "out"):
        for input_index, a_bit, b_bit in SYNAPSES_OF[neuron]:
            print(
                f"  {neuron} <- x{input_index}: "
                f"M{a_bit + 1} + M{b_bit + 1}"
            )

    if DETECTORS:
        for neuron, detector in DETECTORS.items():
            if detector is None:
                print(
                    f"  {neuron} detector: none "
                    f"(suppressed neuron)"
                )
            else:
                label, spikes = PATTERNS[detector]
                print(
                    f"  {neuron} detector target: "
                    f"pattern '{pattern_string(spikes)}' "
                    f"(class 1)"
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
        f"\nSupplies: "
        f"V+ = {aio.channelNodeStatus(0, 1):+.3f} V, "
        f"V- = {aio.channelNodeStatus(1, 1):+.3f} V"
    )

    # -----------------------------------------------------
    # AWG / scope / DIO
    # -----------------------------------------------------

    awg = dev.analogOut
    awg.reset(0)
    set_w1_idle()

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

    dio = dev.digitalIO
    dio.reset()
    dio.outputEnableSet(0xFFFF)
    set_dio(0x0000)

    # -----------------------------------------------------
    # Initial conductances
    # -----------------------------------------------------

    print("\nInitial conductances (before weight init):")

    initial_conductances = read_all_conductances()

    for index, conductance in enumerate(initial_conductances):
        print(f"M{index + 1:02d}: {conductance:.3f} uS")

    # -----------------------------------------------------
    # Weight initialization
    # -----------------------------------------------------

    if INIT_RESET_PULSES > 0:
        print(
            f"\nWeight init: RESET x{INIT_RESET_PULSES} "
            f"on all 14 used memristors..."
        )

        reset_all_weights()

        post_init_conductances = read_all_conductances()

        print("Post-init conductances:")

        for index, conductance in enumerate(
            post_init_conductances
        ):
            print(f"M{index + 1:02d}: {conductance:.3f} uS")
    else:
        post_init_conductances = list(
            initial_conductances
        )

    # -----------------------------------------------------
    # Phase 1: hidden layer (greedy, local targets)
    # -----------------------------------------------------

    phase1_history = {
        neuron: [] for neuron in HIDDEN_NEURONS
    }

    phase1_accuracy = {
        neuron: [] for neuron in HIDDEN_NEURONS
    }

    print("\nPhase 1: hidden layer training started.")

    start_time = time.time()

    for neuron in HIDDEN_NEURONS:

        accuracy_ema = 0.0

        for epoch in range(PHASE1_EPOCHS):

            epoch_values = []

            for pattern_index, (_, spikes) in enumerate(
                PATTERNS
            ):
                active = active_inputs_of(neuron, spikes)

                h = read_neuron(neuron, active)

                target = hidden_target(
                    neuron,
                    pattern_index,
                )

                predicted = h > 0

                if predicted == target:
                    accuracy_ema = (
                        (1.0 - K_EMA) * accuracy_ema
                        + K_EMA
                    )
                else:
                    accuracy_ema = (
                        (1.0 - K_EMA) * accuracy_ema
                    )

                epoch_values.append(h)

                # Perceptron-style local update: only apply a pulse when
                # the neuron's CURRENT sign is WRONG. Updating on every
                # pattern every epoch (even when already correct) over-
                # drives the detector and makes it oscillate against the
                # conflicting class-0 pattern that shares its input.
                # Gating on the prediction error is what lets the
                # separable detector sub-problem converge (exact analogue
                # of the perceptron convergence proof).
                if predicted != target:
                    apply_neuron_learning(
                        neuron,
                        active,
                        target,
                        h,
                    )

                    # Boost the detector pattern: a single-pattern
                    # detector shares its active input with a class-0
                    # pattern, so the net update can still wash out.
                    # Extra excitation on the (wrong) target pattern
                    # tips the balance. See DETECTOR_BOOST.
                    if target:
                        for _ in range(
                            DETECTOR_BOOST - 1
                        ):
                            apply_neuron_learning(
                                neuron,
                                active,
                                target,
                                h,
                            )

            phase1_history[neuron].append(epoch_values)
            phase1_accuracy[neuron].append(accuracy_ema)

            print(
                f"  {neuron} epoch "
                f"{epoch + 1:03d}/{PHASE1_EPOCHS} "
                f"target acc EMA = {accuracy_ema:.3f}"
            )

    # -----------------------------------------------------
    # Phase 2: output layer (hidden frozen)
    # -----------------------------------------------------

    phase2_y_history = []
    phase2_accuracy = []

    if MODE == "two-layer":

        print("\nPhase 2: output layer training started "
              "(hidden frozen).")

        accuracy_ema = 0.0

        for epoch in range(PHASE2_EPOCHS):

            epoch_values = []

            for pattern_index, (label, spikes) in enumerate(
                PATTERNS
            ):
                h1, h2, y, predicted = hardware_forward(spikes)

                value = y if y is not None else 0.0
                epoch_values.append(value)

                if predicted == label:
                    accuracy_ema = (
                        (1.0 - K_EMA) * accuracy_ema
                        + K_EMA
                    )
                else:
                    accuracy_ema = (
                        (1.0 - K_EMA) * accuracy_ema
                    )

                active_out = [
                    index + 1
                    for index, fired in enumerate(
                        (h1 > 0, h2 > 0)
                    )
                    if fired
                ]

                # Perceptron-style gate: only update the output when the
                # current prediction is wrong (same rationale as Phase 1).
                if predicted != label:
                    apply_neuron_learning(
                        "out",
                        active_out,
                        label,
                        y if y is not None else 0.0,
                    )

            phase2_y_history.append(epoch_values)
            phase2_accuracy.append(accuracy_ema)

            print(
                f"  out epoch "
                f"{epoch + 1:03d}/{PHASE2_EPOCHS} "
                f"accuracy EMA = {accuracy_ema:.3f}"
            )

    elapsed = time.time() - start_time

    print(
        f"\nTraining complete in {elapsed:.1f} seconds."
    )

    # -----------------------------------------------------
    # Final evaluation (hardware forward, no pulses)
    # -----------------------------------------------------

    print("\nFinal hardware evaluation:")
    print(
        f"{'pattern':>8}  {'target':>6}  "
        f"{'h1':>7}  {'h2':>7}  {'y':>7}  {'pred':>5}"
    )

    correct = 0

    evaluation_rows = []

    for label, spikes in PATTERNS:

        h1, h2, y, predicted = hardware_forward(spikes)

        if predicted == label:
            correct += 1

        evaluation_rows.append(
            (
                pattern_string(spikes),
                label,
                h1,
                h2,
                y,
                predicted,
            )
        )

        print(
            f"{pattern_string(spikes):>8}  "
            f"{str(label):>6}  "
            f"{h1:7.3f}  "
            f"{'--' if h2 is None else f'{h2:7.3f}'}  "
            f"{'--' if y is None else f'{y:7.3f}'}  "
            f"{str(predicted):>5}"
        )

    final_accuracy = 100.0 * correct / len(PATTERNS)

    print(
        f"\nFINAL {MODE} ACCURACY ON {DATASET}: "
        f"{final_accuracy:.2f}%  "
        f"({correct}/{len(PATTERNS)})"
    )

    if MODE == "two-layer":
        print(
            "Baseline reminder: a single layer cannot "
            f"solve {DATASET} (see DATASET_NOTES)."
        )

    # -----------------------------------------------------
    # Final conductances
    # -----------------------------------------------------

    final_conductances = read_all_conductances()

    print("\nFinal conductances:")

    for index, conductance in enumerate(
        final_conductances
    ):
        print(f"M{index + 1:02d}: {conductance:.3f} uS")

    # -----------------------------------------------------
    # Plot results
    # -----------------------------------------------------

    figure, axes = plt.subplots(
        4,
        1,
        figsize=(11, 14),
    )

    # Panel 1: hidden neuron responses per pattern

    epoch_axis_1 = np.arange(1, PHASE1_EPOCHS + 1)

    line_index = 0

    for neuron in HIDDEN_NEURONS:
        for pattern_index, (_, spikes) in enumerate(
            PATTERNS
        ):
            axes[0].plot(
                epoch_axis_1,
                [
                    epoch_values[pattern_index]
                    for epoch_values in
                    phase1_history[neuron]
                ],
                "o-",
                markersize=3,
                label=(
                    f"{neuron}[{pattern_string(spikes)}]"
                ),
            )

            line_index += 1

    axes[0].axhline(0, color="black", linewidth=0.8)

    axes[0].set_xlabel("Phase 1 epoch")
    axes[0].set_ylabel("Hidden response (Vy)")
    axes[0].set_title("Hidden layer responses per pattern")
    axes[0].legend(ncol=4, fontsize=7)
    axes[0].grid(alpha=0.3)

    # Panel 2: output responses per pattern

    if MODE == "two-layer":
        epoch_axis_2 = np.arange(
            1,
            PHASE2_EPOCHS + 1,
        )

        for pattern_index, (_, spikes) in enumerate(
            PATTERNS
        ):
            axes[1].plot(
                epoch_axis_2,
                [
                    epoch_values[pattern_index]
                    for epoch_values in phase2_y_history
                ],
                "o-",
                markersize=3,
                label=f"y[{pattern_string(spikes)}]",
            )

        axes[1].axhline(0, color="black", linewidth=0.8)

        axes[1].set_xlabel("Phase 2 epoch")
        axes[1].set_ylabel("Output response (Vy)")
        axes[1].set_title(
            "Output layer response per pattern (hidden frozen)"
        )
        axes[1].legend(ncol=4, fontsize=7)
        axes[1].grid(alpha=0.3)
    else:
        axes[1].set_visible(False)

    # Panel 3: accuracy curves

    for neuron in HIDDEN_NEURONS:
        axes[2].plot(
            epoch_axis_1,
            phase1_accuracy[neuron],
            label=f"phase1 {neuron} (vs local target)",
        )

    if MODE == "two-layer":
        axes[2].plot(
            epoch_axis_1[-1] + epoch_axis_2,
            phase2_accuracy,
            "o-",
            markersize=3,
            label="phase2 output (vs label)",
        )

    axes[2].set_ylim(-0.05, 1.05)
    axes[2].set_xlabel("Epoch (phase 1, then phase 2)")
    axes[2].set_ylabel("Accuracy EMA")
    axes[2].set_title("Training accuracy")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    # Panel 4: conductance evolution

    memristor_labels = [
        f"M{i + 1}" for i in range(16)
    ]

    x = np.arange(16)
    width = 0.26

    axes[3].bar(
        x - width,
        initial_conductances,
        width,
        label="Initial",
        color="#95A5A6",
    )

    axes[3].bar(
        x,
        post_init_conductances,
        width,
        label="After weight init",
        color="#7FB3D5",
    )

    axes[3].bar(
        x + width,
        final_conductances,
        width,
        label="Final",
        color="#C0392B",
    )

    axes[3].set_xticks(x)
    axes[3].set_xticklabels(memristor_labels)
    axes[3].set_xlabel("Memristor")
    axes[3].set_ylabel("Conductance (uS)")
    axes[3].set_title(
        "Conductance: initial -> after init -> final"
    )
    axes[3].legend()
    axes[3].grid(alpha=0.3, axis="y")

    figure.tight_layout()

    output_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    plot_path = os.path.join(
        output_directory,
        f"or_{DATASET}_{MODE}_{TARGET_MODE}.png",
    )

    figure.savefig(plot_path, dpi=120)

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
            f"or_{DATASET}_{MODE}_%Y%m%d_%H%M%S.csv"
        ),
    )

    with open(csv_path, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "pattern",
                "target",
                "h1",
                "h2",
                "y",
                "predicted",
            ]
        )

        for row in evaluation_rows:
            writer.writerow(row)

        writer.writerow([])

        writer.writerow(
            [
                "memristor",
                "initial_uS",
                "post_init_uS",
                "final_uS",
            ]
        )

        for index in range(16):
            writer.writerow(
                [
                    f"M{index + 1}",
                    initial_conductances[index],
                    post_init_conductances[index],
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

    # Close the auto-run log opened by the auto-logging block above.
    try:
        _log_file.close()
    except Exception:
        pass

    try:
        _sys.stdout = _sys.__stdout__
    except Exception:
        pass
