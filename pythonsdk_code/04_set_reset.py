"""
Step 4: SET/RESET pulse staircase on one memristor.

Edit the USER CONTROL PANEL below, then press Run.

DIO bit m selects memristor M(m + 1).

M1-M8  -> A branch -> scope 1+
M9-M16 -> B branch -> scope 2+

SET   = negative W1 pulse
RESET = positive W1 pulse
"""

import os
import time
import csv

# =========================================================
# USER CONTROL PANEL
# =========================================================

MEM = 5
# Target memristor: 1 to 16

MODE = "set"
# Choose: "set" or "reset"

PULSE_AMPLITUDE_V = 1.8
# Pulse amplitude in volts

PULSE_WIDTH_US = 5000
# Pulse width in microseconds

MAX_PULSES = 100
# Maximum number of pulses

TARGET_CONDUCTANCE_US = 200
# SET stops when G rises above this value.
# RESET stops when G falls below this value.

SHOW_PLOT_WINDOW = True


# =========================================================
# Validate settings
# =========================================================

if not 1 <= MEM <= 16:
    raise ValueError("MEM must be between 1 and 16.")

if MODE not in ("set", "reset"):
    raise ValueError('MODE must be "set" or "reset".')

if PULSE_AMPLITUDE_V <= 0:
    raise ValueError("Pulse amplitude must be positive.")

if PULSE_WIDTH_US <= 0:
    raise ValueError("Pulse width must be positive.")

if MAX_PULSES <= 0:
    raise ValueError("Maximum pulses must be positive.")

if TARGET_CONDUCTANCE_US <= 0:
    raise ValueError("Target conductance must be positive.")


# =========================================================
# WaveForms DLL setup
# =========================================================

DWF_LIB_DIR = r"D:\Digilent\WaveForms3"

os.environ["PATH"] = (
    DWF_LIB_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DWF_LIB_DIR)


# =========================================================
# Imports
# =========================================================

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
# Fixed measurement settings
# =========================================================

READ_V = -0.10
SERIES_RESISTANCE = 20_000.0
SAMPLE_RATE = 500_000
N_SAMPLES = 8192
SETTLE_TIME_S = 0.05

DIO_BIT = MEM - 1
IS_A_BRANCH = MEM <= 8
SCOPE_CHANNEL = 0 if IS_A_BRANCH else 1

PULSE_WIDTH_S = PULSE_WIDTH_US * 1e-6

if MODE == "set":
    PULSE_V = -PULSE_AMPLITUDE_V
else:
    PULSE_V = +PULSE_AMPLITUDE_V


# =========================================================
# Hardware variables
# =========================================================

dev = None
awg = None
scope = None
dio = None
aio = None


# =========================================================
# Hardware functions
# =========================================================

def set_w1_voltage(voltage):
    """Set W1 to a DC voltage."""

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


def select_memristor():
    """Select exactly one memristor using one-hot DIO control."""

    dio.outputSet(1 << DIO_BIT)
    dio.configure()


def turn_off_all_memristors():
    """Turn off all memristor selection switches."""

    dio.outputSet(0x0000)
    dio.configure()


def apply_pulse(voltage, width_seconds):
    """Apply a pulse and then return W1 to zero volts."""

    set_w1_voltage(voltage)

    start_time = time.perf_counter()

    while time.perf_counter() - start_time < width_seconds:
        pass

    set_w1_voltage(0.0)


def capture_scope_average():
    """Capture both scope channels and return their average voltages."""

    scope.configure(False, True)

    while True:
        status = scope.status(True)

        if status == DwfState.Done:
            break

        time.sleep(0.001)

    number_samples = scope.statusSamplesValid()

    if number_samples <= 0:
        raise RuntimeError("No oscilloscope samples were captured.")

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


def calculate_conductance(node_voltage):
    """Calculate conductance in microSiemens."""

    if abs(node_voltage) < 0.002:
        return float("nan")

    conductance_siemens = (
        READ_V - node_voltage
    ) / (
        SERIES_RESISTANCE * node_voltage
    )

    return conductance_siemens * 1e6


# =========================================================
# Experiment
# =========================================================

try:
    print("Starting experiment...")
    print(f"Target memristor: M{MEM}")
    print(f"DIO bit: {DIO_BIT}")
    print(
        f"Branch: {'A' if IS_A_BRANCH else 'B'}, "
        f"scope channel: {SCOPE_CHANNEL + 1}+"
    )
    print(f"Mode: {MODE.upper()}")
    print(f"Pulse voltage: {PULSE_V:+.3f} V")
    print(f"Pulse width: {PULSE_WIDTH_US} us")
    print(f"Maximum pulses: {MAX_PULSES}")
    print(
        f"Target conductance: "
        f"{TARGET_CONDUCTANCE_US:.3f} uS"
    )

    # -----------------------------------------------------
    # Open AD3
    # -----------------------------------------------------

    dwf = DwfLibrary()
    dev = openDwfDevice(dwf)

    print("\nDevice opened.")

    # -----------------------------------------------------
    # Enable AD3 power rails
    # -----------------------------------------------------

    aio = dev.analogIO
    aio.reset()

    # Positive supply
    aio.channelNodeSet(0, 1, +5.0)
    aio.channelNodeSet(0, 0, 1.0)

    # Negative supply
    aio.channelNodeSet(1, 1, -5.0)
    aio.channelNodeSet(1, 0, 1.0)

    aio.enableSet(True)
    aio.configure()

    time.sleep(0.5)
    aio.status()

    v_plus = aio.channelNodeStatus(0, 1)
    v_minus = aio.channelNodeStatus(1, 1)

    print(
        f"Supplies: V+ = {v_plus:+.3f} V, "
        f"V- = {v_minus:+.3f} V"
    )

    # -----------------------------------------------------
    # Configure AWG
    # -----------------------------------------------------

    awg = dev.analogOut
    awg.reset(0)
    set_w1_voltage(0.0)

    # -----------------------------------------------------
    # Configure oscilloscope
    # -----------------------------------------------------

    scope = dev.analogIn
    scope.reset()
    scope.frequencySet(SAMPLE_RATE)

    scope.channelEnableSet(0, True)
    scope.channelEnableSet(1, True)

    scope.channelRangeSet(0, 1.0)
    scope.channelRangeSet(1, 1.0)

    scope.bufferSizeSet(N_SAMPLES)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)
    scope.triggerSourceSet(DwfTriggerSource.None_)

    # -----------------------------------------------------
    # Configure digital I/O
    # -----------------------------------------------------

    dio = dev.digitalIO
    dio.reset()
    dio.outputEnableSet(0xFFFF)

    turn_off_all_memristors()

    # -----------------------------------------------------
    # Select requested memristor
    # -----------------------------------------------------

    select_memristor()
    set_w1_voltage(0.0)

    time.sleep(SETTLE_TIME_S)

    # -----------------------------------------------------
    # Measure baseline
    # -----------------------------------------------------

    baseline_ch1, baseline_ch2 = capture_scope_average()

    if IS_A_BRANCH:
        baseline = baseline_ch1
    else:
        baseline = baseline_ch2

    print(
        f"\nBaseline: "
        f"{baseline * 1000:+.3f} mV"
    )

    # -----------------------------------------------------
    # Measure initial conductance
    # -----------------------------------------------------

    set_w1_voltage(READ_V)
    time.sleep(SETTLE_TIME_S)

    voltage_ch1, voltage_ch2 = capture_scope_average()

    if IS_A_BRANCH:
        node_voltage = voltage_ch1 - baseline
    else:
        node_voltage = voltage_ch2 - baseline

    initial_g = calculate_conductance(node_voltage)

    print(
        f"Initial conductance: "
        f"{initial_g:.3f} uS"
    )

    records = [
        (0, initial_g, node_voltage)
    ]

    # -----------------------------------------------------
    # Pulse staircase
    # -----------------------------------------------------

    print("\nApplying pulses...")

    for pulse_number in range(1, MAX_PULSES + 1):
        print(
            f"Pulse {pulse_number} "
            f"of {MAX_PULSES}"
        )

        apply_pulse(
            PULSE_V,
            PULSE_WIDTH_S,
        )

        set_w1_voltage(READ_V)
        time.sleep(SETTLE_TIME_S)

        voltage_ch1, voltage_ch2 = capture_scope_average()

        if IS_A_BRANCH:
            node_voltage = voltage_ch1 - baseline
        else:
            node_voltage = voltage_ch2 - baseline

        conductance = calculate_conductance(node_voltage)

        records.append(
            (
                pulse_number,
                conductance,
                node_voltage,
            )
        )

        print(
            f"  Conductance: "
            f"{conductance:.3f} uS"
        )

        if np.isfinite(conductance):
            if MODE == "set":
                if conductance >= TARGET_CONDUCTANCE_US:
                    print(
                        "\nSET target reached."
                    )
                    break

            if MODE == "reset":
                if conductance <= TARGET_CONDUCTANCE_US:
                    print(
                        "\nRESET target reached."
                    )
                    break

    # -----------------------------------------------------
    # Create graph
    # -----------------------------------------------------

    # =========================================================
    # Create graph with starting and ending conductance
    # =========================================================

    pulse_numbers = [
        row[0]
        for row in records
    ]

    conductances = [
        row[1]
        if np.isfinite(row[1])
        else 0.0
        for row in records
    ]

    # Starting conductance
    starting_conductance = conductances[0]

    # Ending conductance
    ending_conductance = conductances[-1]

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    line_color = (
        "#2471A3"
        if MODE == "set"
        else "#C0392B"
    )

    axis.plot(
        pulse_numbers,
        conductances,
        "o-",
        color=line_color,
        linewidth=2,
        label=f"M{MEM} conductance",
    )

    # Target conductance line
    axis.axhline(
        TARGET_CONDUCTANCE_US,
        color="gray",
        linestyle="--",
        alpha=0.7,
        label=(
            f"Target = "
            f"{TARGET_CONDUCTANCE_US:g} uS"
        ),
    )

    # Starting conductance marker
    axis.scatter(
        pulse_numbers[0],
        starting_conductance,
        color="green",
        s=100,
        zorder=5,
        label=(
            f"Start = "
            f"{starting_conductance:.3f} uS"
        ),
    )

    # Ending conductance marker
    axis.scatter(
        pulse_numbers[-1],
        ending_conductance,
        color="black",
        s=100,
        zorder=5,
        label=(
            f"End = "
            f"{ending_conductance:.3f} uS"
        ),
    )

    # Text annotation for starting value
    axis.annotate(
        f"Start\n{starting_conductance:.3f} uS",
        xy=(
            pulse_numbers[0],
            starting_conductance,
        ),
        xytext=(15, 15),
        textcoords="offset points",
        color="green",
        arrowprops={
            "arrowstyle": "->",
            "color": "green",
        },
    )

    # Text annotation for ending value
    axis.annotate(
        f"End\n{ending_conductance:.3f} uS",
        xy=(
            pulse_numbers[-1],
            ending_conductance,
        ),
        xytext=(-70, -35),
        textcoords="offset points",
        color="black",
        arrowprops={
            "arrowstyle": "->",
            "color": "black",
        },
    )

    axis.set_xlabel("Pulse number")
    axis.set_ylabel("Conductance (uS)")
    axis.set_yscale("log")

    axis.set_title(
        f"M{MEM} {MODE.upper()} staircase\n"
        f"Starting G = {starting_conductance:.3f} uS, "
        f"Ending G = {ending_conductance:.3f} uS"
    )

    axis.grid(
        alpha=0.3,
        which="both",
    )

    axis.legend()
    figure.tight_layout()

    output_directory = os.path.dirname(
        os.path.abspath(__file__)
    )

    plot_path = os.path.join(
        output_directory,
        f"setreset_M{MEM}_{MODE}.png",
    )

    figure.savefig(
        plot_path,
        dpi=120,
    )

    print("\nStarting conductance:")
    print(f"  {starting_conductance:.3f} uS")

    print("Ending conductance:")
    print(f"  {ending_conductance:.3f} uS")

    print("\nGraph saved to:")
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
            f"setreset_M{MEM}_{MODE}_"
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
                "pulse",
                "conductance_uS",
                "node_voltage_V",
            ]
        )

        for pulse_number, conductance, voltage in records:
            writer.writerow(
                [
                    pulse_number,
                    conductance,
                    voltage,
                ]
            )

    print("CSV saved to:")
    print(csv_path)

    if SHOW_PLOT_WINDOW:
        plt.show()
    else:
        plt.close(figure)

finally:
    # -----------------------------------------------------
    # Safety shutdown
    # -----------------------------------------------------

    try:
        if awg is not None:
            set_w1_voltage(0.0)
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