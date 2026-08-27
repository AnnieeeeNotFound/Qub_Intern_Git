"""
Step 1: AWG -> Scope loopback test (no memristor board needed)

What this validates:
  - pydwf can open the AD3 and drive the waveform generator
  - pydwf can acquire samples from the oscilloscope
  - the entire Python -> dwf.dll -> AD3 signal chain works

Wiring (you do this with a jumper wire):
  - connect AD3 "W1" pin to "1+" (channel 1 positive input)
  - connect "GND" to "1-" (channel 1 negative input)
  Expected result: measured sine ~1 kHz, ~1.0 V amplitude.

Run:  python 01_loopback.py
"""
import os
import time

# ---------- WaveForms DLL location ----------
DWF_LIB_DIR = r"D:\Digilent\WaveForms3"

os.environ["PATH"] = (
    DWF_LIB_DIR
    + os.pathsep
    + os.environ.get("PATH", "")
)

if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DWF_LIB_DIR)

# ---------- Python libraries ----------
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


# ---------- Test settings ----------
FREQ_HZ = 1000.0
AMPLITUDE_V = 1.0
SAMPLE_RATE = 1_000_000
N_SAMPLES = 8192


# ---------- Open AD3 ----------
dwf = DwfLibrary()
dev = openDwfDevice(dwf)

awg = None

try:
    print("Device opened:", dev)

    # =========================================================
    # 1. Configure waveform generator: W1 = 1 kHz, 1 V sine
    # =========================================================
    awg = dev.analogOut

    awg.reset(0)

    # Explicitly enable AWG channel 0 carrier
    awg.nodeEnableSet(
        0,
        DwfAnalogOutNode.Carrier,
        True,
    )

    awg.nodeFunctionSet(
        0,
        DwfAnalogOutNode.Carrier,
        DwfAnalogOutFunction.Sine,
    )

    awg.nodeFrequencySet(
        0,
        DwfAnalogOutNode.Carrier,
        FREQ_HZ,
    )

    awg.nodeAmplitudeSet(
        0,
        DwfAnalogOutNode.Carrier,
        AMPLITUDE_V,
    )

    awg.nodeOffsetSet(
        0,
        DwfAnalogOutNode.Carrier,
        0.0,
    )

    # Run for 10 seconds
    awg.runSet(0, 10.0)

    # Hold zero volts when stopped
    awg.idleSet(
        0,
        DwfAnalogOutIdle.Offset,
    )

    # Start W1
    awg.configure(0, True)

    # Give the generator time to start
    time.sleep(0.2)

    # =========================================================
    # 2. Configure oscilloscope: channel 1
    # =========================================================
    scope = dev.analogIn

    scope.reset()
    scope.frequencySet(SAMPLE_RATE)

    # Channel index 0 corresponds to AD3 channel 1
    scope.channelEnableSet(0, True)
    scope.channelRangeSet(0, 5.0)
    scope.bufferSizeSet(N_SAMPLES)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)

    # Start immediately without waiting for a trigger
    scope.triggerSourceSet(DwfTriggerSource.None_)

    scope.configure(False, True)

    print("Capturing samples...")

    while True:
        status = scope.status(True)

        if status == DwfState.Done:
            break

        time.sleep(0.001)

    # =========================================================
    # 3. Read captured data
    # =========================================================
    n = scope.statusSamplesValid()

    if n <= 0:
        raise RuntimeError("No samples were captured.")

    data = np.asarray(
        scope.statusData(0, n),
        dtype=float,
    )

    t = np.arange(n) / SAMPLE_RATE

    # =========================================================
    # 4. Analyze signal
    # =========================================================
    amplitude = (data.max() - data.min()) / 2
    dc_offset = data.mean()

    # Remove DC offset before detecting zero crossings
    centered_data = data - dc_offset

    crossings = np.flatnonzero(
        np.diff(np.signbit(centered_data))
    )

    frequency_estimate = 0.0

    if len(crossings) > 2:
        crossing_intervals = np.diff(crossings)

        # A full sine period has two zero crossings
        period_samples = 2 * np.mean(crossing_intervals)
        period_seconds = period_samples / SAMPLE_RATE

        if period_seconds > 0:
            frequency_estimate = 1.0 / period_seconds

    print(f"samples captured  : {n}")
    print(
        f"measured amplitude: {amplitude:.3f} V "
        f"(expected ~{AMPLITUDE_V:.1f} V)"
    )
    print(
        f"measured frequency: {frequency_estimate:.1f} Hz "
        f"(expected ~{FREQ_HZ:.0f} Hz)"
    )
    print(f"DC offset         : {dc_offset:+.3f} V")

    passed = (
        abs(amplitude - AMPLITUDE_V) < 0.15
        and abs(frequency_estimate - FREQ_HZ) < 50
    )

    if passed:
        print("\nLOOPBACK PASSED - signal chain works!")
    else:
        print(
            "\nLOOPBACK FAILED - check that W1 is connected to 1+ "
            "and GND is connected to 1-."
        )

    # =========================================================
    # 5. Save and display plot
    # =========================================================
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "loopback_result.png",
    )

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(
        t * 1000,
        data,
        linewidth=0.8,
    )

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("CH1 voltage (V)")
    ax.set_title(
        f"Loopback: W1 -> CH1 "
        f"(amplitude={amplitude:.3f} V, "
        f"frequency={frequency_estimate:.0f} Hz)"
    )
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=120)

    print("Plot saved to:", output_path)

    # Open the plot in a window
    plt.show()

finally:
    # Stop the waveform generator
    if awg is not None:
        try:
            awg.configure(0, False)
        except Exception:
            pass

    dev.close()
    print("Device closed.")