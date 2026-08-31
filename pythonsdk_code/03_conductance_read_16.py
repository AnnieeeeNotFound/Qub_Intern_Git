"""
Step 3: First contact — read the conductance of all 16 memristors from Python.

   ALL 16 DIO bits are one-hot bilateral-switch enables 
        DIO bit 0-7   -> M1-M8   (A branch, measured on scope 1+)
        DIO bit 8-15  -> M9-M16  (B branch, measured on scope 2+)

    Synapse i (i = 0..7) = M(i+1) + M(i+9), 
  - FLV read voltage in the app is -0.08 V; we use -0.10 V DC (sub-threshold,
    does not disturb the memristor state).
  - Conductance: G = (V_W1 - V_node) / (R_s * V_node)

Prerequisites (hardware):
  - Knowm V2 board mounted on the AD3 header
  - Mode switch = 2 (differential branches)
  - R_A = R_B = 20 kOhm in the two series-resistor sockets
  - (this script enables the +/-5 V rails itself — the switch ICs need them)

Run:  python 03_conductance_read.py
"""
import os
import time

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

# ---------- Test settings ----------
READ_V = -0.10      # DC read voltage (FLV-style, sub-threshold, negative = read)
RS = 20000.0        # series resistance R_A = R_B (both sockets, Mode 2)
N_SAMPLES = 8192
SAMPLE_RATE = 500_000   # ~16 ms of averaging per read
SETTLE_S = 0.05         # settling time after changing DIO / W1

dwf = DwfLibrary()
dev = openDwfDevice(dwf)

scope = None
awg = None
dio = None
aio = None


def set_w1_voltage(v):
    """Drive W1 to a DC level (function DC rides on the offset)."""
    awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, v)
    awg.configure(0, True)


def set_dio(bits):
    dio.outputSet(bits)
    dio.configure()


def capture_means():
    """Acquire one buffer on both scope channels, return (mean_ch1, mean_ch2)."""
    scope.configure(False, True)

    while True:
        status = scope.status(True)
        if status == DwfState.Done:
            break
        time.sleep(0.001)

    n = scope.statusSamplesValid()
    if n <= 0:
        raise RuntimeError("No samples captured.")
    v1 = float(np.mean(scope.statusData(0, n)))
    v2 = float(np.mean(scope.statusData(1, n)))
    return v1, v2


try:
    print("Device opened:", dev)

    # =========================================================
    # 1. Power rails (+/-5 V) — the bilateral switch ICs need them
    # =========================================================
    aio = dev.analogIO
    aio.reset()
    # AD3 layout (verified): ch0 Positive Supply (node0 Enable, node1 Voltage),
    #                       ch1 Negative Supply (node0 Enable, node1 Voltage)
    aio.channelNodeSet(0, 1, 5.0)    # V+ target
    aio.channelNodeSet(0, 0, 1.0)    # V+ enable
    aio.channelNodeSet(1, 1, -5.0)   # V- target
    aio.channelNodeSet(1, 0, 1.0)    # V- enable
    aio.enableSet(True)
    aio.configure()
    time.sleep(0.5)
    aio.status()
    vp = aio.channelNodeStatus(0, 1)
    vm = aio.channelNodeStatus(1, 1)
    print(f"Supplies: V+ = {vp:+.3f} V, V- = {vm:+.3f} V")
    if abs(vp - 5.0) > 0.5 or abs(vm + 5.0) > 0.5:
        print("WARNING: supplies out of range — check board power.")

    # =========================================================
    # 2. Waveform generator: W1 as DC source
    # =========================================================
    awg = dev.analogOut
    awg.reset(0)
    awg.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.DC)
    awg.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, 0.0)
    awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, 0.0)
    awg.nodeEnableSet(0, DwfAnalogOutNode.Carrier, 1)
    awg.runSet(0, 100.0)
    awg.idleSet(0, DwfAnalogOutIdle.Offset)
    awg.triggerSourceSet(0, DwfTriggerSource.None_)
    awg.configure(0, True)

    # =========================================================
    # 3. Oscilloscope: both channels
    # =========================================================
    scope = dev.analogIn
    scope.reset()
    scope.frequencySet(SAMPLE_RATE)
    scope.channelEnableSet(0, True)
    scope.channelEnableSet(1, True)
    try:
        scope.channelRangeSet(0, 1.0)   # smaller range = lower noise
        scope.channelRangeSet(1, 1.0)
    except Exception:
        scope.channelRangeSet(0, 5.0)
        scope.channelRangeSet(1, 5.0)
    scope.bufferSizeSet(N_SAMPLES)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)
    scope.triggerSourceSet(DwfTriggerSource.None_)

    # =========================================================
    # 4. Digital IO: all 16 lines as outputs, all switches off
    # =========================================================
    dio = dev.digitalIO
    dio.reset()
    dio.outputEnableSet(0xFFFF)
    set_dio(0x0000)

    # =========================================================
    # 5. Baseline: both branch nodes tied to W1 = 0 V
    #    (DIO bit 0 = M1 on A branch, DIO bit 8 = M9 on B branch)
    # =========================================================
    set_w1_voltage(0.0)
    set_dio((1 << 0) | (1 << 8))
    time.sleep(SETTLE_S)
    v1_off, v2_off = capture_means()
    print(f"\nBaseline offsets: V(1+) = {v1_off * 1e3:+.2f} mV, "
          f"V(2+) = {v2_off * 1e3:+.2f} mV")

    # =========================================================
    # 6. Read every memristor one by one (one-hot DIO)
    # =========================================================
    results = []   # (memristor 1-16, branch, V_node, G)

    print("\nReading 16 memristors (W1 = %.2f V DC)..." % READ_V)
    print(f"{'M':>3} {'DIO bit':>7} {'branch':>6} {'V_node (mV)':>12} "
          f"{'G (uS)':>10} {'R (kOhm)':>10}")

    set_w1_voltage(READ_V)
    time.sleep(SETTLE_S)

    for m in range(16):                 # m = 0..15 -> memristor M1..M16
        set_dio(1 << m)
        time.sleep(SETTLE_S)
        v1, v2 = capture_means()

        if m < 8:
            branch = "A"
            v_node = v1 - v1_off
            other = v2 - v2_off
        else:
            branch = "B"
            v_node = v2 - v2_off
            other = v1 - v1_off

        if abs(v_node) < 0.002:
            g_us = float("nan")         # below noise floor -> HRS, ~0 G
            r_kohm = float("nan")
        else:
            g = (READ_V - v_node) / (RS * v_node)   # Siemens
            g_us = g * 1e6
            r_kohm = (1.0 / g) / 1e3 if g > 0 else float("nan")

        results.append((m + 1, branch, v_node, g_us))
        print(f"{m + 1:>3} {m:>7} {branch:>6} {v_node * 1e3:>12.2f} "
              f"{g_us:>10.3f} {r_kohm:>10.1f}")

    # =========================================================
    # 7. Synapse view (Mode 2): w_i = G_A - G_B
    # =========================================================
    print("\nSynapse weights (w_i = G_A - G_B, uS):")
    for i in range(8):
        ga = results[i][3]
        gb = results[i + 8][3]
        ga_s = ga if np.isfinite(ga) else 0.0
        gb_s = gb if np.isfinite(gb) else 0.0
        w = ga_s - gb_s
        print(f"  synapse {i + 1} (M{i + 1}/M{i + 9}): "
              f"GA = {ga_s:8.3f}, GB = {gb_s:8.3f}, w = {w:8.3f}")

    # =========================================================
    # 8. Bar chart
    # =========================================================
    labels = [f"M{m + 1}" for m in range(16)]
    values = [r[3] if np.isfinite(r[3]) else 0.0 for r in results]
    colors = ["#378ADD" if r[1] == "A" else "#EF9F27" for r in results]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(labels, values, color=colors, edgecolor="none")
    ax.set_xlabel("Memristor (blue = A branch / 1+, orange = B branch / 2+)")
    ax.set_ylabel("Conductance (uS)")
    ax.set_title(f"Knowm V2 board — 16 memristor conductance "
                 f"(read at {READ_V} V, R_s = {RS / 1000:.0f} kOhm)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()

    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "conductance_result.png",
    )
    fig.savefig(output_path, dpi=120)
    print("\nPlot saved to:", output_path)
    plt.show()

    # =========================================================
    # 9. CSV log
    # =========================================================
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        time.strftime("conductance_%Y%m%d_%H%M%S.csv"),
    )
    with open(csv_path, "w") as f:
        f.write("memristor,branch,dio_bit,V_node_V,G_uS\n")
        for m, branch, v_node, g_us in results:
            g_str = f"{g_us:.6f}" if np.isfinite(g_us) else ""
            f.write(f"{m},{branch},{m - 1},{v_node:.6f},{g_str}\n")
    print("CSV saved to:", csv_path)

finally:
    # Safety cleanup: everything off
    try:
        set_w1_voltage(0.0)
    except Exception:
        pass
    try:
        set_dio(0x0000)
        dio.outputEnableSet(0x0000)
        dio.configure()
    except Exception:
        pass
    try:
        awg.configure(0, False)
    except Exception:
        pass
    try:
        aio.channelNodeSet(0, 0, 0.0)
        aio.channelNodeSet(1, 0, 0.0)
        aio.enableSet(False)
        aio.configure()
    except Exception:
        pass
    dev.close()
    print("\nDevice closed.")
