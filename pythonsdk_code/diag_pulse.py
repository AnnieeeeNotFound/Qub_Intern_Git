"""One-off diagnostic: does the Square 'one-pulse' configuration actually
produce a pulse on W1? Capture 8192 samples at 50 kHz (164 ms window) on
scope 1+, fire the pulse in the middle, print trace stats + save PNG."""
import os
import time

DWF_LIB_DIR = r"D:\Digilent\WaveForms3"
os.environ["PATH"] = DWF_LIB_DIR + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(DWF_LIB_DIR)

import numpy as np
import matplotlib

matplotlib.use("Agg")
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

PERIOD_S = 0.002      # 2 ms
DUTY_PCT = 25.0       # 25% high -> 500 us pulse
PULSE_V = 1.0

dwf = DwfLibrary()
dev = openDwfDevice(dwf)

try:
    awg = dev.analogOut
    awg.reset(0)
    awg.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.Square)
    awg.nodeFrequencySet(0, DwfAnalogOutNode.Carrier, 1.0 / PERIOD_S)
    awg.nodeSymmetrySet(0, DwfAnalogOutNode.Carrier, DUTY_PCT)
    awg.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, PULSE_V / 2.0)
    awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, PULSE_V / 2.0)
    awg.runSet(0, PERIOD_S)              # one cycle
    awg.idleSet(0, DwfAnalogOutIdle.Initial)
    awg.triggerSourceSet(0, DwfTriggerSource.None_)
    awg.nodeEnableSet(0, DwfAnalogOutNode.Carrier, 1)

    # readback
    print("readback:")
    print("  function  :", awg.nodeFunctionGet(0, DwfAnalogOutNode.Carrier))
    print("  frequency :", awg.nodeFrequencyGet(0, DwfAnalogOutNode.Carrier))
    print("  symmetry  :", awg.nodeSymmetryGet(0, DwfAnalogOutNode.Carrier))
    print("  amplitude :", awg.nodeAmplitudeGet(0, DwfAnalogOutNode.Carrier))
    print("  offset    :", awg.nodeOffsetGet(0, DwfAnalogOutNode.Carrier))
    print("  run       :", awg.runGet(0))
    print("  idle      :", awg.idleGet(0))
    print("  enable    :", awg.nodeEnableGet(0, DwfAnalogOutNode.Carrier))

    scope = dev.analogIn
    scope.reset()
    scope.frequencySet(50_000)            # 8192 samples = 164 ms
    scope.channelEnableSet(0, True)
    scope.channelEnableSet(1, True)
    scope.channelRangeSet(0, 5.0)
    scope.channelRangeSet(1, 5.0)
    scope.bufferSizeSet(8192)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)
    scope.triggerSourceSet(DwfTriggerSource.None_)

    print("\nstarting scope, firing pulse in 20 ms...")
    scope.configure(False, True)
    time.sleep(0.020)
    awg.configure(0, True)                # FIRE
    time.sleep(0.120)                     # let pulse + window finish

    while True:
        if scope.status(True) == DwfState.Done:
            break
        time.sleep(0.001)

    n = scope.statusSamplesValid()
    data = np.asarray(scope.statusData(0, n), dtype=float)
    t = np.arange(n) / 50_000.0

    print(f"\ntrace stats: min {data.min():+.4f} V, max {data.max():+.4f} V, "
          f"mean {data.mean():+.4f} V")
    above = np.flatnonzero(data > 0.3)
    if len(above):
        print(f"samples > 0.3 V: {len(above)} "
              f"({len(above) / 50_000.0 * 1e6:.0f} us), "
              f"from t={above[0] / 50_000.0 * 1e3:.1f} ms "
              f"to {above[-1] / 50_000.0 * 1e3:.1f} ms")
    else:
        print("NO samples above 0.3 V -> pulse not generated!")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, data, lw=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Scope 1+ (V)")
    ax.set_title("Diagnostic: single Square pulse on W1")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("pulse_diag.png", dpi=110)
    print("saved pulse_diag.png")

finally:
    try:
        awg.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.DC)
        awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, 0.0)
        awg.configure(0, False)
    except Exception:
        pass
    dev.close()
    print("Device closed.")
