"""
Step 0: AD3 full setup & self-test  (merged 00 + 01 + 02)

This single file replaces the three old scripts:
    - 00_connect_test.py   -> Step 1 (connect / enumerate)
    - 01_loopback.py       -> Step 2 (AWG -> Scope loopback)
    - 02_power_and_dio.py  -> Step 3 (supplies + DIO sanity)

The device is opened ONCE at the top and closed once at the end, so the
three tests run back-to-back without re-plugging or re-enumerating.

Run:  python 00_setup.py
"""

import os
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

from pydwf import (
    DwfLibrary,
    DwfAnalogOutNode,
    DwfAnalogOutFunction,
    DwfAnalogOutIdle,
    DwfAcquisitionMode,
    DwfTriggerSource,
    DwfState,
)
from pydwf.utilities.open_dwf_device import openDwfDevice


# =========================================================
# WaveForms DLL location
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
# Shared hardware handles (filled by main, used by steps)
# =========================================================
dev = None
awg = None
scope = None
aio = None
dio = None


def close_all():
    """Stop every active block and close the device (safe to call anytime)."""
    global awg, aio, dio, dev
    try:
        if awg is not None:
            awg.configure(0, False)
    except Exception:
        pass
    try:
        if aio is not None:
            aio.enableSet(False)
            aio.configure()
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
        if dev is not None:
            dev.close()
    except Exception:
        pass
    print("\nDevice closed.")


# =========================================================
# STEP 1: CONNECT TEST (was 00_connect_test.py)
# =========================================================
def step1_connect():
    """Confirm Python can see the AD3 at all (enumerate, do not open yet)."""
    print("=" * 60)
    print("STEP 1: CONNECT TEST")
    print("=" * 60)

    dwf = DwfLibrary()
    print("DWF library version:", dwf.getVersion())

    n = dwf.deviceEnum.enumerateDevices()
    print(f"Devices found: {n}")
    for i in range(n):
        print(f"  [{i}] name={dwf.deviceEnum.deviceName(i)!r} "
              f"serial={dwf.deviceEnum.serialNumber(i)} "
              f"type={dwf.deviceEnum.deviceType(i)} "
              f"opened={dwf.deviceEnum.deviceIsOpened(i)}")

    if n == 0:
        print("\nNo Digilent device visible. Check USB cable / close WaveForms GUI.")
        sys.exit(1)

    print("\nPYDWF LINK OK - Python can talk to your AD3.\n")


# =========================================================
# STEP 2: AWG -> SCOPE LOOPBACK (was 01_loopback.py)
# =========================================================
def step2_loopback():
    """Validate the full AWG->Scope signal chain with a 1 kHz / 1 V sine.

    Wiring (jumper wire):
        AD3 "W1" pin -> "1+" (channel 1 positive)
        AD3 "GND"    -> "1-" (channel 1 negative)
    """
    global awg, scope
    print("=" * 60)
    print("STEP 2: AWG -> SCOPE LOOPBACK")
    print("=" * 60)

    # ---- test settings ----
    FREQ_HZ = 1000.0
    AMPLITUDE_V = 1.0
    SAMPLE_RATE = 1_000_000
    N_SAMPLES = 8192

    # ---- 1. configure waveform generator: W1 = 1 kHz, 1 V sine ----
    awg = dev.analogOut
    awg.reset(0)

    # ★ MUST enable the carrier node, or no waveform is produced (verified).
    awg.nodeEnableSet(0, DwfAnalogOutNode.Carrier, True)
    awg.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.Sine)
    awg.nodeFrequencySet(0, DwfAnalogOutNode.Carrier, FREQ_HZ)
    awg.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, AMPLITUDE_V)
    awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, 0.0)
    awg.runSet(0, 10.0)
    awg.idleSet(0, DwfAnalogOutIdle.Offset)
    awg.configure(0, True)

    time.sleep(0.2)

    # ---- 2. configure oscilloscope: channel 1 ----
    scope = dev.analogIn
    scope.reset()
    scope.frequencySet(SAMPLE_RATE)
    scope.channelEnableSet(0, True)
    scope.channelRangeSet(0, 5.0)
    scope.bufferSizeSet(N_SAMPLES)
    scope.acquisitionModeSet(DwfAcquisitionMode.Single)
    scope.triggerSourceSet(DwfTriggerSource.None_)
    scope.configure(False, True)

    print("Capturing samples...")

    while True:
        status = scope.status(True)
        if status == DwfState.Done:
            break
        time.sleep(0.001)

    # ---- 3. read captured data ----
    n = scope.statusSamplesValid()
    if n <= 0:
        raise RuntimeError("No samples were captured.")

    data = np.asarray(scope.statusData(0, n), dtype=float)
    t = np.arange(n) / SAMPLE_RATE

    # ---- 4. analyze signal ----
    amplitude = (data.max() - data.min()) / 2
    dc_offset = data.mean()
    centered_data = data - dc_offset
    crossings = np.flatnonzero(np.diff(np.signbit(centered_data)))

    frequency_estimate = 0.0
    if len(crossings) > 2:
        crossing_intervals = np.diff(crossings)
        period_samples = 2 * np.mean(crossing_intervals)
        period_seconds = period_samples / SAMPLE_RATE
        if period_seconds > 0:
            frequency_estimate = 1.0 / period_seconds

    print(f"samples captured  : {n}")
    print(f"measured amplitude: {amplitude:.3f} V (expected ~{AMPLITUDE_V:.1f} V)")
    print(f"measured frequency: {frequency_estimate:.1f} Hz (expected ~{FREQ_HZ:.0f} Hz)")
    print(f"DC offset         : {dc_offset:+.3f} V")

    passed = (abs(amplitude - AMPLITUDE_V) < 0.15
              and abs(frequency_estimate - FREQ_HZ) < 50)
    print("\nLOOPBACK PASSED - signal chain works!" if passed
          else "\nLOOPBACK FAILED - check W1->1+ and GND->1- wiring.")

    # ---- 5. save plot ----
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "setup_step2_loopback.png",
    )
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t * 1000, data, linewidth=0.8)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("CH1 voltage (V)")
    ax.set_title(f"Loopback: W1 -> CH1 "
                 f"(amplitude={amplitude:.3f} V, "
                 f"frequency={frequency_estimate:.0f} Hz)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    print("Plot saved to:", output_path)
    plt.show()

    # stop the generator before moving on to the power/DIO test
    awg.configure(0, False)


# =========================================================
# STEP 3: POWER SUPPLIES + DIGITAL IO (was 02_power_and_dio.py)
# =========================================================
def step3_power_dio():
    """Enable V+/V- rails and verify the DIO selector lines read back.

    Board attached, memristors untouched. Does NOT send any waveform.
    """
    global aio, dio
    print("\n" + "=" * 60)
    print("STEP 3: POWER SUPPLIES + DIGITAL IO")
    print("=" * 60)

    # ---- 1. discover AnalogIO channels and nodes ----
    aio = dev.analogIO
    aio.reset()

    channel_count = aio.channelCount()
    print(f"AnalogIO channels: {channel_count}")

    vplus = vminus = vplus_enable = vminus_enable = None

    for channel in range(channel_count):
        channel_name, _ = aio.channelName(channel)
        print(f"  channel {channel}: {channel_name}")

        node_mask = aio.channelInfo(channel)
        node = 0
        while node_mask:
            if node_mask & 1:
                node_name, _ = aio.channelNodeName(channel, node)
                print(f"    node {node}: {node_name}")

                node_name_upper = str(node_name).upper()
                channel_name_upper = str(channel_name).upper()

                # AD3 actual names (verified live 2026-08-27):
                #   channel 0 "Positive Supply": node 0 = Enable, node 1 = Voltage
                #   channel 1 "Negative Supply": node 0 = Enable, node 1 = Voltage
                if "POSITIVE" in channel_name_upper and "VOLTAGE" in node_name_upper:
                    vplus = (channel, node)
                if "POSITIVE" in channel_name_upper and "ENABLE" in node_name_upper:
                    vplus_enable = (channel, node)
                if "NEGATIVE" in channel_name_upper and "VOLTAGE" in node_name_upper:
                    vminus = (channel, node)
                if "NEGATIVE" in channel_name_upper and "ENABLE" in node_name_upper:
                    vminus_enable = (channel, node)

                # Legacy fallback for other device firmware naming
                if "V+" in node_name_upper:
                    vplus = (channel, node)
                if "V-" in node_name_upper or "V−" in node_name_upper:
                    vminus = (channel, node)

            node_mask >>= 1
            node += 1

    # ---- 2. test AD3 power supplies ----
    if vplus is None or vminus is None:
        print("\nWARNING: Could not locate V+ and V- nodes automatically.")
        print("Use the channel/node list above to identify them.")
    else:
        print(f"\nV+ voltage node : channel {vplus[0]}, node {vplus[1]}")
        print(f"V+ enable node  : channel {vplus_enable[0]}, node {vplus_enable[1]}")
        print(f"V- voltage node : channel {vminus[0]}, node {vminus[1]}")
        print(f"V- enable node  : channel {vminus_enable[0]}, node {vminus_enable[1]}")

        vmin, vmax, _ = aio.channelNodeSetInfo(vplus[0], vplus[1])
        print(f"V+ settable range: {vmin} .. {vmax} V")

        aio.channelNodeSet(vplus[0], vplus[1], 5.0)
        aio.channelNodeSet(vminus[0], vminus[1], -5.0)

        # ★ Enable each supply channel's own Enable node (the real switch).
        #   Writing the Voltage node alone does NOT power the rail.
        aio.channelNodeSet(vplus_enable[0], vplus_enable[1], 1.0)
        aio.channelNodeSet(vminus_enable[0], vminus_enable[1], 1.0)

        aio.enableSet(True)
        aio.configure()

        time.sleep(1.0)
        aio.status()

        measured_plus = aio.channelNodeStatus(vplus[0], vplus[1])
        measured_minus = aio.channelNodeStatus(vminus[0], vminus[1])

        print(f"\nV+ set to +5.0 V -> measured {measured_plus:+.3f} V")
        print(f"V- set to -5.0 V -> measured {measured_minus:+.3f} V")

        power_ok = (abs(measured_plus - 5.0) < 0.3
                    and abs(measured_minus + 5.0) < 0.3)
        print("POWER OK - the board is receiving the expected rails."
              if power_ok else
              "POWER PROBLEM - check the Discovery board power switch "
              "and board connection.")

        # Always switch supplies off after the test
        if vplus_enable is not None:
            aio.channelNodeSet(vplus_enable[0], vplus_enable[1], 0.0)
        if vminus_enable is not None:
            aio.channelNodeSet(vminus_enable[0], vminus_enable[1], 0.0)
        aio.enableSet(False)
        aio.configure()
        print("AnalogIO supplies disabled.")

    # ---- 3. digital IO quick check ----
    print("\nTesting digital IO...")

    dio = dev.digitalIO
    dio.reset()
    dio.outputEnableSet(0xFFFF)

    # DIO15..DIO8 carry the upper eight selector bits.
    # Selector bits:  W2 W1 2+ 1+ = 00 01 11 01
    # Therefore:      0b00011101 << 8
    default_selector = 0b0001110100000000

    dio.outputSet(default_selector)
    dio.configure()
    time.sleep(0.2)

    readback = dio.inputStatus()

    print(f"Digital IO written : {default_selector:016b}")
    print(f"Digital IO readback: {readback:016b}")

    written_upper = (default_selector >> 8) & 0xFF
    readback_upper = (readback >> 8) & 0xFF

    if written_upper == readback_upper:
        print("DIGITAL IO OK - selector bits read back correctly.")
    else:
        print("DIGITAL IO WARNING - readback does not match the "
              "written selector bits.")

    # Return all DIO lines to a safe high-impedance state
    dio.outputSet(0x0000)
    dio.outputEnableSet(0x0000)
    dio.configure()
    print("Digital IO returned to input/high-impedance mode.")


# =========================================================
# MAIN: open once, run all three steps, close once
# =========================================================
def main():
    step1_connect()

    global dev, awg, scope, aio, dio
    dwf = DwfLibrary()
    dev = openDwfDevice(dwf)
    print("Device opened.\n")

    try:
        step2_loopback()
        step3_power_dio()
        print("\n=== ALL SETUP TESTS COMPLETE ===")
    finally:
        close_all()


if __name__ == "__main__":
    main()
