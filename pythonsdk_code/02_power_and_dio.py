"""
Step 2: Power supplies + digital IO sanity (board attached, memristors untouched)

What this validates:
  - AD3 V+ / V- supplies can be enabled from Python and the measured voltages are correct
  - the digital output lines that drive the board's multiplexers respond

Wiring: nothing extra — just have the Discovery board plugged into the AD3.
This script does NOT send any waveform into the memristors.

Run:  python 02_power_and_dio.py
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


from pydwf import DwfLibrary
from pydwf.utilities.open_dwf_device import openDwfDevice


# ---------- Open AD3 ----------
dwf = DwfLibrary()
dev = openDwfDevice(dwf)

print("Device opened.")

try:
    # =========================================================
    # 1. Discover AnalogIO channels and nodes
    # =========================================================
    aio = dev.analogIO

    aio.reset()

    channel_count = aio.channelCount()
    print(f"AnalogIO channels: {channel_count}")

    vplus = None
    vminus = None
    vplus_enable = None
    vminus_enable = None

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

    # =========================================================
    # 2. Test AD3 power supplies
    # =========================================================
    if vplus is None or vminus is None:
        print(
            "\nWARNING: Could not locate V+ and V- nodes automatically."
        )
        print(
            "Use the channel/node list above to identify them."
        )

    else:
        print(f"\nV+ voltage node : channel {vplus[0]}, node {vplus[1]}")
        print(f"V+ enable node  : channel {vplus_enable[0]}, node {vplus_enable[1]}")
        print(f"V- voltage node : channel {vminus[0]}, node {vminus[1]}")
        print(f"V- enable node  : channel {vminus_enable[0]}, node {vminus_enable[1]}")

        # Check the settable voltage range first (sanity)
        vmin, vmax, _ = aio.channelNodeSetInfo(vplus[0], vplus[1])
        print(f"V+ settable range: {vmin} .. {vmax} V")

        # Set the requested supply voltages
        aio.channelNodeSet(
            vplus[0],
            vplus[1],
            5.0,
        )

        aio.channelNodeSet(
            vminus[0],
            vminus[1],
            -5.0,
        )

        # ★ Enable each supply channel's own Enable node (the real switch).
        #   Writing the Voltage node alone does NOT power the rail.
        aio.channelNodeSet(vplus_enable[0], vplus_enable[1], 1.0)
        aio.channelNodeSet(vminus_enable[0], vminus_enable[1], 1.0)

        # Master enable + apply the AnalogIO configuration
        aio.enableSet(True)
        aio.configure()

        time.sleep(1.0)

        aio.status()

        measured_plus = aio.channelNodeStatus(
            vplus[0],
            vplus[1],
        )

        measured_minus = aio.channelNodeStatus(
            vminus[0],
            vminus[1],
        )

        print(
            f"\nV+ set to +5.0 V -> measured "
            f"{measured_plus:+.3f} V"
        )

        print(
            f"V- set to -5.0 V -> measured "
            f"{measured_minus:+.3f} V"
        )

        power_ok = (
            abs(measured_plus - 5.0) < 0.3
            and abs(measured_minus + 5.0) < 0.3
        )

        if power_ok:
            print(
                "POWER OK - the board is receiving the expected rails."
            )
        else:
            print(
                "POWER PROBLEM - check the Discovery board power switch "
                "and board connection."
            )

        # Always switch supplies off after the test
        if vplus_enable is not None:
            aio.channelNodeSet(vplus_enable[0], vplus_enable[1], 0.0)
        if vminus_enable is not None:
            aio.channelNodeSet(vminus_enable[0], vminus_enable[1], 0.0)
        aio.enableSet(False)
        aio.configure()

        print("AnalogIO supplies disabled.")

    # =========================================================
    # 3. Digital IO quick check
    # =========================================================
    print("\nTesting digital IO...")

    dio = dev.digitalIO

    dio.reset()

    # Configure all 16 DIO lines as outputs
    dio.outputEnableSet(0xFFFF)

    # DIO15..DIO8 carry the upper eight selector bits.
    #
    # Selector bits:
    #   W2 W1 2+ 1+ = 00 01 11 01
    #
    # Therefore:
    #   0b00011101 << 8
    #
    default_selector = 0b0001110100000000

    dio.outputSet(default_selector)
    dio.configure()

    time.sleep(0.2)

    readback = dio.inputStatus()

    print(
        f"Digital IO written : {default_selector:016b}"
    )

    print(
        f"Digital IO readback: {readback:016b}"
    )

    written_upper = (default_selector >> 8) & 0xFF
    readback_upper = (readback >> 8) & 0xFF

    if written_upper == readback_upper:
        print("DIGITAL IO OK - selector bits read back correctly.")
    else:
        print(
            "DIGITAL IO WARNING - readback does not match the "
            "written selector bits."
        )

    # Return all DIO lines to a safe high-impedance state
    dio.outputSet(0x0000)
    dio.outputEnableSet(0x0000)
    dio.configure()

    print("Digital IO returned to input/high-impedance mode.")

finally:
    # Safety cleanup
    try:
        if "aio" in locals():
            if vplus_enable is not None:
                aio.channelNodeSet(vplus_enable[0], vplus_enable[1], 0.0)
            if vminus_enable is not None:
                aio.channelNodeSet(vminus_enable[0], vminus_enable[1], 0.0)
            aio.enableSet(False)
            aio.configure()
    except Exception:
        pass

    try:
        if "dio" in locals():
            dio.outputSet(0x0000)
            dio.outputEnableSet(0x0000)
            dio.configure()
    except Exception:
        pass

    dev.close()
    print("\nDevice closed.")