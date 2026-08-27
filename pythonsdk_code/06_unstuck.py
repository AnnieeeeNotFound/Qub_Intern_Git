"""
Step 6: Recover a stuck memristor.

Edit the USER CONTROL PANEL below, then press Run.

Methods:
    lrs:
        For a memristor stuck at high conductance.
        Uses a 10 Hz sine wave with +2 V offset.

    hrs:
        For a memristor stuck at low conductance.
        Uses a 10 Hz sine wave with -2 V offset.

    last_resort:
        Uses sustained DC bias and requires manually shorting
        the 20 kOhm resistor. This method may damage the device.

    adaptive:
        For a "bouncy" memristor that drops to low conductance
        under treatment but immediately grows back. Monitors G
        in real time; the instant G falls below STOP_G_US the
        treatment is stopped, the device is left unbiased for
        2 s, and G is re-read. If it bounced back, retries with
        a slightly higher offset.
"""

import os
import time

# =========================================================
# USER CONTROL PANEL
# =========================================================

TARGET_MEMRISTOR = 11
# Choose M1 to M16

METHOD = "lrs"
# Choose:
# "adaptive"
# "lrs"
# "hrs"
# "last_resort"

CUSTOM_OFFSET_V = 2.5
# None = automatic offset:
# adaptive: +3.0 V
# lrs: +3.0 V
# hrs: -2.0 V
#
# To manually choose an offset, use for example:
# CUSTOM_OFFSET_V = 1.5

SINE_AMPLITUDE_V = 1.0
# Used by adaptive, lrs and hrs methods

SINE_FREQUENCY_HZ = 10.0
# Used by adaptive, lrs and hrs methods

HOLD_SECONDS = 10.0
# Treatment duration per attempt

STOP_G_US = 50.0
# adaptive only: stop treatment the moment G falls below this.

ADAPTIVE_ATTEMPTS = 5
# adaptive only: maximum number of stop-and-check attempts.
# Offset escalates by +0.25 V per attempt (capped at +3.5 V).

LAST_RESORT_OFFSET_V = -1.0
# Used only when METHOD = "last_resort"

SHOW_PLOT_WINDOW = True


# =========================================================
# Validate control panel
# =========================================================

if not 1 <= TARGET_MEMRISTOR <= 16:
    raise ValueError(
        "TARGET_MEMRISTOR must be between 1 and 16."
    )

if METHOD not in (
    "adaptive",
    "lrs",
    "hrs",
    "last_resort",
):
    raise ValueError(
        'METHOD must be "adaptive", "lrs", "hrs", or "last_resort".'
    )

if SINE_AMPLITUDE_V <= 0:
    raise ValueError(
        "SINE_AMPLITUDE_V must be positive."
    )

if SINE_FREQUENCY_HZ <= 0:
    raise ValueError(
        "SINE_FREQUENCY_HZ must be positive."
    )

if HOLD_SECONDS <= 0:
    raise ValueError(
        "HOLD_SECONDS must be positive."
    )


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

from pydwf import (
    DwfLibrary,
    DwfAcquisitionMode,
    DwfAnalogOutFunction,
    DwfAnalogOutIdle,
    DwfAnalogOutNode,
    DwfState,
    DwfTriggerSource,
)

from pydwf.utilities.open_dwf_device import (
    openDwfDevice,
)


# =========================================================
# Measurement settings
# =========================================================

READ_V = -0.10
SERIES_RESISTANCE = 20_000.0

READ_SAMPLE_RATE = 1_000_000
READ_N_SAMPLES = 1024
READ_SETTLE_TIME_S = 0.05

V_PLUS_SET = +5.0
V_MINUS_SET = -5.0


# =========================================================
# Hardware helper functions
# =========================================================

def find_supply_nodes(aio):
    """Find Positive Supply and Negative Supply channels."""

    channel_count = aio.channelCount()

    positive_supply = None
    negative_supply = None

    for channel in range(channel_count):
        name, _ = aio.channelName(channel)
        name = str(name)

        if "Positive" in name:
            positive_supply = (
                channel,
                0,
                1,
            )

        elif "Negative" in name:
            negative_supply = (
                channel,
                0,
                1,
            )

    return positive_supply, negative_supply


def enable_supplies(aio):
    """Enable the AD3 +5 V and -5 V rails."""

    positive, negative = find_supply_nodes(aio)

    if positive is None or negative is None:
        raise RuntimeError(
            "Could not locate AD3 power-supply channels."
        )

    positive_channel, positive_enable, positive_voltage = positive
    negative_channel, negative_enable, negative_voltage = negative

    aio.channelNodeSet(
        positive_channel,
        positive_voltage,
        V_PLUS_SET,
    )

    aio.channelNodeSet(
        positive_channel,
        positive_enable,
        1.0,
    )

    aio.channelNodeSet(
        negative_channel,
        negative_voltage,
        V_MINUS_SET,
    )

    aio.channelNodeSet(
        negative_channel,
        negative_enable,
        1.0,
    )

    aio.enableSet(True)
    aio.configure()

    time.sleep(0.5)
    aio.status()

    measured_plus = aio.channelNodeStatus(
        positive_channel,
        positive_voltage,
    )

    measured_minus = aio.channelNodeStatus(
        negative_channel,
        negative_voltage,
    )

    print(
        f"Supplies: V+ = {measured_plus:+.3f} V, "
        f"V- = {measured_minus:+.3f} V"
    )


def disable_supplies(aio):
    """Disable both AD3 power rails."""

    if aio is None:
        return

    positive, negative = find_supply_nodes(aio)

    for supply in (positive, negative):
        if supply is None:
            continue

        channel = supply[0]
        enable_node = supply[1]

        try:
            aio.channelNodeSet(
                channel,
                enable_node,
                0.0,
            )
        except Exception:
            pass

    try:
        aio.enableSet(False)
        aio.configure()
    except Exception:
        pass


def select_memristor(dio, dio_mask):
    """Select one memristor using one-hot DIO control."""

    dio.outputEnableSet(0xFFFF)
    dio.outputSet(dio_mask)
    dio.configure()


def release_memristor(dio):
    """Turn off all memristor switches."""

    if dio is None:
        return

    try:
        dio.outputSet(0x0000)
        dio.outputEnableSet(0x0000)
        dio.configure()
    except Exception:
        pass


def configure_awg_dc(awg, voltage):
    """Configure W1 as a DC source."""

    awg.reset(0)

    awg.nodeEnableSet(
        0,
        DwfAnalogOutNode.Carrier,
        True,
    )

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

    awg.runSet(0, 5.0)

    awg.idleSet(
        0,
        DwfAnalogOutIdle.Offset,
    )

    awg.triggerSourceSet(
        0,
        DwfTriggerSource.None_,
    )

    awg.configure(0, True)


def read_one_conductance(
    awg,
    scope,
    dio,
    dio_mask,
    memristor_number,
):
    """Read one memristor conductance in Siemens.

    A physically impossible (negative) conductance means the
    scope captured a transient while W1 was still slewing from
    a previous voltage (e.g. +3 V treatment) to the -0.1 V read
    level. Retry with progressively longer settle time.
    """

    scope_channel = (
        0
        if memristor_number <= 8
        else 1
    )

    conductance = 0.0

    for attempt in range(3):

        configure_awg_dc(
            awg,
            READ_V,
        )

        select_memristor(
            dio,
            dio_mask,
        )

        time.sleep(
            READ_SETTLE_TIME_S
            + 0.15 * attempt
        )

        scope.reset()
        scope.frequencySet(READ_SAMPLE_RATE)

        scope.channelEnableSet(0, True)
        scope.channelEnableSet(1, True)

        scope.channelRangeSet(0, 5.0)
        scope.channelRangeSet(1, 5.0)

        scope.bufferSizeSet(READ_N_SAMPLES)

        scope.acquisitionModeSet(
            DwfAcquisitionMode.Single
        )

        scope.triggerSourceSet(
            DwfTriggerSource.None_
        )

        scope.configure(False, True)

        while True:
            status = scope.status(True)

            if status == DwfState.Done:
                break

            time.sleep(0.001)

        number_samples = scope.statusSamplesValid()

        if number_samples <= 0:
            voltage_node = 0.0
        else:
            voltage_node = float(
                np.mean(
                    scope.statusData(
                        scope_channel,
                        number_samples,
                    )
                )
            )

        configure_awg_dc(
            awg,
            0.0,
        )

        release_memristor(dio)

        if abs(voltage_node) < 0.0001:
            return 0.0

        candidate = (
            READ_V - voltage_node
        ) / (
            SERIES_RESISTANCE * voltage_node
        )

        conductance = candidate

        if candidate > 0:
            # Physically valid reading.
            break

        # Negative conductance = measurement artifact.
        # Loop again with a longer settle time.

    return conductance


def apply_treatment_sine(
    awg,
    frequency,
    amplitude,
    offset,
):
    """Apply Method 1 using a sine wave."""

    awg.reset(0)

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
        frequency,
    )

    awg.nodeAmplitudeSet(
        0,
        DwfAnalogOutNode.Carrier,
        amplitude,
    )

    awg.nodeOffsetSet(
        0,
        DwfAnalogOutNode.Carrier,
        offset,
    )

    awg.runSet(0, 60.0)

    awg.idleSet(
        0,
        DwfAnalogOutIdle.Offset,
    )

    awg.configure(0, True)


def apply_treatment_dc(
    awg,
    offset,
):
    """Apply Method 2 using a DC voltage."""

    configure_awg_dc(
        awg,
        offset,
    )


def stop_treatment(awg):
    """Stop the waveform generator."""

    if awg is None:
        return

    try:
        awg.configure(0, False)
        awg.reset(0)
    except Exception:
        pass


# =========================================================
# Main program
# =========================================================

def main():
    global dev
    global aio
    global awg
    global scope
    global dio

    memristor_number = TARGET_MEMRISTOR
    memristor_index = memristor_number - 1
    dio_mask = 1 << memristor_index

    if CUSTOM_OFFSET_V is not None:
        offset = CUSTOM_OFFSET_V
    elif METHOD == "adaptive":
        offset = +3.0
    elif METHOD == "lrs":
        offset = +2.0
    elif METHOD == "hrs":
        offset = -2.0
    else:
        offset = LAST_RESORT_OFFSET_V

    print("Starting Step 6 recovery test.")
    print(f"Target memristor: M{memristor_number}")
    print(f"DIO bit: {memristor_index}")
    print(f"Method: {METHOD}")
    print(f"Offset: {offset:+.2f} V")

    dwf = DwfLibrary()
    dev = openDwfDevice(dwf)

    awg = dev.analogOut
    scope = dev.analogIn
    dio = dev.digitalIO
    aio = dev.analogIO

    conductance_history = []

    try:
        print("\nDevice opened.")

        # -------------------------------------------------
        # Power rails
        # -------------------------------------------------

        enable_supplies(aio)

        # -------------------------------------------------
        # Initial conductance
        # -------------------------------------------------

        print(
            f"\nReading initial conductance of M"
            f"{memristor_number}..."
        )

        initial_g = read_one_conductance(
            awg,
            scope,
            dio,
            dio_mask,
            memristor_number,
        )

        conductance_history.append(
            (0.0, initial_g)
        )

        print(
            f"Initial conductance: "
            f"{initial_g * 1e6:.3f} uS"
        )

        # -------------------------------------------------
        # Select target
        # -------------------------------------------------

        select_memristor(
            dio,
            dio_mask,
        )

        # -------------------------------------------------
        # Apply recovery treatment
        # -------------------------------------------------

        if METHOD == "adaptive":

            print(
                f"\nADAPTIVE MODE: stop the moment "
                f"G < {STOP_G_US:.0f} uS"
            )

            print(
                f"Max attempts: {ADAPTIVE_ATTEMPTS} "
                f"(offset escalates +0.25 V each, "
                f"capped at +3.5 V)"
            )

            cumulative_time = 0.0
            adaptive_success = False
            monitor_interval = 0.4

            for attempt in range(
                1,
                ADAPTIVE_ATTEMPTS + 1,
            ):

                attempt_offset = min(
                    offset + 0.25 * (attempt - 1),
                    3.5,
                )

                print(
                    f"\n--- Attempt {attempt}/"
                    f"{ADAPTIVE_ATTEMPTS} "
                    f"(offset {attempt_offset:+.2f} V) ---"
                )

                apply_treatment_sine(
                    awg,
                    SINE_FREQUENCY_HZ,
                    SINE_AMPLITUDE_V,
                    attempt_offset,
                )

                select_memristor(
                    dio,
                    dio_mask,
                )

                hit_low = False

                start_time = time.perf_counter()

                while True:
                    elapsed = (
                        time.perf_counter()
                        - start_time
                    )

                    if elapsed >= HOLD_SECONDS:
                        break

                    time.sleep(monitor_interval)

                    measured_g = (
                        read_one_conductance(
                            awg,
                            scope,
                            dio,
                            dio_mask,
                            memristor_number,
                        )
                    )

                    conductance_history.append(
                        (
                            cumulative_time
                            + elapsed,
                            measured_g,
                        )
                    )

                    print(
                        f"Time: "
                        f"{cumulative_time + elapsed:5.1f} s"
                        f"    G: "
                        f"{measured_g * 1e6:10.3f} uS"
                    )

                    if (
                        0.0
                        < measured_g * 1e6
                        < STOP_G_US
                    ):
                        hit_low = True
                        break

                    apply_treatment_sine(
                        awg,
                        SINE_FREQUENCY_HZ,
                        SINE_AMPLITUDE_V,
                        attempt_offset,
                    )

                    select_memristor(
                        dio,
                        dio_mask,
                    )

                cumulative_time += (
                    time.perf_counter()
                    - start_time
                )

                stop_treatment(awg)
                release_memristor(dio)

                if not hit_low:
                    print(
                        "Did not reach target "
                        "in this attempt."
                    )

                    continue

                print(
                    f"G dropped below "
                    f"{STOP_G_US:.0f} uS - stopping "
                    f"and leaving unbiased for 2 s..."
                )

                time.sleep(2.0)

                hold_g = read_one_conductance(
                    awg,
                    scope,
                    dio,
                    dio_mask,
                    memristor_number,
                )

                conductance_history.append(
                    (
                        cumulative_time + 2.0,
                        hold_g,
                    )
                )

                print(
                    f"After 2 s unbiased hold: "
                    f"{hold_g * 1e6:.3f} uS"
                )

                if 0.0 < hold_g * 1e6 < STOP_G_US * 2:
                    adaptive_success = True
                    break

                print(
                    "Bounced back - retrying "
                    "with a higher offset."
                )

            print(
                f"\nADAPTIVE RESULT: "
                f"{'SUCCESS' if adaptive_success else 'FAILED'}"
            )

        elif METHOD == "last_resort":
            print(
                "\nWARNING: LAST-RESORT METHOD SELECTED."
            )

            print(
                "This method may permanently damage "
                "the memristor."
            )

            apply_treatment_dc(
                awg,
                offset,
            )

            print(
                f"W1 DC voltage: {offset:+.2f} V"
            )

            print(
                "\nBriefly short the 20 kOhm series "
                "resistor for approximately 0.5-1 second."
            )

            input(
                "\nPress Enter after the short has "
                "been released..."
            )

        else:
            apply_treatment_sine(
                awg,
                SINE_FREQUENCY_HZ,
                SINE_AMPLITUDE_V,
                offset,
            )

            minimum_voltage = (
                offset - SINE_AMPLITUDE_V
            )

            maximum_voltage = (
                offset + SINE_AMPLITUDE_V
            )

            print(
                f"\nSine amplitude: "
                f"{SINE_AMPLITUDE_V:.2f} V"
            )

            print(
                f"Sine frequency: "
                f"{SINE_FREQUENCY_HZ:.2f} Hz"
            )

            print(
                f"W1 voltage range: "
                f"{minimum_voltage:+.2f} V to "
                f"{maximum_voltage:+.2f} V"
            )

            print(
                f"Treatment time: "
                f"{HOLD_SECONDS:.1f} seconds"
            )

        # -------------------------------------------------
        # Monitor conductance (standard methods only;
        # adaptive mode monitors inside its own loop)
        # -------------------------------------------------

        if METHOD != "adaptive":

            start_time = time.perf_counter()
            monitor_interval = 0.5

            while True:
                elapsed = (
                    time.perf_counter()
                    - start_time
                )

                if elapsed >= HOLD_SECONDS:
                    break

                time.sleep(monitor_interval)

                measured_g = read_one_conductance(
                    awg,
                    scope,
                    dio,
                    dio_mask,
                    memristor_number,
                )

                if METHOD == "last_resort":
                    apply_treatment_dc(
                        awg,
                        offset,
                    )
                else:
                    apply_treatment_sine(
                        awg,
                        SINE_FREQUENCY_HZ,
                        SINE_AMPLITUDE_V,
                        offset,
                    )

                select_memristor(
                    dio,
                    dio_mask,
                )

                conductance_history.append(
                    (elapsed, measured_g)
                )

                print(
                    f"Time: {elapsed:5.1f} s    "
                    f"G: {measured_g * 1e6:10.3f} uS"
                )

        # -------------------------------------------------
        # Final reading
        # -------------------------------------------------

        stop_treatment(awg)
        release_memristor(dio)

        time.sleep(1.0)

        print(
            f"\nReading final conductance of M"
            f"{memristor_number}..."
        )

        final_g = read_one_conductance(
            awg,
            scope,
            dio,
            dio_mask,
            memristor_number,
        )

        conductance_history.append(
            (HOLD_SECONDS + 1.0, final_g)
        )

        initial_us = initial_g * 1e6
        final_us = final_g * 1e6

        if abs(initial_g) > 1e-12:
            change_percent = (
                (final_g - initial_g)
                / abs(initial_g)
                * 100
            )
        else:
            change_percent = 0.0

        print(
            f"Initial conductance: {initial_us:.3f} uS"
        )

        print(
            f"Final conductance:   {final_us:.3f} uS"
        )

        print(
            f"Conductance change:   "
            f"{change_percent:+.2f}%"
        )

        # -------------------------------------------------
        # Plot
        # -------------------------------------------------

        times = [
            item[0]
            for item in conductance_history
        ]

        conductances_us = [
            item[1] * 1e6
            for item in conductance_history
        ]

        figure, axis = plt.subplots(
            figsize=(9, 5)
        )

        axis.plot(
            times,
            conductances_us,
            "o-",
            linewidth=1.5,
            markersize=6,
        )

        axis.axhline(
            initial_us,
            color="gray",
            linestyle=":",
            label="Initial conductance",
        )

        axis.scatter(
            times[0],
            conductances_us[0],
            color="green",
            s=100,
            zorder=5,
            label=f"Start: {initial_us:.3f} uS",
        )

        axis.scatter(
            times[-1],
            conductances_us[-1],
            color="black",
            s=100,
            zorder=5,
            label=f"End: {final_us:.3f} uS",
        )

        axis.set_xlabel(
            "Time into treatment (seconds)"
        )

        axis.set_ylabel(
            "Conductance (uS)"
        )

        axis.set_title(
            f"M{memristor_number} recovery test\n"
            f"Method: {METHOD}, "
            f"start = {initial_us:.3f} uS, "
            f"end = {final_us:.3f} uS"
        )

        axis.grid(alpha=0.3)
        axis.legend()
        figure.tight_layout()

        output_directory = os.path.dirname(
            os.path.abspath(__file__)
        )

        plot_path = os.path.join(
            output_directory,
            f"unstuck_M{memristor_number}_{METHOD}.png",
        )

        figure.savefig(
            plot_path,
            dpi=120,
        )

        print(
            "\nPlot saved to:",
            plot_path,
        )

        if SHOW_PLOT_WINDOW:
            plt.show()
        else:
            plt.close(figure)

    finally:
        # -------------------------------------------------
        # Safety shutdown
        # -------------------------------------------------

        stop_treatment(awg)
        release_memristor(dio)
        disable_supplies(aio)

        try:
            if dev is not None:
                dev.close()
        except Exception:
            pass

        print("\nDevice closed.")


if __name__ == "__main__":
    main()