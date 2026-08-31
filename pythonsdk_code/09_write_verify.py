"""
Step 9: WRITE-VERIFY conductance tuning on ONE memristor.

    read G  ->  compare to target  ->  apply 1 correction pulse
            (SET if G too low, RESET if G too high)  ->  read again  ->  ...

Pulse rule (verified on this board):
    SET   = NEGATIVE W1 pulse  ->  conductance INCREASES (toward LRS)
    RESET = POSITIVE W1 pulse  ->  conductance DECREASES (toward HRS)


"""

import os
import time
import csv

# =========================================================
# ===== CONTROL PANEL =====  
# =========================================================

MEM = 6
# Which memristor to tune: 1..16.
#   M1..M8  -> A branch (scope 1+)
#   M9..M16 -> B branch (scope 2+)
# NOTE: M2 and M11 are DEAD (never move). M16 is sticky. Avoid them.

TARGET_G_US = 120.0
# Target conductance you want to "program" the device to (microSiemens).
# e.g. 120.0 means you are trying to write the matrix element 120 uS.

TOLERANCE_US = 1.0
# Absolute tolerance band. Loop stops when |G - TARGET| <= this (uS).
# Smaller = more precise but needs more pulses and risks overshoot.

REL_TOLERANCE = 0.02
# Relative tolerance (fraction of target). Whichever of TOLERANCE_US or
# REL_TOLERANCE*TARGET is LARGER is the effective stop condition.

MAX_ITERATIONS = 50
# Safety cap: maximum number of write-verify rounds.

# =========================================================
# Adaptive step sizing 
# =========================================================
# The loop fires ONE tiny PROBE pulse first, measures how much it moved the
# device (uS), and from that derives a gain estimate. Every later correction
# pulse is sized as:  width = probe_width * (remaining_error / probe_move)
# so the pulse is just big enough to close the gap. Near the target the gap
# shrinks -> the pulse shrinks -> it lands inside the band instead of
# overshooting. Voltage stays at the small safe PROBE level unless a pulse is
# width-capped AND still far away, in which case amplitude is boosted (capped
# at MAX_PULSE_V) to close the distance faster. The gain estimate is refined
# each round (EMA) so drift/wear is tracked automatically.

PROBE_PULSE_V = 0.30       # tiny probe amplitude (V) — used for ALL pulses
PROBE_PULSE_US = 200       # tiny probe width (us) — learns the device gain

MAX_PULSE_V = 1.8         # NEVER fire above this voltage (safety cap)
MAX_PULSE_US = 5000        # NEVER fire longer than this width (safety cap)
MIN_PULSE_US = 50          # don't bother applying a pulse shorter than this

GAIN_SMOOTH = 0.5          # EMA factor for the gain estimate (0=stale,1=instant)

NO_PROGRESS_ITERS = 25
# Saturation guard: if the error has NOT improved for this many consecutive
# rounds, the device is stuck (dead / at LRS or HRS limit) -> stop and warn.

# =========================================================
# Phase 2: settle-and-lock (treats SDC filament relaxation)
# =========================================================
# Phase 1 only guarantees the FRESH (just-pulsed, "hot") value is in band.
# Silver-chalcogenide filaments relax after a pulse, so the value you read
# seconds later -- the value you actually USE -- drifts. Phase 2 fixes this:
#   1. wait SETTLE_LOCK_S for the filament to relax,
#   2. measure the relaxed (HELD) value,
#   3. re-program to target + OVERSHOOT_GAIN * (observed relaxation) so it
#      relaxes BACK onto the target,
#   4. fire one decisive LOCK_PULSE_V / LOCK_PULSE_US pulse in the same
#      direction to form a robust filament sitting away from the switching
#      edge (smaller FUTURE relaxation). Repeat up to LOCK_RETRIES times.
# Set LOCK_RETRIES = 0 to skip Phase 2 (revert to fresh-value behaviour).

LOCK_RETRIES = 1          # relax/re-program cycles to attempt (0 = disable Phase 2)
SETTLE_LOCK_S = 10.0       # seconds to let the filament relax between checks
OVERSHOOT_GAIN = 1.0      # over-program this fraction of the observed relaxation
LOCK_PULSE_V = 0.6        # decisive higher-V pulse to harden the filament
LOCK_PULSE_US = 200       # width of that lock pulse

# --- Fixed measurement settings (usually leave alone) ---
READ_V = -0.10          # sub-threshold DC read voltage (does not disturb state)
RS = 20_000.0          # series resistance (Ohm)
SAMPLE_RATE = 500_000
N_SAMPLES = 8192
SETTLE_S = 0.05         # settle time after switching DIO / changing W1

SHOW_PLOT = True

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
# Hardware state (filled in by the main block / reusable caller)
# =========================================================
dev = None
awg = None
scope = None
dio = None
aio = None

# =========================================================
# Low-level hardware helpers (same pattern as 03 / 04)
# =========================================================

def set_w1_voltage(voltage):
    """Drive W1 to a DC level."""
    awg.nodeFunctionSet(0, DwfAnalogOutNode.Carrier, DwfAnalogOutFunction.DC)
    awg.nodeAmplitudeSet(0, DwfAnalogOutNode.Carrier, 0.0)
    awg.nodeOffsetSet(0, DwfAnalogOutNode.Carrier, voltage)
    awg.runSet(0, 100.0)
    awg.idleSet(0, DwfAnalogOutIdle.Offset)
    awg.triggerSourceSet(0, DwfTriggerSource.None_)
    awg.nodeEnableSet(0, DwfAnalogOutNode.Carrier, True)
    awg.configure(0, True)


def select_memristor(dio_bit):
    """One-hot DIO select for memristor with the given DIO bit (0..15)."""
    dio.outputSet(1 << dio_bit)
    dio.configure()


def turn_off_all_memristors():
    dio.outputSet(0x0000)
    dio.configure()


def apply_pulse(voltage, width_seconds):
    """Apply a single W1 pulse, then return W1 to 0 V."""
    set_w1_voltage(voltage)
    start = time.perf_counter()
    while time.perf_counter() - start < width_seconds:
        pass
    set_w1_voltage(0.0)


def capture_scope_average():
    """Capture both scope channels, return (mean_ch1, mean_ch2)."""
    scope.configure(False, True)
    while True:
        status = scope.status(True)
        if status == DwfState.Done:
            break
        time.sleep(0.001)
    n = scope.statusSamplesValid()
    if n <= 0:
        raise RuntimeError("No oscilloscope samples captured.")
    v1 = float(np.mean(scope.statusData(0, n)))
    v2 = float(np.mean(scope.statusData(1, n)))
    return v1, v2


def calculate_conductance(node_voltage):
    """Conductance from node voltage (uS). nan if below noise floor (HRS)."""
    if abs(node_voltage) < 0.002:
        return float("nan")
    g = (READ_V - node_voltage) / (RS * node_voltage)   # Siemens
    return g * 1e6


# =========================================================
# WRITE-VERIFY CORE
# =========================================================
def write_verify(
    mem,
    target_g_us,
    tol_us=TOLERANCE_US,
    rel_tol=REL_TOLERANCE,
    max_iter=MAX_ITERATIONS,
    no_progress_iters=NO_PROGRESS_ITERS,
    probe_v=PROBE_PULSE_V,
    probe_us=PROBE_PULSE_US,
    max_v=MAX_PULSE_V,
    max_us=MAX_PULSE_US,
    min_us=MIN_PULSE_US,
    gain_smooth=GAIN_SMOOTH,
    lock_retries=LOCK_RETRIES,
    settle_lock_s=SETTLE_LOCK_S,
    overshoot_gain=OVERSHOOT_GAIN,
    lock_pulse_v=LOCK_PULSE_V,
    lock_pulse_us=LOCK_PULSE_US,
):
    """
    Drive memristor `mem` (1..16) to conductance `target_g_us` (uS).

    Phase 1 (adaptive) converges the FRESH value to the band. Phase 2
    (settle-and-lock) then makes the HELD (post-relaxation) value equal the
    target, so the conductance you walk away with is the one you use later.

    Returns (log, final_g, reached) where:
        log      : list of (iter, action, pulse_v, pulse_us, g_us, err_us)
                   (Phase-2 rows are tagged "relax"/"LOCK")
        final_g  : last measured (relaxed/locked) conductance (uS, may be nan)
        reached  : True if the HELD value is within tolerance

    Requires the device + awg/scope/dio/aio to already be open (globals).
    Does NOT touch device open/close, so it is safe to call repeatedly.
    """
    if not 1 <= mem <= 16:
        raise ValueError("mem must be 1..16")
    dio_bit = mem - 1
    is_a = mem <= 8

    # ---- select device, measure baseline offset at W1 = 0 ----
    select_memristor(dio_bit)
    set_w1_voltage(0.0)
    time.sleep(SETTLE_S)
    v1b, v2b = capture_scope_average()
    baseline = v1b if is_a else v2b

    def read_g():
        set_w1_voltage(READ_V)
        time.sleep(SETTLE_S)
        v1, v2 = capture_scope_average()
        node = (v1 - baseline) if is_a else (v2 - baseline)
        return calculate_conductance(node)

    stop_band = max(tol_us, rel_tol * target_g_us)

    log = []
    g = read_g()
    g_eff = 0.0 if not np.isfinite(g) else g
    log.append((0, "init", 0.0, 0, g, target_g_us - g_eff))
    print(f"  init G = {g_eff:.3f} uS (target {target_g_us:.1f} uS, "
          f"band +-{stop_band:.3f})")

    if abs(target_g_us - g_eff) <= stop_band:
        print("  already in band -> nothing to do.")
        return log, g, True

    # ---- PROBE: learn how much ONE small pulse moves THIS device ----
    # err>0 means G too low -> need to RAISE G -> SET (negative W1 pulse).
    # err<0 means G too high -> need to LOWER G -> RESET (positive W1 pulse).
    v_sign0 = -1.0 if (target_g_us - g_eff) > 0 else +1.0
    apply_pulse(v_sign0 * probe_v, probe_us * 1e-6)
    g1 = read_g()
    g1_eff = 0.0 if not np.isfinite(g1) else g1
    dG_probe = g1_eff - g_eff
    log.append((1, "PROBE", v_sign0 * probe_v, probe_us, g1,
                target_g_us - g1_eff))
    print(f"  probe {v_sign0 * probe_v:+5.2f}V {probe_us:4d}us "
          f"-> dG = {dG_probe:+8.3f} uS")
    if abs(dG_probe) < 1e-6:
        print("  WARNING: probe moved ~0 uS -> device may be stuck/dead.")
        gain_mag = 1e-4      # treat as near-zero gain -> wide pulses
    else:
        gain_mag = abs(dG_probe) / probe_us   # uS of movement per us of width
    g_eff = g1_eff

    best_err = abs(target_g_us - g_eff)
    no_progress = 0
    reached = False

    for it in range(2, max_iter + 1):
        err = target_g_us - g_eff
        abs_err = abs(err)

        if abs_err <= stop_band:
            reached = True
            print(f"  target reached: G = {g_eff:.3f} uS "
                  f"(|err| = {abs_err:.3f} <= {stop_band:.3f})")
            break

        # direction for this round (may flip if we overshot last pulse)
        v_sign = -1.0 if err > 0 else +1.0

        # How many probe-widths of pulse energy close the remaining gap?
        #   scale = remaining_error / (gain_mag * probe_us)
        scale = abs_err / (gain_mag * probe_us)
        w_us = probe_us * scale

        # clamp to safe min/max width
        if w_us < min_us:
            w_us = min_us
        elif w_us > max_us:
            w_us = max_us

        # keep voltage at the small safe probe level by default; only boost
        # amplitude (capped at max_v) when width is maxed AND still far away.
        v = v_sign * probe_v
        if (w_us >= max_us and abs(err) > stop_band * 1.5
                and abs(probe_v) < max_v):
            needed_v = probe_v * min(abs(err) / (gain_mag * max_us),
                                     max_v / probe_v)
            v = v_sign * min(needed_v, max_v)

        apply_pulse(v, w_us * 1e-6)
        g = read_g()
        g_new = 0.0 if not np.isfinite(g) else g
        dG = g_new - g_eff
        action = "SET" if v < 0 else "RESET"
        log.append((it, action, v, w_us, g, target_g_us - g_new))
        print(f"  iter {it:2d} {action:5s} {v:+5.2f}V {w_us:6.0f}us "
              f"-> G = {g_new:8.3f} uS  err = {target_g_us - g_new:+8.3f}")

        # refine gain estimate (EMA) so drift/wear is tracked
        if abs(dG) > 1e-9 and w_us > 0:
            inst = abs(dG) / w_us
            gain_mag = gain_smooth * inst + (1.0 - gain_smooth) * gain_mag

        g_eff = g_new

        # saturation / no-progress guard
        new_err = abs(target_g_us - g_eff)
        if new_err < best_err - 1e-6:
            best_err = new_err
            no_progress = 0
        else:
            no_progress += 1
        if no_progress >= no_progress_iters:
            print(f"  WARNING: no progress for {no_progress_iters} rounds "
                  f"-> device likely saturated/dead. Stopping.")
            break

    # =====================================================
    # PHASE 2: anchor the HELD (post-relaxation) value to target.
    # Phase 1 only guarantees the FRESH value is in band. We now let the
    # filament relax, measure it, and re-program to (target + relaxation) so
    # it relaxes back onto target, then fire a decisive lock pulse to harden.
    # =====================================================
    g_final = g_eff
    lock_reached = reached
    prev_relaxed_err = None   # error of the PREVIOUS relaxed reading (None = first)
    in_band = reached         # did any relaxed reading land in band?

    # NOTE on the no-progress guard: it is judged against the PREVIOUS relaxed
    # reading, never against the fresh Phase-1 value. The first relax reading
    # is ALWAYS worse than the just-pulsed fresh value -- that IS the relaxation
    # we are measuring. Comparing it to the Phase-1 error would falsely report
    # "device stuck" and bail before the first re-program ever runs.
    for lock_try in range(1, lock_retries + 1):
        time.sleep(settle_lock_s)
        g_rel = read_g()
        g_rel_eff = 0.0 if not np.isfinite(g_rel) else g_rel
        rel_err = target_g_us - g_rel_eff
        g_final = g_rel_eff   # the value we actually walk away with is the held one
        log.append((f"L{lock_try}", "relax", 0.0, 0, g_rel, rel_err))
        print(f"  [lock {lock_try}] after {settle_lock_s:g}s relax -> "
              f"G = {g_rel_eff:8.3f} uS  err = {rel_err:+8.3f}")

        # already in band -> the held value is good, we are done
        if abs(rel_err) <= stop_band:
            lock_reached = True
            in_band = True
            print("  held value in band -> LOCKED.")
            break

        # no-progress guard: only from the 2nd cycle, and only vs the previous
        # relaxed reading (the correct reference). If a re-program+lock did NOT
        # move the held value closer to target, the device is not converging.
        if prev_relaxed_err is not None and abs(rel_err) >= abs(prev_relaxed_err) - 1e-6:
            print("  WARNING: relaxed value not improving across lock tries "
                  "-> device unlikely to converge. Stopping Phase 2.")
            break
        prev_relaxed_err = rel_err

        # re-program to compensate the observed relaxation (over-program)
        over_target = target_g_us + rel_err * overshoot_gain
        over_err = over_target - g_rel_eff
        v_sign = -1.0 if over_err > 0 else +1.0
        scale = abs(over_err) / (gain_mag * probe_us)
        w_us = probe_us * scale
        if w_us < min_us:
            w_us = min_us
        elif w_us > max_us:
            w_us = max_us
        v = v_sign * probe_v
        if (w_us >= max_us and abs(over_err) > stop_band * 1.5
                and abs(probe_v) < max_v):
            needed_v = probe_v * min(abs(over_err) / (gain_mag * max_us),
                                     max_v / probe_v)
            v = v_sign * min(needed_v, max_v)
        apply_pulse(v, w_us * 1e-6)
        g_a = read_g()
        g_a_eff = 0.0 if not np.isfinite(g_a) else g_a
        log.append((f"L{lock_try}", "SET" if v < 0 else "RESET", v, w_us, g_a,
                    target_g_us - g_a_eff))
        print(f"  [lock {lock_try}] re-program {v:+5.2f}V {w_us:6.0f}us "
              f"-> G = {g_a_eff:8.3f} uS")

        # decisive lock pulse: same direction as the correction just applied,
        # forms a robust filament sitting away from the switching edge.
        if lock_pulse_us > 0:
            v_lock = -lock_pulse_v if over_err > 0 else +lock_pulse_v
            apply_pulse(v_lock, lock_pulse_us * 1e-6)
            g_l = read_g()
            g_l_eff = 0.0 if not np.isfinite(g_l) else g_l
            log.append((f"L{lock_try}", "LOCK", v_lock, lock_pulse_us, g_l,
                        target_g_us - g_l_eff))
            print(f"  [lock {lock_try}] lock pulse {v_lock:+5.2f}V "
                  f"{lock_pulse_us:5d}us -> G = {g_l_eff:8.3f} uS")
            g_final = g_l_eff
        else:
            g_final = g_a_eff

    # If Phase 2 ran relax cycles but never caught the value in band, do ONE
    # final relax-read so the reported G is the TRUE held value (not the
    # just-pulsed fresh value, which would drift again).
    if lock_retries > 0 and not in_band:
        time.sleep(settle_lock_s)
        g_h = read_g()
        g_h_eff = 0.0 if not np.isfinite(g_h) else g_h
        herr = target_g_us - g_h_eff
        log.append(("hold", "relax", 0.0, 0, g_h, herr))
        print(f"  [hold] final held G = {g_h_eff:8.3f} uS  err = {herr:+8.3f}")
        g_final = g_h_eff
        lock_reached = abs(herr) <= stop_band

    return log, g_final, lock_reached


# =========================================================
# Main: open hardware, run, plot, log
# =========================================================
if __name__ == "__main__":
    try:
        print(f"=== Write-verify tuning M{MEM} -> {TARGET_G_US} uS ===")

        dwf = DwfLibrary()
        dev = openDwfDevice(dwf)
        print("Device opened.")

        # power rails
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
        vp = aio.channelNodeStatus(0, 1)
        vm = aio.channelNodeStatus(1, 1)
        print(f"Supplies: V+ = {vp:+.3f} V, V- = {vm:+.3f} V")
        if abs(vp - 5.0) > 0.5 or abs(vm + 5.0) > 0.5:
            print("WARNING: supplies out of range.")

        # AWG
        awg = dev.analogOut
        awg.reset(0)
        set_w1_voltage(0.0)

        # scope
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

        # DIO
        dio = dev.digitalIO
        dio.reset()
        dio.outputEnableSet(0xFFFF)
        turn_off_all_memristors()

        # run the tuning
        log, final_g, reached = write_verify(MEM, TARGET_G_US)

        # ---- plot ----
        # positional index for x: Phase 2 rows are tagged "L1"/"L2" strings,
        # which matplotlib.scatter cannot convert to float.
        iters = list(range(len(log)))
        gs = [r[4] if np.isfinite(r[4]) else 0.0 for r in log]
        colors = ["green" if r[1] == "init" else
                  ("#2471A3" if r[1] == "SET" else "#C0392B") for r in log]

        stop_band = max(TOLERANCE_US, REL_TOLERANCE * TARGET_G_US)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.axhspan(TARGET_G_US - stop_band, TARGET_G_US + stop_band,
                   color="green", alpha=0.12, label=f"tolerance +-{stop_band:.2f} uS")
        ax.axhline(TARGET_G_US, color="gray", ls="--", alpha=0.7,
                   label=f"target {TARGET_G_US:g} uS")
        ax.scatter(iters, gs, c=colors, s=60, zorder=5)
        ax.plot(iters, gs, "-", color="gray", alpha=0.4, lw=1)
        ax.set_xlabel("write-verify iteration")
        ax.set_ylabel("Conductance (uS)")
        ax.set_title(f"M{MEM} write-verify -> target {TARGET_G_US} uS  "
                     f"({'REACHED' if reached else 'NOT REACHED'})")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_dir = os.path.dirname(os.path.abspath(__file__))
        plot_path = os.path.join(out_dir, f"writeverify_M{MEM}_{TARGET_G_US:g}uS.png")
        fig.savefig(plot_path, dpi=120)
        print(f"\nPlot saved: {plot_path}")
        print(f"Final G = {final_g if np.isfinite(final_g) else 'nan'} uS  "
              f"reached = {reached}")

        if SHOW_PLOT:
            plt.show()
        else:
            plt.close(fig)

        # ---- CSV ----
        csv_path = os.path.join(out_dir, time.strftime(
            f"writeverify_M{MEM}_{TARGET_G_US:g}uS_%Y%m%d_%H%M%S.csv"))
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "action", "pulse_v", "pulse_us",
                        "G_uS", "error_uS"])
            for r in log:
                w.writerow([r[0], r[1], r[2], r[3],
                            ("" if not np.isfinite(r[4]) else f"{r[4]:.6f}"),
                            ("" if not np.isfinite(r[5]) else f"{r[5]:.6f}")])
        print(f"CSV saved: {csv_path}")

    finally:
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
