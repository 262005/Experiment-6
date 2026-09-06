"""
Experiment 6 — Digital Line Coding and Power Spectral Density
Software-Based Digital Communication Laboratory

GitHub-ready Python script.
Runs in Google Colab / Jupyter / standard Python environments.

Requirements from the laboratory manual:
- Generate six standard line codes from the same bit sequence
- Do not use toolbox encoders
- Estimate normalized PSD using Welch's method
- Compare DC component, bandwidth, polarity, and self-clocking capability
- Plot aligned waveforms, normalized PSD, running digital sum, and long-run behavior
- Validate an 8-bit word manually against the encoder output
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

# ============================================================
# 1. COMMON PARAMETERS
# ============================================================

BITS = np.array([1, 0, 1, 1, 0, 0, 1, 0])
SPB = 100                  # samples per bit
FS = SPB                   # normalized sampling frequency
AMPLITUDE = 1.0

# Long-run test sequence required by the experiment
LONG_RUN_BITS = np.ones(20, dtype=int)


# ============================================================
# 2. LINE-CODING FUNCTIONS
# ============================================================

def unipolar_nrz(bits, spb=100):
    """1 -> +A, 0 -> 0."""
    bits = np.asarray(bits, dtype=int)
    return np.repeat(bits * AMPLITUDE, spb)


def polar_nrz(bits, spb=100):
    """1 -> +A, 0 -> -A."""
    bits = np.asarray(bits, dtype=int)
    levels = np.where(bits == 1, AMPLITUDE, -AMPLITUDE)
    return np.repeat(levels, spb)


def polar_rz(bits, spb=100):
    """1 -> +A for first half, 0 for second half; 0 -> -A then 0."""
    bits = np.asarray(bits, dtype=int)
    half = spb // 2
    signal = []

    for bit in bits:
        level = AMPLITUDE if bit == 1 else -AMPLITUDE
        signal.extend([level] * half)
        signal.extend([0.0] * (spb - half))

    return np.asarray(signal, dtype=float)


def manchester(bits, spb=100):
    """
    Manchester coding:
    1 -> +A then -A
    0 -> -A then +A
    """
    bits = np.asarray(bits, dtype=int)
    half = spb // 2
    signal = []

    for bit in bits:
        if bit == 1:
            first, second = AMPLITUDE, -AMPLITUDE
        else:
            first, second = -AMPLITUDE, AMPLITUDE

        signal.extend([first] * half)
        signal.extend([second] * (spb - half))

    return np.asarray(signal, dtype=float)


def differential_manchester(bits, spb=100):
    """
    Differential Manchester coding:
    - There is always a transition in the middle of every bit.
    - A 0 causes a transition at the beginning of the bit.
    - A 1 causes no transition at the beginning.

    Initial level is +A.
    """
    bits = np.asarray(bits, dtype=int)
    half = spb // 2
    current = AMPLITUDE
    signal = []

    for bit in bits:
        # Boundary transition for 0
        if bit == 0:
            current *= -1

        # First half
        signal.extend([current] * half)

        # Mandatory mid-bit transition
        current *= -1

        # Second half
        signal.extend([current] * (spb - half))

    return np.asarray(signal, dtype=float)


def ami(bits, spb=100):
    """
    Alternate Mark Inversion:
    0 -> 0
    1 -> alternating +A, -A, +A, -A, ...
    """
    bits = np.asarray(bits, dtype=int)
    last_mark = -AMPLITUDE
    signal = []

    for bit in bits:
        if bit == 0:
            level = 0.0
        else:
            last_mark *= -1
            level = last_mark

        signal.extend([level] * spb)

    return np.asarray(signal, dtype=float)


# ============================================================
# 3. ENCODE THE SAME DATA WITH ALL SIX METHODS
# ============================================================

encoders = {
    "Unipolar NRZ": unipolar_nrz,
    "Polar NRZ": polar_nrz,
    "Polar RZ": polar_rz,
    "Manchester": manchester,
    "Differential Manchester": differential_manchester,
    "AMI": ami,
}

signals = {
    name: encoder(BITS, SPB)
    for name, encoder in encoders.items()
}


# ============================================================
# 4. TIME AXIS
# ============================================================

t = np.arange(len(BITS) * SPB) / FS
bit_centers = (np.arange(len(BITS)) + 0.5) * SPB / FS


# ============================================================
# 5. ALIGNED WAVEFORMS
# ============================================================

fig, axes = plt.subplots(
    len(signals), 1, figsize=(12, 12), sharex=True
)

for ax, (name, signal) in zip(axes, signals.items()):
    ax.plot(t, signal, linewidth=1.5)
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-1.4, 1.4)

    for boundary in np.arange(len(BITS) + 1):
        ax.axvline(boundary, linewidth=0.5, alpha=0.2)

axes[-1].set_xlabel("Bit interval")
fig.suptitle("Experiment 6 — Six Digital Line Codes", fontsize=15)
fig.tight_layout()
plt.show()


# ============================================================
# 6. NORMALIZED PSD USING WELCH'S METHOD
# ============================================================

def normalized_welch_psd(signal, fs=1.0):
    """Return frequency and PSD normalized to a 0 dB maximum."""
    nperseg = min(1024, len(signal))
    frequencies, psd = welch(
        signal,
        fs=fs,
        nperseg=nperseg,
        return_onesided=True,
        scaling="density",
    )

    psd_db = 10 * np.log10(np.maximum(psd, 1e-12))
    psd_db -= np.max(psd_db)

    return frequencies, psd_db


fig, ax = plt.subplots(figsize=(12, 6))

for name, signal in signals.items():
    f, psd_db = normalized_welch_psd(signal, FS)
    ax.plot(f, psd_db, linewidth=1.5, label=name)

ax.set_title("Normalized Power Spectral Density — Welch's Method")
ax.set_xlabel("Normalized frequency")
ax.set_ylabel("Normalized PSD (dB)")
ax.set_ylim(-60, 5)
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()


# ============================================================
# 7. DC / AVERAGE LEVEL AND RUNNING DIGITAL SUM
# ============================================================

def running_sum(signal):
    """Cumulative sum used to visualize DC tendency / imbalance."""
    return np.cumsum(signal)


print("\nAverage level (DC component indicator):")
for name, signal in signals.items():
    print(f"{name:24s}: {np.mean(signal): .4f}")

fig, ax = plt.subplots(figsize=(12, 6))

for name, signal in signals.items():
    rs = running_sum(signal)
    ax.plot(t, rs, linewidth=1.5, label=name)

ax.axhline(0, linewidth=0.8)
ax.set_title("Running Digital Sum")
ax.set_xlabel("Bit interval")
ax.set_ylabel("Cumulative sum")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
plt.show()


# ============================================================
# 8. SIMPLE TRANSITION COUNT / CLOCKING INDICATOR
# ============================================================

def transition_count(signal):
    """Count signal-level changes between adjacent samples."""
    return int(np.count_nonzero(np.diff(signal) != 0))


print("\nTransition count:")
for name, signal in signals.items():
    print(f"{name:24s}: {transition_count(signal)}")


# ============================================================
# 9. LONG RUN OF IDENTICAL BITS
# ============================================================

long_run_signals = {
    name: encoder(LONG_RUN_BITS, SPB)
    for name, encoder in encoders.items()
}

long_t = np.arange(len(LONG_RUN_BITS) * SPB) / FS

fig, axes = plt.subplots(
    len(long_run_signals), 1, figsize=(12, 12), sharex=True
)

for ax, (name, signal) in zip(axes, long_run_signals.items()):
    ax.plot(long_t, signal, linewidth=1.5)
    ax.set_ylabel(name)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(-1.4, 1.4)

axes[-1].set_xlabel("Bit interval")
fig.suptitle("Long Run of Identical Bits — 20 Ones", fontsize=15)
fig.tight_layout()
plt.show()


# ============================================================
# 10. MANDATORY 8-BIT VALIDATION
# ============================================================

validation_bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])

print("\n" + "=" * 65)
print("MANDATORY VALIDATION")
print("=" * 65)
print("Test word:", "".join(map(str, validation_bits)))

# Expected first-symbol patterns for manual checking
print("\nManual transition checks:")
print("Unipolar NRZ: 1 -> +A, 0 -> 0")
print("Polar NRZ:    1 -> +A, 0 -> -A")
print("Polar RZ:     each bit returns to 0 during second half")
print("Manchester:   every bit has one mid-bit transition")
print("Diff. Manchester: every bit has a mid-bit transition;")
print("                  0 adds a boundary transition, 1 does not")
print("AMI:          1s alternate +A/-A; 0 -> 0")

# Direct implementation checks
assert np.allclose(
    unipolar_nrz(validation_bits, SPB)[:SPB],
    AMPLITUDE
)

assert np.allclose(
    polar_nrz(validation_bits, SPB)[:SPB],
    AMPLITUDE
)

assert np.allclose(
    polar_nrz(validation_bits, SPB)[SPB:2 * SPB],
    -AMPLITUDE
)

# Manchester: one always has a mid-bit transition
m = manchester(validation_bits, SPB)
assert m[SPB // 2 - 1] != m[SPB // 2]

# Differential Manchester: every bit has a mid-bit transition
dm = differential_manchester(validation_bits, SPB)
for i in range(len(validation_bits)):
    start = i * SPB
    mid = start + SPB // 2
    assert dm[mid - 1] != dm[mid]

# AMI: consecutive ones alternate polarity
ami_signal = ami(validation_bits, SPB)
ami_levels = []
for i, bit in enumerate(validation_bits):
    if bit == 1:
        ami_levels.append(ami_signal[i * SPB])

for a, b in zip(ami_levels, ami_levels[1:]):
    assert a == -b

print("\nValidation checks passed.")


# ============================================================
# 11. SUMMARY TABLE
# ============================================================

summary = [
    ("Unipolar NRZ", "Positive / zero", "High", "Low", "No"),
    ("Polar NRZ", "Positive / negative", "Low", "Low", "Poor for long runs"),
    ("Polar RZ", "Positive / negative / zero", "Low", "Moderate", "Better"),
    ("Manchester", "Positive / negative", "Very low", "High", "Excellent"),
    ("Differential Manchester", "Positive / negative", "Very low", "High", "Excellent"),
    ("AMI", "Positive / negative / zero", "Very low", "Moderate", "Poor for long zero runs"),
]

print("\n" + "=" * 95)
print("THEORETICAL COMPARISON")
print("=" * 95)
print(
    f"{'Line code':24s} | {'Polarity':24s} | "
    f"{'DC tendency':12s} | {'Bandwidth':10s} | {'Self-clocking'}"
)
print("-" * 95)

for row in summary:
    print(
        f"{row[0]:24s} | {row[1]:24s} | "
        f"{row[2]:12s} | {row[3]:10s} | {row[4]}"
    )

print("\nExperiment 6 completed successfully.")
