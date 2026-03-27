from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal.windows import boxcar, hamming

file_path = Path(__file__).parent / "slowmotion_v2.mat"

# %%
data = loadmat(file_path)

# %%
# Fetch parameters
iq = data["iq"]
pars = data["pars"]
tissue = data["tissue"]

fps = pars[0, 0]["fps"].item()
f0 = pars[0, 0]["f0"].item()
depth_start = pars[0, 0]["depth_start"].item()
depth_inc = pars[0, 0]["depth_inc"].item()

# %%
# dimensions in iq: [range/depth sample, beam number, frame number]
time = np.arange(iq.shape[2]) / fps
depth = depth_start + np.arange(iq.shape[0]) * depth_inc


# %%
def distance_between_fixed_point_and_probe(t, t_0, R, T, z_0):
    """In the lab setup, assuming R << L, it can be shown that the distance between a fixed point
    and the probe varies approximately as

    Parameters:
        - R: Excursion radius [m]
        - t_0: Time offset [s]
        - T: Rotational period [s]
        - z_0: Depth offset [m]
    """
    return -R * np.cos((2 * np.pi * (t - t_0)) / T) + z_0


# %%
# a)
center_beam = iq[:, iq.shape[1] // 2, :]
center_beam_db = 20 * np.log10(np.abs(center_beam))
center_beam_db -= np.max(center_beam_db)

fig, ax = plt.subplots(tight_layout=True)
img = ax.imshow(
    center_beam_db,
    vmin=-40,
    vmax=0,
    extent=[time[0], time[-1], depth[-1] * 1e3, depth[0] * 1e3],
    aspect="auto",
    cmap="Greys",
)
fig.colorbar(img, ax=ax)
ax.set_ylabel("Depth [mm]")
ax.set_xlabel("Time [s]")
plt.savefig("center_beam_m_mode.png")

# Observed from image
z_0 = 35 * 1e-3
t_0 = 0.075
T = 0.98 - t_0
R = 7 * 1e-3
r = distance_between_fixed_point_and_probe(time, t_0, R, T, z_0)

ax.plot(time, r * 1e3, "r")
plt.savefig("center_beam_with_estimate.png")


# %%
def get_spectral_estimate(sample, fs: float = 1, nfft: int = 1024, window_func=boxcar):
    sample *= window_func(len(sample))
    spectral_estimate = np.fft.fftshift(np.abs(np.fft.fft(sample, n=nfft)) ** 2)
    spectral_estimate = spectral_estimate[len(spectral_estimate) // 2 :]

    spectral_estimate_db = 20 * np.log10(spectral_estimate)
    spectral_estimate_db -= np.max(spectral_estimate_db)

    freqs = np.fft.fftshift(np.fft.fftfreq(len(spectral_estimate_db), 1 / fs))

    return freqs, spectral_estimate_db


def plot_spectral_estimate(freqs, spectral_estimate_db, ax: plt.Axes = None):
    if ax is None:
        fig, ax = plt.subplots(tight_layout=True)
    ax.plot(freqs, spectral_estimate_db)
    ax.set_ylim([-40, 5])
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power")


z_0_index = np.where(depth >= z_0)[0][0]
sample_at_depth = iq[z_0_index, iq.shape[1] // 2, :]
plot_spectral_estimate(*get_spectral_estimate(sample_at_depth, fps))
plt.savefig(f"spectral_analysis_z0_{z_0}.png")

# %%
# Spectral analysis only at small time intervals
subsample_length = 16
# Fetch only the indexes around the z_0 index
idx = z_0_index - subsample_length // 2

subsamples_at_depth = sample_at_depth[idx : idx + subsample_length]

plot_spectral_estimate(*get_spectral_estimate(subsamples_at_depth, fps))
plt.savefig("subsamples.png")

# %%
plot_spectral_estimate(
    *get_spectral_estimate(subsamples_at_depth, fps, window_func=hamming)
)
plt.savefig("subsamples_with_hamming.png")

# %%
fig, ax = plt.subplots(3, 2, tight_layout=True)

for ax_idx, subsample_length in enumerate([8, 16, 32, 64, 128, 256]):
    idx = np.max([0, z_0_index - subsample_length // 2])
    subsamples_at_depth = sample_at_depth[idx : idx + subsample_length]

    ax.flatten()[ax_idx].set_title(f"Subsample length: {subsample_length}")
    plot_spectral_estimate(
        *get_spectral_estimate(subsamples_at_depth, fps, window_func=hamming),
        ax=ax.flatten()[ax_idx],
    )
plt.savefig("different_subsample_lengths.png")

# %%
# Sonogram

subsample_length = 64

sample_fft = get_spectral_estimate(
    sample_at_depth[:subsample_length], fps, window_func=hamming
)
num_freq_bins = len(sample_fft[1])
num_time_steps = iq.shape[2] - subsample_length

sonogram = np.zeros((num_freq_bins, num_time_steps))
for jdx in range(num_time_steps):  # Frame / time axis
    subsample_at_depth = sample_at_depth[jdx : jdx + subsample_length,]
    f, fft_db = get_spectral_estimate(subsample_at_depth, fps)

    sonogram[:, jdx] = fft_db


time_sonogram = time[subsample_length // 2 : subsample_length // 2 + num_time_steps]
fig, ax = plt.subplots(tight_layout=True)
img = plt.imshow(
    sonogram,
    vmax=0,
    vmin=-40,
    cmap="Greys",
    interpolation="none",
    extent=[time_sonogram[0], time_sonogram[-1], f[0], f[-1]],
    aspect="auto",
)
fig.colorbar(img, ax=ax)
ax.set_ylabel("Frequency [Hz]")
ax.set_xlabel("Time [s]")


# %%
def get_analytical_solution(t, t_0, R, T):
    """By taking the
    temporal derivative of the analytical displacement function r(t)"""

    return 2 * np.pi * R / T * np.sin(2 * np.pi * (t - t_0) / T)


def get_doppler_shift(v, f_0, c):
    """Calculate doppler shift of signal"""

    return -2 * v * f_0 / c


v_sonogram = get_analytical_solution(time_sonogram, t_0, R, T)
f_d = get_doppler_shift(v_sonogram, 2.5e6, 1490)

ax.plot(time_sonogram, f_d, "r", label="Analytical solution")
plt.savefig("sonogram_with_analytic")
