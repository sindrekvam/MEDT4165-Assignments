import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)

# Load data
file_path = Path(__file__).parent / "slowmotion_v2.mat"
data = loadmat(file_path)

# %%

# Extract data
iq_data = data["iq"]  # depth, beam number, frame
logger.debug(iq_data.shape)

# Extract parameters from the resulting Numpy structured array, use .item() to extract the scalar value (to avoid annoying Numpy errors)
pars = data["pars"]
fps = pars[0, 0]["fps"].item()
d_s = pars[0, 0]["depth_start"].item()
d_inc = pars[0, 0]["depth_inc"].item()
f0 = pars[0, 0]["f0"].item()

# %%

# Create depth and time axis
z = d_s + np.arange(iq_data.shape[0]) * d_inc
t = np.arange(iq_data.shape[2]) / fps

# Fetch only the center beam
center_beam = iq_data[:, iq_data.shape[1] // 2, :]
center_beam_db = 20 * np.log10(np.abs(center_beam))
center_beam_db -= np.max(center_beam_db)

# Plot M-mode data
fig, ax = plt.subplots(tight_layout=True)
img = ax.imshow(
    center_beam_db,
    vmin=-40,
    vmax=0,
    extent=[t[0], t[-1], z[-1] * 1e3, z[0] * 1e3],
    aspect="auto",
    cmap="Greys",
)
fig.colorbar(img, ax=ax)
ax.set_ylabel("Depth [mm]")
ax.set_xlabel("Time [s]")
plt.savefig("center_beam_m_mode.png")

# %%

# Spectral analyses - Welch


# %%
plt.show()
