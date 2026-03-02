import matplotlib.pyplot as plt
import numpy as np
from kwave.kgrid import kWaveGrid
from kwave.kmedium import kWaveMedium
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kspaceFirstOrder2D import kspaceFirstOrder2D
from kwave.kWaveSimulation import SimulationOptions
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.utils.signals import tone_burst

# SIMULATION SETUP
c0 = 1540  # [m/s] Speed of sound
rho0 = 1000  # [kg/m^3] Density of water
source_f0 = 1e6  # [Hz] Source frequency
source_amp = 1e6  # [Pa] Source amplitude
source_cycles = 200  # Number of cycles in the pulse

aperture_size = 10e-3
grid_size_x = 100e-3  # [m] Grid size in x (NB: Depth in k-Wave)
grid_size_y = 40e-3  # [m] Grid size in z (NB: Width in k-Wave)
ppw = 10  # Points per wavelength
cfl = 0.3  # Related to the time resolution, lower is more accurate (no need to change)

viz_t = grid_size_y / 2 / c0 * 0.8  # Visualization time [s]

# SETUP SIMULATION GRID AND TIME STEPS
dx = c0 / (ppw * source_f0)  # Grid resolution
Nx = round(grid_size_x / dx)  # Number of grid points in x, NB: This is depth
Ny = round(grid_size_y / dx)  # Number of grid points in y, NB: This is width (lateral)
Na = round(Ny * aperture_size // grid_size_y)  # Number of grid points in the aperture
kgrid = kWaveGrid([Nx, Ny], [dx, dx])
# Make time array based on the Courant-Friedrichs-Lewy (CFL) condition
kgrid.makeTime(c0, cfl)

# SETUP SOURCE
# Source signal, Gaussian pulse
source_sig = source_amp * tone_burst(1 / kgrid.dt, source_f0, source_cycles)

# Define kWave source object
source = kSource()
source.p_mask = np.zeros_like(kgrid.x)
# Source position, centered in the grid
for y in range(Ny // 2 - Na // 2, Ny // 2 + Na // 2):
    source.p_mask[10, y] = 1
# Source signal (pressure source)
source.p = source_sig

# SETUP SENSOR
sensor = kSensor(
    record=["p", "p_max"]
)  # Sensor object, record pressure and maximum pressure
sensor.mask = np.ones_like(kgrid.x)  # Sensor mask, all grid points

# SETUP MEDIUM
medium = kWaveMedium(sound_speed=c0, density=rho0)
medium.sound_speed = c0 * np.ones((Nx, Ny))
# medium.sound_speed[: Nx // 2, :] = c0 / 2

# RUN SIMULATION
simulation_options = SimulationOptions(
    pml_auto=True,
    pml_inside=False,
    save_to_disk=True,
    data_cast="single",
)
execution_options = SimulationExecutionOptions(is_gpu_simulation=False)

# Run 2D simulation (to save some time)
sensor_data = kspaceFirstOrder2D(
    kgrid=kgrid,
    medium=medium,
    source=source,
    sensor=sensor,
    simulation_options=simulation_options,
    execution_options=execution_options,
)

# Extract pressure field from
p_field = np.reshape(
    sensor_data["p"], (kgrid.Nt, Nx, Ny), order="F"
)  # Stored in Fortran ordering for some reason

p_max = np.reshape(sensor_data["p_max"], (Nx, Ny), order="F")

p_max_depth = np.max(p_max, axis=1, keepdims=True)

normalized_p_max = p_max / p_max_depth
normalized_p_max_db = 20 * np.log10(normalized_p_max)

# VISUALIZATION
plt.style.use("dark_background")

x_axis = (np.arange(Nx) * dx - 10 * dx) * 1e3
y_axis = kgrid.y_vec * 1e3

for alpha in [0.2, 0.5, 0.8, 1.2, 1.5, 1.9]:
    viz_t = grid_size_x / 2 / c0 * alpha  # Visualization time [s]
    # Get frame number to plot
    N_frame = np.round(np.where(kgrid.t_array[0] > viz_t)[0][0]).astype(
        int
    )  # Number of frames to visualize

    # Normalize frames based on the maximum value over all frames
    max_value = np.max(np.abs(p_field))
    p_plot = p_field / max_value

    cmap = "twilight"
    extent = [y_axis[0], y_axis[-1], x_axis[-1], x_axis[0]]

    # Create a figure and axis
    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 8))
    image = plt.imshow(p_plot[N_frame], cmap=cmap, vmin=-0.1, vmax=0.1, extent=extent)
    plt.colorbar()
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title(f"{Na} point sources")
    plt.savefig(f"propagation_{alpha}_{source_cycles}.png")


fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 8))
image = plt.imshow(normalized_p_max_db, cmap=cmap, extent=extent)
plt.colorbar()
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.title("Beam profile")
plt.savefig(f"beam_profile_{Na}_{source_cycles}.png")


# Depth profile
fig, ax = plt.subplots(constrained_layout=True)
center_beam = Ny // 2
center_profile = p_max[:, center_beam]
center_profile /= np.max(center_profile)

distance_to_maximum_pressure = np.pow(aperture_size, 2) / (np.pi * c0 / source_f0)
distance_index = np.argmin(np.abs(x_axis - distance_to_maximum_pressure * 1e3))

ax.plot(x_axis, center_profile, label="center profile")
ax.scatter(
    distance_to_maximum_pressure * 1e3,
    center_profile[distance_index],
    color="red",
    label="$z_t$",
)
plt.xlabel("x [mm]")
plt.ylabel("Amplitude")
plt.title("Depth profile")
plt.legend()
plt.savefig(f"depth_profile_{Na}_{source_cycles}.png")


# Lateral beam profile
fig, ax = plt.subplots(constrained_layout=True)
lateral_profile = p_max[distance_index, :]
# lateral_profile /= np.max(lateral_profile)
lateral_profile_db = 20 * np.log10(lateral_profile)
lateral_profile_db -= np.max(lateral_profile_db)

theoretical_lateral_profile = np.sinc(
    aperture_size * y_axis * 1e-3 / ((c0 / source_f0) * distance_to_maximum_pressure)
)
# theoretical_lateral_profile /= np.max(theoretical_lateral_profile)
theoretical_lateral_profile_db = 20 * np.log10(np.abs(theoretical_lateral_profile))
theoretical_lateral_profile_db -= np.max(theoretical_lateral_profile_db)


ax.plot(y_axis, lateral_profile_db, label="lateral beam profile")
ax.plot(
    y_axis, theoretical_lateral_profile_db, label="theoretical lateral beam profile"
)
ax.set_ylim([-40, 5])
plt.xlabel("y [mm]")
plt.ylabel("Amplitude [dB]")
plt.title(f"Lateral beam profile $z={distance_to_maximum_pressure * 1e3:.2f}$ mm")
plt.legend()
plt.savefig(f"lateral_beam_profile_{Na}_{source_cycles}.png")

plt.show()
