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
source_cycles = 2  # Number of cycles in the pulse

aperture_size = 10e-3
aperture_index = 10
grid_size_x = 100e-3  # [m] Grid size in x (NB: Depth in k-Wave)
grid_size_y = 40e-3  # [m] Grid size in z (NB: Width in k-Wave)
focal_depth = 70e-3  # [m] Focus depth
ppw = 8  # Points per wavelength
cfl = 0.3  # Related to the time resolution, lower is more accurate (no need to change)

viz_t = grid_size_y / 2 / c0 * 1.2  # Visualization time [s]

# SETUP SIMULATION GRID AND TIME STEPS
dx = c0 / (ppw * source_f0)  # Grid resolution
Nx = round(grid_size_x / dx)  # Number of grid points in x, NB: This is depth
Ny = round(grid_size_y / dx)  # Number of grid points in y, NB: This is width (lateral)
Na = round(Ny * aperture_size // grid_size_y)  # Number of grid points in the aperture
kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(
    c0, cfl
)  # Make time array based on the Courant-Friedrichs-Lewy (CFL) condition

# SETUP SOURCE
# Source signal, Gaussian pulse
signal_offset = np.arange(Na // 2)
source_sig = source_amp * tone_burst(
    1 / kgrid.dt,
    source_f0,
    source_cycles,
    signal_offset=signal_offset,
)
plt.figure(tight_layout=True)
plt.title("Source signals")
for i, sig in enumerate(source_sig):
    plt.plot(sig, label=f"signal {i}")
plt.ylabel("Amplitude [Pa]")
plt.xlabel("Sample")

# Define kWave source object
source = kSource()
source.p_mask = np.zeros_like(kgrid.x)
# Start source points at the edge.
# Approximate half of the aperture outside the grid
for point in range(0, Na // 2):
    source.p_mask[aperture_index, point] = 1
source.p = source_sig  # Source signal (pressure source)

# SETUP SENSOR
sensor = kSensor(
    record=["p", "p_max"]
)  # Sensor object, record pressure and maximum pressure
sensor.mask = np.ones_like(kgrid.x)  # Sensor mask, all grid points

# SETUP MEDIUM
medium = kWaveMedium(
    sound_speed=c0, density=rho0
)  # Define medium object, simple homogeneous medium

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

# VISUALIZATION
plt.style.use("dark_background")

# Create updated coordinate axis
x_axis = (np.arange(Nx) * dx - aperture_index * dx) * 1e3
y_axis = kgrid.y_vec * 1e3

for alpha in [0.2, 0.5, 0.8, 1.2, 1.5, 1.8]:
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
    fig, ax = plt.subplots(figsize=(5, 8), constrained_layout=True)
    image = plt.imshow(p_plot[N_frame], cmap=cmap, vmin=-0.1, vmax=0.1, extent=extent)
    plt.colorbar()
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title(f"{Na} point sources")

plt.show()
