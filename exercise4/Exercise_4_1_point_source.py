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
source_cycles = 5  # Number of cycles in the pulse

grid_size_x = 40e-3  # [m] Grid size in x (NB: Depth in k-Wave)
grid_size_y = 40e-3  # [m] Grid size in z (NB: Width in k-Wave)
ppw = 5  # Points per wavelength
cfl = 0.3  # Related to the time resolution, lower is more accurate (no need to change)

viz_t = grid_size_y / 2 / c0 * 0.8  # Visualization time [s]

# SETUP SIMULATION GRID AND TIME STEPS
dx = c0 / (ppw * source_f0)  # Grid resolution
Nx = round(grid_size_x / dx)  # Number of grid points in x, NB: This is depth
Ny = round(grid_size_y / dx)  # Number of grid points in y, NB: This is width (lateral)
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
source.p_mask[round(kgrid.Nx / 2), round(kgrid.Ny / 2)] = 1
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
medium.sound_speed[: Nx // 2, :] = c0 / 2

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

# Get frame number to plot
N_frame = np.round(np.where(kgrid.t_array[0] > viz_t)[0][0]).astype(
    int
)  # Number of frames to visualize

# Normalize frames based on the maximum value over all frames
max_value = np.max(np.abs(p_field))
p_plot = p_field / max_value

cmap = "twilight"
extent = (
    np.array([kgrid.y_vec[0], kgrid.y_vec[-1], kgrid.x_vec[0], kgrid.x_vec[-1]]) * 1e3
)

# Create a figure and axis
fig, ax = plt.subplots()
image = plt.imshow(p_plot[N_frame], cmap=cmap, vmin=-0.1, vmax=0.1, extent=extent)
plt.colorbar()
plt.xlabel("x [mm]")
plt.ylabel("z [mm]")
plt.title("A single point source")
plt.show()
