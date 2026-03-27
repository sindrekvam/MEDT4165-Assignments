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
from scipy.signal.windows import hamming, triang

# SIMULATION SETUP
# %%
c0 = 1540  # [m/s] Speed of sound
rho0 = 1000  # [kg/m^3] Density of water
source_f0 = 1e6  # [Hz] Source frequency
source_amp = 1e6  # [Pa] Source amplitude
source_cycles = 2  # Number of cycles in the pulse

aperture_size = 20e-3
aperture_index = 10
grid_size_x = 100e-3  # [m] Grid size in x (NB: Depth in k-Wave)
grid_size_y = 40e-3  # [m] Grid size in z (NB: Width in k-Wave)
focal_depth = 30e-3  # [m] Focus depth
focal_depth_rx = 30e-3  # [m] Focus depth
steer_angle = 0 * np.pi / 180  # rad
ppw = 8  # Points per wavelength
cfl = 0.3  # Related to the time resolution, lower is more accurate (no need to change)

viz_t = grid_size_y / 2 / c0 * 1.2  # Visualization time [s]

# SETUP SIMULATION GRID AND TIME STEPS
# %%
dx = c0 / (ppw * source_f0)  # Grid resolution
Nx = round(grid_size_x / dx)  # Number of grid points in x, NB: This is depth
Ny = round(grid_size_y / dx)  # Number of grid points in y, NB: This is width (lateral)
Na = round(Ny * aperture_size // grid_size_y)  # Number of grid points in the aperture
kgrid = kWaveGrid([Nx, Ny], [dx, dx])
kgrid.makeTime(
    c0, cfl, t_end=2 * grid_size_x / c0
)  # Make time array based on the Courant-Friedrichs-Lewy (CFL) condition


# Create updated coordinate axis
x_axis = ((np.arange(Nx) - aperture_index) * dx) * 1e3
y_axis = kgrid.y_vec * 1e3

# SETUP SOURCE
# Source signal, Gaussian pulse


def get_delay_profile(
    sources: np.ndarray, focal_depth: float, steer_angle: float, c: float = c0
):
    """
    Calculate delay profile based on focal depth and steering angle
    Equations are from page 37 in lecture 5
    """

    # Focusing
    # 1/c * (r_c - r_i)
    # r_c = distance from focal point to center element
    # r_i = distance from focal point to element i
    r_i = np.sqrt(sources**2 + focal_depth**2)
    r_c = np.sqrt(sources[len(sources) // 2] ** 2 + focal_depth**2)

    _delay_profile = (r_c - r_i) / c

    # Steering
    # 1/c * tan(phi) (x_i - x_end)
    _delay_profile += np.tan(steer_angle) * sources / c

    # Normalize
    _delay_profile -= np.min(_delay_profile)

    return _delay_profile


# %%
aperture_range = range((Ny - Na) // 2, (Ny + Na) // 2)
source_y = kgrid.y_vec[aperture_range]

delay_profile = get_delay_profile(source_y, focal_depth, steer_angle)
signal_offset = np.round(delay_profile / kgrid.dt).astype(int)


def add_triangle_apodization(source_amp: float, signal_offsets: list[float]):
    return source_amp * np.expand_dims(triang(len(signal_offsets)), 1)


def add_hamming_apodization(source_amp: float, signal_offsets: list[float]):
    return source_amp * np.expand_dims(hamming(len(signal_offsets)), 1)


# Task 5.2
# source_amp = add_triangle_apodization(source_amp, signal_offset)
# source_amp = add_hamming_apodization(source_amp, signal_offset)

source_sig = source_amp * tone_burst(
    1 / kgrid.dt,
    source_f0,
    source_cycles,
    signal_offset=signal_offset,
)
plt.figure(tight_layout=True)
plt.title("Signal offset")
plt.plot(
    source_y * 1e3,
    signal_offset * kgrid.dt * 1e6,
    "x-",
    label="Quantized delay profile",
)
plt.plot(source_y * 1e3, delay_profile * 1e6, label="True delay profile")
plt.ylabel("Delay [us]")
plt.xlabel("Width [mm]")
plt.legend()
plt.savefig(
    f"delay_profile_F{np.round(focal_depth * 1e3).astype(int)}_{np.round(steer_angle * 180 / np.pi).astype(int)}.png"
)

# %%
# Define kWave source object
source = kSource()
source.p_mask = np.zeros_like(kgrid.x)
# Center the source points
for point in aperture_range:
    source.p_mask[aperture_index, point] = 1
source.p = source_sig  # Source signal (pressure source)

# SETUP SENSOR
sensor = kSensor(
    record=["p", "p_max"]
)  # Sensor object, record pressure and maximum pressure
sensor.mask = np.ones_like(kgrid.x)  # Sensor mask, all grid points

# SETUP MEDIUM
medium = kWaveMedium(
    sound_speed=np.ones_like(kgrid.x) * c0,
    density=np.ones_like(kgrid.x) * rho0,
)  # Define medium object, simple homogeneous medium

# Add some scatter points with a larger density
r = np.array([10, 30, 40, 50, 70], dtype=int) * 1e-3  # [m]
r_index = np.array([np.where(x_axis >= r_i * 1e3)[0][0] for r_i in r])
medium.density[r_index, Ny // 2] = 4000

# %%
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

# %%
# Extract pressure field from
p_field = np.reshape(
    sensor_data["p"], (kgrid.Nt, Nx, Ny), order="F"
)  # Stored in Fortran ordering for some reason
p_max = np.reshape(sensor_data["p_max"], (Nx, Ny), order="F")


def plot_depth_profile(p_max, focal_depth: float = None):
    center_beam = Ny // 2
    center_profile = p_max[:, center_beam]
    center_profile /= np.max(center_profile)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(x_axis, center_profile, label="center profile")

    if focal_depth is not None:
        ax.axvline(
            focal_depth * 1e3,
            color="red",
            label="Focal depth",
        )
    plt.xlabel("x [mm]")
    plt.ylabel("Amplitude")
    plt.title("Depth profile")
    plt.legend()
    plt.savefig(
        f"depth_profile_F{np.round(focal_depth * 1e3).astype(int)}_{np.round(steer_angle * 180 / np.pi).astype(int)}.png"
    )


def plot_beam_profile(p_max):
    p_max_depth = np.max(p_max, axis=1, keepdims=True)

    normalized_p_max = p_max / p_max_depth
    normalized_p_max_db = 20 * np.log10(normalized_p_max)

    fig, ax = plt.subplots(constrained_layout=True, figsize=(5, 8))
    cmap = "twilight"
    extent = [y_axis[0], y_axis[-1], x_axis[-1], x_axis[0]]
    image = plt.imshow(normalized_p_max_db, cmap=cmap, extent=extent)
    plt.colorbar()
    plt.xlabel("x [mm]")
    plt.ylabel("z [mm]")
    plt.title(f"Beam profile, $F={focal_depth * 1e3} mm$")
    plt.savefig(
        f"beam_profile_F{np.round(focal_depth * 1e3).astype(int)}_{np.round(steer_angle * 180 / np.pi).astype(int)}.png"
    )


def plot_lateral_beam_profile(
    p_max, distance_index: int, distance_to_maximum_pressure: float
):
    lateral_profile = p_max[distance_index, :]

    lateral_profile_db = 20 * np.log10(lateral_profile)
    lateral_profile_db -= np.max(lateral_profile_db)

    theoretical_lateral_profile = np.sinc(
        aperture_size
        * y_axis
        * 1e-3
        / ((c0 / source_f0) * distance_to_maximum_pressure)
    )
    # theoretical_lateral_profile /= np.max(theoretical_lateral_profile)
    theoretical_lateral_profile_db = 20 * np.log10(np.abs(theoretical_lateral_profile))
    theoretical_lateral_profile_db -= np.max(theoretical_lateral_profile_db)

    theoretical_lateral_profile_sinc2 = np.pow(
        np.sinc(
            0.5
            * aperture_size
            * y_axis
            * 1e-3
            / ((c0 / source_f0) * distance_to_maximum_pressure)
        ),
        2,
    )
    # theoretical_lateral_profile /= np.max(theoretical_lateral_profile)
    theoretical_lateral_profile_sinc2_db = 20 * np.log10(
        np.abs(theoretical_lateral_profile_sinc2)
    )
    theoretical_lateral_profile_sinc2_db -= np.max(theoretical_lateral_profile_sinc2_db)

    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(y_axis, lateral_profile_db, label="lateral beam profile")
    ax.plot(
        y_axis, theoretical_lateral_profile_db, label="theoretical lateral beam profile"
    )
    ax.plot(
        y_axis,
        theoretical_lateral_profile_sinc2_db,
        label="theoretical lateral beam profile, sinc²",
    )
    ax.set_ylim([-50, 5])
    plt.xlabel("y [mm]")
    plt.ylabel("Amplitude [dB]")
    plt.title(f"Lateral beam profile $z={distance_to_maximum_pressure * 1e3:.2f}$ mm")
    plt.legend()
    plt.savefig(
        f"lateral_beam_profile_F{np.round(focal_depth * 1e3).astype(int)}_{np.round(steer_angle * 180 / np.pi).astype(int)}.png"
    )


plot_beam_profile(p_max)
plot_depth_profile(p_max, focal_depth=focal_depth)
plot_lateral_beam_profile(
    p_max,
    distance_index=np.where(x_axis >= focal_depth * 1e3)[0][0],
    distance_to_maximum_pressure=focal_depth,
)


# %%
for alpha in [0.2, focal_depth * 2 / grid_size_x, 0.5, 0.8, 1.2, 1.5, 1.8]:
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
    plt.savefig(
        f"propagation_T{alpha}_F{np.round(focal_depth * 1e3).astype(int)}_{np.round(steer_angle * 180 / np.pi).astype(int)}.png"
    )


plt.show()
