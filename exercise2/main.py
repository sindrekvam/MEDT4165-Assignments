import logging

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

logger = logging.getLogger(__name__)


def plot(
    fig: plt.Figure,
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid: bool = True,
    legend: bool = True,
    **kwargs,
) -> tuple[plt.Figure, plt.Axes]:
    """"""

    print(kwargs)

    ax.plot(x, y, **kwargs.get("plot", {}))

    if kwargs.get("xlabel", None) is not None:
        ax.set_xlabel(kwargs["xlabel"])
    if kwargs.get("ylabel", None) is not None:
        ax.set_ylabel(kwargs["ylabel"])

    if kwargs.get("xlim", None) is not None:
        ax.set_xlim(kwargs["xlim"])
    if kwargs.get("ylim", None) is not None:
        ax.set_ylim(kwargs["ylim"])

    if grid:
        ax.grid()
    if legend:
        ax.legend()

    return fig, ax


def get_spectrum(data: np.ndarray, NFFT: int = 2048, f_s: float = 1.0):
    fft = np.fft.fftshift(np.fft.fft(data, NFFT))
    spectrum = 20 * np.log10(abs(fft))
    f = np.fft.fftshift(np.fft.fftfreq(NFFT, 1 / f_s))

    return f, spectrum - np.max(spectrum)


def spatial_pulse_length(
    f: np.ndarray, fft_data: np.ndarray, wave_speed: float = 1540.0
):
    """Calculate the spatial pulse length (SPL) by measuring the
    Full width half maximum of the FFT pulse"""

    rfft = fft_data[len(fft_data) // 2 :]
    r_f = f[len(f) // 2 :]
    peaks, _ = scipy.signal.find_peaks(rfft, height=-6)

    # Find upper limit
    upper_lim = peaks[0]
    while rfft[upper_lim] > -6:
        upper_lim += 1

    # Find lower limit
    lower_lim = peaks[0]
    while rfft[lower_lim] > -6:
        lower_lim -= 1

    fwhm = np.abs(r_f[lower_lim] - r_f[upper_lim])
    signal_envelope = wave_speed / fwhm

    logger.debug(f"Full width half maximum: {fwhm}")

    return signal_envelope


def greens_function(r, t):
    return 1 / (4 * np.pi * r)


class ArbitraryTransmitPulser:
    def __init__(self):
        """"""
        pass

    def gaussian_weighted_sinusoid(
        self, period: float, f_0: float, bw: float, f_s: float
    ):
        """Generate a Gaussian-weighted sinusoidal pulse
        or a \"Gaussian Pulse\" """

        t = np.arange(-period / 2, period / 2, 1 / f_s)
        y = scipy.signal.gausspulse(t, f_0, bw)

        return t, y

    def square_wave(self, f_0, bandwidth: float, f_s: float):
        """Generate a rectangular pulse train"""

        num_periods = int(1 // bandwidth)
        period = num_periods / f_0
        t = np.arange(-period / 2, period / 2, 1 / f_s)

        pulse_train = scipy.signal.square(2 * np.pi * f_0 * t + np.pi / 2)

        return t, pulse_train


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Part 1 - Transmission
    atp = ArbitraryTransmitPulser()

    # Constants
    f_s = 250e6
    f_0 = 2.5e6
    bw = 0.3
    period = 10e-6
    c = 1540.0

    # Gaussian wave
    x, gws = atp.gaussian_weighted_sinusoid(period, f_0, bw, f_s)
    f, fft_gws = get_spectrum(gws, f_s=f_s)

    # Square wave
    _, y = atp.square_wave(f_0, bw, f_s)
    square = np.pad(y, pad_width=(gws.shape[0] - y.shape[0]) // 2, mode="constant")
    _, fft_square = get_spectrum(square, f_s=f_s)

    # Find pulse length of gaussian wave
    spatial_pulse_len = spatial_pulse_length(f, fft_gws)
    logger.info(
        f"Spatial pulse length of the Gaussian wave: {spatial_pulse_len * 1e3:.2f} mm"
    )
    logger.info(f"Radial resolution is {spatial_pulse_len * 1e3 / 2:.2f} mm")

    # Plotting
    fig, ax = plt.subplots(2, 1, tight_layout=True)

    plot(
        fig,
        ax[0],
        x * 1e6,
        gws,
        plot={"label": "Gaussian Weighted Sinusoid"},
        xlabel="Time [us]",
        ylabel="Amplitude",
    )
    plot(fig, ax[0], x * 1e6, square, plot={"label": "Square wave", "alpha": 0.7})

    plot(
        fig,
        ax[1],
        f * 1e-6,
        fft_gws,
        plot={"label": "Gaussian Weighted Sinusoid"},
        xlabel="Frequency [MHz]",
        ylabel="Amplitude [dB]",
        xlim=[-15, 15],
        ylim=[-45, 5],
    )
    plot(fig, ax[1], f * 1e-6, fft_square, plot={"label": "Square Wave", "alpha": 0.7})

    plt.savefig("arbitrary_transmit_pulser.png", dpi=300)

    # Transducer impulse response
    f_xd = 4.0e6
    bw_xd = 0.4
    x_xd, gws_xd = atp.gaussian_weighted_sinusoid(period, f_xd, bw_xd, f_s)
    _, fft_gws_xd = get_spectrum(gws_xd, f_s=f_s)

    # Plotting
    fig, ax = plt.subplots(2, 1, tight_layout=True)

    ax[0].plot(x * 1e6, gws, label=f"Transmitted signal $f_0={f_0 / 1e6:.1f}$ MHz")
    ax[0].plot(
        x * 1e6,
        gws_xd,
        label=f"Transducer impulse response $f_0={f_xd / 1e6:.1f}$ MHz",
        alpha=0.7,
        color="C3",
    )
    ax[1].plot(f * 1e-6, fft_gws, label="Transmitted spectra")
    ax[1].plot(
        f * 1e-6,
        fft_gws_xd,
        label="Transducer Transfer Function",
        alpha=0.7,
        color="C3",
    )

    ax[0].set_xlabel("Time [us]")
    ax[0].set_ylabel("Amplitude")
    ax[1].set_xlabel("Frequency [MHz]")
    ax[1].set_ylabel("Amplitude [dB]")
    ax[1].set_ylim([-45, 5])
    ax[1].set_xlim([-15, 15])

    for a in ax:
        a.legend()
        a.grid()

    plt.savefig("transducer_impulse_response.png", dpi=300)

    # Filtered tx pulse
    filtered_gws_pulse = np.convolve(gws, gws_xd, "same")
    filtered_gws_pulse /= np.max(filtered_gws_pulse)
    filtered_square_pulse = np.convolve(square, gws_xd, "same")
    filtered_square_pulse /= np.max(filtered_square_pulse)

    fig, ax = plt.subplots(2, 2, tight_layout=True, figsize=(12, 8))

    plot(
        fig,
        ax[0, 0],
        x * 1e6,
        gws,
        plot={"label": f"Gaussian weighted sinusoid $f_0={f_0 / 1e6:.1f}$ MHz"},
        xlabel="Time [us]",
        ylabel="Amplitude",
    )
    plot(
        fig,
        ax[0, 0],
        x * 1e6,
        filtered_gws_pulse,
        plot={
            "label": f"Filtered transmitted gaussian $f_0={f_xd / 1e6:.1f}$ MHz",
            "alpha": 0.7,
        },
    )

    plot(
        fig,
        ax[0, 1],
        x * 1e6,
        square,
        plot={"label": f"Transmitted signal $f_0={f_0 / 1e6:.1f}$ MHz"},
        xlabel="Time [us]",
        ylabel="Amplitude",
    )
    plot(
        fig,
        ax[0, 1],
        x * 1e6,
        filtered_square_pulse,
        plot={
            "label": f"Filtered transmitted square $f_0={f_xd / 1e6:.1f}$ MHz",
            "alpha": 0.7,
        },
    )

    f_filtered_gws, fft_filtered_gws = get_spectrum(filtered_gws_pulse, f_s=f_s)
    f_filtered_square, fft_filtered_square = get_spectrum(
        filtered_square_pulse, f_s=f_s
    )

    plot(
        fig,
        ax[1, 0],
        f * 1e-6,
        fft_gws,
        plot={"label": "Gaussian Weighted Sinusoid"},
        xlabel="Frequency [MHz]",
        ylabel="Amplitude [dB]",
        xlim=[-15, 15],
        ylim=[-45, 5],
    )
    plot(
        fig,
        ax[1, 0],
        f_filtered_gws * 1e-6,
        fft_filtered_gws,
        plot={"label": "Filtered Gaussian Weighted Sinusoid", "alpha": 0.7},
        xlabel="Frequency [MHz]",
        ylabel="Amplitude [dB]",
        xlim=[-15, 15],
        ylim=[-45, 5],
    )

    plot(
        fig, ax[1, 1], f * 1e-6, fft_square, plot={"label": "Square Wave", "alpha": 0.7}
    )
    plot(
        fig,
        ax[1, 1],
        f_filtered_square * 1e-6,
        fft_filtered_square,
        plot={"label": "Filtered Square Wave", "alpha": 0.7},
        xlabel="Frequency [MHz]",
        ylabel="Amplitude [dB]",
        xlim=[-15, 15],
        ylim=[-45, 5],
    )

    plt.savefig("filtered_transmit_pulse.png", dpi=300)

    # Part 2 - Reception

    # Constants
    r_max = 15e-2
    t_max = 2 * r_max / c

    scatterer_positions = np.array([1, 3, 5, 7, 9, 11, 13]) * 1e-2
    t_rx = np.arange(0, t_max, 1 / f_s)
    r_rx = (t_rx * c) / 2

    impulse_train = np.zeros(t_rx.shape)
    for r in scatterer_positions:
        # Find the index of the scattering positions
        idx = np.argmin(np.abs(r_rx - r))
        impulse_train[idx] = (1 / (4 * np.pi * r)) ** 2

    # Trim the transmitted pulse to optimize convolution
    # Find indices where pulse is significant (> 1% of peak)
    thresh = 0.01 * np.max(np.abs(filtered_gws_pulse))
    active_idx = np.where(np.abs(filtered_gws_pulse) > thresh)[0]
    # Only use values from the start and end of where the signal is present
    trimmed_pulse = filtered_gws_pulse[active_idx[0] : active_idx[-1]]
    t_tx_trimmed = x[active_idx[0] : active_idx[-1]]

    s_rx_clean = np.convolve(impulse_train, trimmed_pulse)
    # Formula from PDF
    t_rx_conv = np.linspace(
        t_tx_trimmed[0] + t_rx[0], t_tx_trimmed[-1] + t_rx[-1], len(s_rx_clean)
    )
    r_conv = (t_rx_conv * c) / 2

    # Add Gaussian white noise
    noise_std = 0.01 * np.max(np.abs(s_rx_clean))
    s_rx_noisy = s_rx_clean + np.random.normal(0, noise_std, size=len(s_rx_clean))

    # Filtering
    f_cutoff = np.array([f_0 - (f_0 * bw), f_0 + (f_0 * bw)])
    sos = scipy.signal.butter(4, f_cutoff, btype="bandpass", fs=f_s, output="sos")
    s_rx_filtered = scipy.signal.sosfiltfilt(sos, s_rx_noisy)

    # Time-Gain Compensation (TGC)
    s_rx_tgc = s_rx_filtered * (4 * np.pi * r_conv) ** 2

    # IQ Demodulation
    sig_hilbert = scipy.signal.hilbert(s_rx_tgc)
    sig_iq = sig_hilbert * np.exp(-1j * 2 * np.pi * f_0 * t_rx_conv)

    # Envelope Detection and A-mode (dB scale)
    envelope = np.abs(sig_iq)
    amode_db = 20 * np.log10(envelope)
    amode_db -= np.max(amode_db)

    # Plotting
    fig, ax = plt.subplots(3, 1, figsize=(10, 14), tight_layout=True)

    ax[0].plot(
        r_conv * 1e2, s_rx_noisy / np.max(np.abs(s_rx_noisy)), label="Noisy RF Signal"
    )
    ax[0].plot(
        r_conv * 1e2,
        s_rx_filtered / np.max(np.abs(s_rx_filtered)),
        label="Filtered signal",
    )
    ax[0].plot(
        r_rx * 1e2,
        impulse_train / np.max(impulse_train),
        "r--",
        alpha=0.7,
        label="Scatterers",
    )
    ax[0].set_title("Received RF Signal")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Depth [cm]")
    ax[0].legend()

    # TGC
    ax[1].plot(r_conv * 1e2, s_rx_tgc / np.max(np.abs(s_rx_tgc)), color="C3")
    ax[1].set_title("Time-Gain Compensation (TGC)")
    ax[1].set_ylabel("Amplitude")

    # A-image
    ax[2].plot(r_conv * 1e2, amode_db, color="C2")
    ax[2].set_title("A-mode Image")
    ax[2].set_xlabel("Depth [cm]")
    ax[2].set_ylabel("Amplitude [dB]")
    ax[2].set_ylim([-60, 5])

    for a in ax:
        a.grid()

    plt.savefig("reception.png", dpi=300)
