import argparse

import matplotlib.pyplot as plt
import numpy as np
import scipy as scp


def generate_signal(t: np.ndarray, f_0: float = 1e6):
    """Generate a sinusoidal signal with given frequency"""

    return np.sin(2 * np.pi * f_0 * t)


def calculate_power_spectral_density(
    signal: np.ndarray, Nfft: int = 1024, scale: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """"""

    freq = np.fft.fftshift(np.fft.fftfreq(Nfft, scale))
    psd = 1 / Nfft * np.pow(np.abs(np.fft.fftshift(np.fft.fft(signal, Nfft))), 2)

    return freq, psd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampling-frequency", default=50e6, type=float)
    parser.add_argument("--signal-frequency", default=1e6, type=float)
    parser.add_argument("--time", default=10e-6, type=float)
    parser.add_argument("--noise-std", default=0.0, type=float)
    args = parser.parse_args()

    # Generate a time axis
    time = np.linspace(0, args.time, int(args.sampling_frequency * args.time))

    # Generate a noisy signal
    noise = np.random.normal(0.0, args.noise_std, size=time.shape)
    signal = generate_signal(time, args.signal_frequency)
    signal_with_noise = signal + noise

    # Calculate signal to noise ratio
    snr = np.mean(signal**2) / np.mean(noise**2)
    snr_db = 10 * np.log10(snr)

    print(snr_db)

    plt.plot(time * 1e6, signal_with_noise)
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title("$\\sigma_{noise}=" + f"{args.noise_std}$, $SNR = {snr_db:.2f} dB$")
    plt.suptitle("Signal in noise")

    # Calculate PSD
    freq, psd = calculate_power_spectral_density(
        signal + 0.001,  # Add small value to add a slight DC offset
        scale=1 / args.sampling_frequency,
    )
    _, psdn = calculate_power_spectral_density(
        signal_with_noise, scale=1 / args.sampling_frequency
    )
    psd /= np.max(psd)
    psdn /= np.max(psdn)

    fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True)

    ax[0].set_title("PSD of signal")
    ax[0].plot(freq * 1e-6, 10 * np.log10(psd), label="signal")
    ax[1].set_title("PSD of signal with noise")
    ax[1].plot(freq * 1e-6, 10 * np.log10(psdn), label="signal with noise")
    ax[1].set_xlabel("Frequency (MHz)")

    # Do bandpass filtering around
    critical_frequencies = np.array(
        [args.signal_frequency - 250e3, args.signal_frequency + 250e3], dtype=int
    )
    butterworth_filter = scp.signal.butter(
        2, critical_frequencies, btype="bandpass", fs=args.sampling_frequency
    )

    filtered_signal = scp.signal.lfilter(*butterworth_filter, signal_with_noise)

    _, psdf = calculate_power_spectral_density(
        filtered_signal, scale=1 / args.sampling_frequency
    )
    psdf /= np.max(psdf)

    ax[1].plot(freq * 1e-6, 10 * np.log10(psdf), alpha=0.75, label="filtered signal")
    ax[1].set_ylim([-60, 10])

    ax[1].legend()

    plt.figure()

    plt.plot(time * 1e6, filtered_signal)
    plt.xlabel("Time (us)")
    plt.ylabel("Amplitude")
    plt.title(
        "$\\sigma_{noise}="
        + f"{args.noise_std}$, $f_c = {critical_frequencies * 1e-3} kHz$"
    )
    plt.suptitle("Filtered signal from noise")

    plt.show()
