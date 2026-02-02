import matplotlib.pyplot as plt
import numpy as np
import scipy as scp


def get_spectrum(data: np.ndarray, NFFT: int = 2048, f_s: float = 1.0):
    fft = np.fft.fftshift(np.fft.fft(data, NFFT))
    psd = np.abs(fft) ** 2
    f = np.fft.fftshift(np.fft.fftfreq(NFFT))

    return f, psd / np.max(psd)


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
        y = scp.signal.gausspulse(t, f_0, bw)

        return t, y

    def square_wave(self, f_0, bandwidth: float, f_s: float):
        """Generate a rectangular pulse train"""

        num_periods = 1 // bw
        period = num_periods / f_0
        t = np.arange(-period / 2, period / 2, 1 / f_s)

        pulse_train = scp.signal.square(2 * np.pi * f_0 * t + np.pi / 2)

        return t, pulse_train


if __name__ == "__main__":
    atp = ArbitraryTransmitPulser()

    f_s = 250e6
    f_0 = 2.5e6
    bw = 0.3
    period = 10e-6

    fig, ax = plt.subplots(2, 1, tight_layout=True)

    # Gaussian wave
    x, gws = atp.gaussian_weighted_sinusoid(period, f_0, bw, f_s)
    f, fft_gws = get_spectrum(gws)

    # Square wave
    _, y = atp.square_wave(f_0, bw, f_s)
    square = np.pad(y, pad_width=(gws.shape[0] - y.shape[0]) // 2, mode="constant")
    _, fft_square = get_spectrum(square)

    ax[0].plot(x * 1e6, gws, label="Gaussian Weighted Sinusoid")
    ax[0].plot(x * 1e6, square, label="Square Wave", alpha=0.7, color="C3")
    ax[1].plot(f, fft_gws, label="Gaussian Weighted Sinusoid")
    ax[1].plot(f, fft_square, label="Square Wave", alpha=0.7, color="C3")

    ax[0].set_xlabel("Time [us]")
    ax[1].set_xlabel("Frequency [Hz]")

    for a in ax:
        a.legend()
        a.grid()

    plt.show()
