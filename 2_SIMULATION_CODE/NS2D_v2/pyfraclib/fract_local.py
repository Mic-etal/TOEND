import numpy as np
from scipy.fft import fft, ifft
from scipy.ndimage import gaussian_filter
import pywt

def fract_local_fft_windowed(f, alpha_map, dx):
    """
    Approximation locale de la dérivée fractionnaire par Fourier avec fenêtrage.
    """
    n = len(f)
    k = 2 * np.pi * np.fft.fftfreq(n, dx)
    window = np.hanning(n)
    f_windowed = f * window
    f_hat = fft(f_windowed)
    alpha_smooth = gaussian_filter(alpha_map, sigma=2)
    alpha_mean = np.mean(alpha_smooth)
    df = np.real(ifft((1j * k)**alpha_mean * f_hat))
    return df

def fract_local_wavelet(f, alpha_map, wavelet='db4'):
    """
    Approximation par ondelettes discrètes (PyWavelets) avec modulation d'ordre variable.
    """
    coeffs = pywt.wavedec(f, wavelet)
    alpha_mean = np.mean(alpha_map)
    # Appliquer une modulation dans l'espace des coefficients (simplifié)
    for i in range(len(coeffs)):
        coeffs[i] = coeffs[i] * (1j)**alpha_mean  # Approximation à affiner
    return np.real(pywt.waverec(coeffs, wavelet))

def fract_local_fd_adaptive(f, alpha_map, dx):
    """
    Approximation par différences finies adaptatives.
    """
    df = np.zeros_like(f)
    for i in range(1, len(f)-1):
        h = dx * (1 + 0.5 * abs(alpha_map[i] - alpha_map[i-1]))
        df[i] = (f[i+1] - f[i-1]) / (2 * h**alpha_map[i])
    return df

def test_fract_local():
    import matplotlib.pyplot as plt
    x = np.linspace(-5, 5, 512)
    dx = x[1] - x[0]
    f = np.exp(-x**2)
    alpha_map = 0.5 + 0.4 * np.sin(x)

    df_fft = fract_local_fft_windowed(f, alpha_map, dx)
    df_wavelet = fract_local_wavelet(f, alpha_map)
    df_fd = fract_local_fd_adaptive(f, alpha_map, dx)

    plt.plot(x, df_fft, label="FFT localisée")
    plt.plot(x, df_wavelet, label="Ondelette")
    plt.plot(x, df_fd, label="Différences adaptatives")
    plt.legend()
    plt.title("Comparaison des dérivées fractales locales")
    plt.show()

if __name__ == "__main__":
    test_fract_local()
