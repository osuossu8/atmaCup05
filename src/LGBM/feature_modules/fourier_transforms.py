import cmath
from numpy.fft import fftn, ifftn, fftfreq


# DFT
def dft(f):
    n = len(f)
    A = np.arange(n)
    M = cmath.e**(-1j * A.reshape(1, -1) * A.reshape(-1, 1) * 2 * cmath.pi / n)
    return np.sum(f * M, axis=1)

def polar_transform(x):
    return [cmath.polar(arr)[0] for arr in x]

def phi_transform(x):
    return [cmath.polar(arr)[1] for arr in x]


# FFT
def fft_low_pass_diff_transform(x):
    return abs(ifftn(np.where(abs(fftfreq(len(x), d=1 / 1)) > (1/2), 0,  fftn(x))).real - x)


def log_fftn_transform(x):
    return np.log10(abs(fftn(x)[1:int(len(x) / 2)]))
