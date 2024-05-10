import numpy as np
from scipy.signal import convolve2d

def validate_image(img):
    if len(img.shape) != 2:
        raise ValueError('Input must be a grayscale image.')
    return img.astype(np.float64)

def create_filters(win_size):
    radius = (win_size - 1) // 2
    x = np.arange(-radius, radius + 1)
    w0 = np.ones_like(x)
    w1 = np.exp(-2j * np.pi * x / win_size)
    return w0, w1, np.conj(w1)

def apply_filters(img, filters, conv_mode='valid'):
    w0, w1, w2 = filters
    responses = [convolve2d(convolve2d(img, w.reshape(-1, 1), mode=conv_mode), v.reshape(1, -1), mode=conv_mode)
                 for w, v in [(w0, w1), (w1, w0), (w1, w1), (w1, w2)]]
    return np.stack([resp.real for resp in responses] + [resp.imag for resp in responses], axis=-1)

def compute_codewords(freq_resp):
    codewords = np.zeros(freq_resp.shape[:2], dtype=int)
    for i in range(freq_resp.shape[-1]):
        codewords += (freq_resp[:, :, i] > 0) * (2 ** i)
    return codewords

def lpq(img):
    img = validate_image(img)
    filters = create_filters(win_size=3)
    freq_resp = apply_filters(img, filters)
    lpq_codewords = compute_codewords(freq_resp)
    histogram = np.histogram(lpq_codewords.ravel(), bins=np.arange(257))[0]
    normalized_histogram = histogram / histogram.sum()
    return normalized_histogram