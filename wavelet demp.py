import numpy as np
import pywt
import matplotlib.pyplot as plt
from skimage import data, img_as_float, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim, mean_squared_error
from collections import Counter

# --------------------------
# BAYES SHRINK
# --------------------------
def bayes_shrink(detail):
    sigma_n = np.median(np.abs(detail)) / 0.6745
    var = np.var(detail)
    sigma_x = np.sqrt(max(var - sigma_n**2, 1e-6))
    return sigma_n**2 / sigma_x

def soft_threshold(c, t):
    return np.sign(c) * np.maximum(np.abs(c) - t, 0)

# --------------------------
# ADAPTIVE QUANTIZATION
# --------------------------
def adaptive_quantization(detail, base_step):
    variance = np.var(detail)
    return base_step * (1 + variance)

# --------------------------
# RLE
# --------------------------
def rle_encode(arr):
    flat = arr.flatten()
    result = []
    prev = flat[0]
    count = 1

    for val in flat[1:]:
        if val == prev:
            count += 1
        else:
            result.append((prev, count))
            prev = val
            count = 1
    result.append((prev, count))
    return result

def rle_bits(encoded):
    symbols = [val for val, _ in encoded]
    counter = Counter(symbols)
    total = len(symbols)

    entropy = 0
    for count in counter.values():
        p = count / total
        entropy -= p * np.log2(p)

    avg_run_length_bits = np.mean([np.log2(c + 1) for _, c in encoded])
    return total * entropy + len(encoded) * avg_run_length_bits

# --------------------------
# MAIN COMPRESSION FUNCTION
# --------------------------
def compress_wavelet(image, wavelet='db2', step=0.2, level=2):

    coeffs = pywt.wavedec2(image, wavelet, level=level)
    new_coeffs = [coeffs[0]]

    # Thresholding
    for detail in coeffs[1:]:
        cH, cV, cD = detail

        tH = bayes_shrink(cH)
        tV = bayes_shrink(cV)
        tD = bayes_shrink(cD)

        new_coeffs.append((
            soft_threshold(cH, tH),
            soft_threshold(cV, tV),
            soft_threshold(cD, tD)
        ))

    # Quantization
    quantized = [np.round(new_coeffs[0] / step)]
    quant_steps = [step]

    for detail in new_coeffs[1:]:
        cH, cV, cD = detail

        qH = adaptive_quantization(cH, step)
        qV = adaptive_quantization(cV, step)
        qD = adaptive_quantization(cD, step)

        quantized.append((
            np.round(cH / qH),
            np.round(cV / qV),
            np.round(cD / qD)
        ))
        quant_steps.append((qH, qV, qD))

    # Reconstruction (using exact adaptive subband steps)
    rec_coeffs = [quantized[0] * quant_steps[0]]

    for detail, (qH, qV, qD) in zip(quantized[1:], quant_steps[1:]):
        cH, cV, cD = detail
        rec_coeffs.append((cH * qH, cV * qV, cD * qD))

    reconstructed = pywt.waverec2(rec_coeffs, wavelet)
    reconstructed = np.clip(reconstructed[:image.shape[0], :image.shape[1]], 0, 1)

    # Metrics
    mse = mean_squared_error(image, reconstructed)
    rmse = np.sqrt(mse)
    psnr = peak_signal_noise_ratio(image, reconstructed)
    ssim_val = ssim(image, reconstructed, data_range=1.0)

    # RLE Compression
    flat = []
    for c in quantized:
        if isinstance(c, tuple):
            for sub in c:
                flat.extend(sub.flatten())
        else:
            flat.extend(c.flatten())

    flat = np.array(flat, dtype=int)

    encoded = rle_encode(flat)
    compressed_bits = rle_bits(encoded)
    original_bits = image.size * 8

    cr = original_bits / compressed_bits if compressed_bits > 0 else 0

    return reconstructed, psnr, ssim_val, rmse, cr

# --------------------------
# LOAD IMAGES
# --------------------------
images = {
    "Cameraman": img_as_float(data.camera()),
    "Coins": img_as_float(transform.resize(data.coins(), (512, 512))),
    "Moon": img_as_float(transform.resize(data.moon(), (512, 512)))
}

# --------------------------
# RUN FOR 256 SIZE ONLY
# --------------------------
size = 256

for name, img in images.items():

    img_resized = transform.resize(img, (size, size))

    reconstructed, psnr, ssim_val, rmse, cr = compress_wavelet(
        img_resized, wavelet='db2', step=0.2, level=2
    )

    print(f"\n{name}")
    print(f"PSNR: {psnr:.2f}, SSIM: {ssim_val:.3f}, RMSE: {rmse:.4f}, CR: {cr:.2f}")

    # --------------------------
    # SHOW IMAGES
    # --------------------------
    plt.figure(figsize=(8,4))

    plt.subplot(1,2,1)
    plt.imshow(img_resized, cmap='gray')
    plt.title("Original")
    plt.axis('off')

    plt.subplot(1,2,2)
    plt.imshow(reconstructed, cmap='gray')
    plt.title("Reconstructed")
    plt.axis('off')

    plt.suptitle(name)
    plt.show()