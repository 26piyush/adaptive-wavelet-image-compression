import numpy as np
import pywt
import matplotlib.pyplot as plt
import pandas as pd
from skimage import data, img_as_float, transform
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim, mean_squared_error
from collections import Counter
from PIL import Image
import io
from wavelet_codec import compress_image_to_bytes

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
# RUN LENGTH ENCODING
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
# JPEG BASELINE
# --------------------------
def jpeg_compress(image, quality=50):
    img = Image.fromarray((image * 255).astype(np.uint8))

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=quality)

    compressed_size = len(buffer.getvalue()) * 8
    original_size = image.size * 8

    cr = original_size / compressed_size

    buffer.seek(0)
    decompressed = Image.open(buffer)
    decompressed = np.array(decompressed) / 255.0

    psnr = peak_signal_noise_ratio(image, decompressed)
    ssim_val = ssim(image, decompressed, data_range=1.0)

    return psnr, ssim_val, cr

# --------------------------
# COMPRESSION PIPELINE
# --------------------------
def compress_wavelet(image, wavelet, step, level):

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

    # Adaptive Quantization
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

    bpp = compressed_bits / image.size
    cr = original_bits / compressed_bits if compressed_bits > 0 else 0

    return psnr, ssim_val, rmse, bpp, cr

# --------------------------
# ADAPTIVE STEP SEARCH
# --------------------------
def find_balanced_step(image, wavelet, size, level):

    if size == 64:
        steps = np.linspace(0.1, 0.7, 20)
        min_psnr = 28
        min_cr = 10
    elif size == 128:
        steps = np.linspace(0.08, 0.5, 20)
        min_psnr = 30
        min_cr = 10
    else:
        steps = np.linspace(0.05, 0.35, 20)
        min_psnr = 32
        min_cr = 10

    best = None

    for step in steps:
        psnr, ssim_val, rmse, bpp, cr = compress_wavelet(image, wavelet, step, level)

        if cr >= min_cr and psnr >= min_psnr:
            return step, psnr, ssim_val, rmse, bpp, cr

        score = 3 * max(0, min_cr - cr) + abs(psnr - 33)

        if best is None or score < best[0]:
            best = (score, step, psnr, ssim_val, rmse, bpp, cr)

    return best[1:]

# --------------------------
# DATASET
# --------------------------
images = {
    "Cameraman": img_as_float(data.camera()),
    "Coins": transform.resize(img_as_float(data.coins()), (512, 512)),
    "Moon": transform.resize(img_as_float(data.moon()), (512, 512))
}

sizes = [256, 128, 64]
wavelets = ['db1', 'db2', 'db4']
levels = [1, 2, 3]

results = []

# --------------------------
# MAIN LOOP
# --------------------------
for name, img in images.items():
    print(f"\n===== {name} =====")

    for size in sizes:
        img_resized = transform.resize(img, (size, size))

        for wavelet in wavelets:
            for level in levels:

                step, psnr, ssim_val, rmse, bpp, cr = find_balanced_step(
                    img_resized, wavelet, size, level
                )

                # Physical bitstream serialization using Canonical Huffman AWC Codec
                bin_bytes = compress_image_to_bytes(img_resized, wavelet=wavelet, step=step, level=level)
                phys_cr = img_resized.size / len(bin_bytes) if len(bin_bytes) > 0 else 0
                phys_bpp = (len(bin_bytes) * 8.0) / img_resized.size

                jpeg_psnr, jpeg_ssim, jpeg_cr = jpeg_compress(img_resized)

                print(f"{size} | {wavelet} | L{level} | PSNR: {psnr:.2f} | SSIM: {ssim_val:.3f} | CR: {cr:.2f} | PhysCR: {phys_cr:.2f}")

                results.append([
                    name, size, wavelet, level,
                    psnr, ssim_val, rmse, bpp, cr,
                    phys_bpp, phys_cr,
                    jpeg_psnr, jpeg_ssim, jpeg_cr
                ])

# --------------------------
# SAVE RESULTS
# --------------------------
df = pd.DataFrame(results, columns=[
    "Image", "Size", "Wavelet", "Level",
    "PSNR", "SSIM", "RMSE", "BPP", "CR",
    "Physical_BPP", "Physical_CR",
    "JPEG_PSNR", "JPEG_SSIM", "JPEG_CR"
])

df.to_csv("final_research_results.csv", index=False)
print("\nSaved final_research_results.csv")
# --------------------------
# RATE-DISTORTION GRAPH
# --------------------------
plt.figure(figsize=(8,6))

for wavelet in wavelets:
    psnrs = []
    crs = []

    for step in np.linspace(0.05, 0.5, 15):
        psnr, _, _, _, cr = compress_wavelet(images["Cameraman"], wavelet, step, level=2)
        psnrs.append(psnr)
        crs.append(cr)

    plt.plot(crs, psnrs, marker='o', label=wavelet)

plt.xlabel("Compression Ratio")
plt.ylabel("PSNR (dB)")
plt.title("Rate-Distortion Curve (Final Model)")
plt.legend()
plt.grid()

plt.savefig("rate_distortion_curve.png", dpi=300, bbox_inches='tight')
print("Saved rate_distortion_curve.png")
plt.close()