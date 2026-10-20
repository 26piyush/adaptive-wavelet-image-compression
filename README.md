# Adaptive Wavelet-Based Image Compression

An end-to-end lossy image compression framework integrating **Discrete Wavelet Transform (DWT)**, **Bayesian Thresholding (BayesShrink)**, **Variance-Driven Adaptive Quantization**, and **Physical Bitstream Serialization** using **Run-Length Encoding (RLE)** and **Canonical Huffman Coding**.

---

## Key Features

- **Multi-Resolution Analysis**: Decomposes images using Daubechies orthogonal wavelets (`db1`, `db2`, `db4`) across multiple decomposition levels (Levels 1–3).
- **BayesShrink Soft Thresholding**: Calculates Bayesian subband thresholds based on noise standard deviation ($\sigma_n$) and signal standard deviation ($\sigma_x$) to suppress noise and sparsify detail subbands.
- **Adaptive Subband Quantization**: Varies quantization step sizes dynamically according to local subband variance.
- **Physical Bitstream Codec (`.bin`)**: Encodes quantized runs into a standalone binary format (`AWC1`) with bit-level Canonical Huffman coding, achieving true file compression on disk.
- **Comprehensive Benchmarks**: Evaluated on *Cameraman*, *Coins*, and *Moon* against standard **JPEG (quality=50)** over 81 parameter configurations.

---

## Repository Structure

```
├── wavelet_codec.py          # Standalone CLI binary compressor & decompressor (.bin)
├── wavelwt.py                # Full 81-run research benchmark runner & RD curve generator
├── wavelet demp.py           # Interactive visual demo (Original vs Reconstructed)
├── final_research_results.csv# Benchmark results dataset across all configurations
├── rate_distortion_curve.png # Rate-Distortion curves comparing db1, db2, and db4
└── requirements.txt          # Python dependencies
```

---

## Installation & Requirements

Ensure you have Python 3.8+ installed. Install the dependencies via:

```bash
pip install -r requirements.txt
```

---

## Usage

### 1. Standalone Binary Codec (`wavelet_codec.py`)

Compress any image into a standalone `.bin` file:
```bash
python wavelet_codec.py compress "input.jpg" "output.bin" --wavelet db2 --level 2 --step 0.2
```

Decompress the `.bin` file back to an image:
```bash
python wavelet_codec.py decompress "output.bin" "reconstructed.png"
```

### 2. Interactive Visual Demo (`wavelet demp.py`)

Run the visualizer to view side-by-side reconstructions for test images:
```bash
python "wavelet demp.py"
```

### 3. Run Benchmark Suite (`wavelwt.py`)

Executes the full grid search, evaluates against JPEG, and exports updated results:
```bash
python wavelwt.py
```

---

## Experimental Results Summary

- **Wavelet Selection**: `db2` and `db4` outperform `db1` (Haar), preventing blocking artifacts and achieving higher PSNR and SSIM.
- **Small-Image Advantage over JPEG**: At low resolutions ($64 \times 64$), the wavelet framework sustains high compression ratios ($\text{CR} \approx 13.37\times$), whereas JPEG drops to $\approx 5.35\times$ due to fixed JFIF header overhead.
- **Physical Disk Compression**: Reaches up to **$24.33\times$** real disk compression ratio ($513\text{ KB} \to 21.1\text{ KB}$) with $\text{PSNR} > 34\text{ dB}$.
