"""
Adaptive Wavelet-Based Image Compression Codec (AWC)
Integrates Discrete Wavelet Transform (DWT), Bayesian Thresholding (BayesShrink),
Variance-Driven Adaptive Quantization, and Physical Canonical Huffman Bitstream Serialization.
"""

import os
import io
import struct
import heapq
import numpy as np
import pywt
from collections import Counter
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim, mean_squared_error


# =====================================================================
# 1. BITSTREAM I/O (BitWriter & BitReader)
# =====================================================================

class BitWriter:
    """Packs individual bits into bytes."""
    def __init__(self):
        self.buffer = bytearray()
        self.current_byte = 0
        self.bit_count = 0
        self.total_bits = 0

    def write_bit(self, bit: int):
        self.current_byte = (self.current_byte << 1) | (1 if bit else 0)
        self.bit_count += 1
        self.total_bits += 1
        if self.bit_count == 8:
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0

    def write_bits(self, bit_str: str):
        for b in bit_str:
            self.write_bit(b == '1')

    def flush(self) -> bytes:
        if self.bit_count > 0:
            self.current_byte = self.current_byte << (8 - self.bit_count)
            self.buffer.append(self.current_byte)
            self.current_byte = 0
            self.bit_count = 0
        return bytes(self.buffer)


class BitReader:
    """Reads individual bits sequentially from a byte array."""
    def __init__(self, data_bytes: bytes):
        self.data = data_bytes
        self.byte_idx = 0
        self.current_byte = 0
        self.bit_count = 0

    def read_bit(self):
        if self.bit_count == 0:
            if self.byte_idx >= len(self.data):
                return None
            self.current_byte = self.data[self.byte_idx]
            self.byte_idx += 1
            self.bit_count = 8
        self.bit_count -= 1
        return (self.current_byte >> self.bit_count) & 1


# =====================================================================
# 2. CANONICAL HUFFMAN TREE & CODEBOOK
# =====================================================================

class _HuffmanNode:
    def __init__(self, sym, freq):
        self.sym = sym
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


def compute_huffman_lengths(frequencies: dict) -> dict:
    """Builds a Huffman tree and returns symbol -> code_length."""
    if not frequencies:
        return {}
    if len(frequencies) == 1:
        sym = next(iter(frequencies))
        return {sym: 1}

    heap = [_HuffmanNode(sym, freq) for sym, freq in frequencies.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        n1 = heapq.heappop(heap)
        n2 = heapq.heappop(heap)
        parent = _HuffmanNode(None, n1.freq + n2.freq)
        parent.left = n1
        parent.right = n2
        heapq.heappush(heap, parent)

    root = heap[0]
    lengths = {}

    def _traverse(node, depth):
        if node.sym is not None:
            lengths[node.sym] = depth
            return
        if node.left:
            _traverse(node.left, depth + 1)
        if node.right:
            _traverse(node.right, depth + 1)

    _traverse(root, 0)
    return lengths


def canonical_huffman_codes(lengths_dict: dict) -> dict:
    """
    Constructs canonical prefix-free codes from symbol lengths.
    Deterministic ordering: sorted by (code_length, symbol).
    """
    sorted_syms = sorted(lengths_dict.keys(), key=lambda s: (lengths_dict[s], s))
    codes = {}
    code = 0
    prev_len = 0
    for sym in sorted_syms:
        l = lengths_dict[sym]
        code = code << (l - prev_len)
        codes[sym] = bin(code)[2:].zfill(l)
        code += 1
        prev_len = l
    return codes


def build_canonical_decode_tree(lengths_dict: dict) -> dict:
    """Constructs a fast prefix-trie for decoding canonical Huffman bitstreams."""
    codes = canonical_huffman_codes(lengths_dict)
    tree = {}
    for sym, code_str in codes.items():
        curr = tree
        for bit in code_str:
            if bit not in curr:
                curr[bit] = {}
            curr = curr[bit]
        curr['sym'] = sym
    return tree


# =====================================================================
# 3. MATHEMATICAL WAVELET ALGORITHMS
# =====================================================================

def bayes_shrink(detail: np.ndarray) -> float:
    """
    Bayesian Thresholding (BayesShrink) estimator.
    T = sigma_n^2 / sigma_x
    """
    sigma_n = np.median(np.abs(detail)) / 0.6745
    var = np.var(detail)
    sigma_x = np.sqrt(max(var - sigma_n**2, 1e-6))
    return sigma_n**2 / sigma_x


def soft_threshold(c: np.ndarray, t: float) -> np.ndarray:
    """Applies soft-thresholding to wavelet coefficients."""
    return np.sign(c) * np.maximum(np.abs(c) - t, 0)


def adaptive_quantization(detail: np.ndarray, base_step: float) -> float:
    """Calculates adaptive quantization step size based on subband variance."""
    variance = np.var(detail)
    return base_step * (1.0 + variance)


def rle_encode(arr: np.ndarray) -> list:
    """Run-Length Encodes a 1D integer array into [(val, count), ...]."""
    flat = arr.flatten()
    if len(flat) == 0:
        return []
    result = []
    prev = int(flat[0])
    count = 1
    for val in flat[1:]:
        val_int = int(val)
        if val_int == prev:
            count += 1
        else:
            result.append((prev, count))
            prev = val_int
            count = 1
    result.append((prev, count))
    return result


def rle_decode(runs: list) -> list:
    """Decodes [(val, count), ...] back into 1D integer array."""
    out = []
    for val, count in runs:
        out.extend([val] * count)
    return out


# =====================================================================
# 4. PHYSICAL BITSTREAM ENCODER & DECODER
# =====================================================================

MAGIC_HEADER = b"AWC1"

def compress_image_to_bytes(image: np.ndarray, wavelet='db2', step=0.2, level=2) -> bytes:
    """
    Compresses a 2D grayscale image (float in [0, 1]) into a complete,
    self-describing binary bitstream using DWT, BayesShrink, Adaptive Quantization,
    RLE, and Canonical Huffman Bit-packing.
    """
    h_orig, w_orig = image.shape
    coeffs = pywt.wavedec2(image, wavelet, level=level)

    # 1. Bayesian Thresholding on detail subbands
    new_coeffs = [coeffs[0]]
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

    # 2. Adaptive Quantization
    quantized = [np.round(new_coeffs[0] / step)]
    quant_steps = [step]
    shapes = [coeffs[0].shape]

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
        shapes.append(cH.shape)

    # 3. Serialization to 1D stream and RLE
    flat = []
    for c in quantized:
        if isinstance(c, tuple):
            for sub in c:
                flat.extend(sub.flatten())
        else:
            flat.extend(c.flatten())
    flat = np.array(flat, dtype=int)
    runs = rle_encode(flat)

    # 4. Canonical Huffman Encoding
    counts = Counter(runs)
    lengths = compute_huffman_lengths(counts)
    codes = canonical_huffman_codes(lengths)

    writer = BitWriter()
    for r in runs:
        writer.write_bits(codes[r])
    payload_bytes = writer.flush()

    # 5. Pack Binary Container
    buf = io.BytesIO()
    # Magic (4B)
    buf.write(MAGIC_HEADER)
    # Header: h(2B), w(2B), wavelet_len(1B), level(1B)
    w_name_bytes = wavelet.encode('ascii')
    buf.write(struct.pack('>HHBB', h_orig, w_orig, len(w_name_bytes), level))
    buf.write(w_name_bytes)
    buf.write(struct.pack('>f', float(step)))

    # Subband shapes and exact dequantization steps
    buf.write(struct.pack('>B', len(shapes)))
    for i, sh in enumerate(shapes):
        buf.write(struct.pack('>HH', sh[0], sh[1]))
        if i == 0:
            buf.write(struct.pack('>f', float(quant_steps[0])))
        else:
            qH, qV, qD = quant_steps[i]
            buf.write(struct.pack('>fff', float(qH), float(qV), float(qD)))

    # Stream metadata: total_coeffs(4B), num_runs(4B), num_symbols(2B)
    buf.write(struct.pack('>III', len(flat), len(runs), len(lengths)))

    # Compact Canonical Codebook Table:
    # For each symbol: val (int32: 4B), count (uint32: 4B), code_length (uint8: 1B) = 9 bytes
    for sym, l in sorted(lengths.items(), key=lambda s: (s[1], s[0])):
        val, count = sym
        buf.write(struct.pack('>iIB', int(val), int(count), int(l)))

    # Payload metadata: total_bits(4B), payload_bytes_len(4B)
    buf.write(struct.pack('>II', writer.total_bits, len(payload_bytes)))
    buf.write(payload_bytes)

    return buf.getvalue()


def decompress_image_from_bytes(data_bytes: bytes) -> tuple:
    """
    Decompresses a binary bitstream back into a 2D float image [0, 1].
    Returns (reconstructed_image, metadata_dict).
    """
    stream = io.BytesIO(data_bytes)
    magic = stream.read(4)
    if magic != MAGIC_HEADER:
        raise ValueError(f"Invalid magic header: {magic}. Expected {MAGIC_HEADER}.")

    h_orig, w_orig, w_name_len, level = struct.unpack('>HHBB', stream.read(6))
    wavelet = stream.read(w_name_len).decode('ascii')
    base_step = struct.unpack('>f', stream.read(4))[0]

    num_shapes = struct.unpack('>B', stream.read(1))[0]
    shapes = []
    quant_steps = []
    for i in range(num_shapes):
        sh = struct.unpack('>HH', stream.read(4))
        shapes.append(sh)
        if i == 0:
            s0 = struct.unpack('>f', stream.read(4))[0]
            quant_steps.append(s0)
        else:
            qH, qV, qD = struct.unpack('>fff', stream.read(12))
            quant_steps.append((qH, qV, qD))

    total_coeffs, total_runs, num_symbols = struct.unpack('>III', stream.read(12))

    # Read Canonical Codebook
    lengths = {}
    for _ in range(num_symbols):
        val, count, code_len = struct.unpack('>iIB', stream.read(9))
        lengths[(val, count)] = code_len

    total_bits, payload_len = struct.unpack('>II', stream.read(8))
    payload = stream.read(payload_len)

    # Decode Canonical Huffman Stream
    decode_tree = build_canonical_decode_tree(lengths)
    reader = BitReader(payload)
    decoded_runs = []
    curr = decode_tree

    for _ in range(total_bits):
        b = reader.read_bit()
        b_str = '1' if b == 1 else '0'
        curr = curr[b_str]
        if 'sym' in curr:
            decoded_runs.append(curr['sym'])
            curr = decode_tree

    if len(decoded_runs) != total_runs:
        raise RuntimeError(f"Run count mismatch: expected {total_runs}, got {len(decoded_runs)}")

    decoded_flat = rle_decode(decoded_runs)
    if len(decoded_flat) != total_coeffs:
        raise RuntimeError(f"Coefficient count mismatch: expected {total_coeffs}, got {len(decoded_flat)}")

    # Unpack 1D array back into multi-resolution wavelet subbands
    quantized = []
    offset = 0
    sh0 = shapes[0]
    n0 = sh0[0] * sh0[1]
    quantized.append(np.array(decoded_flat[offset:offset + n0], dtype=float).reshape(sh0))
    offset += n0

    for i in range(1, num_shapes):
        sh = shapes[i]
        n = sh[0] * sh[1]
        cH = np.array(decoded_flat[offset:offset + n], dtype=float).reshape(sh)
        offset += n
        cV = np.array(decoded_flat[offset:offset + n], dtype=float).reshape(sh)
        offset += n
        cD = np.array(decoded_flat[offset:offset + n], dtype=float).reshape(sh)
        offset += n
        quantized.append((cH, cV, cD))

    # CORRECT INVERSE DEQUANTIZATION:
    # Multiply each detail subband by its corresponding exact adaptive step (qH, qV, qD)
    rec_coeffs = [quantized[0] * quant_steps[0]]
    for detail, (qH, qV, qD) in zip(quantized[1:], quant_steps[1:]):
        cH, cV, cD = detail
        rec_coeffs.append((cH * qH, cV * qV, cD * qD))

    # Inverse DWT
    reconstructed = pywt.waverec2(rec_coeffs, wavelet)
    reconstructed = np.clip(reconstructed[:h_orig, :w_orig], 0.0, 1.0)

    metadata = {
        "height": h_orig,
        "width": w_orig,
        "wavelet": wavelet,
        "level": level,
        "base_step": base_step,
        "total_coefficients": total_coeffs,
        "num_runs": total_runs,
        "num_unique_symbols": num_symbols,
        "payload_bytes": payload_len,
        "total_compressed_bytes": len(data_bytes)
    }
    return reconstructed, metadata


def compress_image_file(image_path: str, output_bin_path: str, wavelet='db2', step=0.2, level=2) -> dict:
    """Compresses an image file from disk and saves the standalone .bin bitstream."""
    img_pil = Image.open(image_path).convert('L')
    img_arr = np.array(img_pil, dtype=float) / 255.0

    compressed_bytes = compress_image_to_bytes(img_arr, wavelet=wavelet, step=step, level=level)

    with open(output_bin_path, 'wb') as f:
        f.write(compressed_bytes)

    # Decompress to evaluate quality metrics
    rec_arr, meta = decompress_image_from_bytes(compressed_bytes)

    raw_bytes = img_arr.size  # 1 byte per pixel for 8-bit grayscale
    bin_size = len(compressed_bytes)
    cr = raw_bytes / bin_size if bin_size > 0 else 0
    bpp = (bin_size * 8.0) / img_arr.size
    psnr_val = peak_signal_noise_ratio(img_arr, rec_arr)
    ssim_val = ssim(img_arr, rec_arr, data_range=1.0)
    rmse_val = np.sqrt(mean_squared_error(img_arr, rec_arr))

    stats = {
        "original_path": image_path,
        "output_path": output_bin_path,
        "raw_bytes": raw_bytes,
        "bin_bytes": bin_size,
        "compression_ratio": cr,
        "bpp": bpp,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "rmse": rmse_val,
        **meta
    }
    return stats


def decompress_image_file(bin_path: str, output_image_path: str) -> dict:
    """Reads a standalone .bin bitstream and writes the reconstructed image to disk."""
    with open(bin_path, 'rb') as f:
        data_bytes = f.read()

    rec_arr, meta = decompress_image_from_bytes(data_bytes)
    rec_pil = Image.fromarray((rec_arr * 255.0).astype(np.uint8))
    rec_pil.save(output_image_path)

    meta["output_image_path"] = output_image_path
    return meta


# =====================================================================
# 5. CLI INTERFACE
# =====================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Adaptive Wavelet Compression CLI")
    subparsers = parser.add_subparsers(dest="command")

    # Compress
    p_comp = subparsers.add_parser("compress", help="Compress image to .bin file")
    p_comp.add_argument("input", help="Path to input image")
    p_comp.add_argument("output", help="Path to output .bin file")
    p_comp.add_argument("--wavelet", default="db2", help="Wavelet basis (default: db2)")
    p_comp.add_argument("--level", type=int, default=2, help="Decomposition level (default: 2)")
    p_comp.add_argument("--step", type=float, default=0.2, help="Quantization step (default: 0.2)")

    # Decompress
    p_decomp = subparsers.add_parser("decompress", help="Decompress .bin file to image")
    p_decomp.add_argument("input", help="Path to input .bin file")
    p_decomp.add_argument("output", help="Path to reconstructed image output")

    args = parser.parse_args()
    if args.command == "compress":
        stats = compress_image_file(args.input, args.output, wavelet=args.wavelet, step=args.step, level=args.level)
        print(f"\n[Compression Complete]")
        print(f"Raw image size:    {stats['raw_bytes']:,} bytes")
        print(f"Output .bin size:  {stats['bin_bytes']:,} bytes")
        print(f"Physical CR:       {stats['compression_ratio']:.2f}x")
        print(f"Physical BPP:      {stats['bpp']:.3f} bits/pixel")
        print(f"Reconstructed PSNR:{stats['psnr']:.2f} dB")
        print(f"Reconstructed SSIM:{stats['ssim']:.4f}")
    elif args.command == "decompress":
        meta = decompress_image_file(args.input, args.output)
        print(f"\n[Decompression Complete]")
        print(f"Restored image saved to: {meta['output_image_path']}")
        print(f"Image dimensions:        {meta['width']} x {meta['height']}")
        print(f"Wavelet:                 {meta['wavelet']} (Level {meta['level']})")
    else:
        parser.print_help()
