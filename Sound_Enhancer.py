import librosa
import librosa.core
import soundfile as sf
import numpy as np
import torch
import os
import subprocess
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from scipy.signal import butter, sosfilt, sosfiltfilt, lfilter, iirfilter, fftconvolve
from scipy.ndimage import gaussian_filter1d
from scipy import signal as scipy_signal
from typing import Tuple, Optional, Union, List # MODIFIED: Added List
import warnings
import gc
from functools import lru_cache
import psutil
import logging
from scipy.signal import firwin, filtfilt, hilbert, remez
from scipy.fft import fft, ifft, fftfreq
import scipy.interpolate

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Control warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- GPU Support Check (Improved Stability) ---
def initialize_gpu():
    """GPU initialization process (enhanced error handling)."""
    try:
        if torch.backends.mps.is_available():
            # Check if MPS device is working
            test_tensor = torch.zeros(1).to("mps")
            del test_tensor
            torch.mps.empty_cache()
            logger.info("MPS device is available and working.") # ADDED: Log success
            return True
    except Exception as e:
        logger.warning(f"MPS initialization failed: {e}")
        return False
    logger.info("MPS device not available.") # ADDED: Log if not available
    return False

USE_GPU = initialize_gpu()
DEVICE = "mps" if USE_GPU else "cpu"
logger.info(f"Using device: {DEVICE}") # ADDED: Log device in use

# Memory management settings
if USE_GPU:
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0" # Recommended for MPS to avoid out-of-memory
    # torch.mps.set_per_process_memory_fraction(0.8) # REMOVED: Not applicable to MPS or deprecated

# CPU parallel processing settings
NUM_CORES = min(psutil.cpu_count(logical=False), 8)  # Limit to physical cores, max 8
torch.set_num_threads(NUM_CORES)

class MemoryManager:
    """Memory management class."""
    
    @staticmethod
    def clear_cache():
        """Clear memory cache."""
        gc.collect()
        if USE_GPU:
            torch.mps.empty_cache()
    
    @staticmethod
    def check_memory():
        """Check memory usage."""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            logger.warning(f"High memory usage: {memory.percent}%")
            MemoryManager.clear_cache()
            return False
        return True

# MODIFIED: safe_to_gpu function to handle negative strides
def safe_to_gpu(array: Union[np.ndarray, torch.Tensor, list], dtype=torch.float32) -> torch.Tensor:
    """Safe GPU transfer (with memory check, handles negative strides)."""
    try:
        # Ensure contiguity for NumPy arrays
        if isinstance(array, np.ndarray):
            # Address negative strides or non-contiguous memory layouts
            processed_array = np.ascontiguousarray(array)
        else:
            processed_array = array

        if not MemoryManager.check_memory():
            # If memory is low, process on CPU. Convert processed_array to tensor.
            return torch.tensor(processed_array, dtype=dtype)
        
        if isinstance(processed_array, torch.Tensor):
            # If already a tensor
            if USE_GPU and processed_array.device.type != DEVICE: # Check against current DEVICE
                return processed_array.to(DEVICE, dtype=dtype)
            return processed_array.to(dtype=dtype) # Convert dtype or maintain current
        
        # Create tensor from NumPy array or list
        tensor = torch.tensor(processed_array, dtype=dtype)
        if USE_GPU:
            return tensor.to(DEVICE)
        return tensor
    except Exception as e:
        logger.warning(f"GPU transfer failed, processing on CPU: {e}")
        # On fallback, also create tensor from a contiguous array
        if isinstance(array, np.ndarray):
            cpu_array = np.ascontiguousarray(array)
        else:
            cpu_array = array
        return torch.tensor(cpu_array, dtype=dtype)

def safe_to_cpu(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Safe CPU transfer."""
    try:
        if isinstance(tensor, torch.Tensor):
            # Ensure tensor is contiguous before converting to numpy to avoid potential stride issues
            return tensor.contiguous().cpu().detach().numpy()
        # If already a NumPy array or other, convert to NumPy array
        return np.array(tensor) 
    except Exception as e:
        logger.error(f"CPU transfer error: {e}")
        # Fallback: try to convert to array directly if error
        return np.array(tensor)


# --- Integrated Effect Settings (with validation) ---
class ParameterValidator:
    """Parameter validation class."""
    
    @staticmethod
    def validate_params(params):
        """Validate parameter plausibility."""
        validated = params.copy()
        
        # Validate numerical ranges
        validated["target_sr"] = max(22050, min(192000, validated["target_sr"]))
        validated["oversample_factor"] = max(1, min(4, validated["oversample_factor"]))
        validated["exciter_amount"] = np.clip(validated["exciter_amount"], 0.0, 1.0)
        validated["saturation_amount"] = np.clip(validated["saturation_amount"], 0.0, 1.0)
        validated["stereo_width"] = np.clip(validated["stereo_width"], 0.5, 3.0)
        validated["compression_ratio"] = max(1.0, min(20.0, validated["compression_ratio"]))
        
        return validated

PARAMS = ParameterValidator.validate_params({
    # [Common Settings]
    "target_sr": 96000,             # Target sampling rate for processing
    "oversample_factor": 2,         # Oversampling factor
    "chunk_size": 8192,             # Processing chunk size

    # [BBE Processor Settings]
    "bbe_on": True,                 # Enable BBE processing
    "headroom_db": 3.0,             # Headroom in dB for normalization

    "low_shelf_on": True,           # Enable low-shelf filter
    "low_cutoff_hz": 100,           # Low-shelf cutoff frequency (Hz)
    "low_gain_db": 4.0,             # Low-shelf gain (dB)
    "low_q": 0.7,                   # Low-shelf Q factor

    "high_shelf_on": True,          # Enable high-shelf filter
    "high_cutoff_hz": 6000,         # High-shelf cutoff frequency (Hz)
    "high_gain_db": 5.0,            # High-shelf gain (dB)
    "high_q": 0.8,                  # High-shelf Q factor

    "bass_enhance_on": True,        # Enable bass enhancement
    "bass_freq": 80,                # Bass enhancement center frequency (Hz)
    "bass_intensity": 0.7,          # Bass enhancement intensity (0.0-1.0)

    "exciter_on": True,             # Enable harmonic exciter
    "exciter_amount": 0.5,          # Exciter amount (0.0-1.0)
    "exciter_low_freq": 3000,       # Exciter low frequency limit (Hz)
    "exciter_high_freq": 12000,     # Exciter high frequency limit (Hz)
    "preemph_strength": 0.8,        # Pre-emphasis strength for exciter
    "hpss_harmonic_margin": 2.5,    # HPSS margin for exciter (if HPSS is used)

    "dynamic_eq_on": True,          # Enable dynamic EQ
    "dynamic_threshold": -24,       # Dynamic EQ threshold (dB)
    "dynamic_ratio": 1.5,           # Dynamic EQ ratio
    "dynamic_attack": 10,           # Dynamic EQ attack time (ms)
    "dynamic_release": 150,         # Dynamic EQ release time (ms)
    "dynamic_bands": 3,             # Number of dynamic EQ bands

    "spectral_processor_on": True,  # Enable spectral processor
    "spectral_low": 1.2,            # Spectral gain for low frequencies
    "spectral_mid": 0.9,            # Spectral gain for mid frequencies
    "spectral_high": 1.3,           # Spectral gain for high frequencies
    "spectral_brightness": 1.1,     # Overall spectral brightness factor

    "stereo_width_on": True,        # Enable stereo width adjustment
    "stereo_width": 1.3,            # Stereo width factor (1.0 = original)

    # [Compressor Settings]
    "compressor_on": True,          # Enable compressor
    "compression_ratio": 3.0,       # Main compression ratio
    "mid_compression_ratio": 1.8,   # Mid channel compression ratio (for M/S)
    "side_compression_ratio": 1.3,  # Side channel compression ratio (for M/S)

    # [Limiter Settings]
    "limiter_on": True,             # Enable limiter
    "limiter_threshold": 0.95,      # Limiter threshold (linear amplitude)
    "output_gain": 1.0,             # Final output gain

    # [Saturation Settings]
    "saturation_on": True,          # Enable saturation
    "saturation_amount": 0.5,       # Saturation amount (0.0-1.0)
    "saturation_drive": 1.0,        # Saturation drive
    "saturation_mix": 1.0,          # Saturation mix (wet/dry)

    # Transient Shaper Settings
    "transient_shaper_on": True,    # Enable transient shaper
    "attack_gain": 1.5,             # Gain for attack portion
    "sustain_gain": 0.7,            # Gain for sustain portion

    # Vocal Presence EQ Settings
    "vocal_presence_eq_on": True,   # Enable vocal presence EQ
    "vocal_presence_center_freq": 3000, # Center frequency for vocal presence (Hz)
    "vocal_presence_gain": 2.0,     # Gain for vocal presence (dB)
    "vocal_presence_q": 1.4,        # Q factor for vocal presence EQ
})

# --- Utility Functions (Optimized Version) ---
# MODIFIED: get_butter_filter to handle list/tuple for cutoff (bandpass/bandstop)
@lru_cache(maxsize=32)
def get_butter_filter(cutoff: Union[float, Tuple[float, float], List[float]], 
                      fs: float, order: int, btype: str) -> np.ndarray:
    """Cached filter design (Butterworth)."""
    nyq = 0.5 * fs
    if isinstance(cutoff, (list, tuple)): # bandpass, bandstop
        if len(cutoff) != 2:
            raise ValueError("Bandpass/bandstop cutoff must be a list/tuple of two frequencies.")
        normal_cutoff = [c / nyq for c in cutoff]
        # Ensure band frequencies are clipped and in ascending order
        normal_cutoff = [np.clip(c, 0.001, 0.999) for c in normal_cutoff]
        if normal_cutoff[0] >= normal_cutoff[1]:
            logger.debug(f"Bandpass/bandstop cutoff frequencies reordered: {normal_cutoff}")
            normal_cutoff.sort()
    else: # low, high
        normal_cutoff = cutoff / nyq
        normal_cutoff = np.clip(normal_cutoff, 0.001, 0.999) # Clip to avoid issues at Nyquist
    return butter(order, normal_cutoff, btype=btype, output='sos') # Use second-order sections


def safe_normalize_audio(y: np.ndarray, headroom_db: float = 1.0) -> np.ndarray:
    """Safe audio normalization."""
    try:
        y = np.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0)  # Handle NaN/Inf
        peak = np.max(np.abs(y))
        if peak > 1e-10:  # Prevent division by zero
            target_peak = np.power(10, -headroom_db/20)
            return y * (target_peak / peak)
        return y
    except Exception as e:
        logger.error(f"Normalization error: {e}")
        return y

def apply_filter_safe(data: np.ndarray, cutoff: Union[float, Tuple[float, float], List[float]], 
                      fs: float, btype:str = 'low', order:int = 4) -> np.ndarray:
    """Safe filter application."""
    try:
        # sosfiltfilt requires len(x) > 3 * max(len(sos[0]), len(sos[1]))
        # A simpler check: len(data) must be greater than 3 times the filter order for SOS.
        # Each SOS section has order 2. Total order is num_sections * 2.
        # The number of sections is roughly order/2.
        # A common rule of thumb is len(data) > 3 * filter_order (referring to the overall order).
        if len(data) < order * 3 + 1: 
            logger.warning(f"Signal length ({len(data)}) too short for filter ({btype}, cutoff={cutoff}). Skipping.")
            return data
        
        sos = get_butter_filter(cutoff, fs, order, btype)
        return sosfiltfilt(sos, data) # Zero-phase filtering
    except Exception as e:
        logger.warning(f"Filter application error ({btype}, cutoff={cutoff}): {e}")
        return data

def encode_mid_side(left: np.ndarray, right: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side encoding (improved numerical stability)."""
    try:
        mid = ((left + right) * 0.5).astype(np.float32)
        side = ((left - right) * 0.5).astype(np.float32)
        return mid, side
    except Exception as e:
        logger.error(f"M/S encoding error: {e}")
        return left.astype(np.float32), np.zeros_like(left, dtype=np.float32) # Fallback

def decode_mid_side(mid: np.ndarray, side: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Mid/Side decoding (improved numerical stability)."""
    try:
        left = (mid + side).astype(np.float32)
        right = (mid - side).astype(np.float32)
        return left, right
    except Exception as e:
        logger.error(f"M/S decoding error: {e}")
        return mid.astype(np.float32), mid.astype(np.float32) # Fallback: output mid for both

# --- High-Performance Effect Functions ---
def harmonic_exciter_effect_optimized(signal, params, sr):
    """Optimized harmonic exciter."""
    try:
        signal_cpu = safe_to_cpu(signal)
        amount = params["exciter_amount"]
        freq_range = [params["exciter_low_freq"], params["exciter_high_freq"]]
        
        if amount < 0.01:  # Skip if effect is negligible
            return safe_to_gpu(signal_cpu)
        
        nyquist = sr / 2
        low_cutoff_norm = np.clip(freq_range[0] / nyquist, 0.001, 0.999)
        high_cutoff_norm = np.clip(freq_range[1] / nyquist, 0.001, 0.999)
        
        if low_cutoff_norm >= high_cutoff_norm: # ADDED: Ensure valid band
            logger.warning(f"Harmonic exciter freq_range invalid: {freq_range}. Processing full signal instead of bandpass.")
            target_band = signal_cpu # Process full signal if band is invalid
        else:
            # Efficient band splitting
            sos_band = butter(4, [low_cutoff_norm, high_cutoff_norm], btype='bandpass', output='sos')
            target_band = sosfiltfilt(sos_band, signal_cpu)

        # Non-linear processing (optimized)
        # 1. Soft clipping
        drive_factor = 1.0 + amount * 2.0
        soft_clip = np.tanh(target_band * drive_factor) 
        if abs(drive_factor) > 1e-6 : soft_clip = soft_clip / drive_factor # Avoid division by zero

        # 2. Even harmonic generation (simplified)
        even_harm = np.sign(target_band) * np.power(np.abs(target_band), 1.5) * amount * 0.3

        # 3. Dynamic mix based on envelope
        envelope = np.abs(target_band)
        envelope = apply_filter_safe(envelope, 50, sr, 'low', 2) # Smooth envelope
        
        dynamic_mix = amount * (1.0 - 0.2 * np.clip(envelope, 0, 1)) # Reduce mix for louder parts
        
        # Final mix of processed components
        processed_harmonics = (soft_clip + even_harm) 
        processed_band_mixed = target_band * (1 - dynamic_mix) + processed_harmonics * dynamic_mix
        
        # Combine with original signal
        # If full signal was processed, 'processed_band_mixed' is the result
        if low_cutoff_norm >= high_cutoff_norm:
             result = processed_band_mixed
        else:
            # Replace the original band content with the processed band content
            # This avoids adding the band on top of itself.
            # result = (signal_cpu - target_band) + processed_band_mixed
            # Simpler approach from before: adding a scaled version of the *difference* or just the processed part
            # The original `result = signal_cpu + processed * 0.5` implied adding something on top.
            # Let's assume 'processed' here refers to the *additional* harmonic content.
            # The 'processed_band_mixed' is the *new* band content.
            # A common way is: original_signal + (processed_band - original_band_in_that_range) * mix_amount
            # The line `result = signal_cpu + processed * 0.5` (where `processed` was `processed_band_mixed` in this context)
            # adds the processed band, scaled, to the full original signal.
            # This is effectively mixing the processed band back in.
            # For additive harmonics, usually it's:
            # result = signal_cpu + (processed_band_mixed - target_band) * amount
            # Given the structure, let's assume the intention was to add the *generated harmonics* to the signal
            # The variable `processed` in the original code was `processed_band_mixed`.
            # Let's refine the mix to be more controlled.
            # If `processed_band_mixed` is the *new* band content, replace old band:
            result = (signal_cpu - target_band) + processed_band_mixed

        return safe_to_gpu(np.clip(result, -1.0, 1.0))
        
    except Exception as e:
        logger.error(f"Harmonic exciter error: {e}")
        return signal # Return original tensor on error

def dynamic_eq_effect_optimized(signal, sr, params):
    """Optimized dynamic EQ."""
    try:
        signal_cpu = safe_to_cpu(signal)
        if not params.get("dynamic_eq_on", False): # Use .get for safety
            return signal # Return original tensor if effect is off
        
        bands = params.get("dynamic_bands", 3)
        threshold_db = params.get("dynamic_threshold", -24)
        ratio = params.get("dynamic_ratio", 1.5)
        # attack_ms = params.get("dynamic_attack", 10) # Not directly used in this simplified envelope detection
        # release_ms = params.get("dynamic_release", 150) # Not directly used

        # Efficient band splitting
        band_edges = np.logspace(np.log10(80), np.log10(min(sr/2.5, 16000)), bands + 1) # Cap high freq
        
        threshold_linear = 10 ** (threshold_db / 20.0)
        
        # Function for parallel processing of bands
        def process_band_deq(args):
            b_idx, low_edge, high_edge = args
            
            # Extract band signal from the original full signal
            current_band_signal_component = np.zeros_like(signal_cpu) 
            if b_idx == 0: # Low band
                current_band_signal_component = apply_filter_safe(signal_cpu, high_edge, sr, 'low', order=2)
            elif b_idx == bands - 1: # High band
                current_band_signal_component = apply_filter_safe(signal_cpu, low_edge, sr, 'high', order=2)
            else: # Mid band(s)
                if low_edge < high_edge: # Ensure valid bandpass
                    sos_bp = get_butter_filter((low_edge, high_edge), sr, order=2, btype='bandpass')
                    current_band_signal_component = sosfiltfilt(sos_bp, signal_cpu)
                else: 
                    return np.zeros_like(signal_cpu) # Should not happen with logspace if sr reasonable

            # Simplified envelope detection
            envelope = np.abs(current_band_signal_component)
            sigma_val = max(1, int(sr / 1000)) # Sigma for Gaussian filter, must be > 0
            envelope_smooth = gaussian_filter1d(envelope, sigma=sigma_val) 
            
            # Compression logic
            gain = np.ones_like(envelope_smooth)
            mask = envelope_smooth > threshold_linear
            
            if np.any(mask):
                env_masked_safe = np.maximum(envelope_smooth[mask], 1e-9) # Avoid division by zero
                gain[mask] = np.power(threshold_linear / env_masked_safe, (ratio - 1.0) / ratio)
            
            # Return the gain-adjusted component for this band
            return current_band_signal_component * gain 

        band_args_list = [(b, band_edges[b], band_edges[b+1]) for b in range(bands)]
        
        # The result is the sum of the processed (gain-adjusted) bands.
        # This implicitly means the original signal is decomposed, each band processed, and then recomposed.
        result_cpu = np.zeros_like(signal_cpu)

        if NUM_CORES > 1 and len(signal_cpu) > 8192 and bands > 1: # Use threading for long signals and multiple bands
            with ThreadPoolExecutor(max_workers=min(NUM_CORES, bands)) as executor:
                processed_bands_components = list(executor.map(process_band_deq, band_args_list))
        else: # Single-threaded execution
            processed_bands_components = [process_band_deq(args) for args in band_args_list]
        
        # Sum the processed band components
        for comp in processed_bands_components:
            result_cpu += comp

        # Clipping might be necessary as dynamic EQ can alter levels.
        # Allowing some headroom here, final clipping/normalization happens later.
        return safe_to_gpu(np.clip(result_cpu, -1.5, 1.5)) 

    except Exception as e:
        logger.error(f"Dynamic EQ error: {e}", exc_info=True)
        return signal # Return original tensor on error


def spectral_processor_effect_optimized(signal, sr, params):
    """Optimized spectral processor."""
    try:
        signal_cpu = safe_to_cpu(signal)
        
        if not params.get("spectral_processor_on", False): # Use .get for safety
            return signal # Return original tensor if effect is off

        # Get parameters
        low_enhance = params.get("spectral_low", 1.0)
        mid_cut = params.get("spectral_mid", 1.0) # Assuming 1.0 is no change
        high_enhance = params.get("spectral_high", 1.0)
        brightness = params.get("spectral_brightness", 1.0)
        
        # Efficient STFT settings
        # Ensure nperseg is not too large for short signals, and not too small overall
        nperseg = min(2048, len(signal_cpu) // 2 if len(signal_cpu) >= 512 else 256)
        nperseg = max(256, nperseg) # Minimum nperseg
        noverlap = nperseg // 2
        
        # Pad signal if shorter than nperseg to allow STFT
        padded_signal_cpu = signal_cpu
        if len(signal_cpu) < nperseg:
            pad_width = nperseg - len(signal_cpu)
            padded_signal_cpu = np.pad(signal_cpu, (0, pad_width), mode='reflect')
        
        # STFT
        f, t, Zxx = scipy_signal.stft(padded_signal_cpu, sr, nperseg=nperseg, noverlap=noverlap)
        
        # Frequency band masks
        low_mask = f < 150
        mid_mask = (f >= 150) & (f < 4000)
        high_mask = f >= 4000
        
        # Spectral processing
        Zxx_processed = Zxx.copy()
        if np.any(low_mask): Zxx_processed[low_mask,:] *= low_enhance
        if np.any(mid_mask): Zxx_processed[mid_mask,:] *= mid_cut
        if np.any(high_mask): Zxx_processed[high_mask,:] *= high_enhance
        
        # Brightness adjustment
        if abs(brightness - 1.0) > 1e-3: # Apply if significantly different from 1.0
            nyq = sr / 2.0
            if nyq > 0: # Avoid division by zero if sr is 0 (should not happen)
                # Create a brightness curve dependent on frequency
                # Ensure freq_ratios are positive for power calculation
                freq_ratios = f / nyq 
                brightness_curve = np.power(np.maximum(freq_ratios, 1e-9), 0.5) * (brightness - 1.0) + 1.0
                # Apply curve to all time frames: needs (freq, 1) to broadcast with (freq, time)
                Zxx_processed *= brightness_curve[:, np.newaxis] 
            else:
                logger.warning("Nyquist frequency is zero, skipping brightness adjustment.")

        # ISTFT
        _, processed_np = scipy_signal.istft(Zxx_processed, sr, nperseg=nperseg, noverlap=noverlap)
        
        # Adjust length to original signal_cpu length
        original_len = len(signal_cpu) # Length of signal_cpu, which might be different from `signal` if padding occurred for STFT
        if len(processed_np) > original_len:
            processed_np = processed_np[:original_len]
        elif len(processed_np) < original_len:
            # Pad if ISTFT result is shorter (can happen with some STFT params/libraries or edge cases)
            processed_np = np.pad(processed_np, (0, original_len - len(processed_np)), 'constant')
        
        return safe_to_gpu(processed_np)
        
    except Exception as e:
        logger.error(f"Spectral processor error: {e}", exc_info=True)
        return signal # Return original tensor on error


def chunk_processing(signal: torch.Tensor, chunk_size: int, process_func, *args, **kwargs) -> torch.Tensor:
    """Chunk-wise processing for memory efficiency using overlap-add."""
    try:
        signal_cpu = safe_to_cpu(signal) # Convert tensor to numpy for chunking
        
        if len(signal_cpu) <= chunk_size:
            # If signal is smaller than chunk_size, process directly
            # process_func expects tensor, so convert back
            return process_func(safe_to_gpu(signal_cpu), *args, **kwargs)
        
        num_samples = len(signal_cpu)
        overlap = chunk_size // 4 # 25% overlap is common
        
        # Window for overlap-add (Hann window for smooth transitions)
        window = scipy_signal.hann(chunk_size)

        result_cpu = np.zeros_like(signal_cpu, dtype=np.float32)
        sum_of_windows = np.zeros_like(signal_cpu, dtype=np.float32) # For normalizing overlap regions

        for i in range(0, num_samples, chunk_size - overlap): # Step by chunk_size - overlap
            start_idx = i
            end_idx = min(i + chunk_size, num_samples)
            current_chunk_len = end_idx - start_idx
            
            chunk_data_np = signal_cpu[start_idx:end_idx]

            # Pad chunk if it's shorter than chunk_size (typically the last chunk)
            if current_chunk_len < chunk_size:
                pad_len = chunk_size - current_chunk_len
                # Pad with reflection to maintain signal characteristics at edges
                chunk_data_np_padded = np.pad(chunk_data_np, (0, pad_len), mode='reflect')
            else:
                chunk_data_np_padded = chunk_data_np
            
            # Process the (potentially padded) chunk
            # process_func expects a tensor
            processed_chunk_tensor = process_func(safe_to_gpu(chunk_data_np_padded), *args, **kwargs)
            processed_chunk_np_padded = safe_to_cpu(processed_chunk_tensor)

            # Truncate processed chunk back to original chunk length (before padding) if it was padded
            # The windowing should apply to the chunk_size length
            processed_chunk_for_windowing = processed_chunk_np_padded 
            
            # Apply window to the processed chunk (full chunk_size window)
            windowed_processed_chunk = processed_chunk_for_windowing * window

            # Add to result buffer, only up to current_chunk_len (actual data part)
            result_cpu[start_idx:end_idx] += windowed_processed_chunk[:current_chunk_len]
            sum_of_windows[start_idx:end_idx] += window[:current_chunk_len]
            
            del processed_chunk_tensor, processed_chunk_np_padded
            MemoryManager.clear_cache() # Clear cache periodically

        # Normalize by sum_of_windows for proper overlap-add reconstruction
        # Avoid division by zero where sum_of_windows is zero (should not happen in main signal area)
        result_cpu = result_cpu / np.maximum(sum_of_windows, 1e-9) # Epsilon for safety
        
        return safe_to_gpu(result_cpu)
        
    except Exception as e:
        logger.error(f"Chunk processing error: {e}", exc_info=True)
        return signal # Return original tensor on error


def advanced_bbe_processor_optimized(y: torch.Tensor, sr: int, params: dict) -> torch.Tensor:
    """Optimized advanced BBE processor."""
    try:
        if not params.get("bbe_on", False): # Use .get for safety
            return y # Return original tensor if BBE is off

        if not MemoryManager.check_memory():
            logger.warning("Low memory, simplified processing will be executed (advanced_bbe_processor_optimized)")
            # Potentially return y or a more simplified version if this itself is too heavy.
            # For now, if memory is low, it might be better to skip some heavy effects or return y.
            # This function is already the "optimized" one.
            return y

        y_cpu = safe_to_cpu(y) # Convert input tensor to numpy
        original_shape = y_cpu.shape
        # Assumes channels-first: (channels, samples) or (samples,) for mono
        is_stereo = len(y_cpu.shape) > 1 and y_cpu.shape[0] == 2 
        
        if not is_stereo:
            if len(y_cpu.shape) == 1: # Mono (samples,)
                 y_cpu = np.expand_dims(y_cpu, axis=0) # Convert to (1, samples) for consistent processing
        
        processed_channels_np_list = []
        
        # Per-channel processing function (takes numpy, returns numpy)
        def process_channel_opt(channel_data_np: np.ndarray) -> np.ndarray: 
            ch = channel_data_np.copy() # Work on a copy
            
            # Effect chain (order can be important)
            if params.get("bass_enhance_on", False):
                bass_freq = params.get("bass_freq", 80)
                # apply_filter_safe expects numpy
                bass_comp = apply_filter_safe(ch, bass_freq, sr, 'low', order=2) # Isolate bass frequencies
                # Simplified harmonic generation for bass
                harmonic = np.sign(bass_comp) * np.power(np.abs(bass_comp), 0.7) * params.get("bass_intensity", 0.7) * 0.5
                ch += harmonic # Add generated harmonics
            
            ch_tensor = safe_to_gpu(ch) # Convert to tensor for GPU-capable effects

            if params.get("exciter_on", False):
                # harmonic_exciter_effect_optimized expects tensor
                # Use chunk_processing if signal is long and chunk_size is positive
                if len(ch_tensor) > params.get("chunk_size", 8192) and params.get("chunk_size", 0) > 0 :
                    ch_tensor = chunk_processing(ch_tensor, params["chunk_size"], 
                                             harmonic_exciter_effect_optimized, params, sr)
                else:
                    ch_tensor = harmonic_exciter_effect_optimized(ch_tensor, params, sr)
            
            if params.get("dynamic_eq_on", False):
                # dynamic_eq_effect_optimized expects tensor
                ch_tensor = dynamic_eq_effect_optimized(ch_tensor, sr, params)
            
            if params.get("spectral_processor_on", False):
                # spectral_processor_effect_optimized expects tensor
                ch_tensor = spectral_processor_effect_optimized(ch_tensor, sr, params)
            
            ch = safe_to_cpu(ch_tensor) # Convert back to numpy for further CPU processing if any

            # Shelving filters (simplified gain application after filtering)
            # Note: This is a simplified approach and not a true shelving filter response.
            # A true shelf would typically use a different filter design (e.g., from `iirfilter` with specific `ftype`).
            if params.get("low_shelf_on", False):
                # This applies a high-pass filter, then boosts the result.
                # It will attenuate lows and boost what's left, which is not a standard low-shelf boost.
                # For a "boost below X Hz", one would filter for lows and add, or use a proper shelf design.
                # Keeping existing logic, but noting its characteristics.
                # filtered_signal = apply_filter_safe(ch, params["low_cutoff_hz"], sr, 'high', 2) # High-pass
                # ch_boosted = filtered_signal * (10**(params["low_gain_db"]/20.0))
                # ch = ch * (1-mix) + ch_boosted * mix # This would be a more standard way for parallel processing
                # Current simple approach:
                # apply_filter_safe is zero-phase, so it's not just a simple IIR filter.
                # For a shelf boost, you usually add a filtered version or use a specific shelf filter design.
                # Example of a simple boost:
                #   low_freq_component = apply_filter_safe(ch, params["low_cutoff_hz"], sr, 'low', 2)
                #   ch = ch + low_freq_component * (10**(params["low_gain_db"]/20.0) - 1.0)
                pass # Placeholder - Shelving logic needs review for true shelving response. Current is more of a tilt.

            if params.get("high_shelf_on", False):
                # Similar logic issue as low shelf.
                pass # Placeholder - Shelving logic needs review.

            return np.clip(ch, -1.0, 1.0) # Clip to standard audio range
        
        # Parallel processing for stereo channels
        if y_cpu.shape[0] > 0: # Check if there are channels
            if is_stereo and NUM_CORES > 1: # Process stereo channels in parallel if multiple cores
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = [executor.submit(process_channel_opt, y_cpu[i]) for i in range(y_cpu.shape[0])] 
                    processed_channels_np_list = [future.result() for future in futures]
            else: # Mono or single core processing
                for i in range(y_cpu.shape[0]): # Iterate through channels (1 for mono, 2 for stereo)
                     processed_channels_np_list.append(process_channel_opt(y_cpu[i]))
            
            if not processed_channels_np_list: # Should not happen if y_cpu.shape[0] > 0
                 logger.error("No channels processed in advanced_bbe_processor_optimized.")
                 return y # Return original tensor

            y_processed_np = np.array(processed_channels_np_list)
        else: # Input has no channels (e.g., empty array)
            logger.warning("Input to advanced_bbe_processor_optimized has no channels.")
            y_processed_np = y_cpu # Continue with original (empty or malformed) data

        # Stereo width processing
        if is_stereo and params.get("stereo_width_on", False):
            if y_processed_np.shape[0] == 2: # Ensure it's still stereo after processing
                left, right = y_processed_np[0], y_processed_np[1]
                mid, side = encode_mid_side(left, right) # Expects numpy, returns numpy
                
                side_gain = params.get("stereo_width", 1.3)
                side *= side_gain # Adjust side channel gain for width
                # Optional: filter side channel to avoid excessive low-end width or harsh high-end
                # Example: side = apply_filter_safe(side, (80, 8000), sr, 'bandpass', order=2) 
                
                left_out, right_out = decode_mid_side(mid, side) # Expects numpy, returns numpy
                y_processed_np = np.array([left_out, right_out])
            else:
                logger.warning("Stereo width processing skipped: data is not stereo or became non-stereo.")
        
        # Final normalization
        y_processed_np = safe_normalize_audio(y_processed_np, params.get("headroom_db", 3.0)) # Expects numpy
        
        # Restore original mono shape if needed: (1, samples) -> (samples,)
        if not is_stereo and len(original_shape) == 1 and y_processed_np.shape[0] == 1:
            y_processed_np = y_processed_np.squeeze(0)
        
        return safe_to_gpu(y_processed_np) # Convert final numpy back to tensor
        
    except Exception as e:
        logger.error(f"BBE processor error (optimized): {e}", exc_info=True)
        return y # Return original tensor on error

# Main processing function for "optimized" mode
def effect_process_optimized(input_file: str, output_file: str, params: dict = None):
    """Optimized main processing function."""
    current_params = PARAMS.copy() # Start with default parameters
    if params: # Update with user-provided parameters if any
        current_params.update(params)
    
    logger.info(f"Processing started (optimized): {input_file}")
    
    temp_files_optimized = [] # List to keep track of temporary files
    try:
        effective_input_optimized = input_file
        # Convert MP3 to WAV if necessary
        if input_file.lower().endswith('.mp3'):
            base_opt, ext_opt = os.path.splitext(input_file)
            # Use a more unique temp file name to avoid conflicts
            temp_wav_opt = f"{base_opt}_temp_opt_{os.getpid()}.wav" 
            temp_files_optimized.append(temp_wav_opt)
            
            logger.info("Converting MP3 to WAV (optimized)...")
            # Target SR for ffmpeg conversion should be the pre-oversampling SR
            ffmpeg_sr_opt = current_params["target_sr"] // current_params["oversample_factor"]
            subprocess.run([
                "ffmpeg", "-y", "-i", input_file, 
                "-acodec", "pcm_s16le", # Using PCM 16-bit for intermediate WAV
                "-ar", str(ffmpeg_sr_opt), 
                temp_wav_opt
            ], check=True, capture_output=True, text=True) # Capture output for debugging
            effective_input_optimized = temp_wav_opt

        # Load audio file
        # Load at the pre-oversampling SR
        load_sr_opt = current_params["target_sr"] // current_params["oversample_factor"]
        y_opt, sr_loaded_opt = librosa.load(effective_input_optimized, sr=load_sr_opt, mono=False) # Load as float32 by default
        logger.info(f"Loading complete (optimized) - SR: {sr_loaded_opt}Hz, Shape: {y_opt.shape}")
        
        # Check memory and adjust parameters if necessary
        if not MemoryManager.check_memory():
            logger.warning("Low memory, adjusting parameters (optimized)")
            current_params["oversample_factor"] = 1 # Reduce oversampling
            # Reduce chunk size further if memory is an issue
            current_params["chunk_size"] = max(4096, current_params.get("chunk_size", 8192) // 2)

        # Determine processing SR (after oversampling)
        processing_sr_opt = current_params["target_sr"] 
        y_os_opt = y_opt # Default if no oversampling

        # Oversampling if factor > 1 and loaded SR differs from target processing SR
        if current_params["oversample_factor"] > 1 and sr_loaded_opt != processing_sr_opt:
            logger.info("Applying oversampling (optimized)...")
            y_os_opt = librosa.resample(
                y_opt, orig_sr=sr_loaded_opt, target_sr=processing_sr_opt, 
                res_type='kaiser_fast' # Faster resampling for optimized mode
            )
        else: # No oversampling or already at target SR
            processing_sr_opt = sr_loaded_opt # Actual processing SR is the loaded SR
            current_params["target_sr"] = sr_loaded_opt # Update param to reflect actual processing SR


        y_gpu_opt = safe_to_gpu(y_os_opt) # Convert to tensor for processing
        
        logger.info("Executing BBE processing (optimized)...")
        # Call the BBE processing function
        y_processed_opt_tensor = advanced_bbe_processor_optimized(y_gpu_opt, processing_sr_opt, current_params)
        
        y_processed_opt_cpu = safe_to_cpu(y_processed_opt_tensor) # Convert back to CPU/numpy
        
        # Downsampling if oversampling was applied
        output_sr_opt = load_sr_opt # Target output SR is the original pre-oversampling SR
        y_final_opt = y_processed_opt_cpu

        if current_params["oversample_factor"] > 1 and processing_sr_opt != output_sr_opt:
            logger.info("Downsampling (optimized)...")
            y_final_opt = librosa.resample(
                y_processed_opt_cpu,
                orig_sr=processing_sr_opt, 
                target_sr=output_sr_opt,
                res_type='kaiser_fast' # Faster resampling
            )
        
        # Final normalization
        y_final_opt = safe_normalize_audio(y_final_opt, 1.0) # Numpy in, numpy out
        
        logger.info("Writing output file (optimized)...")
        # sf.write expects channels last: (samples, channels)
        # librosa loads as (channels, samples) for stereo if mono=False
        if len(y_final_opt.shape) > 1 and y_final_opt.shape[0] < y_final_opt.shape[1]: # (channels, samples)
            sf.write(output_file, y_final_opt.T, output_sr_opt, subtype='PCM_24') # Transpose
        else: # Mono (samples,) or already channels-last
            sf.write(output_file, y_final_opt, output_sr_opt, subtype='PCM_24')
        
        logger.info("Processing complete (optimized)")
        
    except subprocess.CalledProcessError as e_spe: # Catch FFmpeg errors
        logger.error(f"FFmpeg error (optimized): {e_spe.stderr}")
        raise # Re-raise the exception
    except Exception as e_opt: # Catch other processing errors
        logger.error(f"Processing error (optimized): {e_opt}", exc_info=True)
        raise # Re-raise the exception
    finally:
        # Clean up temporary files
        for temp_f_opt in temp_files_optimized:
            if os.path.exists(temp_f_opt):
                os.remove(temp_f_opt)
        MemoryManager.clear_cache() # Clear memory cache


# === 3. Advanced Stereo Imaging ===
class AdvancedStereoProcessor:
    """Advanced stereo processor."""
    
    @staticmethod
    def binaural_widening(left: np.ndarray, right: np.ndarray, sr: int, width_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Binaural stereo widening."""
        try:
            if abs(width_factor - 1.0) < 1e-3: # No significant widening needed
                return left, right

            # Simplified M/S based widening:
            # Encode to Mid/Side
            mid, side = encode_mid_side(left, right)
            
            # Enhance side signal based on width_factor
            # A non-linear gain or filtering on the side channel can also be used for more nuanced effects.
            side_gain = width_factor 
            side_processed = side * side_gain
            
            # Optional: Filter side channel to control width at different frequencies.
            # e.g., reduce low-frequency width to maintain focus, boost high-frequency for air.
            # Example:
            # side_low = PhaseLinearFilters.apply_fir_filter(side_processed, 200, sr, 'lowpass') * 0.8
            # side_mid_raw = PhaseLinearFilters.apply_fir_filter(side_processed, (200, 5000), sr, 'bandpass')
            # side_high = PhaseLinearFilters.apply_fir_filter(side_processed, 5000, sr, 'highpass') * 1.2
            # side_final = side_low + side_mid_raw + side_high # Reconstruct side
            # For simplicity, using direct gain for now:
            side_final = side_processed

            # Decode back to Left/Right
            left_out, right_out = decode_mid_side(mid, side_final)
            
            # Normalize to prevent clipping if width_factor is large and increases overall level.
            # This maintains the original peak level if widening causes an increase.
            max_amp_in = np.max(np.abs(np.concatenate((left, right)))) if left.size > 0 and right.size > 0 else 0
            max_amp_out = np.max(np.abs(np.concatenate((left_out, right_out)))) if left_out.size > 0 and right_out.size > 0 else 0
            
            if max_amp_out > max_amp_in and max_amp_in > 1e-6 : # If output is louder and input was not silent
                norm_factor = max_amp_in / max_amp_out
                left_out *= norm_factor
                right_out *= norm_factor

            return np.clip(left_out, -1.0, 1.0), np.clip(right_out, -1.0, 1.0)
            
        except Exception as e:
            logger.error(f"Binaural stereo error: {e}", exc_info=True)
            return left, right # Fallback to original L/R
            
            
# === 1. Phase-Linear Filter Implementation ===
class PhaseLinearFilters:
    """Phase-linear filter class."""
    
    @staticmethod
    @lru_cache(maxsize=64) # Cache filter designs
    # MODIFIED: cutoff type hint, numtaps calculation, normalization
    def design_fir_filter(cutoff: Union[float, Tuple[float, float]], 
                          fs: float, filter_type: str, numtaps: Optional[int]=None) -> Optional[np.ndarray]:
        """Optimized FIR filter design using firwin."""
        nyq = fs / 2.0
        
        # Determine number of taps if not provided
        if numtaps is None:
            # Heuristic for numtaps: transition bandwidth related.
            # A common rule of thumb: N ≈ A / (Δf/fs), where Δf is transition width.
            # Simpler: N ≈ C * (fs / cutoff_freq)
            # Ensure ref_cutoff is reasonable for calculation.
            ref_cutoff = cutoff[0] if isinstance(cutoff, (tuple, list)) else cutoff
            if ref_cutoff <= 1e-6: # Avoid division by zero or very small numbers
                logger.warning(f"Invalid or very small cutoff frequency {ref_cutoff}. Using default taps 257.")
                numtaps = 257 # Default odd number of taps
            else:
                # Calculate taps, ensure it's odd for Type I/II FIR (symmetric)
                # Wider transition for lower cutoffs might need more taps.
                # This is a heuristic, aiming for reasonable filter length.
                calculated_taps = int(fs / ref_cutoff * 4.0) # Factor 4 is empirical
                if calculated_taps % 2 == 0:
                    calculated_taps += 1 # Ensure odd number of taps
                # Clip taps to a reasonable range to prevent excessive computation or too short filters.
                numtaps = int(np.clip(calculated_taps, 63, 1023)) # Ensure it's odd. Both 63 and 1023 are odd.
        
        # Normalize cutoff frequency/frequencies to Nyquist
        if isinstance(cutoff, (tuple, list)): # bandpass
            if len(cutoff) != 2:
                logger.error(f"Bandpass filter cutoff must be a tuple/list of two frequencies: {cutoff}")
                return None
            normalized_cutoff = [c / nyq for c in cutoff]
            if normalized_cutoff[0] >= normalized_cutoff[1]: # Ensure f_low < f_high
                logger.warning(f"Bandpass filter frequencies out of order: {cutoff}. Correcting.")
                normalized_cutoff.sort()
            # Clip normalized frequencies to be within (0, 1) exclusive, for firwin
            normalized_cutoff = [np.clip(c, 1e-5, 1.0 - 1e-5) for c in normalized_cutoff] 
        else: # lowpass, highpass
            normalized_cutoff = cutoff / nyq
            normalized_cutoff = np.clip(normalized_cutoff, 1e-5, 1.0 - 1e-5) # Clip to (0,1)

        try:
            # Design FIR filter using Hamming window (good general-purpose window)
            if filter_type == 'lowpass':
                return firwin(numtaps, normalized_cutoff, window='hamming', pass_zero=True)
            elif filter_type == 'highpass':
                return firwin(numtaps, normalized_cutoff, window='hamming', pass_zero=False)
            elif filter_type == 'bandpass':
                # For bandpass, normalized_cutoff should be [low_freq_norm, high_freq_norm]
                return firwin(numtaps, normalized_cutoff, window='hamming', pass_zero=False) 
        except ValueError as e: # firwin can raise ValueError for invalid parameters
            logger.error(f"FIR filter coefficient generation error (type={filter_type}, norm_cutoff={normalized_cutoff}, taps={numtaps}): {e}")
            return None
        
        logger.error(f"Unsupported FIR filter type: {filter_type}")
        return None # Should not be reached if filter_type is one of the above
    
    @staticmethod
    # MODIFIED: apply_fir_filter to handle tuple for cutoff, IIR fallback
    def apply_fir_filter(signal: np.ndarray, cutoff: Union[float, List[float], Tuple[float,float]], 
                         fs: float, filter_type: str) -> np.ndarray:
        """Apply phase-linear FIR filter (using filtfilt for zero phase)."""
        try:
            # Convert list cutoff to tuple for lru_cache compatibility in design_fir_filter
            cacheable_cutoff = tuple(cutoff) if isinstance(cutoff, list) else cutoff
            
            fir_coeffs = PhaseLinearFilters.design_fir_filter(cacheable_cutoff, fs, filter_type)
            
            if fir_coeffs is not None and len(fir_coeffs) > 0 :
                 # filtfilt requires signal to be longer than 3 times filter order.
                 # Filter order for FIR is numtaps - 1.
                if len(signal) <= len(fir_coeffs) * 3: 
                    logger.warning(f"Signal length ({len(signal)}) too short for FIR filter taps ({len(fir_coeffs)}). FIR filter ({filter_type}) skipped.")
                    return signal
                return filtfilt(fir_coeffs, [1.0], signal) # Apply FIR filter with zero phase
            
            logger.warning(f"Failed to get FIR filter coefficients ({filter_type}, cutoff={cutoff}). Skipping FIR processing.")
            return signal # Return original signal if coeffs are None or empty

        except Exception as e:
            logger.warning(f"FIR filter application error ({filter_type}, cutoff={cutoff}, fs={fs}). Attempting IIR fallback: {e}")
            # Map FIR filter_type to IIR btype
            btype_map = {'lowpass': 'low', 'highpass': 'high', 'bandpass': 'band'}
            iir_btype = btype_map.get(filter_type)
            
            if iir_btype:
                # Ensure cutoff format is correct for apply_filter_safe / get_butter_filter
                current_cutoff = cutoff
                if filter_type == 'bandpass':
                    if not (isinstance(cutoff, (list, tuple)) and len(cutoff)==2):
                        logger.error(f"IIR fallback: Invalid cutoff format for bandpass: {cutoff}")
                        return signal # Cannot proceed with IIR either
                elif not isinstance(cutoff, (int,float)): # lowpass/highpass expect single float
                     logger.error(f"IIR fallback: Invalid cutoff format for {filter_type}: {cutoff}")
                     return signal
                
                logger.info(f"Applying IIR ({iir_btype}) fallback for {filter_type} filter.")
                return apply_filter_safe(signal, current_cutoff, fs, btype=iir_btype, order=4) # Use a typical order for IIR fallback
            
            logger.warning(f"IIR fallback not possible (unknown type or invalid cutoff format). Filter not applied: {filter_type}, {cutoff}")
            return signal


# === 2. Advanced Harmonic Exciter ===
class AdvancedHarmonicExciter:
    """Advanced harmonic exciter with multiband processing."""
    
    @staticmethod
    def multiband_exciter(signal: torch.Tensor, sr: int, params: dict) -> torch.Tensor:
        """Multiband exciter using phase-linear crossovers."""
        try:
            signal_cpu = safe_to_cpu(signal)
            
            # Define crossover frequencies for 3-band split
            # These can be adjusted or made configurable
            crossover_freqs = [800, 3000]  # Hz (e.g., Low/Mid crossover, Mid/High crossover)
            
            # Low band (signal below first crossover)
            low_band = PhaseLinearFilters.apply_fir_filter(
                signal_cpu, crossover_freqs[0], sr, 'lowpass')
            
            # Mid band (signal between crossover_freqs[0] and crossover_freqs[1])
            # PhaseLinearFilters.apply_fir_filter expects a tuple for bandpass
            mid_band = PhaseLinearFilters.apply_fir_filter(
                signal_cpu, (crossover_freqs[0], crossover_freqs[1]), sr, 'bandpass') 
            
            # High band (signal above second crossover)
            high_band = PhaseLinearFilters.apply_fir_filter(
                signal_cpu, crossover_freqs[1], sr, 'highpass')
            
            bands = [low_band, mid_band, high_band]
            
            # Define different excitation settings for each band
            # 'drive': input gain to non-linearity
            # 'harmonics': list of harmonic orders to generate (can be symbolic e.g., 2 for 2nd)
            # 'amount': mix level for generated harmonics in that band
            excite_settings = [
                {'drive': 1.2, 'harmonics': [2, 3], 'amount': 0.3 * params.get("exciter_amount", 0.5)}, # Low band settings
                {'drive': 1.8, 'harmonics': [2, 3, 4], 'amount': 0.5 * params.get("exciter_amount", 0.5)}, # Mid band settings
                {'drive': 2.5, 'harmonics': [2, 4, 6], 'amount': 0.7 * params.get("exciter_amount", 0.5)}  # High band settings
            ] # Individual amounts are scaled by the global exciter_amount
            
            processed_bands = []
            for i, (band_signal, setting) in enumerate(zip(bands, excite_settings)):
                # Ensure band signal is valid before processing
                if band_signal is None or len(band_signal) < 10: # Arbitrary short length check
                    logger.warning(f"Skipping empty or too short band {i} in multiband exciter.")
                    # Add zeros or original short band to maintain structure for summation
                    processed_bands.append(np.zeros_like(signal_cpu) if band_signal is None else band_signal) 
                    continue

                excited_band = AdvancedHarmonicExciter.apply_harmonic_generation(
                    band_signal, setting, sr)
                processed_bands.append(excited_band)
            
            # Sum bands to reconstruct the signal.
            # Since filtfilt (used in FIR filters) is phase-linear, direct summation is generally acceptable.
            # Ensure all processed bands have the same length as original signal_cpu for summation.
            # This should ideally be handled by apply_fir_filter returning same-length signals or original signal on error.
            # Adding an explicit check and alignment if lengths differ.
            target_len = len(signal_cpu)
            aligned_processed_bands = []
            for b_idx, b_arr in enumerate(processed_bands):
                if len(b_arr) != target_len:
                    logger.warning(f"Band {b_idx} length {len(b_arr)} differs from target {target_len}. Aligning.")
                    if len(b_arr) < target_len:
                        pad_width = target_len - len(b_arr)
                        aligned_processed_bands.append(np.pad(b_arr, (0, pad_width), 'constant'))
                    else: # len(b_arr) > target_len
                        aligned_processed_bands.append(b_arr[:target_len])
                else:
                    aligned_processed_bands.append(b_arr)
            
            result = np.sum(aligned_processed_bands, axis=0) # Sum along bands
            
            return safe_to_gpu(np.clip(result, -1.0, 1.0)) # Clip and return as tensor
            
        except Exception as e:
            logger.error(f"Multiband exciter error: {e}", exc_info=True)
            return signal # Return original tensor on error
    
    @staticmethod
    def apply_harmonic_generation(signal: np.ndarray, setting: dict, sr: int) -> np.ndarray:
        """Applies harmonic generation to a signal based on settings."""
        drive = setting['drive']
        harmonics_to_generate = setting['harmonics'] # List of harmonic orders
        # Amount from setting is already pre-scaled by global exciter_amount in multiband_exciter
        amount = setting['amount'] 
        
        if amount < 1e-3: # Skip if amount is negligible
            return signal

        # Pre-process signal with drive (soft clipping)
        # np.tanh is a common choice for soft clipping.
        driven_signal = np.tanh(signal * drive) 
        if abs(drive) > 1e-6: # Avoid division by zero if drive is very small
             driven_signal = driven_signal / drive # Scale back to roughly original level range
        
        harmonic_sum = np.zeros_like(signal) # Accumulator for generated harmonics
        
        # Generate specified harmonics. These are simplified formulas.
        # More complex waveshaping or polynomial functions could be used.
        for h_order in harmonics_to_generate:
            h_component = np.zeros_like(signal)
            if h_order == 2: # Second harmonic (even)
                h_component = np.sign(driven_signal) * np.power(np.abs(driven_signal), 1.5) # Or simply driven_signal**2
            elif h_order == 3: # Third harmonic (odd)
                h_component = driven_signal - np.power(driven_signal, 3) / 3.0 # Approx. from Chebyshev or Taylor
            elif h_order == 4: # Fourth harmonic (even)
                h_component = np.power(driven_signal, 2) - np.power(driven_signal, 4) / 2.0
            elif h_order == 6: # Sixth harmonic (even)
                h_component = np.power(driven_signal, 3) - np.power(driven_signal, 6) / 4.0
            else: # Generic power for other orders (can be fractional)
                # For audio, usually abs(driven_signal) is used for even-like powers if negative inputs.
                # Here, direct power is used, which might lead to complex numbers if not handled for negative bases and non-integer exponents.
                # Assuming positive h_order for simplicity.
                if h_order > 0:
                    # A more general way to create harmonics might involve phase considerations
                    # or specific waveshaping functions.
                    # This power law is a simplification.
                    h_component = np.power(np.abs(driven_signal), h_order / 2.0) * np.sign(driven_signal if h_order % 2 != 0 else 1)
            
            # Weight harmonics (higher orders often weaker)
            weight = 1.0 / h_order if h_order > 0 else 1.0
            harmonic_sum += h_component * weight
        
        # Dynamic mix: reduce harmonic amount for louder parts of the original signal
        envelope = np.abs(signal) # Envelope of the original band signal
        sigma_val = max(1, sr // 500) # Sigma for Gaussian smoothing, ensure positive integer
        envelope_smooth = gaussian_filter1d(envelope, sigma=sigma_val)
        dynamic_amount = amount * (1.0 - 0.3 * np.clip(envelope_smooth, 0, 1)) # Reduce amount for louder sections
        
        # Add scaled harmonics to the original signal
        return signal + harmonic_sum * dynamic_amount


# === 4. Adaptive Dynamics Processing ===
class AdaptiveDynamicsProcessor:
    """Adaptive dynamics processing, e.g., spectral compression."""
    
    @staticmethod
    def spectral_compressor(signal, sr, params):
        """Spectral compressor (frequency-dependent compression)."""
        try:
            signal_cpu = safe_to_cpu(signal)
            
            # STFT parameters
            nperseg = 2048 # FFT window size
            noverlap = nperseg // 2 # Overlap between windows
            
            # Perform STFT
            f, t, Zxx = scipy_signal.stft(signal_cpu, sr, nperseg=nperseg, noverlap=noverlap)
            
            # Frequency band-specific compression settings
            # These can be made more configurable via `params`
            freq_bands_settings = [
                {'range': (0, 200), 'ratio': params.get("spectral_comp_low_ratio", 2.0), 'threshold': params.get("spectral_comp_low_thresh", -20)},    # Lows
                {'range': (200, 2000), 'ratio': params.get("spectral_comp_midlow_ratio", 3.0), 'threshold': params.get("spectral_comp_midlow_thresh", -18)}, # Mid-lows
                {'range': (2000, 6000), 'ratio': params.get("spectral_comp_midhigh_ratio", 2.5), 'threshold': params.get("spectral_comp_midhigh_thresh", -15)},# Mid-highs
                {'range': (6000, sr/2), 'ratio': params.get("spectral_comp_high_ratio", 1.8), 'threshold': params.get("spectral_comp_high_thresh", -12)}  # Highs
            ] # Thresholds in dB
            
            Zxx_compressed = Zxx.copy() # Work on a copy of the STFT data
            
            for band_setting in freq_bands_settings:
                # Create frequency mask for the current band
                freq_mask = (f >= band_setting['range'][0]) & (f < band_setting['range'][1])
                
                if not np.any(freq_mask): # Skip if band is empty (e.g., due to low SR)
                    continue
                
                # Get spectrum for the current band
                band_spectrum_complex = Zxx_compressed[freq_mask, :]
                magnitude = np.abs(band_spectrum_complex)
                phase = np.angle(band_spectrum_complex) # Preserve phase
                
                # Compression processing
                threshold_linear = 10 ** (band_setting['threshold'] / 20.0) # Convert dB to linear
                ratio = band_setting['ratio']
                
                # Smooth magnitude over time (simple attack/release approximation)
                # Savitzky-Golay filter can smooth, but true attack/release needs stateful envelope following.
                # For simplicity, using Sav-Gol as a smoother. Window length controls "responsiveness".
                # A short window (e.g., 5-11) will be more reactive.
                magnitude_smooth = scipy_signal.savgol_filter(
                    magnitude, window_length=max(5, int(sr*0.005/ (nperseg-noverlap) ) |1 ), # Approx 5ms, ensure odd and >=5
                    polyorder=2, axis=1) # Smooth along time axis (axis=1)
                
                # Calculate compression gain
                gain = np.ones_like(magnitude_smooth)
                over_threshold_mask = magnitude_smooth > threshold_linear
                
                if np.any(over_threshold_mask):
                    # Standard compressor gain formula
                    gain[over_threshold_mask] = np.power(
                        threshold_linear / np.maximum(magnitude_smooth[over_threshold_mask], 1e-9), # Avoid div by zero
                        (ratio - 1.0) / ratio
                    )
                
                # Apply gain to original (non-smoothed) magnitude
                compressed_magnitude = magnitude * gain
                
                # Reconstruct complex spectrum for the band with original phase
                Zxx_compressed[freq_mask, :] = compressed_magnitude * np.exp(1j * phase)
            
            # Perform Inverse STFT
            _, compressed_signal = scipy_signal.istft(
                Zxx_compressed, sr, nperseg=nperseg, noverlap=noverlap)
            
            # Adjust length to match original signal (ISTFT can sometimes have slight length differences)
            if len(compressed_signal) != len(signal_cpu):
                # A simple resize might not be ideal for audio.
                # Truncate or pad as done in spectral_processor_effect_optimized.
                original_len = len(signal_cpu)
                if len(compressed_signal) > original_len:
                    compressed_signal = compressed_signal[:original_len]
                else:
                    compressed_signal = np.pad(compressed_signal, (0, original_len - len(compressed_signal)), 'constant')

            return safe_to_gpu(compressed_signal)
            
        except Exception as e:
            logger.error(f"Spectral compressor error: {e}", exc_info=True)
            return signal # Return original tensor on error
            
# === 5. Advanced Limiter ===
class AdvancedLimiter:
    """Advanced limiter with lookahead capabilities."""
    
    @staticmethod
    def lookahead_limiter(signal: torch.Tensor, sr: int, params: dict, 
                          threshold_default: float = 0.95, 
                          lookahead_ms_default: float = 5.0) -> torch.Tensor:
        """Lookahead limiter to prevent peaks without audible distortion.
        
        Args:
            signal: Input audio signal as a PyTorch tensor.
            sr: Sampling rate of the audio signal.
            params: Dictionary containing processing parameters.
                    Expected keys: "limiter_threshold", "limiter_lookahead_ms",
                                   "limiter_attack_ms", "limiter_release_ms".
            threshold_default: Default threshold if not in params.
            lookahead_ms_default: Default lookahead time in ms if not in params.
        
        Returns:
            Processed (limited) audio signal as a PyTorch tensor.
        """
        try:
            signal_cpu = safe_to_cpu(signal)
            
            # Get limiter parameters from the params dictionary, using defaults if not found.
            actual_threshold = params.get("limiter_threshold", threshold_default)
            actual_lookahead_ms = params.get("limiter_lookahead_ms", lookahead_ms_default)
            attack_time_s = params.get("limiter_attack_ms", 1.0) / 1000.0  # Default 1ms
            release_time_s = params.get("limiter_release_ms", 100.0) / 1000.0 # Default 100ms

            lookahead_samples = int(sr * actual_lookahead_ms / 1000.0)
            
            # Pad signal at the beginning for lookahead
            # The lookahead portion of the padded signal will be used to anticipate peaks.
            padded_signal = np.pad(signal_cpu, (lookahead_samples, 0), mode='reflect')
            
            # Envelope detection on the padded signal
            envelope = np.abs(padded_signal) # Simple peak envelope

            # Coefficients for one-pole filter (exp(-1/(time_constant * sr)))
            attack_coeff = np.exp(-1.0 / (sr * attack_time_s)) if attack_time_s > 0 and sr > 0 else 0.0
            release_coeff = np.exp(-1.0 / (sr * release_time_s)) if release_time_s > 0 and sr > 0 else 0.0
            
            gain_reduction_envelope = np.ones_like(envelope) # Stores the gain to apply
            current_gain = 1.0

            # Iterate through the envelope (which includes lookahead)
            for i in range(len(envelope)):
                # Max peak in the lookahead window starting from current sample `i`
                # up to `i + lookahead_samples`.
                if lookahead_samples > 0 and i < len(envelope) - lookahead_samples:
                    future_peak = np.max(envelope[i : i + lookahead_samples + 1])
                else: # Not enough samples for full lookahead (near end of signal) or no lookahead
                    future_peak = envelope[i]
                
                target_gain = 1.0
                if future_peak > actual_threshold:
                    # Required gain to bring peak to threshold, avoid division by zero
                    target_gain = actual_threshold / np.maximum(future_peak, 1e-9) 
                
                # Apply attack/release smoothing to the gain
                if target_gain < current_gain: # Peak detected, gain needs to decrease (attack)
                    current_gain = (target_gain * (1.0 - attack_coeff) + 
                                    current_gain * attack_coeff)
                else: # Peak passed, gain can increase (release)
                    current_gain = (target_gain * (1.0 - release_coeff) +
                                    current_gain * release_coeff)
                
                gain_reduction_envelope[i] = current_gain
            
            # Apply gain reduction to the padded signal
            limited_padded_signal = padded_signal * gain_reduction_envelope
            
            # Remove the lookahead padding to align with original signal
            limited_signal = limited_padded_signal[lookahead_samples:]
            
            return safe_to_gpu(limited_signal)
            
        except Exception as e:
            logger.error(f"Lookahead limiter error: {e}", exc_info=True)
            return signal # Return original tensor on error
            
                        

# === 6. Main Processing Function Update (Ultra Advanced) ===
# MODIFIED: ultra_advanced_bbe_processor to use updated functions
def ultra_advanced_bbe_processor(y: torch.Tensor, sr: int, params: dict) -> torch.Tensor:
    """Ultra-advanced BBE processor combining multiple advanced effects."""
    try:
        if not params.get("bbe_on", False): # Check if BBE processing is enabled
            return y
        
        if not MemoryManager.check_memory(): # Check memory before proceeding
            logger.warning("Low memory, falling back to standard optimized processing.")
            # Fallback to the less resource-intensive optimized processor
            return advanced_bbe_processor_optimized(y, sr, params)
        
        y_cpu = safe_to_cpu(y) # Convert input tensor to numpy array for processing
        # Determine if stereo: assumes channels-first format (channels, samples)
        is_stereo = len(y_cpu.shape) > 1 and y_cpu.shape[0] == 2 
        
        original_shape = y_cpu.shape # Store original shape for potential mono restoration
        if not is_stereo: # If mono, ensure it's (1, num_samples) for consistent channel processing
            if len(y_cpu.shape) == 1: # Input is (num_samples,)
                 y_cpu = np.expand_dims(y_cpu, axis=0) # Reshape to (1, num_samples)
        
        processed_channels_list = [] # To store processed audio for each channel
        
        # Define per-channel processing function (operates on numpy arrays)
        def ultra_process_channel_fn(channel_data: np.ndarray) -> np.ndarray: 
            channel = channel_data.copy() # Work on a copy to avoid modifying original
            
            # 1. Advanced Harmonic Exciter (Multiband)
            if params.get("exciter_on", False) and params.get("multiband_exciter", True): # Check specific flag
                # AdvancedHarmonicExciter.multiband_exciter expects tensor, sr, params
                channel_gpu = safe_to_gpu(channel) # Convert numpy to tensor
                channel_gpu = AdvancedHarmonicExciter.multiband_exciter(channel_gpu, sr, params)
                channel = safe_to_cpu(channel_gpu) # Convert back to numpy
            
            # 2. Adaptive Spectral Compression
            # Ensure compressor_on is also true for this to be meaningful
            if params.get("compressor_on", False) and params.get("spectral_compression", False):
                # AdaptiveDynamicsProcessor.spectral_compressor expects tensor, sr, params
                channel_gpu = safe_to_gpu(channel)
                channel_gpu = AdaptiveDynamicsProcessor.spectral_compressor(channel_gpu, sr, params)
                channel = safe_to_cpu(channel_gpu)
            
            # 3. Phase-Linear EQ (Shelving Filters)
            # Note: Current implementation of FIR shelves might be simplified.
            # True phase-linear shelving requires specific FIR design.
            if params.get("low_shelf_on", False) and params.get("use_phase_linear", False):
                # PhaseLinearFilters.apply_fir_filter expects numpy
                # A proper low-shelf boost would typically add a low-passed version of the signal, scaled by gain.
                # Or use a dedicated FIR shelf design.
                # Example (conceptual) for a boost:
                # low_component = PhaseLinearFilters.apply_fir_filter(channel, params["low_cutoff_hz"], sr, 'lowpass')
                # gain_factor = (10**(params["low_gain_db"]/20.0)) - 1.0 # Gain above 0dB
                # channel = channel + low_component * gain_factor
                pass # Placeholder - Review FIR shelf implementation for accuracy. Current may act as a tilt.

            if params.get("high_shelf_on", False) and params.get("use_phase_linear", False):
                # Similar to low-shelf, a true high-shelf FIR needs specific design.
                pass # Placeholder - Review FIR shelf implementation.

            # 4. Advanced Limiter (Lookahead)
            # Ensure limiter_on is also true
            if params.get("limiter_on", False) and params.get("lookahead_limiting", False):
                # AdvancedLimiter.lookahead_limiter expects tensor, sr, params, threshold, lookahead_ms
                channel_gpu = safe_to_gpu(channel)
                # Pass the full params dictionary. The function itself will extract
                # 'limiter_threshold', 'limiter_lookahead_ms', 'limiter_attack_ms', 'limiter_release_ms'
                # using its own defaults if they are not present in params.
                channel_gpu = AdvancedLimiter.lookahead_limiter(
                    signal=channel_gpu, 
                    sr=sr, 
                    params=params
                    # threshold_default and lookahead_ms_default are handled by the function's signature
                )
                channel = safe_to_cpu(channel_gpu)
            
            return np.clip(channel, -1.0, 1.0) # Clip final channel output

        # Process channels (mono or stereo)
        if y_cpu.shape[0] > 0: # Check if there are any channels to process
            if is_stereo and NUM_CORES > 1: # Parallel processing for stereo if multiple cores
                with ThreadPoolExecutor(max_workers=2) as executor: # Max 2 threads for stereo
                    futures = [executor.submit(ultra_process_channel_fn, y_cpu[i]) 
                              for i in range(y_cpu.shape[0])] # y_cpu.shape[0] should be 2 for stereo
                    processed_channels_list = [future.result() for future in futures]
            else: # Single-threaded processing for mono or if NUM_CORES <= 1
                for i in range(y_cpu.shape[0]): # Iterate through channels
                    processed_channels_list.append(ultra_process_channel_fn(y_cpu[i]))
            
            if not processed_channels_list: # Safety check, should not be empty if y_cpu had channels
                 logger.error("No channels were processed in ultra_advanced_bbe_processor.")
                 return y # Return original tensor

            y_processed_np = np.array(processed_channels_list) # Combine processed channels
        else: # Input had no channels (e.g. empty array)
            logger.warning("Input to ultra_advanced_bbe_processor has no channels.")
            y_processed_np = y_cpu # Continue with the (malformed) original data

        # Advanced Stereo Processing (Binaural Widening)
        # Ensure stereo_width_on is also true
        if is_stereo and params.get("stereo_width_on", False) and params.get("binaural_stereo", False):
            if y_processed_np.shape[0] == 2: # Double-check it's still stereo
                left, right = y_processed_np[0], y_processed_np[1]
                # AdvancedStereoProcessor.binaural_widening expects numpy L, R arrays
                left_out, right_out = AdvancedStereoProcessor.binaural_widening(
                    left, right, sr, params.get("stereo_width", 1.3)) # Pass stereo width param
                y_processed_np = np.array([left_out, right_out]) # Recombine to stereo array
            else:
                logger.warning("Stereo processing (binaural) skipped: input no longer stereo after channel processing.")
        
        # Final normalization of the processed audio
        y_processed_np = safe_normalize_audio(y_processed_np, params.get("headroom_db", 1.0))
        
        # Restore original mono shape if input was (num_samples,)
        if not is_stereo and len(original_shape) == 1 and y_processed_np.shape[0] == 1:
            y_processed_np = y_processed_np.squeeze(0) # (1, num_samples) -> (num_samples,)
        
        return safe_to_gpu(y_processed_np) # Convert final numpy array back to tensor
        
    except Exception as e:
        logger.error(f"Ultra advanced BBE processor error: {e}", exc_info=True)
        # Fallback to the optimized processor if ultra fails
        logger.info("Falling back to advanced_bbe_processor_optimized due to error.")
        return advanced_bbe_processor_optimized(y, sr, params) # Fallback returns tensor
        
        
# === 7. New Main Processing Function (Ultra Mode) ===
def ultra_effect_process(input_file: str, output_file: str, params: dict = None):
    """Ultra advanced effect processing pipeline."""
    current_params = PARAMS.copy() # Start with base default parameters
    if params is not None:
        current_params.update(params) # Override with any user-supplied parameters
    
    # Default settings for "ultra" mode features (can be overridden by user params if they exist)
    # These ensure that if the flags are not in PARAMS or user params, they default to True for ultra mode.
    default_ultra_settings = {
        "use_phase_linear": True,       # Use phase-linear filters for EQ if applicable
        "multiband_exciter": True,      # Use multiband exciter (implies exciter_on)
        "spectral_compression": True,   # Use spectral compressor (implies compressor_on)
        "lookahead_limiting": True,     # Use lookahead limiter (implies limiter_on)
        "binaural_stereo": True,        # Use binaural stereo widening (implies stereo_width_on)
    }
    for key, value in default_ultra_settings.items():
        if key not in current_params: # Add if not already set by user or base PARAMS
            current_params[key] = value

    logger.info(f"Ultra advanced processing started: {input_file}")
    
    temp_files = [] # To keep track of temporary files for cleanup
    try:
        effective_input_file = input_file
        # Pre-processing: Convert MP3 to WAV if necessary
        if input_file.lower().endswith('.mp3'):
            base, ext = os.path.splitext(input_file)
            # Use a more unique temp file name (e.g., including PID)
            temp_wav = f"{base}_temp_ultra_{os.getpid()}.wav" 
            temp_files.append(temp_wav)
            
            logger.info("Converting MP3 to WAV (for ultra mode)...")
            # Target SR for ffmpeg conversion: pre-oversampling SR
            ffmpeg_sr = current_params["target_sr"] // current_params["oversample_factor"]
            subprocess.run([
                "ffmpeg", "-y", "-i", input_file, 
                "-acodec", "pcm_s24le", # Use 24-bit PCM for intermediate WAV in ultra mode
                "-ar", str(ffmpeg_sr), 
                temp_wav
            ], check=True, capture_output=True, text=True) # Capture output for better error info
            effective_input_file = temp_wav
        
        logger.info("Loading audio with high precision...")
        # Load audio at the pre-oversampling SR.
        # Use float64 for higher precision during loading if available/beneficial, though librosa defaults to float32.
        load_sr = current_params["target_sr"] // current_params["oversample_factor"]
        y, sr_loaded = librosa.load(effective_input_file, sr=load_sr, mono=False, dtype=np.float32) # librosa loads as float32
        
        # Processing SR is the target SR after oversampling
        processing_sr = current_params["target_sr"] 
        
        # High-quality oversampling if needed
        y_os = y # Default if no oversampling
        if current_params["oversample_factor"] > 1 and sr_loaded != processing_sr:
            logger.info(f"High-quality oversampling from {sr_loaded}Hz to {processing_sr}Hz...")
            # y is already (channels, samples) or (samples,) from librosa
            try:
                # Use high-quality resampler if available (requires libresample/soxr)
                y_os = librosa.resample(
                    y, orig_sr=sr_loaded, target_sr=processing_sr, 
                    res_type='soxr_hq' 
                )
            except Exception as resample_err: # Fallback if soxr_hq fails (e.g., not installed)
                logger.warning(f"soxr_hq resampling failed: {resample_err}. Falling back to kaiser_best.")
                y_os = librosa.resample(
                    y, orig_sr=sr_loaded, target_sr=processing_sr,
                    res_type='kaiser_best' # High-quality alternative
                )
        else: # No oversampling needed or already at target SR
            processing_sr = sr_loaded # Actual processing SR is the loaded SR
            current_params["target_sr"] = sr_loaded # Update param to reflect actual SR

        logger.info("Executing ultra advanced BBE processing...")
        # Ensure y_os is float32 for GPU processing; safe_to_gpu handles dtype conversion if needed.
        # y_os is already float32 from librosa.load or librosa.resample.
        y_gpu = safe_to_gpu(y_os) 
        y_processed = ultra_advanced_bbe_processor(y_gpu, processing_sr, current_params)
        
        # High-quality downsampling if audio was oversampled
        output_sr = load_sr # Target output SR is the original (pre-oversample) SR
        y_processed_cpu = safe_to_cpu(y_processed) # Convert processed tensor to numpy array

        if current_params["oversample_factor"] > 1 and processing_sr != output_sr:
            logger.info(f"High-quality downsampling from {processing_sr}Hz to {output_sr}Hz...")
            try:
                y_final = librosa.resample(
                    y_processed_cpu,
                    orig_sr=processing_sr, 
                    target_sr=output_sr,
                    res_type='soxr_hq'
                )
            except Exception as resample_err: # Fallback for downsampling
                logger.warning(f"soxr_hq resampling failed for downsampling: {resample_err}. Falling back to kaiser_best.")
                y_final = librosa.resample(
                    y_processed_cpu,
                    orig_sr=processing_sr,
                    target_sr=output_sr,
                    res_type='kaiser_best'
                )
        else: # No downsampling needed
            y_final = y_processed_cpu
        
        # Final mastering (normalization)
        y_final = safe_normalize_audio(y_final, 0.5) # Normalize with 0.5dB headroom
        
        logger.info("Writing high-quality output file...")
        # sf.write expects channels-last (num_samples, num_channels)
        # librosa loads as (num_channels, num_samples) for stereo if mono=False
        if len(y_final.shape) > 1 and y_final.shape[0] < y_final.shape[1]: # (channels, samples)
            sf.write(output_file, y_final.T, output_sr, subtype='PCM_32') # Transpose and write as 32-bit PCM
        else: # Mono (samples,) or already (samples, channels)
            sf.write(output_file, y_final, output_sr, subtype='PCM_32') # Write as 32-bit PCM
        
        logger.info("Ultra advanced processing complete.")
        
    except subprocess.CalledProcessError as spe: # Specifically for FFmpeg errors
        logger.error(f"FFmpeg execution error: {spe}")
        if hasattr(spe, 'stdout') and spe.stdout: logger.error(f"FFmpeg stdout: {spe.stdout.decode(errors='ignore')}")
        if hasattr(spe, 'stderr') and spe.stderr: logger.error(f"FFmpeg stderr: {spe.stderr.decode(errors='ignore')}")
        raise # Re-raise to indicate failure
    except Exception as e: # General error catch for the ultra pipeline
        logger.error(f"Ultra advanced processing error: {e}", exc_info=True) # Log traceback
        logger.info("Falling back to optimized processing due to error in ultra mode...")
        # Fallback to effect_process_optimized. Ensure input_file for fallback is the original,
        # not a temporary WAV if conversion failed very early.
        try:
            effect_process_optimized(input_file, output_file, PARAMS) # Use original PARAMS for fallback
            logger.info("Optimized fallback processing completed.")
        except Exception as fallback_e:
            logger.error(f"Optimized fallback processing also failed: {fallback_e}", exc_info=True)
            raise fallback_e # Re-raise the fallback error
            
    finally:
        # Clean up any temporary files created
        for temp_file_path in temp_files:
            if os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.info(f"Temporary file removed: {temp_file_path}")
                except OSError as e_os:
                    logger.warning(f"Failed to remove temporary file {temp_file_path}: {e_os}")
        MemoryManager.clear_cache() # Clear memory cache at the end of processing

# === Updated Usage Example and CLI ===
if __name__ == '__main__':
    import argparse
    import sys # Import sys module for sys.exit

    parser = argparse.ArgumentParser(description="Sound Enhancer - Applies BBE-like effects to audio files.")
    parser.add_argument("input_file", help="Path to the input audio file (MP3 or WAV).")
    parser.add_argument(
        "--mode", 
        choices=['ultra', 'optimized'], 
        default='ultra', 
        help="Processing mode ('ultra' or 'optimized'). Default: 'ultra'."
    )
    parser.add_argument(
        "--suffix",
        default="_enhanced",
        help="Suffix to add to the output file name. Default: '_enhanced'."
    )

    args = parser.parse_args()

    input_path = args.input_file
    processing_mode = args.mode
    output_suffix = args.suffix

    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1) # Exit with error code 1

    # Generate output file path
    input_dir = os.path.dirname(input_path)
    input_filename, input_ext = os.path.splitext(os.path.basename(input_path))
    
    # Output will always be WAV format
    output_filename = f"{input_filename}{output_suffix}.wav"
    output_path = os.path.join(input_dir, output_filename)

    logger.info(f"--- Processing Start ---")
    logger.info(f"Input file: {input_path}")
    logger.info(f"Output file: {output_path}")
    logger.info(f"Processing mode: {processing_mode}")

    try:
        if processing_mode == 'ultra':
            logger.info("Executing ultra-quality processing...")
            ultra_effect_process(input_path, output_path)
        elif processing_mode == 'optimized':
            logger.info("Executing optimized processing...")
            effect_process_optimized(input_path, output_path)
        
        logger.info(f"Processing completed successfully. Output file: {output_path}")

    except subprocess.CalledProcessError as spe: # Catch errors from external processes like FFmpeg
        logger.error(f"External process error (e.g., FFmpeg): {spe}")
        if hasattr(spe, 'stdout') and spe.stdout:
            logger.error(f"Stdout: {spe.stdout.decode(errors='ignore') if isinstance(spe.stdout, bytes) else spe.stdout}")
        if hasattr(spe, 'stderr') and spe.stderr:
            logger.error(f"Stderr: {spe.stderr.decode(errors='ignore') if isinstance(spe.stderr, bytes) else spe.stderr}")
        logger.error("Processing failed. Please check the logs.")
        sys.exit(1) # Exit with error code 1
    except Exception as e: # Catch any other unexpected errors during main processing
        logger.error(f"An unexpected error occurred during main processing: {e}", exc_info=True)
        logger.error("Processing failed. Please check the logs.")
        sys.exit(1) # Exit with error code 1
    finally:
        MemoryManager.clear_cache() # Ensure memory is cleared at the very end
        logger.info("--- Processing End ---")