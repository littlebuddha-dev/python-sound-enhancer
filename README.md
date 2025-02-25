# python-sound-enhancer
High-Quality Audio Enhancement Test on Intel Mac with Python 3.10

GPU Processing Support
Uses PyTorch MPS backend to accelerate processing on Intel Mac.

Oversampling
Processes at a high sampling rate (96kHz) to improve processing quality.

Multiple Effects:
Bass Enhancer (Low-frequency enhancement)
Shelving Filter (Low and high-frequency adjustment)
Harmonic Exciter (Harmonic generation)
Dynamic EQ
Spectral Processor
Stereo Width Expansion
Transient Shaper
Multiband Compressor
Saturation
Limiter
Effect Evaluation
Bass Enhancer

Settings: bass_freq=80Hz, intensity=0.7
Effect: Psychoacoustic bass enhancement boosts low frequencies around 80Hz and adds artificial harmonics at 160Hz (80Hz × 2). This makes bass sound "thicker" even on small speakers.
Shelving Filter
Low shelf: +4dB below 100Hz (Q=0.7)
High shelf: +5dB above 6000Hz (Q=0.8)
Effect: Creates a "smile curve EQ" effect, making the sound more dynamic and "lively."
Harmonic Exciter
Settings: Nonlinear distortion with intensity=0.5 in the 3000Hz–12000Hz range.
Effect: Adds harmonics in the high-frequency range, enhancing clarity and detail, especially in vocals and string instruments.
Dynamic EQ
Settings: 3-band EQ, threshold=-24dB, ratio=1.5
Effect: Dynamically adjusts gain based on volume, reducing overly loud frequencies and enhancing quieter ones, leading to a more balanced sound.
Spectral Processor
Settings: Low frequencies ×1.2, mid frequencies ×0.9, high frequencies ×1.3, brightness ×1.1
Effect: Enhances lows and highs while slightly reducing mids for a "modern" sound profile. The brightness parameter fine-tunes the overall high-frequency tilt.
Stereo Width Expansion
Settings: Width ×1.3
Effect: Processes mid and side signals separately, boosting the side signal to enhance stereo imaging and spatial depth.
Transient Shaper
Settings: Attack ×1.5, Sustain ×0.7
Effect: Emphasizes the attack phase while reducing sustain, enhancing the punchiness of drums and percussion.
Multiband Compressor
Settings: Overall ratio = 1.5, Mid ratio = 1.8, Side ratio = 1.3
Effect: Balances volume levels by compressing loud signals and boosting quieter ones. The stronger mid compression enhances the presence of vocals and main instruments.
Saturation
Settings: Amount = 0.5, Drive = 1.0, Mix = 1.0
Effect: Adds dynamic, "warm" non-linear distortion, mimicking the natural thickness of analog processing.
Limiter
Settings: Threshold = 0.95, Output gain = 1.0
Effect: Optimizes the final volume while preventing clipping (digital distortion), allowing for safe loudness maximization.
Overall Effects
By combining multiple stages of audio processing, this system delivers the following benefits:

Improved Clarity: Enhances high and low frequencies while refining details through transient shaping.
Added Warmth and Thickness: Introduces analog-like characteristics through saturation and bass enhancement.
Expanded Stereo Imaging: Creates a wider spatial sound field using stereo width expansion.
Optimized Dynamic Range: Controls audio dynamics through compressors and limiters.
Enhanced Frequency Balance: Adjusts tonal balance using multiple EQs and spectral processing.
This system achieves results similar to professional mastering processes. The 96kHz oversampling minimizes aliasing (digital artifacts), ensuring high-quality sound processing.

Requirements
FFmpeg is required.
Replace input.mp3 with your music file.
The output will be saved as output.wav.
