```markdown
# Sound Enhancer (BBE-like Effects)

This Python script provides a suite of audio processing tools designed to enhance sound quality, offering effects reminiscent of BBE (Sonic Maximizer) processors and other advanced audio mastering techniques. It includes functionalities for harmonic excitation, dynamic equalization, spectral processing, phase-linear filtering, advanced stereo imaging, adaptive dynamics, and lookahead limiting.

The script supports two processing modes:
1.  **Optimized Mode**: A balanced approach offering significant enhancement with good performance.
2.  **Ultra Mode**: A high-quality mode that utilizes more advanced, computationally intensive algorithms for superior results, including phase-linear filters and high-quality resampling.

## Features

*   **BBE-like Processing**: Core engine for emulating the sonic characteristics of BBE processors, including:
    *   Low-end contouring and bass enhancement.
    *   High-frequency clarity and harmonic excitation.
*   **Harmonic Exciter**:
    *   Optimized single-band harmonic exciter.
    *   Advanced multi-band harmonic exciter with phase-linear crossovers (Ultra Mode).
*   **Dynamic Equalizer**: Adjusts EQ dynamically based on the input signal's content across multiple bands.
*   **Spectral Processor**: Modifies the audio spectrum with controls for low, mid, high frequencies, and overall brightness.
*   **Phase-Linear Filters**: FIR-based filters for equalization tasks where phase coherence is critical (Ultra Mode). Includes fallback to IIR filters if FIR design fails.
*   **Advanced Stereo Imaging**:
    *   Mid/Side based stereo width adjustment.
    *   Binaural stereo widening for an immersive soundstage (Ultra Mode).
*   **Adaptive Dynamics**:
    *   Spectral Compressor: Frequency-dependent compression for targeted dynamic control (Ultra Mode).
*   **Advanced Limiter**:
    *   Lookahead Limiter: Prevents peaks effectively with minimal audible distortion by anticipating them (Ultra Mode).
*   **GPU Acceleration**: Supports MPS (Apple Silicon GPUs) for PyTorch-based operations, with CPU fallback.
*   **Memory Management**: Includes utilities to monitor and clear memory, especially important for large audio files and intensive processing.
*   **Chunk Processing**: Processes audio in chunks for memory efficiency, especially for long files, using overlap-add with Hann windowing.
*   **MP3 Support**: Automatically converts MP3 input files to WAV for processing using FFmpeg (FFmpeg must be installed and in your system's PATH).
*   **Configurable Parameters**: A wide range of parameters can be tuned to achieve desired sonic results.
*   **Two Processing Modes**:
    *   `optimized`: Faster processing with good quality.
    *   `ultra`: Highest quality processing using advanced algorithms like SOXR resampling and phase-linear filters.

## Prerequisites

*   Python 3.8 or newer.
*   FFmpeg: Required for MP3 file input. Ensure it's installed and accessible from your system's PATH.
    *   You can download FFmpeg from [ffmpeg.org](https://ffmpeg.org/download.html).

## Installation

1.  **Clone the repository (or download the script):**
    ```bash
    git clone https://github.com/littlebuddha-dev/python-sound-enhancer.git
    cd your-repository-name
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    The script uses several libraries for audio processing, numerical computation, and system utilities. Install them using the provided `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file includes:
    ```
    numpy>=1.20.0,<2.0.0
    scipy>=1.7.0
    librosa>=0.9.0
    soundfile>=0.11.0
    torch>=1.12.0
    psutil>=5.8.0
    ```

## Usage

The script is run from the command line.

**Basic syntax:**

```bash
python Sound_Enhancer_new.py <input_file> [--mode <mode>] [--suffix <suffix>]
```

**Arguments:**

*   `input_file`: Path to the input audio file (MP3 or WAV).
*   `--mode <mode>`: (Optional) Processing mode.
    *   `ultra`: For ultra-high quality processing (default).
    *   `optimized`: For faster, optimized processing.
*   `--suffix <suffix>`: (Optional) Suffix to add to the output filename (before the `.wav` extension). Default is `_enhanced`.

**Examples:**

1.  **Process an MP3 file in Ultra mode (default):**
    ```bash
    python Sound_Enhancer_new.py audio/my_song.mp3
    ```
    This will create `audio/my_song_enhanced.wav`.

2.  **Process a WAV file in Optimized mode:**
    ```bash
    python Sound_Enhancer_new.py samples/track.wav --mode optimized
    ```
    This will create `samples/track_enhanced.wav`.

3.  **Process an MP3 file and specify a custom output suffix:**
    ```bash
    python Sound_Enhancer_new.py song.mp3 --suffix _bbe_ultra
    ```
    This will create `song_bbe_ultra.wav`.

The script will output a WAV file with the enhancements applied. Log messages during processing will indicate the steps being performed and any warnings or errors.

## Customization

The core processing parameters are defined in the `PARAMS` dictionary within the `Sound_Enhancer_new.py` script. You can modify these default values directly in the script if you need to fine-tune the effects for specific applications.

Advanced features in "Ultra" mode (like `use_phase_linear`, `multiband_exciter`, etc.) are enabled by default when running in that mode but can also be adjusted within the `ultra_effect_process` function or by modifying `PARAMS` if those flags are exposed there.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

**MIT License**

Copyright (c) [2025] [littlebuddha]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

**To make this fully ready for GitHub:**

1.  **Replace Placeholders:**
    *   In the "Installation" section: `https://your-github-username/your-repository-name.git` should be replaced with the actual URL of your repository.
    *   In the MIT License text at the end: `[Year]` should be the current year (e.g., `2023` or `2024`) and `[Your Name/Organization]` should be your GitHub username, your name, or your organization's name.
2.  **Create a `LICENSE` file:**
    Copy the MIT License text (starting from "MIT License" and ending with "DEALINGS IN THE SOFTWARE.") into a new file named `LICENSE` (no extension) in the root of your repository. The `README.md` already links to this.
3.  **Ensure `Sound_Enhancer_new.py` and `requirements.txt` are in the repository root.**

This `README.md` aims to be comprehensive, covering the purpose, key features, setup, usage, and licensing of your script.
