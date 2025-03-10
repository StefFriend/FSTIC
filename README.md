# FSTIC - Forensic Speech Transmission Index Calculator

![FLOUC Banner](https://img.shields.io/badge/FSTIC-Forensic%20Speech%20Transmission%20Index%20Calculator-blue)
![License](https://img.shields.io/badge/License-GNU%20GPLv3-green)
![Python](https://img.shields.io/badge/Python-3.6%2B-blue)

FSTIC is a Python tool for analyzing speech intelligibility in audio files, particularly for forensic applications. It implements the Speech Transmission Index (STI) methodology based on the research paper by Costantini, Paoloni, and Todisco: ["Objective Speech Intelligibility Measures Based on Speech Transmission Index for Forensic Applications"](https://www.researchgate.net/publication/279467055_Objective_Speech_Intelligibility_Measures_Based_on_Speech_Transmission_Index_for_Forensic_Applications).

## Overview

This tool is designed to provide objective measurements of speech intelligibility in challenging audio conditions, such as recordings from lawful interceptions or other forensic audio sources. It calculates the Speech Transmission Index (STI) over time for audio files and generates comprehensive analysis reports.

## Features

- **Multi-format Audio Support**: Analyzes WAV, MP3, and other common audio formats
- **Octave Band Analysis**: Filters audio into standard octave bands (125Hz-8kHz)
- **STI Calculation**: Implements the STI algorithm as described in the reference paper
- **Temporal Analysis**: Shows how intelligibility varies over time in the recording
- **Visualization**: Generates detailed charts showing waveform, spectrograms, and STI values
- **PDF Reports**: Creates comprehensive PDF reports for documentation
- **Batch Processing**: Supports processing multiple files in a directory
- **Audio Comparison**: Directly compare STI metrics and visualizations between two audio files

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/FSTIC.git
cd FSTIC

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
python fstic.py path/to/audiofile.wav --output ./results
```

### Advanced Options

```bash
python fstic.py path/to/audio/directory --output ./results --window 500 --hop 250 --nopdf --file-ext wav,mp3
```

### Comparing Two Audio Files

```bash
python fstic.py --compare path/to/file1.wav path/to/file2.wav --output ./comparison --window 500 --hop 250
```

### Arguments

- `input`: Path to audio file or directory containing audio files
- `--output`: Directory where results will be saved (default: "./output")
- `--window`: Analysis window length in milliseconds (default: 500)
- `--hop`: Hop size in milliseconds (default: 250)
- `--nopdf`: Flag to disable PDF report generation
- `--file-ext`: Comma-separated list of file extensions to process (default: processes common audio formats)
- `--compare`: Compare two audio files side by side (requires two file paths as arguments)

## Core Functions

### Audio Processing

- `read_audio_file(filepath)`: Reads audio from various formats, with fallback handling for unsupported formats
- `design_octave_band(fs, center_freq)`: Creates Butterworth bandpass filters for octave band analysis

### STI Calculation

- `compute_sti(audio, fs, window_dur, hop_dur)`: The main function that calculates STI values over time
  - Divides the signal into octave bands
  - Extracts envelope function for each band
  - Calculates modulation transfer function
  - Computes STI according to the standard formula

### Visualization and Reporting

- `create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename)`: Generates analysis plots including waveform, spectrograms, and STI over time
- `process_audio_file(audio_path, output_dir, window_ms, hop_ms, create_pdf)`: Processes a single audio file and generates all outputs
- `compare_two_audio_files(file1, file2, output_dir, window_ms, hop_ms, create_pdf)`: Compares two audio files by processing both and generating side-by-side visualizations and comparison metrics

## Output Files

For each audio file, FSTIC generates:
- CSV file with STI values over time
- PNG image with analysis charts
- PDF report with complete analysis and metadata (optional)
- Summary CSV for batch processing

For audio file comparisons, FSTIC generates:
- Comparison CSV with STI values from both files
- Side-by-side comparison chart in PNG format
- Comprehensive comparison PDF report with detailed metrics for both files

## Implementation Details

The STI calculation follows the methodology described in the reference paper:
1. The audio is filtered into seven octave bands (125Hz to 8000Hz)
2. For each band, the envelope function is extracted
3. The modulation transfer function is calculated for 14 modulation frequencies (0.63-12.5Hz)
4. These values are converted to equivalent signal-to-noise ratios
5. The SNRs are mapped to Transmission Indices
6. The weighted average across bands gives the final STI value

## Reference

This implementation is based on the method described in:

Costantini, G., Paoloni, A., & Todisco, M. (2010). "Objective Speech Intelligibility Measures Based on Speech Transmission Index for Forensic Applications." AES 39th International Conference, Hillerød, Denmark.

## License

This project is licensed under the GNU General Public License v3.0 (GNU GPLv3) - see the LICENSE file for details.

If you use this software please cite this repo: [FSTIC - Forensic Speech Transmission Index Calculator (https://github.com/StefFriend/FSTIC)](https://github.com/StefFriend/FSTIC)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
