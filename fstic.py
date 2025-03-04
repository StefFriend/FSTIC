import numpy as np
import math
import soundfile as sf
from scipy.signal import butter, sosfilt
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import datetime
from matplotlib.backends.backend_pdf import PdfPages
import wave
import contextlib
import hashlib
import argparse
import glob

# --- Function to read an audio file (supports WAV, MP3, etc.) ---
def read_audio_file(filepath):
    """
    Reads the audio file from the specified 'filepath'. 
    It can handle multiple formats (WAV, MP3, etc.). 
    If the format is not directly supported by soundfile, it falls back to pydub.
    
    Parameters:
    filepath (str): Path to the audio file.
    
    Returns:
    tuple: A tuple (data, fs) where 'data' is a NumPy array of audio samples (mono) 
           and 'fs' is the sampling rate in Hz.
    """
    try:
        data, fs = sf.read(filepath)
    except Exception:
        # Fallback using pydub for unsupported formats (e.g., MP3)
        audio = AudioSegment.from_file(filepath)
        fs = audio.frame_rate
        if audio.channels > 1:
            audio = audio.set_channels(1)
        data = np.array(audio.get_array_of_samples(), dtype=np.float32)
        if audio.sample_width == 2:  # 16-bit PCM
            data /= 2**15
        elif audio.sample_width == 3:  # 24-bit
            data /= 2**23
        elif audio.sample_width == 4:  # 32-bit
            data /= 2**31
    else:
        if data.ndim > 1:
            data = data.mean(axis=1)
    return np.array(data, dtype=np.float64), fs

# --- Function to design a one-octave bandpass filter (Butterworth) ---
def design_octave_band(fs, center_freq):
    """
    Designs a 4th-order Butterworth bandpass filter for one-octave band 
    around 'center_freq'.
    
    The frequency boundaries are center_freq / sqrt(2) and center_freq * sqrt(2).
    
    Parameters:
    fs (int): Sampling rate in Hz.
    center_freq (float): Center frequency of the octave band.
    
    Returns:
    sos (ndarray or None): Second-order sections representation of the filter.
                           Returns None if the filter cutoffs exceed the Nyquist limit.
    """
    # Calculate the frequency boundaries
    low = center_freq / math.sqrt(2)
    high = center_freq * math.sqrt(2)
    nyq = 0.5 * fs
    low_cut = max(low / nyq, 1e-5)
    high_cut = min(high / nyq, 0.99999)
    if low_cut >= 1:
        return None
    if high_cut >= 1:
        high_cut = 0.99999
    sos = butter(N=4, Wn=[low_cut, high_cut], btype='bandpass', output='sos')
    return sos

# --- Function to compute the STI according to the referenced paper ---
def compute_sti(audio, fs, window_dur=0.5, hop_dur=0.25):
    """
    Computes the Speech Transmission Index (STI) for the given audio signal 'audio' 
    sampled at 'fs'. 
    
    The main steps are:
      1. Split into octave bands (centers: 125, 250, 500, 1000, 2000, 4000, 8000 Hz).
      2. For each band: extract the local envelope by windowing the signal power
         (Hanning window, equation (2) in the paper).
      3. Compute the normalized envelope spectrum at 14 modulation frequencies (0.63–12.5 Hz).
      4. Map to SNR in dB (limited between -15 and +15 dB), then convert to Transmission Index (TI) 
         as in equation (5).
      5. Compute the Modulation Transfer Index (MTI) for each band and finally compute 
         the overall STI (equation (7)).
    
    The calculation is done over overlapping windows (default 0.5 s with 0.25 s hop).
    
    Parameters:
    audio (ndarray): The audio signal (must be mono).
    fs (int): Sampling rate in Hz.
    window_dur (float): Window duration in seconds (default 0.5s).
    hop_dur (float): Hop duration in seconds (default 0.25s).
    
    Returns:
    tuple: Two arrays (time_stamps, sti_values) giving the STI over time.
    """
    # Define octave band centers and weights (empirical weighting across bands)
    band_centers = [125, 250, 500, 1000, 2000, 4000, 8000]
    band_weights = [0.01, 0.04, 0.146, 0.212, 0.308, 0.244, 0.04]
    nyquist = 0.5 * fs
    valid_bands = []
    valid_weights = []
    for center, w in zip(band_centers, band_weights):
        if center / math.sqrt(2) < nyquist * 0.999:
            valid_bands.append(center)
            valid_weights.append(w)
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    # Filter the signal in each octave band
    band_signals = {}
    for center, w in zip(valid_bands, valid_weights):
        sos = design_octave_band(fs, center)
        if sos is None:
            continue
        y = sosfilt(sos, audio)
        band_signals[center] = y

    # Standard modulation frequencies (14 values, 0.63 to 12.5 Hz)
    mod_freqs = np.array([0.63, 0.8, 1.0, 1.25, 1.6, 2.0, 2.5, 
                          3.15, 4.0, 5.0, 6.3, 8.0, 10.0, 12.5])
    
    # Parameters for local envelope extraction
    env_window = int(0.05 * fs)  # ~50 ms window
    if env_window < 1:
        env_window = 1
    env_hop = int(0.01 * fs)     # ~10 ms hop
    if env_hop < 1:
        env_hop = 1
    hann = np.hanning(env_window)
    
    # Parameters for STI sliding window computation
    frame_length = int(window_dur * fs)
    frame_step = int(hop_dur * fs)
    num_frames = 1 + max(0, (len(audio) - frame_length) // frame_step)
    
    sti_values = []
    time_stamps = []
    
    # Compute STI for each time window
    for i in range(num_frames):
        start = i * frame_step
        end = start + frame_length
        if end > len(audio):
            break
        segment_sti = 0.0
        # Process each band
        for center, Wk in zip(valid_bands, valid_weights):
            x_band = band_signals[center][start:end]
            power = x_band**2
            # Local envelope computation via moving window with a Hanning window
            if len(power) < len(hann):
                pad_width = len(hann) - len(power)
                power_padded = np.pad(power, (0, pad_width), 'constant', constant_values=0)
            else:
                power_padded = power
            envelope = np.convolve(power_padded, hann, mode='valid')[::env_hop]
            envelope = np.clip(envelope, a_min=0.0, a_max=None)
            
            # Normalize envelope and compute its spectrum
            E = envelope
            if len(E) == 0:
                continue
            sumE = np.sum(E)
            if sumE <= 1e-8:
                continue
            M_f = []
            N = len(E)
            env_dt = env_hop / fs  # time interval between envelope samples
            t = np.arange(N) * env_dt
            # For each modulation frequency, compute the complex component
            for f in mod_freqs:
                phi = 2 * np.pi * f * t
                comp = np.dot(E, np.exp(-1j * phi))
                m_val = (2.0 * abs(comp)) / sumE
                if m_val > 1.0:
                    m_val = 1.0
                M_f.append(m_val)
            M_f = np.array(M_f)
            
            # Compute SNR in dB (formula (4)) for each modulation frequency
            eps = 1e-12
            M_sq = M_f**2
            M_sq = np.clip(M_sq, 0.0, 1.0 - 1e-9)
            snr_values = 10.0 * np.log10((M_sq + eps) / (1.0 - M_sq + eps))
            # Limit SNR between -15 and +15 dB
            snr_values = np.clip(snr_values, -15.0, 15.0)
            # Compute the Transmission Index (TI) (formula (5))
            TI = (snr_values + 15.0) / 30.0
            # Average TI for the band's Modulation Transfer Index (MTI)
            MTI_k = np.mean(TI)
            segment_sti += Wk * MTI_k
        # Center time of the window
        time_center = (start + frame_length/2) / fs
        time_stamps.append(time_center)
        sti_values.append(segment_sti)
    
    return np.array(time_stamps), np.array(sti_values)

# --- Function to create analysis plots ---
def create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename):
    """
    Creates analysis plots for the audio file:
      1. Waveform
      2. Spectrogram (viridis colormap)
      3. Spectrogram (magma colormap) limited to 20-4000 Hz
      4. STI over time
    
    Parameters:
    audio_signal (ndarray): Audio data (mono).
    sample_rate (int): Sampling rate in Hz.
    times (ndarray): Time array corresponding to STI calculations.
    sti (ndarray): Computed STI values over time.
    overall_sti (float): The average STI over the entire audio file.
    audio_filename (str): Name of the analyzed audio file.
    
    Returns:
    matplotlib.figure.Figure: The figure containing the analysis plots.
    """
    fig, axs = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    
    # Main title with the name of the file
    fig.suptitle(f"STI Analysis: {audio_filename}", fontsize=14, fontweight='bold')
    
    # Formatter for the x-axis to display time in seconds with two decimals
    time_formatter = ticker.FormatStrFormatter('%.2f')
    
    # 1. Waveform
    t_audio = np.arange(len(audio_signal)) / sample_rate
    axs[0].plot(t_audio, audio_signal, color='blue')
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform")
    axs[0].grid(True)
    axs[0].xaxis.set_major_formatter(time_formatter)
    
    # 2. Spectrogram with viridis colormap
    NFFT = 2048
    noverlap = 1024
    Pxx, freqs, bins, im = axs[1].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, 
                                           noverlap=noverlap, cmap='viridis')
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectrogram")
    axs[1].grid(True)
    axs[1].xaxis.set_major_formatter(time_formatter)
    
    # 3. Another spectrogram with magma colormap, limited to 20-4000 Hz
    Pxx, freqs, bins, im = axs[2].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, 
                                           noverlap=noverlap, cmap='magma')
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_title("Spectrogram (20-4000 Hz)")
    axs[2].set_ylim(20, 4000)
    axs[2].grid(True)
    axs[2].xaxis.set_major_formatter(time_formatter)
    
    # 4. STI step plot: each step represents a window ('post' style)
    axs[3].step(times, sti, where='post', color='red')
    axs[3].set_xlabel("Time (s)")
    axs[3].set_ylabel("STI")
    axs[3].set_title("STI Over Time")
    axs[3].set_ylim(0, 1)
    axs[3].grid(True)
    axs[3].xaxis.set_major_formatter(time_formatter)
    
    # Horizontal line at the overall STI value
    axs[3].axhline(y=overall_sti, color='darkred', linestyle='--', alpha=0.7)
    
    # Text box for the average STI
    text_box = axs[3].text(
        0.95, 0.95, f'Mean STI: {overall_sti:.3f}', 
        transform=axs[3].transAxes,
        color='darkred', fontsize=10, fontweight='bold',
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkred', boxstyle='round,pad=0.5')
    )
    
    # Force x-axis labels for all subplots
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=True)
    
    # Set x-axis labels
    for ax in axs:
        ax.set_xlabel("Time (s)")
    
    # Adequate spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, top=0.9)
    
    return fig

# --- Function to process a single audio file ---
def process_audio_file(audio_path, output_dir, window_ms, hop_ms, create_pdf=True):
    """
    Processes a single audio file by computing STI and generating output files.
    
    Parameters:
    audio_path (str): Path to the audio file to process.
    output_dir (str): Folder to save the results.
    window_ms (int): Window duration in milliseconds.
    hop_ms (int): Hop duration in milliseconds.
    create_pdf (bool): If True, a PDF report is generated.
    
    Returns:
    tuple: (success, overall_sti) 
           where 'success' is a boolean indicating if processing succeeded 
           and 'overall_sti' is the computed average STI (or None on failure).
    """
    try:
        # Convert milliseconds to seconds
        window_dur = window_ms / 1000.0
        hop_dur = hop_ms / 1000.0
        
        # Create output folder if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Read the audio file
        audio_signal, sample_rate = read_audio_file(audio_path)
        
        # Extract file name without extension/path
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        # Compute STI over time
        print(f"Processing {audio_filename} with window={window_ms}ms, hop={hop_ms}ms...")
        times, sti = compute_sti(audio_signal, sample_rate, window_dur=window_dur, hop_dur=hop_dur)
        
        # Compute overall STI (mean of the STI values)
        overall_sti = np.mean(sti)
        
        # Calculate file SHA256 hash
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"Overall STI (mean): {overall_sti:.3f}")
        print(f"SHA256: {file_hash}")
        
        # Save STI results to a CSV file
        csv_filename = os.path.join(output_dir, f"sti_results_{audio_name}.csv")
        with open(csv_filename, "w") as f:
            f.write("Time_s,STI\n")
            for t, val in zip(times, sti):
                f.write(f"{t:.3f},{val:.3f}\n")
        
        print(f"STI results saved to {csv_filename}")
        
        # Create analysis plots (once)
        fig_plots = create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename)
        
        # Save the plot with a name including the analyzed audio file
        plot_filename = os.path.join(output_dir, f"chart_{audio_name}.png")
        fig_plots.savefig(plot_filename)
        print(f"Plot saved to {plot_filename}")
        
        # Create a PDF report (A4) if requested
        if create_pdf:
            pdf_filename = os.path.join(output_dir, f"report_{audio_name}.pdf")
            with PdfPages(pdf_filename) as pdf:
                # A4 document dimensions in inches
                a4_width_inch, a4_height_inch = 8.27, 11.69
                
                # First page: header and audio file info
                fig_info = plt.figure(figsize=(a4_width_inch, a4_height_inch))
                
                # Layout with appropriate margins
                ax_header = plt.axes([0.1, 0.8, 0.8, 0.15])
                ax_header.axis('off')
                ax_info = plt.axes([0.1, 0.3, 0.8, 0.45])
                ax_info.axis('off')
                ax_footer = plt.axes([0.1, 0.05, 0.8, 0.1])
                ax_footer.axis('off')
                
                # Header text
                ax_header.text(0.5, 0.8, "STI ANALYSIS REPORT", 
                               horizontalalignment='center', fontsize=18, fontweight='bold')
                ax_header.text(0.5, 0.5, f"File: {audio_filename}", 
                               horizontalalignment='center', fontsize=14)
                
                # Separator line
                ax_header.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # Gather audio file info
                frames = len(audio_signal)
                rate = sample_rate
                duration = frames / float(rate)
                
                # Convert duration to hh:mm:ss.mmm format
                hours, remainder = divmod(duration, 3600)
                minutes, seconds = divmod(remainder, 60)
                milliseconds = int((seconds - int(seconds)) * 1000)
                seconds = int(seconds)
                formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"
                
                # Additional info if the file is WAV
                try:
                    with contextlib.closing(wave.open(audio_path, 'r')) as f:
                        frames = f.getnframes()
                        rate = f.getframerate()
                        duration = frames / float(rate)
                        channels = f.getnchannels()
                        sampwidth = f.getsampwidth()
                        format_info = f"{channels} channels, {sampwidth*8} bit"

                        hours, remainder = divmod(duration, 3600)
                        minutes, seconds = divmod(remainder, 60)
                        milliseconds = int((seconds - int(seconds)) * 1000)
                        seconds = int(seconds)
                        formatted_duration = f"{int(hours):02d}:{int(minutes):02d}:{seconds:02d}.{milliseconds:03d}"

                except:
                    # If not a WAV file, use the previously calculated info
                    channels = 1  # After mono conversion
                    format_info = "Converted to mono"
                
                # Analysis parameters
                analysis_params = (
                    f"• Analysis window: {window_ms} ms\n"
                    f"• Hop interval: {hop_ms} ms\n"
                )
                
                # File info text
                info_text = (
                    "TECHNICAL SPECIFICATIONS\n\n"
                    f"• File name: {audio_filename}\n"
                    f"• Format: {format_info}\n"
                    f"• Sampling rate: {rate} Hz\n"
                    f"• Number of samples: {frames:,}\n"
                    f"• Duration: {formatted_duration}\n\n"
                    f"ANALYSIS PARAMETERS\n\n"
                    f"{analysis_params}\n"
                    f"HASH\n\n"
                    f"• SHA-256: {file_hash}\n\n"
                    f"STI ANALYSIS RESULTS\n\n"
                    f"• Mean STI: {overall_sti:.3f}\n"
                )
                
                # Additional STI statistics
                sti_min = np.min(sti)
                sti_max = np.max(sti)
                sti_std = np.std(sti)
                
                info_text += f"• Min STI: {sti_min:.3f}\n"
                info_text += f"• Max STI: {sti_max:.3f}\n"
                info_text += f"• Standard Deviation: {sti_std:.3f}\n"
                
                ax_info.text(0, 1, info_text, fontsize=11, verticalalignment='top', 
                             horizontalalignment='left', linespacing=1.5)
                
                # Footer with page number
                ax_footer.text(0.5, 0.2, f"Page 1/2", 
                               horizontalalignment='center', fontsize=8)
                
                # Save the first page
                pdf.savefig(fig_info)
                plt.close(fig_info)
                
                # Second page: reuse the already created figure
                # Resize figure to A4
                fig_plots.set_size_inches(a4_width_inch, a4_height_inch)
                
                # Update title for the PDF
                fig_plots.suptitle(f"STI Analysis: {audio_filename}", fontsize=12, fontweight='bold', y=0.98)
                
                # Footer for page 2
                fig_plots.text(0.5, 0.01, "Page 2/2", ha='center', fontsize=8)
                
                # Save the second page
                pdf.savefig(fig_plots)
                
                print(f"PDF report saved to {pdf_filename}")
        
        plt.close('all')  # Close all Matplotlib figures
        return True, overall_sti
        
    except Exception as e:
        print(f"Error while processing {audio_path}: {str(e)}")
        return False, None

# --- Main: Processes audio files based on command-line parameters ---
if __name__ == "__main__":
    # Configure command-line argument parser
    parser = argparse.ArgumentParser(description="STI analysis for audio files")
    
    # Required parameters
    parser.add_argument("input", help="Audio file or folder containing audio files to process")
    parser.add_argument("--output", help="Output folder for the results", default="./output")
    
    # Optional parameters
    parser.add_argument("--window", type=int, help="Analysis window length in milliseconds", default=500)
    parser.add_argument("--hop", type=int, help="Hop length in milliseconds", default=250)
    parser.add_argument("--nopdf", action="store_true", help="Do not generate a PDF report")
    parser.add_argument("--file-ext", help="Filter only these file extensions (comma-separated). If omitted, it processes all common audio formats", default=None)
    
    args = parser.parse_args()
    
    # List of common audio file extensions (used if --file-ext is not specified)
    common_audio_extensions = ["wav", "mp3", "ogg", "flac", "aac", "wma", "m4a", "aiff", "opus"]
    
    # If --file-ext is specified, use those extensions; otherwise, use the default list
    if args.file_ext:
        file_extensions = args.file_ext.lower().split(',')
        file_extensions = [ext.strip() for ext in file_extensions]
        print(f"Processing limited to extensions: {', '.join(file_extensions)}")
    else:
        file_extensions = common_audio_extensions
        print(f"Processing all common audio formats: {', '.join(file_extensions)}")
    
    # Check if the input is a file or a folder
    if os.path.isfile(args.input):
        # Process a single file
        print(f"Processing single file: {args.input}")
        success, sti = process_audio_file(
            args.input, 
            args.output, 
            args.window, 
            args.hop, 
            not args.nopdf
        )
        if success:
            print(f"Processing successfully completed. STI: {sti:.3f}")
        else:
            print("Error while processing the file")
    
    elif os.path.isdir(args.input):
        # Process all audio files in the folder with the specified extensions
        print(f"Searching for audio files in folder: {args.input}")
        
        # Create a list of patterns to find the files
        patterns = [os.path.join(args.input, f"*.{ext}") for ext in file_extensions]
        
        # Gather all matching audio files
        audio_files = []
        for pattern in patterns:
            audio_files.extend(glob.glob(pattern))
        
        if not audio_files:
            print(f"No audio files found in {args.input} with extensions {file_extensions}")
            exit(1)
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Create a summary CSV file
        summary_csv = os.path.join(args.output, "sti_summary.csv")
        os.makedirs(args.output, exist_ok=True)
        
        with open(summary_csv, "w") as f:
            f.write("Filename,STI_Mean,Success\n")
            
            # Process each audio file
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)
                print(f"\nProcessing {filename}...")
                
                success, sti = process_audio_file(
                    audio_file, 
                    args.output, 
                    args.window, 
                    args.hop, 
                    not args.nopdf
                )
                
                # Add result to summary CSV
                if success and sti is not None:
                    f.write(f"{filename},{sti:.3f},1\n")
                    print(f"  -> Success: STI = {sti:.3f}")
                else:
                    f.write(f"{filename},N/A,0\n")
                    print(f"  -> Error: unable to compute STI")
        
        print(f"\nSummary saved to {summary_csv}")
    
    else:
        print(f"Error: '{args.input}' is neither a valid file nor a valid directory")
        exit(1)
    
    print("\nProcessing completed")
