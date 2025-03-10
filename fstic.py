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

# -----------------------------------------------------------------------------------
# Utility function to format time as HH:MM:SS.mmm
# -----------------------------------------------------------------------------------
def format_time_hhmmssmmm(seconds_float):
    """
    Convert a floating-point 'seconds_float' into a time string HH:MM:SS.mmm
    where HH, MM, SS are zero-padded and mmm = milliseconds.
    """
    hours = int(seconds_float // 3600)
    remainder = seconds_float % 3600
    minutes = int(remainder // 60)
    seconds = remainder % 60
    millis = int(round((seconds - int(seconds)) * 1000))
    seconds = int(seconds)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{millis:03d}"

# -----------------------------------------------------------------------------------
# Function to read an audio file (supports WAV, MP3, etc.)
# -----------------------------------------------------------------------------------
def read_audio_file(filepath):
    """
    Reads the audio file from the specified 'filepath'. 
    It can handle multiple formats (WAV, MP3, etc.). 
    If the format is not directly supported by soundfile, it falls back to pydub.
    
    Returns:
    tuple: (data, fs) where 'data' is a NumPy array of audio samples (mono),
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

# -----------------------------------------------------------------------------------
# Function to design a one-octave bandpass filter (Butterworth)
# -----------------------------------------------------------------------------------
def design_octave_band(fs, center_freq):
    """
    Designs a 4th-order Butterworth bandpass filter for one-octave band 
    around 'center_freq'.
    
    The frequency boundaries are center_freq / sqrt(2) and center_freq * sqrt(2).
    
    Returns:
    sos (ndarray or None): second-order sections representation of the filter.
                           Returns None if the filter cutoffs exceed Nyquist.
    """
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

# -----------------------------------------------------------------------------------
# Function to compute STI (speech transmission index) for a single file
# -----------------------------------------------------------------------------------
def compute_sti(audio, fs, window_dur=0.5, hop_dur=0.25):
    """
    Computes the Speech Transmission Index (STI) for the given audio signal 'audio' 
    sampled at 'fs'. 
    
    Steps:
      1. Split into octave bands (centers: 125, 250, 500, 1000, 2000, 4000, 8000 Hz).
      2. For each band: extract the local envelope by windowing the signal power.
      3. Compute the normalized envelope spectrum at 14 modulation frequencies (0.63–12.5 Hz).
      4. Map to SNR in dB, then convert to Transmission Index (TI).
      5. Compute the Modulation Transfer Index (MTI) for each band and the overall STI.
    
    The calculation is done over overlapping windows of duration 'window_dur' 
    with a hop of 'hop_dur'.
    
    Returns:
      (time_stamps, sti_values)
    """
    band_centers = [125, 250, 500, 1000, 2000, 4000, 8000]
    band_weights = [0.01, 0.04, 0.146, 0.212, 0.308, 0.244, 0.04]
    nyquist = 0.5 * fs
    
    # Filter out any band that can't be designed properly for the given fs
    valid_bands = []
    valid_weights = []
    for center, w in zip(band_centers, band_weights):
        if center / math.sqrt(2) < nyquist * 0.999:
            valid_bands.append(center)
            valid_weights.append(w)
    valid_weights = np.array(valid_weights)
    valid_weights = valid_weights / valid_weights.sum()
    
    # Filter the signal in each valid octave band
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
    env_window = max(env_window, 1)
    env_hop = int(0.01 * fs)     # ~10 ms hop
    env_hop = max(env_hop, 1)
    hann = np.hanning(env_window)
    
    # Parameters for STI sliding window
    frame_length = int(window_dur * fs)
    frame_step = int(hop_dur * fs)
    num_frames = 1 + max(0, (len(audio) - frame_length) // frame_step)
    
    sti_values = []
    time_stamps = []
    
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
            
            # Local envelope computation
            if len(power) < len(hann):
                pad_width = len(hann) - len(power)
                power_padded = np.pad(power, (0, pad_width), 'constant', constant_values=0)
            else:
                power_padded = power
            
            envelope = np.convolve(power_padded, hann, mode='valid')[::env_hop]
            envelope = np.clip(envelope, a_min=0.0, a_max=None)
            
            E = envelope
            if len(E) == 0:
                continue
            sumE = np.sum(E)
            if sumE <= 1e-8:
                continue
            
            # Compute the modulation index for standard mod freqs
            M_f = []
            N = len(E)
            env_dt = env_hop / fs
            t = np.arange(N) * env_dt
            for f in mod_freqs:
                phi = 2 * np.pi * f * t
                comp = np.dot(E, np.exp(-1j * phi))
                m_val = (2.0 * abs(comp)) / sumE
                m_val = min(m_val, 1.0)
                M_f.append(m_val)
            M_f = np.array(M_f)
            
            # Compute SNR in dB
            eps = 1e-12
            M_sq = M_f**2
            M_sq = np.clip(M_sq, 0.0, 1.0 - 1e-9)
            snr_values = 10.0 * np.log10((M_sq + eps) / (1.0 - M_sq + eps))
            # Clip SNR between -15 and +15 dB
            snr_values = np.clip(snr_values, -15.0, 15.0)
            
            # Transmission Index for each modulation frequency
            TI = (snr_values + 15.0) / 30.0
            MTI_k = np.mean(TI)
            segment_sti += Wk * MTI_k
        
        # Center time of the window (in seconds)
        time_center_sec = (start + frame_length/2) / fs
        
        time_stamps.append(time_center_sec)
        sti_values.append(segment_sti)
    
    return np.array(time_stamps), np.array(sti_values)

# -----------------------------------------------------------------------------------
# Single-file analysis plots (only 3 time ticks: start, mid, end)
# -----------------------------------------------------------------------------------
def create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename):
    """
    Creates analysis plots for a single audio file:
      1. Waveform
      2. Spectrogram (viridis)
      3. Spectrogram (magma, limited to 20-4000 Hz)
      4. STI over time
    Each subplot will show x-axis ticks only at 0s, mid, and end.
    Returns a matplotlib Figure.
    """
    # Calculate total duration
    total_duration = len(audio_signal) / sample_rate
    mid_time = total_duration / 2.0
    
    fig, axs = plt.subplots(4, 1, figsize=(9, 10))  # Further reduced width for better margins
    
    # Removed main title
    
    # 1. Waveform
    t_audio = np.arange(len(audio_signal)) / sample_rate
    axs[0].plot(t_audio, audio_signal)
    axs[0].set_ylabel("Amplitude")
    axs[0].set_title("Waveform", pad=10)  # Added padding to the title
    axs[0].grid(True)
    
    # 2. Spectrogram with viridis
    NFFT = 2048
    noverlap = 1024
    axs[1].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='viridis')
    axs[1].set_ylabel("Frequency (Hz)")
    axs[1].set_title("Spectrogram", pad=10)  # Added padding to the title
    axs[1].grid(True)
    
    # 3. Spectrogram with magma, limited to 20-4000 Hz
    axs[2].specgram(audio_signal, NFFT=NFFT, Fs=sample_rate, noverlap=noverlap, cmap='magma')
    axs[2].set_ylabel("Frequency (Hz)")
    axs[2].set_title("Spectrogram (20-4000 Hz)", pad=10)  # Added padding to the title
    axs[2].set_ylim(20, 4000)
    axs[2].grid(True)
    
    # 4. STI over time (step)
    axs[3].step(times, sti, where='post', color='red')
    axs[3].set_ylabel("STI")
    axs[3].set_title("STI Over Time", pad=10)  # Added padding to the title
    axs[3].set_ylim(0, 1)
    axs[3].grid(True)
    
    # Horizontal line for overall STI
    axs[3].axhline(y=overall_sti, color='darkred', linestyle='--', alpha=0.7)
    
    # Text box
    axs[3].text(
        0.95, 0.95, f'Mean STI: {overall_sti:.3f}', 
        transform=axs[3].transAxes,
        color='darkred', fontsize=10, fontweight='bold',
        ha='right', va='top',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='darkred', boxstyle='round,pad=0.5')
    )
    
    # Set x-ticks and x-labels for each subplot
    for ax in axs:
        ax.set_xticks([0, mid_time, total_duration])
        ax.set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time),
            format_time_hhmmssmmm(total_duration)
        ])
        ax.set_xlim([0, total_duration])
        ax.set_xlabel("Time")
    
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, top=0.95, left=0.15, right=0.92)  # Adjusted margins to prevent cut-off
    return fig

# -----------------------------------------------------------------------------------
# Comparison plots for two files side by side (only 3 time ticks per axis)
# -----------------------------------------------------------------------------------
def create_comparison_plots(
    audio_signal1, fs1, times1, sti1, overall_sti1, name1,
    audio_signal2, fs2, times2, sti2, overall_sti2, name2
):
    """
    Creates a figure with 4 rows and 2 columns, comparing:
      Row 1: Waveform (File1 left, File2 right)
      Row 2: Spectrogram (File1 left, File2 right)
      Row 3: Spectrogram (limited freq) (File1 left, File2 right)
      Row 4: STI over time (File1 left, File2 right)
    
    Only 3 x-axis ticks (start, mid, end) for each column.
    """
    total_duration1 = len(audio_signal1) / fs1
    mid_time1 = total_duration1 / 2.0
    
    total_duration2 = len(audio_signal2) / fs2
    mid_time2 = total_duration2 / 2.0
    
    fig, axs = plt.subplots(4, 2, figsize=(10.5, 12))  # Further reduced width for better margins
    
    # Removed main title
    
    NFFT = 2048
    noverlap = 1024
    
    # Row 1: Waveforms
    t1 = np.arange(len(audio_signal1)) / fs1
    t2 = np.arange(len(audio_signal2)) / fs2
    
    # File1 waveform (left)
    axs[0, 0].plot(t1, audio_signal1)
    axs[0, 0].set_title(f"Waveform\n{name1}", pad=10)  # Added padding to the title
    axs[0, 0].set_ylabel("Amplitude")
    axs[0, 0].grid(True)
    
    # File2 waveform (right)
    axs[0, 1].plot(t2, audio_signal2)
    axs[0, 1].set_title(f"Waveform\n{name2}", pad=10)  # Added padding to the title
    axs[0, 1].grid(True)
    
    # Row 2: Spectrogram (File1 left, File2 right)
    axs[1, 0].specgram(audio_signal1, NFFT=NFFT, Fs=fs1, noverlap=noverlap, cmap='viridis')
    axs[1, 0].set_ylabel("Frequency (Hz)")
    axs[1, 0].set_title(f"Spectrogram\n{name1}", pad=10)  # Added padding to the title
    axs[1, 0].grid(True)
    
    axs[1, 1].specgram(audio_signal2, NFFT=NFFT, Fs=fs2, noverlap=noverlap, cmap='viridis')
    axs[1, 1].set_title(f"Spectrogram\n{name2}", pad=10)  # Added padding to the title
    axs[1, 1].grid(True)
    
    # Row 3: Spectrogram limited 20-4000 Hz
    axs[2, 0].specgram(audio_signal1, NFFT=NFFT, Fs=fs1, noverlap=noverlap, cmap='magma')
    axs[2, 0].set_ylabel("Frequency (Hz)")
    axs[2, 0].set_ylim(20, 4000)
    axs[2, 0].set_title(f"Spectrogram (20-4000 Hz)\n{name1}", pad=10)  # Added padding to the title
    axs[2, 0].grid(True)
    
    axs[2, 1].specgram(audio_signal2, NFFT=NFFT, Fs=fs2, noverlap=noverlap, cmap='magma')
    axs[2, 1].set_ylim(20, 4000)
    axs[2, 1].set_title(f"Spectrogram (20-4000 Hz)\n{name2}", pad=10)  # Added padding to the title
    axs[2, 1].grid(True)
    
    # Row 4: STI over time (File1 left, File2 right)
    axs[3, 0].step(times1, sti1, where='post', color='red')
    axs[3, 0].axhline(y=overall_sti1, color='darkred', linestyle='--', alpha=0.7)
    axs[3, 0].set_ylim(0, 1)
    axs[3, 0].set_ylabel("STI")
    axs[3, 0].set_title(f"STI Over Time\n{name1}", pad=10)  # Added padding to the title
    axs[3, 0].grid(True)
    
    axs[3, 1].step(times2, sti2, where='post', color='red')
    axs[3, 1].axhline(y=overall_sti2, color='darkred', linestyle='--', alpha=0.7)
    axs[3, 1].set_ylim(0, 1)
    axs[3, 1].set_ylabel("STI")
    axs[3, 1].set_title(f"STI Over Time\n{name2}", pad=10)  # Added padding to the title
    axs[3, 1].grid(True)
    
    # Set x-ticks for each column in each row
    # Left column -> file1
    for row in range(4):
        axs[row, 0].set_xticks([0, mid_time1, total_duration1])
        axs[row, 0].set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time1),
            format_time_hhmmssmmm(total_duration1)
        ])
        axs[row, 0].set_xlim([0, total_duration1])
        axs[row, 0].set_xlabel("Time")
    
    # Right column -> file2
    for row in range(4):
        axs[row, 1].set_xticks([0, mid_time2, total_duration2])
        axs[row, 1].set_xticklabels([
            format_time_hhmmssmmm(0),
            format_time_hhmmssmmm(mid_time2),
            format_time_hhmmssmmm(total_duration2)
        ])
        axs[row, 1].set_xlim([0, total_duration2])
        axs[row, 1].set_xlabel("Time")
    
    # Adjust spacing to reduce overlap
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.8, wspace=0.5, top=0.95, left=0.15, right=0.92)  # Adjusted all margins to prevent cut-off
    return fig

# -----------------------------------------------------------------------------------
# Single-file processing (returns: success, overall_sti)
# -----------------------------------------------------------------------------------
def process_audio_file(audio_path, output_dir, window_ms, hop_ms, create_pdf=True):
    """
    Processes a single audio file by computing STI and generating output files.
    Returns (success, overall_sti).
    """
    try:
        # Convert ms to s
        window_dur = window_ms / 1000.0
        hop_dur = hop_ms / 1000.0
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Read audio
        audio_signal, sample_rate = read_audio_file(audio_path)
        audio_filename = os.path.basename(audio_path)
        audio_name = os.path.splitext(audio_filename)[0]
        
        # Compute STI
        print(f"Processing {audio_filename} with window={window_ms}ms, hop={hop_ms}ms...")
        times, sti = compute_sti(audio_signal, sample_rate, window_dur=window_dur, hop_dur=hop_dur)
        overall_sti = np.mean(sti) if len(sti) > 0 else 0.0
        
        # SHA256
        with open(audio_path, 'rb') as f:
            file_hash = hashlib.sha256(f.read()).hexdigest()
        
        print(f"Overall STI (mean): {overall_sti:.3f}")
        print(f"SHA256: {file_hash}")
        
        # Save STI results to CSV, times in HH:MM:SS.mmm
        csv_filename = os.path.join(output_dir, f"sti_results_{audio_name}.csv")
        with open(csv_filename, "w") as f:
            f.write("Time,STI\n")
            for t, val in zip(times, sti):
                t_formatted = format_time_hhmmssmmm(t)
                f.write(f"{t_formatted},{val:.3f}\n")
        print(f"STI results saved to {csv_filename}")
        
        # Create plots
        fig_plots = create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename)
        plot_filename = os.path.join(output_dir, f"chart_{audio_name}.png")
        fig_plots.savefig(plot_filename)
        plt.close(fig_plots)
        print(f"Plot saved to {plot_filename}")
        
        # PDF report if requested
        if create_pdf:
            pdf_filename = os.path.join(output_dir, f"report_{audio_name}.pdf")
            with PdfPages(pdf_filename) as pdf:
                # A4 page
                a4_width_inch, a4_height_inch = 8.27, 11.69
                
                # First page: textual info
                fig_info = plt.figure(figsize=(a4_width_inch, a4_height_inch))
                ax_header = plt.axes([0.1, 0.8, 0.8, 0.15])
                ax_header.axis('off')
                ax_info = plt.axes([0.1, 0.3, 0.8, 0.45])
                ax_info.axis('off')
                ax_footer = plt.axes([0.1, 0.05, 0.8, 0.1])
                ax_footer.axis('off')
                
                ax_header.text(0.5, 0.8, "STI ANALYSIS REPORT", 
                               horizontalalignment='center', fontsize=18, fontweight='bold')
                ax_header.text(0.5, 0.5, f"File: {audio_filename}", 
                               horizontalalignment='center', fontsize=14)
                ax_header.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # Audio info
                frames = len(audio_signal)
                rate = sample_rate
                duration = frames / float(rate)
                
                # Attempt more detailed info if WAV
                try:
                    with contextlib.closing(wave.open(audio_path, 'r')) as wf:
                        frames_w = wf.getnframes()
                        rate_w = wf.getframerate()
                        duration_w = frames_w / float(rate_w)
                        channels = wf.getnchannels()
                        sampwidth = wf.getsampwidth()
                        format_info = f"{channels} channels, {sampwidth*8} bit"
                        # Overwrite stats with wave data
                        frames = frames_w
                        rate = rate_w
                        duration = duration_w
                except:
                    format_info = "Converted/Unknown (Mono)"
                
                # Format duration
                dur_str = format_time_hhmmssmmm(duration)
                
                analysis_params = (
                    f"• Analysis window: {window_ms} ms\n"
                    f"• Hop interval: {hop_ms} ms\n"
                )
                
                info_text = (
                    "TECHNICAL SPECIFICATIONS\n\n"
                    f"• File name: {audio_filename}\n"
                    f"• Format: {format_info}\n"
                    f"• Sampling rate: {rate} Hz\n"
                    f"• Number of samples: {frames:,}\n"
                    f"• Duration: {dur_str}\n\n"
                    f"ANALYSIS PARAMETERS\n\n"
                    f"{analysis_params}\n"
                    f"HASH\n\n"
                    f"• SHA-256: {file_hash}\n\n"
                    f"STI ANALYSIS RESULTS\n\n"
                    f"• Mean STI: {overall_sti:.3f}\n"
                )
                sti_min = np.min(sti) if len(sti) > 0 else 0
                sti_max = np.max(sti) if len(sti) > 0 else 0
                sti_std = np.std(sti) if len(sti) > 0 else 0
                info_text += f"• Min STI: {sti_min:.3f}\n"
                info_text += f"• Max STI: {sti_max:.3f}\n"
                info_text += f"• Standard Deviation: {sti_std:.3f}\n"
                
                ax_info.text(0, 1, info_text, fontsize=11, verticalalignment='top', 
                             horizontalalignment='left', linespacing=1.5)
                
                ax_footer.text(0.5, 0.2, "Page 1/2", horizontalalignment='center', fontsize=8)
                
                pdf.savefig(fig_info)
                plt.close(fig_info)
                
                # Second page: the plots
                # Recreate or reuse the same figure? We already closed it, so let's re-create:
                fig_plots = create_analysis_plots(audio_signal, sample_rate, times, sti, overall_sti, audio_filename)
                fig_plots.set_size_inches(a4_width_inch, a4_height_inch)
                # Adjust the figure margins for PDF output
                plt.subplots_adjust(left=0.15, right=0.90, hspace=0.8, top=0.95)  # Further adjusted margins to prevent time cut-off
                # Removed suptitle
                fig_plots.text(0.5, 0.01, "Page 2/2", ha='center', fontsize=8)
                pdf.savefig(fig_plots)
                plt.close(fig_plots)
                
            print(f"PDF report saved to {pdf_filename}")
        
        return True, overall_sti
    except Exception as e:
        print(f"Error while processing {audio_path}: {str(e)}")
        return False, None

# -----------------------------------------------------------------------------------
# Function to compare two audio files side by side
# -----------------------------------------------------------------------------------
def compare_two_audio_files(file1, file2, output_dir, window_ms, hop_ms, create_pdf=True):
    """
    Computes STI for two audio files and generates a side-by-side comparison plot and PDF.
    Also creates a combined CSV with times in HH:MM:SS.mmm format, using the actual filenames in headers.
    
    Returns (success, sti_mean_file1, sti_mean_file2).
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert ms to s
        window_dur = window_ms / 1000.0
        hop_dur = hop_ms / 1000.0
        
        # Read both files
        audio_signal1, fs1 = read_audio_file(file1)
        audio_signal2, fs2 = read_audio_file(file2)
        filename1 = os.path.basename(file1)  # keep extension as user requested
        filename2 = os.path.basename(file2)
        
        name1 = os.path.splitext(filename1)[0]  # used for some titles
        name2 = os.path.splitext(filename2)[0]
        
        print(f"Comparing:\n  File A: {file1}\n  File B: {file2}")
        print(f"Using window={window_ms}ms, hop={hop_ms}ms...")
        
        # Compute STI for both
        times1, sti1 = compute_sti(audio_signal1, fs1, window_dur=window_dur, hop_dur=hop_dur)
        times2, sti2 = compute_sti(audio_signal2, fs2, window_dur=window_dur, hop_dur=hop_dur)
        
        overall_sti1 = np.mean(sti1) if len(sti1) > 0 else 0.0
        overall_sti2 = np.mean(sti2) if len(sti2) > 0 else 0.0
        
        # Calculate STI statistics
        sti1_min = np.min(sti1) if len(sti1) > 0 else 0
        sti1_max = np.max(sti1) if len(sti1) > 0 else 0
        sti1_std = np.std(sti1) if len(sti1) > 0 else 0
        
        sti2_min = np.min(sti2) if len(sti2) > 0 else 0
        sti2_max = np.max(sti2) if len(sti2) > 0 else 0
        sti2_std = np.std(sti2) if len(sti2) > 0 else 0
        
        # Calculate SHA256 for both
        with open(file1, 'rb') as f:
            hash1 = hashlib.sha256(f.read()).hexdigest()
        with open(file2, 'rb') as f:
            hash2 = hashlib.sha256(f.read()).hexdigest()
        
        print(f"File A STI: {overall_sti1:.3f}")
        print(f"File B STI: {overall_sti2:.3f}")
        
        # Create combined CSV for comparison
        # We'll merge times1 & times2 side by side. If one is shorter, we'll pad with empty.
        csv_filename = os.path.join(output_dir, f"sti_comparison_{name1}_vs_{name2}.csv")
        
        max_len = max(len(times1), len(times2))
        
        # Use underscores for any spaces in filenames to keep CSV simpler
        header_timeA = f"Time_{filename1.replace(' ', '_')}"
        header_stiA = f"STI_{filename1.replace(' ', '_')}"
        header_timeB = f"Time_{filename2.replace(' ', '_')}"
        header_stiB = f"STI_{filename2.replace(' ', '_')}"
        
        with open(csv_filename, "w") as f:
            f.write(f"{header_timeA},{header_stiA},{header_timeB},{header_stiB}\n")
            for i in range(max_len):
                if i < len(times1):
                    tA_str = format_time_hhmmssmmm(times1[i])
                    stiA_str = f"{sti1[i]:.3f}"
                else:
                    tA_str = ""
                    stiA_str = ""
                
                if i < len(times2):
                    tB_str = format_time_hhmmssmmm(times2[i])
                    stiB_str = f"{sti2[i]:.3f}"
                else:
                    tB_str = ""
                    stiB_str = ""
                
                f.write(f"{tA_str},{stiA_str},{tB_str},{stiB_str}\n")
        
        print(f"Comparison CSV saved to {csv_filename}")
        
        # Create side-by-side comparison plots
        fig_compare = create_comparison_plots(
            audio_signal1, fs1, times1, sti1, overall_sti1, name1,
            audio_signal2, fs2, times2, sti2, overall_sti2, name2
        )
        
        plot_filename = os.path.join(output_dir, f"chart_comparison_{name1}_vs_{name2}.png")
        fig_compare.savefig(plot_filename)
        plt.close(fig_compare)
        print(f"Comparison plot saved to {plot_filename}")
        
        if create_pdf:
            pdf_filename = os.path.join(output_dir, f"report_comparison_{name1}_vs_{name2}.pdf")
            with PdfPages(pdf_filename) as pdf:
                a4_width_inch, a4_height_inch = 8.27, 11.69
                
                # First page: textual summary for both files
                fig_info = plt.figure(figsize=(a4_width_inch, a4_height_inch))
                ax_header = plt.axes([0.1, 0.8, 0.8, 0.15])
                ax_header.axis('off')
                ax_info = plt.axes([0.1, 0.2, 0.8, 0.55])
                ax_info.axis('off')
                ax_footer = plt.axes([0.1, 0.05, 0.8, 0.1])
                ax_footer.axis('off')
                
                ax_header.text(0.5, 0.8, "STI COMPARISON REPORT", 
                               horizontalalignment='center', fontsize=18, fontweight='bold')
                ax_header.text(0.5, 0.5, f"Files: {filename1} vs {filename2}", 
                               horizontalalignment='center', fontsize=14)
                ax_header.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
                
                # Info about File A
                framesA = len(audio_signal1)
                durA_sec = framesA / fs1 if fs1 > 0 else 0
                durA = format_time_hhmmssmmm(durA_sec)
                try:
                    with contextlib.closing(wave.open(file1, 'r')) as wfA:
                        chA = wfA.getnchannels()
                        swA = wfA.getsampwidth()
                        infoA = f"{chA} channels, {swA*8} bit"
                except:
                    infoA = "Converted/Unknown (Mono)"
                
                # Info about File B
                framesB = len(audio_signal2)
                durB_sec = framesB / fs2 if fs2 > 0 else 0
                durB = format_time_hhmmssmmm(durB_sec)
                try:
                    with contextlib.closing(wave.open(file2, 'r')) as wfB:
                        chB = wfB.getnchannels()
                        swB = wfB.getsampwidth()
                        infoB = f"{chB} channels, {swB*8} bit"
                except:
                    infoB = "Converted/Unknown (Mono)"
                
                info_text = (
                    f"FILE A: {filename1}\n"
                    f"  • Format: {infoA}\n"
                    f"  • Sampling rate: {fs1} Hz\n"
                    f"  • Number of samples: {framesA:,}\n"
                    f"  • Duration: {durA}\n"
                    f"  • SHA-256: {hash1}\n"
                    f"  • Mean STI: {overall_sti1:.3f}\n"
                    f"  • Min STI: {sti1_min:.3f}\n"
                    f"  • Max STI: {sti1_max:.3f}\n"
                    f"  • Standard Deviation: {sti1_std:.3f}\n\n"
                    
                    f"FILE B: {filename2}\n"
                    f"  • Format: {infoB}\n"
                    f"  • Sampling rate: {fs2} Hz\n"
                    f"  • Number of samples: {framesB:,}\n"
                    f"  • Duration: {durB}\n"
                    f"  • SHA-256: {hash2}\n"
                    f"  • Mean STI: {overall_sti2:.3f}\n"
                    f"  • Min STI: {sti2_min:.3f}\n"
                    f"  • Max STI: {sti2_max:.3f}\n"
                    f"  • Standard Deviation: {sti2_std:.3f}\n\n"
                    
                    f"ANALYSIS PARAMETERS\n"
                    f"  • Analysis window: {window_ms} ms\n"
                    f"  • Hop interval: {hop_ms} ms\n"
                )
                
                ax_info.text(0, 1, info_text, fontsize=11, verticalalignment='top', 
                             horizontalalignment='left', linespacing=1.5)
                ax_footer.text(0.5, 0.2, "Page 1/2", horizontalalignment='center', fontsize=8)
                
                pdf.savefig(fig_info)
                plt.close(fig_info)
                
                # Second page: the comparison figure
                # We'll recreate the comparison figure since we've closed it
                fig_compare = create_comparison_plots(
                    audio_signal1, fs1, times1, sti1, overall_sti1, name1,
                    audio_signal2, fs2, times2, sti2, overall_sti2, name2
                )
                fig_compare.set_size_inches(a4_width_inch, a4_height_inch)
                # Adjust the figure margins for PDF output
                plt.subplots_adjust(left=0.15, right=0.90, hspace=0.8, wspace=0.5, top=0.95)  # Further adjusted margins to prevent time cut-off
                # Removed suptitle
                fig_compare.text(0.5, 0.01, "Page 2/2", ha='center', fontsize=8)
                pdf.savefig(fig_compare)
                plt.close(fig_compare)
            
            print(f"PDF comparison report saved to {pdf_filename}")
        
        return True, overall_sti1, overall_sti2
    except Exception as e:
        print(f"Error while comparing files:\n  {file1}\n  {file2}\n  Error: {e}")
        return False, None, None

# -----------------------------------------------------------------------------------
# Main CLI
# -----------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="STI analysis for audio files (single, folder, or compare two).")
    # Positional/optional
    parser.add_argument("input", nargs="?", help="Audio file or folder. Not used when --compare is set.", default=None)
    
    parser.add_argument("--output", help="Output folder for the results", default="./output")
    parser.add_argument("--window", type=int, help="Analysis window length in milliseconds", default=500)
    parser.add_argument("--hop", type=int, help="Hop length in milliseconds", default=250)
    parser.add_argument("--nopdf", action="store_true", help="Do not generate a PDF report")
    parser.add_argument("--file-ext", help="Filter only these file extensions (comma-separated). If omitted, it processes all common audio formats", default=None)
    
    # Parameter for 2-file comparison
    parser.add_argument("--compare", nargs=2, help="Compare two audio files side by side. Usage: --compare fileA fileB")
    
    args = parser.parse_args()
    
    # List of common audio file extensions (used if --file-ext is not specified)
    common_audio_extensions = ["wav", "mp3", "ogg", "flac", "aac", "wma", "m4a", "aiff", "opus"]
    if args.file_ext:
        file_extensions = [ext.strip().lower() for ext in args.file_ext.split(',')]
        print(f"Processing limited to extensions: {', '.join(file_extensions)}")
    else:
        file_extensions = common_audio_extensions
        print(f"Processing all common audio formats: {', '.join(file_extensions)}")
    
    if args.compare:
        # Comparison mode
        fileA, fileB = args.compare
        if not (os.path.isfile(fileA) and os.path.isfile(fileB)):
            print("Error: --compare requires two valid files.")
            exit(1)
        
        success, stiA, stiB = compare_two_audio_files(
            fileA, fileB,
            args.output,
            args.window,
            args.hop,
            create_pdf=(not args.nopdf)
        )
        if success:
            print(f"Comparison completed. STI {os.path.basename(fileA)}={stiA:.3f}, {os.path.basename(fileB)}={stiB:.3f}")
        else:
            print("Comparison failed.")
        exit(0)
    
    # If not in compare mode, check the 'input'
    if args.input is None:
        print("Error: you must specify either an input file/folder or use --compare")
        exit(1)
    
    # Check if input is a file or a directory
    if os.path.isfile(args.input):
        # Single file
        print(f"Processing single file: {args.input}")
        success, sti_val = process_audio_file(
            args.input,
            args.output,
            args.window,
            args.hop,
            create_pdf=(not args.nopdf)
        )
        if success and sti_val is not None:
            print(f"Processing successfully completed. STI: {sti_val:.3f}")
        else:
            print("Error while processing the file")
    
    elif os.path.isdir(args.input):
        # Folder mode
        print(f"Searching for audio files in folder: {args.input}")
        patterns = [os.path.join(args.input, f"*.{ext}") for ext in file_extensions]
        
        audio_files = []
        for pattern in patterns:
            audio_files.extend(glob.glob(pattern))
        
        if not audio_files:
            print(f"No audio files found in {args.input} with extensions {file_extensions}")
            exit(1)
        
        print(f"Found {len(audio_files)} audio files to process")
        
        # Summary CSV
        summary_csv = os.path.join(args.output, "sti_summary.csv")
        os.makedirs(args.output, exist_ok=True)
        
        with open(summary_csv, "w") as f:
            f.write("Filename,STI_Mean,Success\n")
            for audio_file in audio_files:
                filename = os.path.basename(audio_file)
                print(f"\nProcessing {filename}...")
                
                success, sti_val = process_audio_file(
                    audio_file,
                    args.output,
                    args.window,
                    args.hop,
                    create_pdf=(not args.nopdf)
                )
                if success and sti_val is not None:
                    f.write(f"{filename},{sti_val:.3f},1\n")
                    print(f"  -> Success: STI = {sti_val:.3f}")
                else:
                    f.write(f"{filename},N/A,0\n")
                    print(f"  -> Error: unable to compute STI")
        
        print(f"\nSummary saved to {summary_csv}")
    
    else:
        print(f"Error: '{args.input}' is neither a valid file nor a valid directory")
        exit(1)
    
    print("\nProcessing completed.")
