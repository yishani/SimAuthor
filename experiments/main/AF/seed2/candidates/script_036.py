import numpy as np
import os
from scipy.signal import butter, filtfilt, sosfiltfilt

def generate_af_rr_intervals(duration, mean_hr, irregularity, rng):
    """
    Generate irregularly irregular RR intervals typical of Atrial Fibrillation.
    Uses a shifted Gamma distribution to model the intervals.
    """
    mean_rr = 60.0 / mean_hr
    t_ref = 0.25  # Refractory period in seconds
    
    # Ensure the refractory period is strictly less than the mean RR interval
    if mean_rr <= t_ref + 0.05:
        t_ref = mean_rr - 0.05
        
    gamma_mean = mean_rr - t_ref
    # Variance scales with irregularity parameter
    gamma_var = irregularity * gamma_mean
    
    theta = gamma_var / gamma_mean
    k = gamma_mean / theta
    
    # Estimate number of beats needed to cover the duration
    n_beats = int(duration / mean_rr * 1.5) + 10
    
    rr = rng.gamma(k, theta, n_beats) + t_ref
    return rr

def generate_ppg_signal(rr_intervals, fs, duration, dicrotic_ratio, rng):
    """
    Synthesize the PPG waveform by placing pulses at the generated RR intervals.
    Pulse amplitude depends on the preceding RR interval (diastolic filling time).
    Uses asymmetric Gaussian functions to model systolic and diastolic waves.
    """
    t_total = int(duration * fs)
    signal = np.zeros(t_total)
    
    beat_times = np.cumsum(rr_intervals)
    # Start first beat slightly after t=0
    beat_times = np.insert(beat_times, 0, 0.1)
    
    t = np.arange(t_total) / fs
    
    for i in range(len(beat_times)):
        b_time = beat_times[i]
        if b_time > duration:
            break
            
        # Amplitude variation based on previous RR interval (Frank-Starling mechanism)
        if i == 0:
            prev_rr = rr_intervals[0]
        else:
            prev_rr = rr_intervals[i-1]
            
        # Moderated amplitude variation to prevent extreme peaks from squashing 
        # the signal during peak normalization, maintaining a realistic ~20% CV.
        amp = 0.5 + 0.5 * (1.0 - np.exp(-prev_rr / 0.5))
        a2 = amp * dicrotic_ratio
        
        # Add pulse in a localized window to optimize computation
        window_start = int(b_time * fs)
        window_end = int((b_time + 2.0) * fs)
        window_end = min(window_end, t_total)
        
        if window_start >= t_total:
            break
            
        t_window = t[window_start:window_end]
        
        # Time relative to beat start
        dt = t_window - b_time
        
        # Asymmetric Gaussian for systolic wave
        # Narrowed right side to reduce overall pulse width closer to reference
        t_sys = 0.16
        sigma_sys_left = 0.06
        sigma_sys_right = 0.09
        
        sys_wave = np.where(dt < t_sys,
                            amp * np.exp(- (dt - t_sys)**2 / (2 * sigma_sys_left**2)),
                            amp * np.exp(- (dt - t_sys)**2 / (2 * sigma_sys_right**2)))
        
        # Asymmetric Gaussian for diastolic wave
        # Narrowed and shifted earlier to prevent excessive pulse overlap and width
        t_dias = 0.32
        sigma_dias_left = 0.08
        sigma_dias_right = 0.15
        
        dias_wave = np.where(dt < t_dias,
                             a2 * np.exp(- (dt - t_dias)**2 / (2 * sigma_dias_left**2)),
                             a2 * np.exp(- (dt - t_dias)**2 / (2 * sigma_dias_right**2)))
                
        pulse = sys_wave + dias_wave
        signal[window_start:window_end] += pulse
        
    return signal

def add_noise_and_baseline(signal, fs, bw_amp, bw_freq, noise_amp, rng):
    """
    Add baseline wander (respiration/movement) and high-frequency sensor noise.
    """
    t = np.arange(len(signal)) / fs
    
    # Baseline wander
    baseline = bw_amp * np.sin(2 * np.pi * bw_freq * t + rng.uniform(0, 2*np.pi))
    # Add a secondary lower frequency wander component
    baseline += (bw_amp / 2) * np.sin(2 * np.pi * (bw_freq / 3) * t + rng.uniform(0, 2*np.pi))
    
    # High frequency noise
    noise = rng.normal(0, noise_amp, len(signal))
    
    return signal + baseline + noise

def main():
    output_dir = "[PROJECT_ROOT]/artifacts/af_ppg/runs/signal/seed2/generated/"
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = 100
    fs_synth = 500
    fs_out = 125
    duration = 30.0
    
    # Explicit random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Anti-aliasing and DC-removal filter setup for downsampling.
    # The high-pass component (0.1 Hz) mimics the AC coupling of real PPG devices,
    # centering the pulses and creating the deep negative valleys (undershoot) 
    # required to match the reference peak-to-trough amplitude.
    nyq = 0.5 * fs_out
    sos = butter(2, [0.1 / (0.5 * fs_synth), nyq / (0.5 * fs_synth)], btype='bandpass', output='sos')
    
    for i in range(n_samples):
        # Sample-specific physiological parameters
        mean_hr = rng.uniform(60, 105)
        irregularity = rng.uniform(0.04, 0.16)
        dicrotic_ratio = rng.uniform(0.15, 0.45)
        
        # Artifact parameters (slightly reduced wander to ensure AC signal dominance)
        bw_amp = rng.uniform(0.01, 0.08)
        bw_freq = rng.uniform(0.15, 0.4)  # ~9 to 24 breaths per minute
        noise_amp = rng.uniform(0.002, 0.012)
        
        # 1. Generate RR intervals
        # Generate extra intervals to cover the padding
        pad_duration = 4.0
        rr_intervals = generate_af_rr_intervals(duration + pad_duration + 5.0, mean_hr, irregularity, rng)
        
        # 2. Synthesize PPG waveform (pad to avoid filter edge effects)
        signal = generate_ppg_signal(rr_intervals, fs_synth, duration + pad_duration, dicrotic_ratio, rng)
        
        # 3. Add noise and baseline wander
        signal = add_noise_and_baseline(signal, fs_synth, bw_amp, bw_freq, noise_amp, rng)
        
        # 4. Anti-aliasing and DC-removal filter
        signal_filt = sosfiltfilt(sos, signal)
        
        # Crop to target duration to remove filter transient edges
        target_samples = int(duration * fs_synth)
        signal_filt = signal_filt[:target_samples]
        
        # 5. Downsample
        downsample_factor = fs_synth // fs_out
        signal_down = signal_filt[::downsample_factor]
        
        # 6. Mean center and Peak normalize
        signal_down = signal_down - np.mean(signal_down)
        max_val = np.max(np.abs(signal_down))
        if max_val > 0:
            signal_down = signal_down / max_val
            
        # 7. Save
        filename = os.path.join(output_dir, f"af_ppg_{i:03d}.npy")
        np.save(filename, signal_down)

if __name__ == "__main__":
    main()