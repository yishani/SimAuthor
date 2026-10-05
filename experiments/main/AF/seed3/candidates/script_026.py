import numpy as np
import os
from scipy.signal import butter, filtfilt

def generate_af_rr_intervals(duration, mean_hr, irregularity, rng):
    """
    Generate irregularly irregular RR intervals typical of Atrial Fibrillation.
    Uses a shifted Gamma distribution to model the intervals.
    """
    mean_rr = 60.0 / mean_hr
    t_ref = 0.30  # Refractory period in seconds
    
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
    Pulse amplitude depends on the preceding RR interval (diastolic filling time),
    which is a key physiological manifestation of AF in pulsatile signals.
    """
    t_total = int(duration * fs)
    signal = np.zeros(t_total)
    
    tau = 0.5  # Time constant for ventricular filling (seconds) - tuned for AFib amplitude variability
    
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
            
        amp = 1.0 - np.exp(-prev_rr / tau)
        
        # Pulse morphology parameters (3-component model for physiological realism)
        # Adjusted to create a wider diastolic tail, which raises the signal mean and 
        # deepens the troughs after high-pass filtering, increasing peak-to-trough amplitude.
        
        # 1. Systolic peak
        w1 = 0.050 + 0.015 * (1.0 - amp)
        t1 = b_time + 0.10
        
        # 2. Reflected wave (shapes the descent and notch)
        w2 = 0.12
        t2 = t1 + 0.14
        a2 = amp * 0.45
        
        # 3. Diastolic wave (wide and sustained to fill the gap between beats)
        w3 = 0.35
        t3 = t1 + 0.32
        a3 = amp * (dicrotic_ratio + 0.1)
        
        # Add pulse in a localized window to optimize computation
        # Extended window_end to 2.0s to fully capture the wider diastolic tail
        window_start = max(0, int((b_time - 0.15) * fs))
        window_end = int((b_time + 2.0) * fs)
        window_end = min(window_end, t_total)
        
        if window_start >= t_total:
            break
            
        t_window = t[window_start:window_end]
        
        pulse = amp * np.exp(-0.5 * ((t_window - t1) / w1)**2) + \
                a2 * np.exp(-0.5 * ((t_window - t2) / w2)**2) + \
                a3 * np.exp(-0.5 * ((t_window - t3) / w3)**2)
                
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
    baseline += (bw_amp / 2) * np.sin(2 * np.pi * (bw_freq / 2) * t + rng.uniform(0, 2*np.pi))
    
    # High frequency noise
    noise = rng.normal(0, noise_amp, len(signal))
    
    return signal + baseline + noise

def main():
    output_dir = "[PROJECT_ROOT]/artifacts/af_ppg/canonical_signal/generated/"
    os.makedirs(output_dir, exist_ok=True)
    
    n_samples = 100
    fs_synth = 500
    fs_out = 125
    duration = 30.0
    
    # Explicit random seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Filter setup
    nyq = 0.5 * fs_synth
    # Highpass filter (0.3 Hz) to remove artificial DC shifts from varying pulse areas
    # Applied BEFORE adding physiological baseline wander
    b_hp, a_hp = butter(3, 0.3 / nyq, btype='high')
    # Lowpass filter (60.0 Hz) for anti-aliasing and high-frequency noise removal
    b_lp, a_lp = butter(3, 60.0 / nyq, btype='low')
    
    for i in range(n_samples):
        # Sample-specific physiological parameters
        mean_hr = rng.uniform(60, 105) # Adjusted slightly to match reference mean HR
        irregularity = rng.uniform(0.05, 0.18)
        dicrotic_ratio = rng.uniform(0.3, 0.6)
        
        # Artifact parameters
        # Reduced baseline wander amplitude to prevent squashing normalized peaks
        bw_amp = rng.uniform(0.005, 0.025) 
        bw_freq = rng.uniform(0.15, 0.3)  # ~9 to 18 breaths per minute
        noise_amp = rng.uniform(0.001, 0.01)
        
        # 1. Generate RR intervals
        rr_intervals = generate_af_rr_intervals(duration + 5.0, mean_hr, irregularity, rng)
        
        # 2. Synthesize PPG waveform
        signal = generate_ppg_signal(rr_intervals, fs_synth, duration, dicrotic_ratio, rng)
        
        # 3. Highpass filter to remove artificial stacking wander
        # Subtract mean first to minimize edge transients in filtfilt
        signal = signal - np.mean(signal)
        signal = filtfilt(b_hp, a_hp, signal)
        
        # 4. Add noise and physiological baseline wander
        signal = add_noise_and_baseline(signal, fs_synth, bw_amp, bw_freq, noise_amp, rng)
        
        # 5. Lowpass filter
        signal_filt = filtfilt(b_lp, a_lp, signal)
        
        # 6. Downsample
        downsample_factor = fs_synth // fs_out
        signal_down = signal_filt[::downsample_factor]
        
        # 7. Peak normalize
        max_val = np.max(np.abs(signal_down))
        if max_val > 0:
            signal_down = signal_down / max_val
            
        # 8. Save
        filename = os.path.join(output_dir, f"af_ppg_{i:03d}.npy")
        np.save(filename, signal_down)

if __name__ == "__main__":
    main()