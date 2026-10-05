import os
import numpy as np

def gaussian(x, a, mu, sigma):
    """
    Evaluates a Gaussian function.
    
    Parameters:
    x : array-like - Time vector
    a : float - Amplitude
    mu : float - Mean (center)
    sigma : float - Standard deviation (width)
    """
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2)

def generate_wpw_ecg(seed, fs=500, duration=10.0):
    """
    Generates a synthetic 2-lead ECG signal exhibiting Wolff-Parkinson-White (WPW) Syndrome.
    
    Physiological characteristics modeled:
    1. Shortened PR interval (< 120 ms) due to accessory pathway bypass.
    2. Delta wave (slurred upstroke of QRS) due to early ventricular pre-excitation.
    3. Widened QRS complex (fusion of pre-excitation and normal conduction).
    4. Secondary ST-T wave abnormalities (discordant to the delta/QRS vector).
    5. Variations in delta wave polarity based on accessory pathway location.
    
    Parameters:
    seed : int - Random seed for reproducibility
    fs : int - Sampling frequency in Hz
    duration : float - Duration of the signal in seconds
    
    Returns:
    signal : np.ndarray - Array of shape (n_samples, 2) containing Lead I and Lead II
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to create diverse manifestations
    # 0: Left lateral (Positive delta in I and II)
    # 1: Posteroseptal (Positive in I, Negative delta in II - pseudo-inferior infarct)
    # 2: Right lateral (Negative/flat delta in I, Positive in II)
    pathway = rng.choice([0, 1, 2])
    
    # Heart rate and HRV
    hr = rng.uniform(60, 95)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with slight natural variability
    rr_intervals = rng.normal(rr_mean, 0.02, int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    # Shift first peak to start early in the recording
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # WPW Hallmark: Short PR interval
    # Normal is 0.12 - 0.20s. WPW is typically 0.08 - 0.11s.
    pr_interval = rng.uniform(0.08, 0.11)
    p_mu = -pr_interval
    
    for lead_idx in range(2):
        # Base parameters for normal-ish conduction (Amplitude, Center, Width)
        # Centers are relative to the R peak (0.0)
        p_a = rng.uniform(0.08, 0.15)
        p_sigma = 0.015
        
        q_a = rng.uniform(-0.1, 0.0)
        q_mu = -0.015
        q_sigma = 0.01
        
        r_a = rng.uniform(0.8, 1.5)
        r_mu = 0.0
        r_sigma = 0.015
        
        s_a = rng.uniform(-0.3, -0.05)
        s_mu = 0.025
        s_sigma = 0.015
        
        t_a = rng.uniform(0.15, 0.3)
        t_mu = rng.uniform(0.22, 0.26)
        t_sigma = 0.035
        
        # WPW Hallmark: Delta wave
        # Positioned just before the QRS complex to create the slurred upstroke
        delta_mu = -0.035
        delta_sigma = rng.uniform(0.018, 0.025)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.15, 0.35)
                t_a = rng.uniform(-0.2, -0.05) # Discordant T wave inversion
            else: # Lead II
                delta_a = rng.uniform(0.2, 0.4)
                t_a = rng.uniform(-0.15, 0.05)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.25)
            else: # Lead II
                delta_a = rng.uniform(-0.4, -0.2) # Negative delta wave
                r_a = rng.uniform(0.2, 0.5)       # Attenuated R wave
                s_a = rng.uniform(-0.8, -0.4)     # Deep S wave
                t_a = rng.uniform(0.2, 0.4)       # Discordant positive T wave
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.2, -0.05)
                r_a = rng.uniform(0.4, 0.8)
                t_a = rng.uniform(0.1, 0.25)
            else: # Lead II
                delta_a = rng.uniform(0.15, 0.35)
                t_a = rng.uniform(-0.15, 0.0)
                
        # Respiratory modulation (baseline wander and amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_var = rng.uniform(0.95, 1.05)
            
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, q_a * beat_var, r_time + q_mu, q_sigma)
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            lead_signal += gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Low frequency baseline wander
        baseline = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
        baseline += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * powerline_freq * t)
        # 3. High frequency muscle artifact (EMG)
        emg = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    return signal

def main():
    # Configuration
    output_dir = "[PROJECT_ROOT]/artifacts_3.1/wpw_ecg/runs/rep/seed2/generated/"
    n_samples = 100
    fs = 500
    duration = 10.0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save samples
    for i in range(n_samples):
        # Use a deterministic seed for each sample for reproducibility
        seed = 42 + i
        
        # Generate WPW ECG signal
        signal = generate_wpw_ecg(seed=seed, fs=fs, duration=duration)
        
        # Save to .npy file
        filename = f"wpw_ecg_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, signal)

if __name__ == "__main__":
    main()