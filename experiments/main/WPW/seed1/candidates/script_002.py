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
    1. Shortened PR interval (80-110 ms) due to accessory pathway bypass.
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
    
    # Temporal parameters (consistent across leads for physiological accuracy)
    # WPW Hallmark: Delta wave positioned to create a slurred upstroke fusing with the R-wave
    delta_mu = rng.uniform(-0.07, -0.05)
    delta_sigma = rng.uniform(0.022, 0.028)
    
    # WPW Hallmark: Short PR interval (80 - 110 ms) measured from P onset to Delta onset
    pr_interval = rng.uniform(0.08, 0.11)
    p_mu = delta_mu - pr_interval
    p_sigma = rng.uniform(0.015, 0.02)
    
    # Sharper R-wave to contrast with the slurred delta wave
    r_mu = 0.0
    r_sigma = rng.uniform(0.008, 0.012)
    
    s_mu = rng.uniform(0.02, 0.035)
    s_sigma = rng.uniform(0.012, 0.018)
    
    t_mu = rng.uniform(0.22, 0.28)
    t_sigma = rng.uniform(0.035, 0.045)
    
    for lead_idx in range(2):
        # Amplitude parameters (can vary by lead)
        p_a = rng.uniform(0.08, 0.15)
        r_a = rng.uniform(1.0, 2.0)
        s_a = rng.uniform(-0.4, -0.05)
        t_a = rng.uniform(0.15, 0.3)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.15, 0.35)
                t_a = rng.uniform(-0.3, -0.1) # Discordant T wave inversion
            else: # Lead II
                delta_a = rng.uniform(0.2, 0.4)
                r_a = rng.uniform(1.2, 2.2)
                t_a = rng.uniform(-0.2, 0.0)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.25)
            else: # Lead II
                delta_a = rng.uniform(-0.4, -0.2) # Negative delta wave
                r_a = rng.uniform(0.2, 0.6)       # Attenuated R wave
                s_a = rng.uniform(-0.8, -0.4)     # Deep S wave
                t_a = rng.uniform(0.2, 0.5)       # Discordant positive T wave
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.2, -0.05)
                r_a = rng.uniform(0.4, 0.8)
                t_a = rng.uniform(0.1, 0.3)
            else: # Lead II
                delta_a = rng.uniform(0.15, 0.35)
                r_a = rng.uniform(1.0, 1.8)
                t_a = rng.uniform(-0.15, 0.05)
                
        # Respiratory modulation (amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_var = rng.uniform(0.95, 1.05)
            
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            # Q-wave is omitted as it is typically obscured by the delta wave in WPW
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            lead_signal += gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Low frequency baseline wander (enhanced for realism and spread)
        bw_freq1 = rng.uniform(0.1, 0.3)
        bw_freq2 = rng.uniform(0.02, 0.08)
        baseline = rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * bw_freq1 * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * bw_freq2 * t + rng.uniform(0, 2*np.pi))
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG)
        emg = rng.normal(0, rng.uniform(0.01, 0.025), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    return signal

def main():
    # Configuration
    output_dir = "artifacts/wpw_ecg/visual_representation_100iter/generated/"
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