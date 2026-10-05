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
    6. Continuous spectrum of pre-excitation severity, affecting all morphological features.
    
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
    
    # Degree of pre-excitation (0.2 to 1.0)
    # Modulates delta wave size, QRS width, and PR interval to create a continuum of severity
    # This directly addresses the spread ratio by increasing inter-recording diversity
    pre_ex = rng.uniform(0.2, 1.0)
    
    # Heart rate and HRV
    hr = rng.uniform(50, 110)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with Respiratory Sinus Arrhythmia (RSA)
    num_beats = int(duration / rr_mean) + 5
    resp_freq = rng.uniform(0.2, 0.35)
    t_beats = np.arange(num_beats) * rr_mean
    rsa_modulation = rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t_beats)
    rr_intervals = rr_mean + rsa_modulation + rng.normal(0, 0.015, num_beats)
    
    r_peaks = np.cumsum(rr_intervals)
    # Shift first peak to start early in the recording
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    for lead_idx in range(2):
        # Base parameters for conduction (Amplitude, Center, Width)
        p_a = rng.uniform(0.05, 0.15)
        p_sigma = rng.uniform(0.015, 0.022)
        
        r_mu = 0.0
        s_sigma = rng.uniform(0.012, 0.02)
        
        t_mu = rng.uniform(0.18, 0.28)
        t_sigma = rng.uniform(0.035, 0.055)
        
        # U wave for added morphological diversity
        u_a = rng.uniform(0.0, 0.02)
        u_mu = t_mu + rng.uniform(0.12, 0.16)
        u_sigma = rng.uniform(0.03, 0.05)
        
        st_mu = rng.uniform(0.08, 0.12)
        st_sigma = rng.uniform(0.06, 0.08)
        
        # Base morphological offsets (constant for the lead to prevent unrealistic beat-to-beat jitter)
        base_delta_mu_offset = rng.uniform(-0.035, -0.02)
        base_delta_sigma_offset = rng.uniform(0.02, 0.03)
        base_pr_interval_offset = rng.uniform(0.10, 0.14)
        base_r_sigma_offset = rng.uniform(0.012, 0.016)
        base_s_mu_offset = rng.uniform(0.025, 0.04)
        
        # Adjust base amplitudes based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                base_delta_a = rng.uniform(0.1, 0.35)
                r_a = rng.uniform(0.5, 1.5)
                s_a = rng.uniform(-0.2, 0.0)
            else: # Lead II
                base_delta_a = rng.uniform(0.15, 0.45)
                r_a = rng.uniform(0.6, 1.8)
                s_a = rng.uniform(-0.2, 0.0)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                base_delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.4, 1.2)
                s_a = rng.uniform(-0.4, -0.1)
            else: # Lead II
                base_delta_a = rng.uniform(-0.4, -0.15) # Negative delta wave (pseudo-Q)
                r_a = rng.uniform(0.2, 0.8)
                s_a = rng.uniform(-0.6, -0.2)
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                base_delta_a = rng.uniform(-0.3, -0.05)
                r_a = rng.uniform(0.3, 1.0)
                s_a = rng.uniform(-0.5, -0.1)
            else: # Lead II
                base_delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.6, 1.6)
                s_a = rng.uniform(-0.3, 0.0)
                
        # Respiratory modulation (amplitude modulation)
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            beat_var = rng.uniform(0.95, 1.05)
            
            # Slight beat-to-beat variation in pre-excitation severity
            beat_pre_ex = np.clip(pre_ex * rng.uniform(0.95, 1.05), 0.1, 1.0)
            
            # Dynamically couple morphological features to the degree of pre-excitation
            b_delta_mu = base_delta_mu_offset - 0.015 * beat_pre_ex
            b_delta_sigma = base_delta_sigma_offset + 0.01 * beat_pre_ex
            b_pr_interval = base_pr_interval_offset - 0.04 * beat_pre_ex
            b_p_mu = b_delta_mu - b_pr_interval
            
            b_r_sigma = base_r_sigma_offset + 0.005 * beat_pre_ex
            b_s_mu = base_s_mu_offset + 0.01 * beat_pre_ex
            b_delta_a = base_delta_a * beat_pre_ex
            
            # P, Delta, R, S waves
            lead_signal += gaussian(t, p_a * beat_var, r_time + b_p_mu, p_sigma)
            lead_signal += gaussian(t, b_delta_a * beat_var, r_time + b_delta_mu, b_delta_sigma)
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, b_r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + b_s_mu, s_sigma)
            
            # ST Segment and T wave Discordance: strictly coupled to the net QRS vector
            b_net_qrs = b_delta_a + r_a + s_a
            
            b_st_a = -0.08 * b_net_qrs * beat_var
            lead_signal += gaussian(t, b_st_a, r_time + st_mu, st_sigma)
            
            b_t_a = -0.15 * b_net_qrs + rng.uniform(-0.05, 0.05)
            lead_signal += gaussian(t, b_t_a * beat_var, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, b_t_a * 0.4 * beat_var, r_time + t_mu + 0.02, t_sigma * 1.2)
            
            # U wave
            if u_a != 0:
                u_sign = np.sign(b_t_a) if b_t_a != 0 else 1
                lead_signal += gaussian(t, u_a * u_sign * beat_var, r_time + u_mu, u_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Complex low frequency baseline wander
        baseline = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
        baseline += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        for _ in range(2):
            baseline += rng.uniform(0.01, 0.04) * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(-0.1, 0.1) * (t / duration) # Slow linear drift
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG - pinkish noise)
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(5)/5, mode='same')
        emg = rng.uniform(0.015, 0.04) * pink_noise
        
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