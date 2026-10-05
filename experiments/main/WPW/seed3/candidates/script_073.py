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
    hr = rng.uniform(50, 110)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with Respiratory Sinus Arrhythmia (RSA) to increase diversity
    num_beats = int(duration / rr_mean) + 5
    resp_freq = rng.uniform(0.15, 0.35)
    t_beats = np.arange(num_beats) * rr_mean
    rsa_modulation = rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t_beats)
    rr_intervals = rr_mean + rsa_modulation + rng.normal(0, 0.015, num_beats)
    
    r_peaks = np.cumsum(rr_intervals)
    # Shift first peak to start early in the recording
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Global amplitude scale for this patient to increase inter-sample variance
    amp_scale = rng.uniform(0.5, 2.0)
    
    for lead_idx in range(2):
        # Base parameters for conduction (Amplitude, Center, Width)
        # Centers are relative to the R peak (0.0)
        p_a = rng.uniform(0.05, 0.15) * amp_scale
        p_sigma = rng.uniform(0.02, 0.025)
        
        r_mu = 0.0
        r_sigma = rng.uniform(0.01, 0.018) 
        
        s_mu = rng.uniform(0.015, 0.035)
        s_sigma = rng.uniform(0.01, 0.025)
        
        t_mu = rng.uniform(0.18, 0.28)
        t_sigma = rng.uniform(0.035, 0.06)
        
        # U wave for added morphological diversity
        u_a = rng.uniform(0.0, 0.02) * amp_scale
        u_mu = t_mu + rng.uniform(0.12, 0.18)
        u_sigma = rng.uniform(0.03, 0.06)
        
        # WPW Hallmark: Delta wave
        # Positioned earlier and wider to ensure a true slurred upstroke (fusion) 
        # and to guarantee total QRS duration > 120ms.
        delta_mu = rng.uniform(-0.05, -0.03)
        delta_sigma = rng.uniform(0.025, 0.035)
        
        # WPW Hallmark: Short PR interval (80 - 110 ms)
        # Clinically measured from P onset to QRS (Delta) onset.
        pr_interval = rng.uniform(0.08, 0.11)
        p_mu = delta_mu - 2*delta_sigma + 2*p_sigma - pr_interval
        
        # ST Segment parameters
        st_mu = rng.uniform(0.08, 0.14)
        st_sigma = rng.uniform(0.04, 0.08)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.5, 1.2)
                s_a = rng.uniform(-0.15, 0.0)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.8, 1.8)
                s_a = rng.uniform(-0.15, 0.0)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.05, 0.25)
                r_a = rng.uniform(0.4, 1.4)
                s_a = rng.uniform(-0.25, -0.05)
            else: # Lead II
                delta_a = rng.uniform(-0.4, -0.15) # Negative delta wave (pseudo-Q)
                r_a = rng.uniform(0.2, 0.8)       # Attenuated R wave
                s_a = rng.uniform(-0.4, -0.1)     # Deep S wave
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.25, -0.05)
                r_a = rng.uniform(0.2, 1.2)
                s_a = rng.uniform(-0.35, -0.1)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.5, 1.8)
                s_a = rng.uniform(-0.25, 0.0)
                
        # Apply global amplitude scaling
        delta_a *= amp_scale
        r_a *= amp_scale
        s_a *= amp_scale
                
        # Respiratory modulation (amplitude modulation)
        resp_mod = 1.0 + rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_var = rng.uniform(0.92, 1.08)
            
            # P, Delta, R, S waves
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            
            # WPW Rule of Discordance: ST segment and T wave are opposite to the net QRS vector
            net_qrs = delta_a + r_a + s_a
            
            st_a = rng.uniform(-0.08, -0.02) * net_qrs * beat_var
            lead_signal += gaussian(t, st_a, r_time + st_mu, st_sigma)
            
            # Dynamic T wave amplitude based on Rule of Discordance
            t_a = -0.25 * net_qrs + rng.uniform(-0.05, 0.05)
            t_a = np.clip(t_a, -0.5, 0.5) * beat_var
            
            # Asymmetric T wave
            lead_signal += gaussian(t, t_a, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, t_a * 0.4, r_time + t_mu + 0.02, t_sigma * 1.2)
            
            # U wave
            if u_a != 0:
                u_sign = np.sign(t_a) if t_a != 0 else 1
                lead_signal += gaussian(t, u_a * u_sign * beat_var, r_time + u_mu, u_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Complex low frequency baseline wander (smoothed to match real signals)
        baseline = rng.uniform(0.1, 0.25) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
        baseline += rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t)
        for _ in range(2):
            baseline += rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(-0.1, 0.1) * (t / duration) # Slow linear drift
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG - pinkish noise)
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(5)/5, mode='same')
        emg = rng.uniform(0.005, 0.025) * pink_noise
        
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