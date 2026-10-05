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
       - The delta wave is modeled with a wide Gaussian centered close to the R-wave to ensure fusion (no notching).
    3. Widened QRS complex (fusion of pre-excitation and normal conduction).
    4. Secondary ST-T wave abnormalities (discordant to the delta/QRS vector).
    5. Variations in delta wave polarity based on accessory pathway location.
    6. Variable degree of pre-excitation (competition between AV node and accessory pathway).
    
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
    # 3: Anteroseptal (Positive delta in I and II, normal R)
    pathway = rng.choice([0, 1, 2, 3])
    
    # Degree of pre-excitation (0.3 to 1.0)
    # 1.0 = max pre-excitation (wide QRS, huge delta, short PR)
    # 0.3 = min pre-excitation (almost normal conduction, subtle delta)
    pre_ex = rng.uniform(0.3, 1.0)
    
    # Heart rate and HRV (Respiratory Sinus Arrhythmia)
    hr = rng.uniform(55, 100)
    rr_mean = 60.0 / hr
    
    resp_freq = rng.uniform(0.2, 0.35)
    t_rr = np.arange(0, duration + 2, rr_mean)
    rr_intervals = rr_mean + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t_rr)
    rr_intervals += rng.normal(0, 0.015, len(rr_intervals)) # Add random HRV noise
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    for lead_idx in range(2):
        # Base parameters for conduction
        p_a = rng.uniform(0.08, 0.18)
        p_sigma = rng.uniform(0.015, 0.022)
        # P-wave moves slightly further away from QRS when pre-excitation is low (AV node takes over)
        p_mu = rng.uniform(-0.15, -0.12) - (1.0 - pre_ex) * 0.03
        
        r_a = rng.uniform(0.8, 2.0)
        r_mu = 0.0
        r_sigma = rng.uniform(0.012, 0.018)
        
        s_a = rng.uniform(-0.4, -0.05)
        s_mu = rng.uniform(0.03, 0.06)
        s_sigma = rng.uniform(0.015, 0.025)
        
        t_a = rng.uniform(0.15, 0.4)
        t_mu = rng.uniform(0.22, 0.28)
        t_sigma = rng.uniform(0.035, 0.055)
        
        # WPW Hallmark: Delta wave
        # Centered very close to the R-wave but with a much wider sigma to create a slurred upstroke (fusion)
        delta_mu = rng.uniform(-0.02, -0.01)
        delta_sigma = rng.uniform(0.03, 0.04)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.15, 0.4)
                r_a = rng.uniform(1.0, 2.0)
                t_a = rng.uniform(-0.3, -0.05) # Discordant T wave inversion
                s_a = rng.uniform(-0.2, 0.0)
            else: # Lead II
                delta_a = rng.uniform(0.2, 0.45)
                r_a = rng.uniform(1.2, 2.2)
                t_a = rng.uniform(-0.25, 0.0)
                s_a = rng.uniform(-0.2, 0.0)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.8, 1.5)
                t_a = rng.uniform(0.1, 0.3)
            else: # Lead II
                delta_a = rng.uniform(-0.5, -0.2) # Negative delta wave (pseudo-Q)
                delta_mu = rng.uniform(-0.025, -0.015) 
                r_a = rng.uniform(0.3, 0.8)       # Attenuated R wave
                s_a = rng.uniform(-0.8, -0.3)     # Deep S wave
                t_a = rng.uniform(0.2, 0.5)       # Discordant positive T wave
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.2, 0.05)
                r_a = rng.uniform(0.5, 1.2)
                s_a = rng.uniform(-0.5, -0.1)
                t_a = rng.uniform(0.1, 0.3)
            else: # Lead II
                delta_a = rng.uniform(0.15, 0.4)
                r_a = rng.uniform(1.0, 2.0)
                t_a = rng.uniform(-0.2, 0.1)
        elif pathway == 3: # Anteroseptal
            if lead_idx == 0:
                delta_a = rng.uniform(0.1, 0.25)
                r_a = rng.uniform(0.8, 1.6)
                t_a = rng.uniform(-0.1, 0.2)
            else:
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.8, 1.6)
                t_a = rng.uniform(-0.1, 0.2)

        # Apply pre-excitation scaling
        delta_a *= pre_ex
        delta_sigma *= (0.7 + 0.3 * pre_ex) # Delta is wider when more pre-excited
        r_a *= (1.0 + (1.0 - pre_ex) * 0.3) # R is taller when less pre-excited (normal conduction dominates)
        r_sigma *= (1.0 - (1.0 - pre_ex) * 0.2) # R is narrower when less pre-excited
        
        # Respiratory amplitude modulation
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_p_a = p_a * rng.uniform(0.9, 1.1)
            beat_delta_a = delta_a * rng.uniform(0.9, 1.1)
            beat_r_a = r_a * rng.uniform(0.95, 1.05)
            beat_s_a = s_a * rng.uniform(0.9, 1.1)
            beat_t_a = t_a * rng.uniform(0.9, 1.1)
            
            lead_signal += gaussian(t, beat_p_a, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, beat_delta_a, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, beat_r_a, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, beat_s_a, r_time + s_mu, s_sigma)
            
            # Asymmetric T-wave (main peak + wider tail)
            lead_signal += gaussian(t, beat_t_a, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, beat_t_a * 0.4, r_time + t_mu + 0.04, t_sigma * 1.3)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Realistic multi-frequency baseline wander
        baseline = np.zeros_like(t)
        for _ in range(rng.integers(3, 6)):
            freq = rng.uniform(0.01, 0.5)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.02, 0.1)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
            
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG)
        emg = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    return signal

def main():
    # Configuration
    output_dir = "[PROJECT_ROOT]/artifacts_3.1/wpw_ecg/runs/rep/seed1/generated/"
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