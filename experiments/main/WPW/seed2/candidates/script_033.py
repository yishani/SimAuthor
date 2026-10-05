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

def project_lead(mag, angle_deg, lead_angle_deg):
    """
    Projects a 2D heart vector onto a specific ECG lead axis.
    
    Parameters:
    mag : float - Magnitude of the vector
    angle_deg : float - Angle of the vector in the frontal plane (degrees)
    lead_angle_deg : float - Angle of the recording lead (degrees)
    """
    angle_rad = np.deg2rad(angle_deg)
    lead_angle_rad = np.deg2rad(lead_angle_deg)
    return mag * np.cos(angle_rad - lead_angle_rad)

def generate_wpw_ecg(seed, fs=500, duration=10.0):
    """
    Generates a synthetic 2-lead ECG signal exhibiting Wolff-Parkinson-White (WPW) Syndrome.
    
    Refinements:
    1. Delta Wave Fusion: Shifted delta_mu closer to the R-wave and adjusted widths to ensure 
       a monotonic slurred upstroke, eliminating the unnatural dip/notch between the delta and R waves.
    2. T-Wave Discordance: Increased the discordance shift towards 180 degrees to better reflect 
       the secondary repolarization abnormalities in manifest WPW.
    3. Noise Reduction: Lowered EMG and baseline wander amplitudes to match the cleaner 
       signal-to-noise ratio observed in the empirical reference set.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to determine delta wave axis
    pathway = rng.choice([0, 1, 2, 3])
    
    # Degree of pre-excitation (0.3 to 1.0)
    pre_ex = rng.uniform(0.3, 1.0)
    
    # Heart rate and HRV (Respiratory Sinus Arrhythmia)
    hr = rng.uniform(50, 105)
    rr_mean = 60.0 / hr
    
    resp_freq = rng.uniform(0.15, 0.35)
    t_rr = np.arange(0, duration + 2, rr_mean)
    rr_intervals = rr_mean + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t_rr)
    rr_intervals += rng.normal(0, 0.015, len(rr_intervals)) # Add random HRV noise
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # --- Vector Parameters (Magnitude in mV, Angle in degrees) ---
    
    p_mag = rng.uniform(0.08, 0.2)
    p_angle = rng.uniform(15, 75)
    p_width = rng.uniform(0.015, 0.022)
    # P-wave positioned to maintain appropriate PR interval (80-110 ms)
    p_mu = rng.uniform(-0.15, -0.12) - (1.0 - pre_ex) * 0.02
    
    # Delta wave axis depends on pathway location
    if pathway == 0: # Left lateral (Positive in I and II)
        delta_angle = rng.uniform(15, 75)
    elif pathway == 1: # Posteroseptal (Positive in I, Negative in II)
        delta_angle = rng.uniform(-105, -15)
    elif pathway == 2: # Right lateral (Negative/flat in I, Positive in II)
        delta_angle = rng.uniform(75, 135)
    else: # Anteroseptal (Positive in I and II)
        delta_angle = rng.uniform(-15, 90)
        
    # Delta wave shifted closer to R-wave to ensure monotonic fusion (slurred upstroke)
    delta_mag = rng.uniform(0.15, 0.4) * pre_ex
    delta_width = rng.uniform(0.025, 0.035)
    delta_mu = rng.uniform(-0.045, -0.03)
    
    r_angle = rng.uniform(-45, 120)
    r_mag = rng.uniform(0.8, 2.5) * (1.0 + (1.0 - pre_ex) * 0.2)
    r_width = rng.uniform(0.01, 0.015)
    r_mu = 0.0
    
    # Q-wave is intentionally omitted. In WPW, the delta wave obscures the normal septal Q-wave.
    
    s_angle = r_angle + rng.uniform(120, 240)
    s_mag = rng.uniform(0.0, 0.15)
    s_width = rng.uniform(0.015, 0.025)
    s_mu = rng.uniform(0.03, 0.05)
    
    # T wave discordance: Axis shifts opposite to QRS as pre-excitation increases
    discordance_shift = rng.uniform(120, 180) * pre_ex
    t_angle = r_angle + discordance_shift + rng.uniform(-20, 20)
    t_mag = rng.uniform(0.1, 0.3)
    t_width = rng.uniform(0.035, 0.05)
    t_mu = rng.uniform(0.18, 0.24)
    
    # Lead angles for projection (with slight variation for diversity)
    lead_angles = [rng.uniform(-5, 5), 60 + rng.uniform(-5, 5)] # Lead I, Lead II
    
    for lead_idx, lead_angle in enumerate(lead_angles):
        lead_signal = np.zeros_like(t)
        
        # Respiratory amplitude modulation
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            b_p_mag = p_mag * rng.uniform(0.9, 1.1)
            b_p_ang = p_angle + rng.uniform(-5, 5)
            
            b_delta_mag = delta_mag * rng.uniform(0.9, 1.1)
            b_delta_ang = delta_angle + rng.uniform(-5, 5)
            
            b_r_mag = r_mag * rng.uniform(0.95, 1.05)
            b_r_ang = r_angle + rng.uniform(-3, 3)
            
            b_s_mag = s_mag * rng.uniform(0.9, 1.1)
            b_s_ang = s_angle + rng.uniform(-5, 5)
            
            b_t_mag = t_mag * rng.uniform(0.9, 1.1)
            b_t_ang = t_angle + rng.uniform(-5, 5)
            
            # Project 2D vectors to the current lead
            p_amp = project_lead(b_p_mag, b_p_ang, lead_angle)
            delta_amp = project_lead(b_delta_mag, b_delta_ang, lead_angle)
            r_amp = project_lead(b_r_mag, b_r_ang, lead_angle)
            s_amp = project_lead(b_s_mag, b_s_ang, lead_angle)
            t_amp = project_lead(b_t_mag, b_t_ang, lead_angle)
            
            # Add waveforms to signal
            lead_signal += gaussian(t, p_amp, r_time + p_mu, p_width)
            lead_signal += gaussian(t, delta_amp, r_time + delta_mu, delta_width)
            lead_signal += gaussian(t, r_amp, r_time + r_mu, r_width)
            lead_signal += gaussian(t, s_amp, r_time + s_mu, s_width)
            
            # Explicit ST segment (concordant with T-wave, discordant with QRS)
            st_amp = t_amp * rng.uniform(0.3, 0.6)
            lead_signal += gaussian(t, st_amp, r_time + 0.08, 0.06)
            
            # Asymmetric T-wave (main peak + slightly wider tail)
            lead_signal += gaussian(t, t_amp, r_time + t_mu, t_width)
            lead_signal += gaussian(t, t_amp * 0.3, r_time + t_mu + 0.03, t_width * 1.3)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        baseline = np.zeros_like(t)
        for _ in range(rng.integers(1, 3)):
            freq = rng.uniform(0.05, 0.2)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.01, 0.03)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
            
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.0005, 0.002) * np.sin(2 * np.pi * powerline_freq * t)
        
        emg = rng.normal(0, rng.uniform(0.002, 0.008), len(t))
        
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