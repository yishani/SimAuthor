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
    
    Physiological characteristics modeled:
    1. Shortened PR interval (< 120 ms) due to accessory pathway bypass.
    2. Delta wave (slurred upstroke of QRS) due to early ventricular pre-excitation.
       - Carefully timed and widened to prevent notching and ensure a smooth fusion upstroke.
    3. Widened QRS complex (fusion of pre-excitation and normal conduction).
    4. Secondary ST-T wave abnormalities (discordant to the delta/QRS vector).
       - Includes ST segment depression and T-wave inversion proportional to pre-excitation.
    5. 2D Vector Projection: Ensures physiological consistency between Lead I (0 deg) 
       and Lead II (60 deg).
    
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
    
    # Degree of pre-excitation (0.2 to 1.0)
    # 1.0 = max pre-excitation (wide QRS, huge delta, short PR, strong discordance)
    pre_ex = rng.uniform(0.2, 1.0)
    
    # Heart rate and HRV (Respiratory Sinus Arrhythmia)
    hr = rng.uniform(50, 100)
    rr_mean = 60.0 / hr
    
    resp_freq = rng.uniform(0.15, 0.4)
    t_rr = np.arange(0, duration + 2, rr_mean)
    hrv_amount = rng.uniform(0.02, 0.10)
    rr_intervals = rr_mean + hrv_amount * np.sin(2 * np.pi * resp_freq * t_rr)
    rr_intervals += rng.normal(0, rng.uniform(0.01, 0.03), len(rr_intervals)) # Add random HRV noise
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # --- Vector Parameters (Magnitude in mV, Angle in degrees) ---
    # Frontal plane angles: Lead I = 0 deg, Lead II = 60 deg
    
    p_mag = rng.uniform(0.1, 0.22)
    p_angle = rng.uniform(20, 70)
    p_width = rng.uniform(0.015, 0.022)
    p_mu = rng.uniform(-0.16, -0.14) # Ensures PR interval < 120ms when combined with delta_mu
    
    # Delta wave: Constrained to be positive in both I and II (classic manifest WPW)
    delta_angle = rng.uniform(20, 70)
    delta_mag = rng.uniform(0.2, 0.6) * pre_ex
    delta_width = rng.uniform(0.025, 0.035)
    # Positioned carefully to create a slurred upstroke without a notch before the R-peak
    delta_mu = rng.uniform(-0.045, -0.035) 
    
    # Main R-wave vector
    r_angle = rng.uniform(30, 70)
    r_mag = rng.uniform(1.2, 2.5)
    r_width = rng.uniform(0.012, 0.016) # Sharp peak to contrast with slurred delta
    r_mu = 0.0
    
    # Q-wave is intentionally omitted as the delta wave replaces normal early septal depolarization
    
    # S-wave
    s_angle = r_angle + rng.uniform(150, 210)
    s_mag = rng.uniform(0.0, 0.3) # Kept small to match typical WPW morphology in I/II
    s_width = rng.uniform(0.015, 0.025)
    s_mu = rng.uniform(0.03, 0.05)
    
    # ST segment depression (Discordant to QRS)
    st_angle = r_angle + 180
    st_mag = rng.uniform(0.05, 0.15) * pre_ex
    st_width = rng.uniform(0.04, 0.06)
    st_mu = rng.uniform(0.08, 0.12)
    
    # T wave discordance: Axis shifts opposite to QRS as pre-excitation increases
    normal_t_angle = rng.uniform(20, 70)
    discordant_t_angle = r_angle + 180 + rng.uniform(-20, 20)
    t_angle = normal_t_angle * (1.0 - pre_ex) + discordant_t_angle * pre_ex
    t_mag = rng.uniform(0.15, 0.4)
    t_width = rng.uniform(0.035, 0.05)
    t_mu = rng.uniform(0.22, 0.26)
    
    # Lead angles for projection
    lead_angles = [0, 60] # Lead I, Lead II
    
    for lead_idx, lead_angle in enumerate(lead_angles):
        lead_signal = np.zeros_like(t)
        
        # Respiratory amplitude modulation
        resp_mod = 1.0 + rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * resp_freq * t)
        
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
            
            b_st_mag = st_mag * rng.uniform(0.9, 1.1)
            b_st_ang = st_angle + rng.uniform(-5, 5)
            
            b_t_mag = t_mag * rng.uniform(0.9, 1.1)
            b_t_ang = t_angle + rng.uniform(-5, 5)
            
            # Project 2D vectors to the current lead
            p_amp = project_lead(b_p_mag, b_p_ang, lead_angle)
            delta_amp = project_lead(b_delta_mag, b_delta_ang, lead_angle)
            r_amp = project_lead(b_r_mag, b_r_ang, lead_angle)
            s_amp = project_lead(b_s_mag, b_s_ang, lead_angle)
            st_amp = project_lead(b_st_mag, b_st_ang, lead_angle)
            t_amp = project_lead(b_t_mag, b_t_ang, lead_angle)
            
            # Add waveforms to signal
            lead_signal += gaussian(t, p_amp, r_time + p_mu, p_width)
            lead_signal += gaussian(t, delta_amp, r_time + delta_mu, delta_width)
            lead_signal += gaussian(t, r_amp, r_time + r_mu, r_width)
            lead_signal += gaussian(t, s_amp, r_time + s_mu, s_width)
            lead_signal += gaussian(t, st_amp, r_time + st_mu, st_width)
            
            # Asymmetric T-wave (main peak + wider tail)
            lead_signal += gaussian(t, t_amp, r_time + t_mu, t_width)
            lead_signal += gaussian(t, t_amp * 0.4, r_time + t_mu + 0.04, t_width * 1.3)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        baseline = np.zeros_like(t)
        for _ in range(rng.integers(2, 5)):
            freq = rng.uniform(0.02, 0.4)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.02, 0.1)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
            
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * powerline_freq * t)
        
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