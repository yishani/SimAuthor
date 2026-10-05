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
    1. Axis Constraining: Constrained the electrical axis of the delta wave and R-wave to +30 to +75 degrees 
       to consistently produce the characteristic tall, positive QRS complexes in Leads I and II.
    2. Delta Wave Fusion: Widened the delta wave and adjusted its timing to create a smooth, slurred 
       upstroke that seamlessly fuses with the R-wave, avoiding artificial notching.
    3. S-wave Attenuation: Significantly reduced the S-wave amplitude to match the dominant R-wave 
       morphology typical of manifest WPW in these leads.
    4. ST-T Discordance: Enforced strict ST-segment and T-wave discordance (axis shift of 135-180 degrees 
       relative to the R-wave) to model secondary repolarization abnormalities accurately.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Degree of pre-excitation (0.4 to 1.0)
    pre_ex = rng.uniform(0.4, 1.0)
    
    # Heart rate and HRV (Respiratory Sinus Arrhythmia)
    hr = rng.uniform(60, 95)
    rr_mean = 60.0 / hr
    
    resp_freq = rng.uniform(0.15, 0.35)
    t_rr = np.arange(0, duration + 2, rr_mean)
    rr_intervals = rr_mean + rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * resp_freq * t_rr)
    rr_intervals += rng.normal(0, 0.01, len(rr_intervals)) # Add random HRV noise
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # --- Vector Parameters (Magnitude in mV, Angle in degrees) ---
    
    p_mag = rng.uniform(0.1, 0.18)
    p_angle = rng.uniform(30, 60)
    p_width = rng.uniform(0.015, 0.022)
    p_mu = rng.uniform(-0.16, -0.14) # Ensures PR interval is 80-110 ms
    
    # Delta wave axis constrained to be positive in I and II
    delta_angle = rng.uniform(30, 60)
    delta_mag = rng.uniform(0.3, 0.6) * pre_ex
    delta_width = rng.uniform(0.025, 0.035) # Wider for slurred upstroke
    delta_mu = rng.uniform(-0.05, -0.035)
    
    # R wave axis aligned with Lead II for tall R waves
    r_angle = rng.uniform(30, 75)
    r_mag = rng.uniform(1.2, 2.2)
    r_width = rng.uniform(0.015, 0.02)
    r_mu = 0.0
    
    # Attenuated S wave
    s_angle = r_angle + rng.uniform(150, 210)
    s_mag = rng.uniform(0.0, 0.15)
    s_width = rng.uniform(0.015, 0.025)
    s_mu = rng.uniform(0.03, 0.05)
    
    # T wave discordance: Axis shifts opposite to QRS
    t_angle = r_angle + rng.uniform(135, 180)
    t_mag = rng.uniform(0.15, 0.3)
    t_width = rng.uniform(0.04, 0.05)
    t_mu = rng.uniform(0.22, 0.28)
    
    # Lead angles for projection (with slight variation for diversity)
    lead_angles = [rng.uniform(-5, 5), 60 + rng.uniform(-5, 5)] # Lead I, Lead II
    
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
            
            # Explicit ST segment (discordant with QRS, concordant with T)
            st_amp = t_amp * rng.uniform(0.2, 0.5)
            lead_signal += gaussian(t, st_amp, r_time + 0.12, 0.06)
            
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
            amp = rng.uniform(0.02, 0.05)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
            
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.0005, 0.002) * np.sin(2 * np.pi * powerline_freq * t)
        
        emg = rng.normal(0, rng.uniform(0.005, 0.01), len(t))
        
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