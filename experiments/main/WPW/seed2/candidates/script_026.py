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
    1. Delta Wave Fusion: Aligned the delta wave electrical axis with the R-wave and adjusted 
       its timing/width to create a smooth, monotonically increasing slurred upstroke, 
       eliminating the unnatural notch/separation seen in previous versions.
    2. QRS Axis & Morphology: Constrained the R-wave axis to ensure tall, positive R-waves 
       in both Lead I and II, matching typical manifest WPW presentations.
    3. Noise Reduction: Decreased the amplitude of baseline wander and high-frequency noise 
       to better match the clean physiological baseline of the reference empirical samples.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Degree of pre-excitation (0.6 to 1.0) to ensure manifest WPW features are clear
    pre_ex = rng.uniform(0.6, 1.0)
    
    # Heart rate and HRV (Respiratory Sinus Arrhythmia)
    hr = rng.uniform(55, 95)
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
    p_angle = rng.uniform(30, 60) # Positive in I and II
    p_width = rng.uniform(0.015, 0.022)
    p_mu = rng.uniform(-0.16, -0.13) # Positioned for 80-110ms PR interval
    
    # R-wave axis constrained to produce tall positive complexes in Lead I and II
    r_angle = rng.uniform(20, 70) 
    r_mag = rng.uniform(1.2, 2.2)
    r_width = rng.uniform(0.012, 0.018)
    r_mu = 0.0
    
    # Delta wave axis aligned with R-wave to ensure concordant slurring (no notch)
    delta_angle = r_angle + rng.uniform(-15, 15)
    delta_mag = rng.uniform(0.25, 0.5) * pre_ex
    delta_width = rng.uniform(0.022, 0.028)
    delta_mu = rng.uniform(-0.04, -0.03)
    
    s_angle = r_angle + rng.uniform(120, 240)
    s_mag = rng.uniform(0.05, 0.2) # Small S-wave
    s_width = rng.uniform(0.012, 0.02)
    s_mu = rng.uniform(0.025, 0.04)
    
    # T wave discordance: Axis shifts opposite to QRS
    discordance_shift = rng.uniform(135, 180)
    t_angle = r_angle + discordance_shift + rng.uniform(-10, 10)
    t_mag = rng.uniform(0.15, 0.35)
    t_width = rng.uniform(0.035, 0.05)
    t_mu = rng.uniform(0.20, 0.26)
    
    # Lead angles for projection (with slight variation for diversity)
    lead_angles = [rng.uniform(-5, 5), 60 + rng.uniform(-5, 5)] # Lead I, Lead II
    
    for lead_idx, lead_angle in enumerate(lead_angles):
        lead_signal = np.zeros_like(t)
        
        # Respiratory amplitude modulation
        resp_mod = 1.0 + rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * resp_freq * t)
        
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
            st_amp = t_amp * rng.uniform(0.2, 0.4)
            lead_signal += gaussian(t, st_amp, r_time + 0.08, 0.04)
            
            # Asymmetric T-wave (main peak + slightly wider tail)
            lead_signal += gaussian(t, t_amp, r_time + t_mu, t_width)
            lead_signal += gaussian(t, t_amp * 0.4, r_time + t_mu + 0.03, t_width * 1.4)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise (Reduced for cleaner baseline)
        baseline = np.zeros_like(t)
        for _ in range(rng.integers(1, 3)):
            freq = rng.uniform(0.05, 0.2)
            phase = rng.uniform(0, 2 * np.pi)
            amp = rng.uniform(0.01, 0.04)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
            
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.0005, 0.002) * np.sin(2 * np.pi * powerline_freq * t)
        
        emg = rng.normal(0, rng.uniform(0.002, 0.006), len(t))
        
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