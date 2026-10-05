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
    5. 2D Frontal Plane Vector Projection: Models the heart's electrical activity as 2D vectors 
       and projects them onto Lead I (0 deg) and Lead II (60 deg) to ensure perfect physiological 
       consistency and infinite continuous morphological diversity across samples.
    
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
    
    # Heart rate and HRV
    hr = rng.uniform(55, 105)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with Respiratory Sinus Arrhythmia (RSA) to increase diversity
    num_beats = int(duration / rr_mean) + 5
    resp_freq = rng.uniform(0.15, 0.4)
    t_beats = np.arange(num_beats) * rr_mean
    rsa_modulation = rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t_beats)
    rr_intervals = rr_mean + rsa_modulation + rng.normal(0, 0.015, num_beats)
    
    r_peaks = np.cumsum(rr_intervals)
    # Shift first peak to start early in the recording
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # --- 2D Vector Model for ECG Generation ---
    # Angles in radians. Lead I is at 0 rad, Lead II is at pi/3 (60 degrees) rad.
    
    # P wave vector
    p_mag = rng.uniform(0.08, 0.18)
    p_angle = rng.uniform(0, 75) * np.pi / 180
    p_sigma = rng.uniform(0.015, 0.025)
    
    # Delta wave vector (Accessory pathway location)
    # -90 to 180 degrees covers all typical pathway axes (left, right, septal)
    delta_angle = rng.uniform(-90, 180) * np.pi / 180
    delta_mag = rng.uniform(0.15, 0.5)
    delta_mu = rng.uniform(-0.05, -0.03)
    delta_sigma = rng.uniform(0.02, 0.035)
    
    # WPW Hallmark: Short PR interval (80 - 110 ms)
    pr_interval = rng.uniform(0.08, 0.11)
    p_mu = delta_mu - pr_interval
    
    # R wave vector
    r_angle = rng.uniform(-30, 120) * np.pi / 180
    r_mag = rng.uniform(1.0, 2.5)
    r_mu = 0.0
    r_sigma = rng.uniform(0.01, 0.016) # Sharper R wave to contrast with the slurred delta wave
    
    # S wave vector
    s_angle = rng.uniform(120, 270) * np.pi / 180
    s_mag = rng.uniform(0.1, 0.8)
    s_mu = rng.uniform(0.02, 0.04)
    s_sigma = rng.uniform(0.01, 0.018)
    
    # Calculate net QRS vector to apply Rule of Discordance for ST-T
    qrs_x = delta_mag * np.cos(delta_angle) + r_mag * np.cos(r_angle) + s_mag * np.cos(s_angle)
    qrs_y = delta_mag * np.sin(delta_angle) + r_mag * np.sin(r_angle) + s_mag * np.sin(s_angle)
    qrs_angle = np.arctan2(qrs_y, qrs_x)
    qrs_mag = np.sqrt(qrs_x**2 + qrs_y**2)
    
    # ST segment vector (Discordant to QRS)
    st_angle = qrs_angle + np.pi + rng.uniform(-0.3, 0.3)
    st_mag = rng.uniform(0.05, 0.15) * (qrs_mag / 1.5)
    st_mu = rng.uniform(0.08, 0.12)
    st_sigma = rng.uniform(0.06, 0.08)
    
    # T wave vector (Discordant to QRS)
    t_angle = qrs_angle + np.pi + rng.uniform(-0.4, 0.4)
    t_mag = rng.uniform(0.15, 0.4) * (qrs_mag / 1.5)
    t_mu = rng.uniform(0.18, 0.28)
    t_sigma = rng.uniform(0.035, 0.055)
    
    # U wave vector
    u_angle = t_angle + rng.uniform(-0.5, 0.5)
    u_mag = rng.uniform(0.0, 0.04)
    u_mu = t_mu + rng.uniform(0.12, 0.16)
    u_sigma = rng.uniform(0.03, 0.05)
    
    # Respiratory modulation (amplitude modulation)
    resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
    
    # Projection angles for Lead I and Lead II
    lead_angles = [0.0, np.pi / 3]
    
    for lead_idx in range(2):
        proj_angle = lead_angles[lead_idx]
        
        # Project 2D vectors onto the specific lead axis
        p_a = p_mag * np.cos(p_angle - proj_angle)
        delta_a = delta_mag * np.cos(delta_angle - proj_angle)
        r_a = r_mag * np.cos(r_angle - proj_angle)
        s_a = s_mag * np.cos(s_angle - proj_angle)
        st_a = st_mag * np.cos(st_angle - proj_angle)
        t_a = t_mag * np.cos(t_angle - proj_angle)
        u_a = u_mag * np.cos(u_angle - proj_angle)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_var = rng.uniform(0.95, 1.05)
            
            # P, Delta, R, S waves
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            
            # ST Segment
            lead_signal += gaussian(t, st_a * beat_var, r_time + st_mu, st_sigma)
            
            # Asymmetric T wave (sum of main peak and a delayed, wider tail)
            lead_signal += gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, t_a * 0.4 * beat_var, r_time + t_mu + 0.02, t_sigma * 1.2)
            
            # U wave
            if u_mag != 0:
                lead_signal += gaussian(t, u_a * beat_var, r_time + u_mu, u_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Complex low frequency baseline wander
        baseline = rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
        baseline += rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * resp_freq * t)
        for _ in range(3):
            baseline += rng.uniform(0.01, 0.06) * np.sin(2 * np.pi * rng.uniform(0.05, 0.4) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(-0.2, 0.2) * (t / duration) # Slow linear drift
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG - pinkish noise)
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(5)/5, mode='same')
        emg = rng.uniform(0.01, 0.04) * pink_noise
        
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