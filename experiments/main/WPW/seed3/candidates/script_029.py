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
    5. Continuous vector projection model for infinite morphological diversity.
    6. Beat-to-beat variability in pre-excitation degree (AV node vs accessory pathway competition).
    
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
    
    # Randomize electrical axes for continuous morphological diversity (addresses mode collapse)
    # Accessory pathway pre-excitation axis (can be anywhere, 0 to 2pi)
    pre_exc_angle = rng.uniform(0, 2 * np.pi)
    
    # Normal ventricular depolarization axis (typically -30 to +90 degrees, i.e., -pi/6 to pi/2)
    qrs_angle = rng.uniform(-np.pi/6, np.pi/2)
    
    # P wave axis (typically 0 to 75 degrees, i.e., 0 to 5pi/12)
    p_angle = rng.uniform(0, 5 * np.pi / 12)
    
    for lead_idx in range(2):
        # Lead I is at 0 degrees, Lead II is at 60 degrees (pi/3)
        lead_angle = 0.0 if lead_idx == 0 else np.pi / 3.0
        
        # Project the 2D electrical vectors onto the 1D lead axis
        p_proj = np.cos(p_angle - lead_angle)
        delta_proj = np.cos(pre_exc_angle - lead_angle)
        qrs_proj = np.cos(qrs_angle - lead_angle)
        s_proj = np.cos(qrs_angle + np.pi - lead_angle) # S wave is roughly opposite to main QRS
        
        # Base parameters for conduction (Amplitude, Center, Width)
        p_a = p_proj * rng.uniform(0.08, 0.20)
        p_sigma = rng.uniform(0.015, 0.022)
        
        # WPW Hallmark: Delta wave
        delta_a = delta_proj * rng.uniform(0.15, 0.50)
        delta_mu = rng.uniform(-0.045, -0.025)
        delta_sigma = rng.uniform(0.025, 0.040)
        
        # Main R wave
        r_a = max(0.0, qrs_proj * rng.uniform(0.8, 2.5))
        r_mu = 0.0
        r_sigma = rng.uniform(0.012, 0.020)
        
        # S wave
        s_a = min(0.0, s_proj * rng.uniform(0.2, 1.0))
        # If R wave is small, ensure S wave is deep to maintain QRS energy (e.g., QS complex)
        if r_a < 0.4:
            s_a -= rng.uniform(0.4, 1.0)
            
        s_mu = rng.uniform(0.025, 0.045)
        s_sigma = rng.uniform(0.01, 0.018)
        
        # T wave timing
        t_mu = rng.uniform(0.16, 0.26)
        t_sigma = rng.uniform(0.03, 0.055)
        
        # U wave for added morphological diversity
        u_a = rng.uniform(0.0, 0.02)
        u_mu = t_mu + rng.uniform(0.12, 0.16)
        u_sigma = rng.uniform(0.03, 0.05)
        
        # WPW Hallmark: Short PR interval (80 - 110 ms)
        pr_interval = rng.uniform(0.08, 0.11)
        
        # Calculate P wave center such that the distance between P onset and Delta onset strictly matches pr_interval
        delta_onset = delta_mu - 2 * delta_sigma
        p_onset = delta_onset - pr_interval
        p_mu = p_onset + 2 * p_sigma
        
        # ST Segment timing
        st_mu = rng.uniform(0.06, 0.10)
        st_sigma = rng.uniform(0.05, 0.08)
        
        # Respiratory modulation (amplitude modulation)
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add slight beat-to-beat morphological variation
            beat_var = rng.uniform(0.95, 1.05)
            
            # Vary the degree of pre-excitation slightly beat-to-beat 
            # (models physiological competition between AV node and accessory pathway)
            pre_exc_var = rng.uniform(0.8, 1.2)
            c_delta_a = delta_a * pre_exc_var * beat_var
            c_r_a = r_a * (2.0 - pre_exc_var) * beat_var # If delta increases, R decreases slightly (more fusion)
            c_s_a = s_a * beat_var
            
            # P, Delta, R, S waves
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, c_delta_a, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, c_r_a, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, c_s_a, r_time + s_mu, s_sigma)
            
            # Calculate net QRS vector for this specific beat to enforce discordance
            net_qrs = c_delta_a + c_r_a + c_s_a
            
            # ST Segment Discordance: ST level is opposite to the net QRS vector
            st_a = rng.uniform(-0.08, -0.03) * net_qrs
            lead_signal += gaussian(t, st_a, r_time + st_mu, st_sigma)
            
            # T wave Discordance: T wave polarity is opposite to the net QRS vector
            t_a = -rng.uniform(0.1, 0.25) * net_qrs + rng.uniform(-0.05, 0.05)
            
            # Asymmetric T wave (sum of main peak and a delayed, wider tail)
            lead_signal += gaussian(t, t_a, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, t_a * 0.4, r_time + t_mu + 0.02, t_sigma * 1.2)
            
            # U wave
            if u_a != 0:
                u_sign = np.sign(t_a) if t_a != 0 else 1
                lead_signal += gaussian(t, u_a * u_sign * beat_var, r_time + u_mu, u_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        baseline = np.zeros_like(t)
        # 1. Very slow drift
        baseline += rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * rng.uniform(0.005, 0.02) * t + rng.uniform(0, 2*np.pi))
        # 2. Respiratory wander
        baseline += rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t + rng.uniform(0, 2*np.pi))
        # 3. Additional wandering components
        for _ in range(3):
            baseline += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * rng.uniform(0.05, 0.3) * t + rng.uniform(0, 2*np.pi))
        
        # Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * powerline_freq * t)
        
        # High frequency muscle artifact (EMG - pinkish noise)
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(3)/3, mode='same')
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