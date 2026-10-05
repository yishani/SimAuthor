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
    1. Variable degree of pre-excitation, controlling PR shortening, delta wave size, and QRS width.
    2. Shortened PR interval (< 120 ms) due to accessory pathway bypass.
    3. Delta wave (slurred upstroke of QRS) due to early ventricular pre-excitation.
    4. Secondary ST-T wave abnormalities proportional to the degree of pre-excitation.
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
    # 1: Posteroseptal (Positive in I, Negative delta in II)
    # 2: Right lateral (Negative/flat delta in I, Positive in II)
    pathway = rng.choice([0, 1, 2])
    
    # Degree of pre-excitation (0.2 = mild/intermittent, 1.0 = severe/manifest)
    # This single variable drives the physiological continuum of WPW severity,
    # increasing intra-class diversity and realism.
    pre_excitation = rng.uniform(0.2, 1.0)
    
    # Heart rate and HRV
    hr = rng.uniform(55, 105)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with Respiratory Sinus Arrhythmia (RSA)
    num_beats = int(duration / rr_mean) + 5
    resp_freq = rng.uniform(0.2, 0.35)
    t_beats = np.arange(num_beats) * rr_mean
    rsa_modulation = rng.uniform(0.02, 0.06) * np.sin(2 * np.pi * resp_freq * t_beats)
    rr_intervals = rr_mean + rsa_modulation + rng.normal(0, 0.01, num_beats)
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    for lead_idx in range(2):
        # Base parameters for conduction
        p_a = rng.uniform(0.05, 0.15)
        p_sigma = rng.uniform(0.015, 0.025)
        
        r_mu = 0.0
        r_sigma = rng.uniform(0.008, 0.012) # Sharper R wave to contrast with delta slur
        
        s_mu = rng.uniform(0.015, 0.03)
        s_sigma = rng.uniform(0.01, 0.015)
        
        t_mu = rng.uniform(0.18, 0.28)
        t_sigma = rng.uniform(0.035, 0.06)
        
        u_a = rng.uniform(0.0, 0.02)
        u_mu = t_mu + rng.uniform(0.12, 0.16)
        u_sigma = rng.uniform(0.03, 0.05)
        
        # Delta wave parameters scaled by pre-excitation
        delta_mu = rng.uniform(-0.045, -0.035)
        delta_sigma = rng.uniform(0.02, 0.025) + 0.01 * pre_excitation
        
        # PR interval inversely proportional to pre-excitation (0.08s to 0.112s)
        pr_interval = 0.12 - 0.04 * pre_excitation
        p_mu = delta_mu - pr_interval - 2*delta_sigma + 2*p_sigma
        
        st_mu = rng.uniform(0.08, 0.14)
        st_sigma = rng.uniform(0.06, 0.10)
        
        # Adjust morphology based on pathway with realistic, less extreme amplitude ranges
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                base_delta = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.6, 1.5)
                s_a = rng.uniform(-0.15, 0.0)
            else: # Lead II
                base_delta = rng.uniform(0.15, 0.35)
                r_a = rng.uniform(0.8, 1.8)
                s_a = rng.uniform(-0.15, 0.0)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                base_delta = rng.uniform(0.05, 0.25)
                r_a = rng.uniform(0.5, 1.2)
                s_a = rng.uniform(-0.2, -0.05)
            else: # Lead II
                base_delta = rng.uniform(-0.3, -0.1) # Negative delta wave (pseudo-Q)
                r_a = rng.uniform(0.4, 1.2)
                s_a = rng.uniform(-0.3, -0.1)
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                base_delta = rng.uniform(-0.15, 0.0)
                r_a = rng.uniform(0.4, 1.2)
                s_a = rng.uniform(-0.2, -0.05)
            else: # Lead II
                base_delta = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.7, 1.6)
                s_a = rng.uniform(-0.15, 0.0)
                
        # Actual delta amplitude depends on the degree of pre-excitation
        delta_a = base_delta * pre_excitation
        
        # T-wave discordance is proportional to pre-excitation
        normal_t = rng.uniform(0.15, 0.35)
        net_qrs_direction = np.sign(base_delta + r_a + s_a)
        if net_qrs_direction >= 0:
            discordant_t = rng.uniform(-0.3, -0.05)
        else:
            discordant_t = rng.uniform(0.1, 0.4)
            
        # Blend normal and discordant T-wave based on pre-excitation severity
        t_a = (1 - pre_excitation) * normal_t + pre_excitation * discordant_t
        
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        lead_signal = np.zeros_like(t)
        
        for r_time in r_peaks:
            beat_var = rng.uniform(0.95, 1.05)
            
            lead_signal += gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            
            # ST Segment Discordance scales with pre-excitation
            net_qrs = delta_a + r_a + s_a
            st_a = -0.08 * net_qrs * beat_var * pre_excitation
            lead_signal += gaussian(t, st_a, r_time + st_mu, st_sigma)
            
            # Asymmetric T wave
            lead_signal += gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, t_a * 0.4 * beat_var, r_time + t_mu + 0.02, t_sigma * 1.2)
            
            if u_a != 0:
                u_sign = np.sign(t_a) if t_a != 0 else 1
                lead_signal += gaussian(t, u_a * u_sign * beat_var, r_time + u_mu, u_sigma)
            
        lead_signal *= resp_mod
        
        # Noise generation
        baseline = rng.uniform(0.1, 0.25) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t)
        baseline += rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * resp_freq * t)
        for _ in range(2):
            baseline += rng.uniform(0.01, 0.05) * np.sin(2 * np.pi * rng.uniform(0.05, 0.2) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(-0.15, 0.15) * (t / duration)
        
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.008) * np.sin(2 * np.pi * powerline_freq * t)
        
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(5)/5, mode='same')
        emg = rng.uniform(0.015, 0.04) * pink_noise
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    return signal

def main():
    output_dir = "[PROJECT_ROOT]/artifacts_3.1/wpw_ecg/runs/rep/seed2/generated/"
    n_samples = 100
    fs = 500
    duration = 10.0
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(n_samples):
        seed = 42 + i
        signal = generate_wpw_ecg(seed=seed, fs=fs, duration=duration)
        
        filename = f"wpw_ecg_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, signal)

if __name__ == "__main__":
    main()