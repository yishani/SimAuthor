import os
import numpy as np

def gaussian(x, a, mu, sigma):
    """
    Evaluates a Gaussian function.
    """
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2)

def generate_wpw_ecg(seed, fs=500, duration=10.0):
    """
    Generates a synthetic 2-lead ECG signal exhibiting Wolff-Parkinson-White (WPW) Syndrome.
    
    Refinements in this version:
    1. Sharpened R-wave (reduced r_sigma) to create a distinct physiological inflection point 
       where the slurred delta wave transitions into normal His-Purkinje conduction.
    2. Synchronized global timing parameters across leads to ensure physiological validity 
       (electrical events occur simultaneously in 3D space).
    3. Expanded morphological diversity (added a 4th accessory pathway, widened amplitude ranges, 
       and increased HRV/noise variance) to address the low spread ratio and mode collapse.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Expanded WPW accessory pathway locations for higher inter-sample diversity
    # 0: Left lateral, 1: Posteroseptal, 2: Right lateral, 3: Anteroseptal
    pathway = rng.choice([0, 1, 2, 3])
    
    # Heart rate and HRV (widened range for more diversity)
    hr = rng.uniform(50, 110)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with increased natural variability (RSA)
    rr_intervals = rng.normal(rr_mean, rng.uniform(0.02, 0.06), int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Global timing parameters (consistent across leads to maintain physiological validity)
    p_sigma = rng.uniform(0.015, 0.025)
    r_sigma = rng.uniform(0.008, 0.014) # Sharpened R wave to create distinct delta inflection
    s_sigma = rng.uniform(0.015, 0.025)
    t_sigma = rng.uniform(0.035, 0.055)
    delta_sigma = rng.uniform(0.02, 0.035)
    
    delta_mu = rng.uniform(-0.05, -0.03)
    r_mu = 0.0
    s_mu = rng.uniform(0.02, 0.04)
    t_mu = rng.uniform(0.20, 0.30)
    
    # WPW Hallmark: Short PR interval (80 - 110 ms)
    pr_interval = rng.uniform(0.08, 0.11)
    p_mu = delta_mu - pr_interval
    
    st_mu = rng.uniform(0.06, 0.10)
    st_sigma = rng.uniform(0.04, 0.07)
    
    # Asymmetric T-wave tail parameters
    t_tail_a_ratio = rng.uniform(0.2, 0.5)
    t_tail_mu_offset = rng.uniform(0.03, 0.06)
    t_tail_sigma_ratio = rng.uniform(1.2, 1.6)
    
    resp_freq = rng.uniform(0.2, 0.35)
    
    for lead_idx in range(2):
        p_a = rng.uniform(0.05, 0.2)
        
        # Adjust morphology based on the accessory pathway location with expanded ranges
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.8, 2.0)
                s_a = rng.uniform(-0.3, 0.0)
                t_a = rng.uniform(-0.3, 0.1)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.5)
                r_a = rng.uniform(1.0, 2.5)
                s_a = rng.uniform(-0.3, 0.0)
                t_a = rng.uniform(-0.3, 0.1)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.05, 0.3)
                r_a = rng.uniform(0.6, 1.5)
                s_a = rng.uniform(-0.4, -0.1)
                t_a = rng.uniform(-0.2, 0.2)
            else: # Lead II
                delta_a = rng.uniform(-0.5, -0.1) # Negative delta wave (pseudo-Q)
                r_a = rng.uniform(0.1, 0.8)       # Attenuated R wave
                s_a = rng.uniform(-1.0, -0.2)     # Deep S wave
                t_a = rng.uniform(0.1, 0.4)       # Discordant positive T wave
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.3, -0.05)
                r_a = rng.uniform(0.4, 1.5)
                s_a = rng.uniform(-0.5, -0.1)
                t_a = rng.uniform(-0.2, 0.3)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.8, 2.0)
                s_a = rng.uniform(-0.3, 0.0)
                t_a = rng.uniform(-0.3, 0.1)
        elif pathway == 3: # Anteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.15, 0.05)
                r_a = rng.uniform(0.4, 1.2)
                s_a = rng.uniform(-0.6, -0.1)
                t_a = rng.uniform(-0.1, 0.25)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.35)
                r_a = rng.uniform(0.8, 1.8)
                s_a = rng.uniform(-0.4, -0.1)
                t_a = rng.uniform(-0.2, 0.15)
                
        # Respiratory modulation (amplitude modulation)
        resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t + rng.uniform(0, 2*np.pi))
        
        # Rule of Discordance factor
        st_discordance_factor = rng.uniform(-0.08, -0.03)
        
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
            
            # ST Segment Discordance: ST level is opposite to the net QRS vector
            net_qrs = delta_a + r_a + s_a
            st_a = st_discordance_factor * net_qrs * beat_var
            lead_signal += gaussian(t, st_a, r_time + st_mu, st_sigma)
            
            # Asymmetric T wave (sum of main peak and a delayed, wider tail)
            lead_signal += gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma)
            lead_signal += gaussian(t, t_a * t_tail_a_ratio * beat_var, r_time + t_mu + t_tail_mu_offset, t_sigma * t_tail_sigma_ratio)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise (enhanced for realism)
        # 1. Complex low frequency baseline wander
        baseline = rng.uniform(0.05, 0.25) * np.sin(2 * np.pi * rng.uniform(0.01, 0.05) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * resp_freq * t + rng.uniform(0, 2*np.pi))
        for _ in range(4):
            baseline += rng.uniform(0.01, 0.06) * np.sin(2 * np.pi * rng.uniform(0.05, 0.5) * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(-0.2, 0.2) * (t / duration) # Slow linear drift
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.015) * np.sin(2 * np.pi * powerline_freq * t + rng.uniform(0, 2*np.pi))
        
        # 3. High frequency muscle artifact (EMG - pinkish noise with movement envelopes)
        white_noise = rng.normal(0, 1, len(t))
        pink_noise = np.convolve(white_noise, np.ones(5)/5, mode='same')
        emg_env = 1.0 + rng.uniform(0.5, 2.0) * np.sin(2 * np.pi * rng.uniform(0.1, 0.5) * t + rng.uniform(0, 2*np.pi))
        emg = rng.uniform(0.005, 0.025) * pink_noise * emg_env
        
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