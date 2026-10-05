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
    
    Physiological characteristics modeled:
    1. Variable degree of pre-excitation coupling PR interval, delta wave, and QRS width.
    2. Shortened PR interval (80-110 ms) strictly maintained by calculating exact QRS onset.
    3. Delta wave (slurred upstroke of QRS) masking normal Q-waves during pre-excitation.
    4. Widened QRS complex (fusion of pre-excitation and normal conduction).
    5. Secondary ST-T wave abnormalities (discordant to the delta/QRS vector).
    6. Diverse accessory pathway locations (Left lateral, Posteroseptal, Right lateral, Anteroseptal).
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to create diverse manifestations
    # 0: Left lateral (Type A-like, positive delta in I and II)
    # 1: Posteroseptal (Type B-like, positive in I, negative delta in II - pseudo-inferior infarct)
    # 2: Right lateral (Type B, negative/flat delta in I, positive in II)
    # 3: Anteroseptal (Positive delta in I and II, but smaller amplitudes)
    pathway = rng.choice([0, 1, 2, 3])
    
    # Heart rate and HRV (wider range for increased inter-sample diversity)
    hr = rng.uniform(55, 105)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with slight natural variability
    rr_intervals = rng.normal(rr_mean, 0.03, int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Global physiological parameters defining the severity of the condition
    pre_excitation_base = rng.uniform(0.2, 1.0)
    is_intermittent = rng.random() < 0.2 # 20% chance of intermittent WPW
    
    # Temporal parameters (consistent across leads for physiological synchrony)
    p_sigma = rng.uniform(0.015, 0.022)
    r_mu = 0.0
    r_sigma_base = rng.uniform(0.008, 0.012)
    s_mu_base = rng.uniform(0.015, 0.025)
    s_sigma = rng.uniform(0.010, 0.018)
    t_mu = rng.uniform(0.22, 0.28)
    t_sigma = rng.uniform(0.035, 0.055)
    st_sigma = rng.uniform(0.04, 0.06)
    
    # Pre-calculate synchronized beat parameters across leads to ensure physiological validity
    beat_params = []
    for _ in r_peaks:
        if is_intermittent and rng.random() < 0.25:
            # Sudden drop in pre-excitation (normal or near-normal conduction)
            beat_pre_ex = rng.uniform(0.0, 0.1)
        else:
            # Normal variation around base severity
            beat_pre_ex = np.clip(pre_excitation_base + rng.uniform(-0.1, 0.1), 0.0, 1.0)
            
        # High pre-excitation -> shorter PR, wider delta, wider R, wider QRS
        b_pr_interval = rng.uniform(0.12, 0.16) - 0.05 * beat_pre_ex
        b_delta_mu = -0.02 - 0.03 * beat_pre_ex
        b_delta_sigma = 0.015 + 0.015 * beat_pre_ex
        b_r_sigma = r_sigma_base + 0.005 * beat_pre_ex
        b_s_mu = s_mu_base + 0.015 * beat_pre_ex
        
        # Calculate exact QRS onset to strictly maintain the PR interval
        q_onset = -0.02 - 2 * 0.01 # Approximate onset of normal Q wave
        delta_onset = b_delta_mu - 2 * b_delta_sigma
        
        # True QRS onset blends between normal Q wave and pre-excited delta wave
        actual_qrs_onset = q_onset * (1 - beat_pre_ex) + delta_onset * beat_pre_ex
        
        # Position P wave relative to the true QRS onset
        b_p_mu = actual_qrs_onset + 2*p_sigma - b_pr_interval
        
        # Scale delta wave amplitude based on pre-excitation degree
        b_delta_scale = beat_pre_ex
        
        beat_params.append({
            'pre_ex': beat_pre_ex,
            'p_mu': b_p_mu,
            'delta_mu': b_delta_mu,
            'delta_sigma': b_delta_sigma,
            'r_sigma': b_r_sigma,
            's_mu': b_s_mu,
            'delta_scale': b_delta_scale,
            'beat_var': rng.uniform(0.95, 1.05)
        })
        
    for lead_idx in range(2):
        p_a = rng.uniform(0.08, 0.18)
        normal_t_a = rng.uniform(0.15, 0.35)
        
        # Normal Q wave parameters (masked during pre-excitation)
        q_a = rng.uniform(-0.15, 0.0)
        q_mu = -0.02
        q_sigma = rng.uniform(0.008, 0.012)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.8, 1.8)
                s_a = rng.uniform(-0.2, 0.0)
                discordant_t = rng.uniform(-0.3, -0.1)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(1.0, 2.2)
                s_a = rng.uniform(-0.2, 0.0)
                discordant_t = rng.uniform(-0.4, -0.1)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(0.8, 1.8)
                s_a = rng.uniform(-0.3, 0.0)
                discordant_t = rng.uniform(-0.3, -0.05)
            else: # Lead II
                delta_a = rng.uniform(-0.4, -0.15) # Negative delta wave (pseudo-infarct)
                r_a = rng.uniform(0.1, 0.6)
                s_a = rng.uniform(-1.0, -0.3)
                discordant_t = rng.uniform(0.1, 0.4)
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.2, 0.05)
                r_a = rng.uniform(0.3, 1.0)
                s_a = rng.uniform(-0.8, -0.2)
                discordant_t = rng.uniform(0.1, 0.3)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.3)
                r_a = rng.uniform(1.0, 2.0)
                s_a = rng.uniform(-0.3, 0.0)
                discordant_t = rng.uniform(-0.3, -0.1)
        elif pathway == 3: # Anteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.05, 0.2)
                r_a = rng.uniform(0.6, 1.4)
                s_a = rng.uniform(-0.4, -0.1)
                discordant_t = rng.uniform(-0.2, 0.1)
            else: # Lead II
                delta_a = rng.uniform(0.05, 0.25)
                r_a = rng.uniform(0.6, 1.5)
                s_a = rng.uniform(-0.4, -0.1)
                discordant_t = rng.uniform(-0.2, 0.1)
                
        # Respiratory modulation (amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for beat_idx, r_time in enumerate(r_peaks):
            bp = beat_params[beat_idx]
            
            # Blend normal T wave with discordant T wave based on pre-excitation severity
            b_t_a = normal_t_a * (1 - bp['pre_ex']) + discordant_t * bp['pre_ex']
            
            # Mask normal Q wave when pre-excitation is present
            b_q_a = q_a * (1 - bp['pre_ex'])
            
            # ST segment shift (discordant to QRS vector)
            st_shift = discordant_t * 0.3 * bp['pre_ex']
            st_mu = bp['s_mu'] + 0.04
            
            lead_signal += gaussian(t, p_a * bp['beat_var'], r_time + bp['p_mu'], p_sigma)
            lead_signal += gaussian(t, b_q_a * bp['beat_var'], r_time + q_mu, q_sigma)
            lead_signal += gaussian(t, delta_a * bp['delta_scale'] * bp['beat_var'], r_time + bp['delta_mu'], bp['delta_sigma'])
            lead_signal += gaussian(t, r_a * bp['beat_var'], r_time + r_mu, bp['r_sigma'])
            lead_signal += gaussian(t, s_a * bp['beat_var'], r_time + bp['s_mu'], s_sigma)
            lead_signal += gaussian(t, st_shift * bp['beat_var'], r_time + st_mu, st_sigma)
            lead_signal += gaussian(t, b_t_a * bp['beat_var'], r_time + t_mu, t_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise (realistic amplitudes)
        bw_freq1 = rng.uniform(0.1, 0.3)
        bw_freq2 = rng.uniform(0.02, 0.08)
        baseline = rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * bw_freq1 * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * bw_freq2 * t + rng.uniform(0, 2*np.pi))
        
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.001, 0.005) * np.sin(2 * np.pi * powerline_freq * t)
        
        emg = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    return signal

def main():
    # Configuration
    output_dir = "artifacts/wpw_ecg/visual_representation_100iter/generated/"
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