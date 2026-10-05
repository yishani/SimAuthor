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
    2. Shortened PR interval (80-110 ms) due to accessory pathway bypass.
    3. Delta wave (slurred upstroke of QRS) due to early ventricular pre-excitation.
    4. Widened QRS complex (fusion of pre-excitation and normal conduction).
    5. Secondary ST-T wave abnormalities (discordant to the delta/QRS vector).
    6. Intermittent WPW with beat-to-beat variability in pre-excitation.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to create diverse manifestations
    # 0: Left lateral (Positive delta in I and II)
    # 1: Posteroseptal (Positive in I, Negative delta in II - pseudo-inferior infarct)
    # 2: Right lateral (Negative/flat delta in I, Positive in II)
    pathway = rng.choice([0, 1, 2])
    
    # Heart rate and HRV
    hr = rng.uniform(60, 95)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with slight natural variability
    rr_intervals = rng.normal(rr_mean, 0.02, int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Global physiological parameters defining the severity of the condition
    # Expanded range to increase inter-sample diversity
    pre_excitation_base = rng.uniform(0.1, 1.0)
    is_intermittent = rng.random() < 0.25 # Increased chance of intermittent WPW
    
    p_sigma = rng.uniform(0.015, 0.025)
    r_mu = 0.0
    r_sigma_base = rng.uniform(0.008, 0.015)
    s_mu_base = rng.uniform(0.015, 0.030)
    s_sigma = rng.uniform(0.010, 0.020)
    t_mu = rng.uniform(0.20, 0.30)
    t_sigma = rng.uniform(0.035, 0.060)
    st_sigma = rng.uniform(0.04, 0.08)
    
    # Pre-calculate synchronized beat parameters across leads to ensure physiological validity
    beat_params = []
    for _ in r_peaks:
        if is_intermittent and rng.random() < 0.4:
            # Sudden drop in pre-excitation (normal or near-normal conduction)
            beat_pre_ex = rng.uniform(0.0, 0.15)
        else:
            # Normal variation around base severity
            beat_pre_ex = np.clip(pre_excitation_base + rng.uniform(-0.1, 0.1), 0.0, 1.0)
            
        # High pre-excitation -> shorter PR, wider delta, wider R, wider QRS
        b_pr_interval = rng.uniform(0.09, 0.14) - 0.03 * beat_pre_ex
        b_delta_mu = -0.02 - 0.04 * beat_pre_ex
        b_delta_sigma = 0.015 + 0.015 * beat_pre_ex
        b_r_sigma = r_sigma_base + 0.01 * beat_pre_ex
        b_s_mu = s_mu_base + 0.015 * beat_pre_ex
        
        # Calculate P wave position to strictly maintain the PR interval (onset to onset)
        b_p_mu = b_delta_mu - 2*b_delta_sigma + 2*p_sigma - b_pr_interval
        
        # Scale delta wave amplitude based on pre-excitation degree
        b_delta_scale = 0.1 + 0.9 * beat_pre_ex
        
        beat_params.append({
            'pre_ex': beat_pre_ex,
            'p_mu': b_p_mu,
            'delta_mu': b_delta_mu,
            'delta_sigma': b_delta_sigma,
            'r_sigma': b_r_sigma,
            's_mu': b_s_mu,
            'delta_scale': b_delta_scale,
            'beat_var': rng.uniform(0.92, 1.08)
        })
        
    for lead_idx in range(2):
        p_a = rng.uniform(0.05, 0.2)
        normal_t_a = rng.uniform(0.1, 0.4)
        
        # Continuous axis shift to increase morphological diversity between samples
        axis_shift = rng.uniform(-0.3, 0.3)
        
        # Adjust morphology based on the accessory pathway location with expanded ranges
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.8, 2.5) * (1.0 + axis_shift)
                s_a = rng.uniform(-0.6, 0.0)
                discordant_t = rng.uniform(-0.5, -0.1)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.5)
                r_a = rng.uniform(0.8, 2.5) * (1.0 - axis_shift)
                s_a = rng.uniform(-0.6, 0.0)
                discordant_t = rng.uniform(-0.5, -0.1)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.05, 0.3)
                r_a = rng.uniform(0.5, 2.0) * (1.0 + axis_shift)
                s_a = rng.uniform(-0.8, -0.1)
                discordant_t = rng.uniform(-0.4, 0.1)
            else: # Lead II
                delta_a = rng.uniform(-0.5, -0.1) # Negative delta wave
                r_a = rng.uniform(0.1, 0.8) * (1.0 - axis_shift)
                s_a = rng.uniform(-1.5, -0.3)
                discordant_t = rng.uniform(0.1, 0.6) # Discordant to negative QRS is positive
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.4, -0.05)
                r_a = rng.uniform(0.1, 1.0) * (1.0 + axis_shift)
                s_a = rng.uniform(-1.2, -0.2)
                discordant_t = rng.uniform(0.1, 0.5)
            else: # Lead II
                delta_a = rng.uniform(0.1, 0.4)
                r_a = rng.uniform(0.8, 2.2) * (1.0 - axis_shift)
                s_a = rng.uniform(-0.6, 0.0)
                discordant_t = rng.uniform(-0.4, 0.1)
                
        # Lead-specific morphological nuances
        t_mu_lead = t_mu + rng.uniform(-0.02, 0.02)
        t_sigma_lead = t_sigma * rng.uniform(0.8, 1.2)
        p_sigma_lead = p_sigma * rng.uniform(0.8, 1.2)
        
        # Fixed offsets for QRS sharpening to prevent unnatural beat-to-beat jitter
        r_notch_offset = rng.uniform(-0.005, 0.005)
        s_notch_offset = rng.uniform(-0.005, 0.005)
        st_shift_factor = rng.uniform(0.15, 0.35)
        st_mu_offset = rng.uniform(0.04, 0.08)
        
        # Respiratory modulation (amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for beat_idx, r_time in enumerate(r_peaks):
            bp = beat_params[beat_idx]
            
            # Blend normal T wave with discordant T wave based on pre-excitation severity
            b_t_a = normal_t_a * (1 - bp['pre_ex']) + discordant_t * bp['pre_ex']
            
            # ST segment shift (discordant to QRS vector)
            st_shift = discordant_t * st_shift_factor * bp['pre_ex']
            st_mu = bp['s_mu'] + st_mu_offset
            
            lead_signal += gaussian(t, p_a * bp['beat_var'], r_time + bp['p_mu'], p_sigma_lead)
            lead_signal += gaussian(t, delta_a * bp['delta_scale'] * bp['beat_var'], r_time + bp['delta_mu'], bp['delta_sigma'])
            
            # Main R wave and sharpening notch
            lead_signal += gaussian(t, r_a * bp['beat_var'], r_time + r_mu, bp['r_sigma'])
            lead_signal += gaussian(t, r_a * 0.3 * bp['beat_var'], r_time + r_mu + r_notch_offset, bp['r_sigma'] * 0.4)
            
            # Main S wave and sharpening notch
            lead_signal += gaussian(t, s_a * bp['beat_var'], r_time + bp['s_mu'], s_sigma)
            lead_signal += gaussian(t, s_a * 0.3 * bp['beat_var'], r_time + bp['s_mu'] + s_notch_offset, s_sigma * 0.5)
            
            lead_signal += gaussian(t, st_shift * bp['beat_var'], r_time + st_mu, st_sigma)
            lead_signal += gaussian(t, b_t_a * bp['beat_var'], r_time + t_mu_lead, t_sigma_lead)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add diverse physiological and environmental noise
        bw_freq1 = rng.uniform(0.05, 0.2)
        bw_freq2 = rng.uniform(0.2, 0.5)
        bw_freq3 = rng.uniform(0.01, 0.05)
        baseline = rng.uniform(0.1, 0.5) * np.sin(2 * np.pi * bw_freq1 * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.05, 0.2) * np.sin(2 * np.pi * bw_freq2 * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * bw_freq3 * t + rng.uniform(0, 2*np.pi))
        
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.0, 0.02) * np.sin(2 * np.pi * powerline_freq * t + rng.uniform(0, 2*np.pi))
        
        emg = rng.normal(0, rng.uniform(0.005, 0.03), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    # Apply global scale for inter-patient amplitude variability (addresses spread ratio discrepancy)
    global_scale = rng.uniform(0.5, 1.5)
    signal *= global_scale
        
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