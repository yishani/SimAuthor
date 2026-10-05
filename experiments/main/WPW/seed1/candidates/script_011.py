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
    1. Continuous 3D-like vector projection (independent R and S axes) for infinite morphological diversity.
    2. Variable degree of pre-excitation coupling PR interval, delta wave, and QRS width.
    3. Shortened PR interval (80-110 ms) due to accessory pathway bypass.
    4. Delta wave (slurred upstroke of QRS) seamlessly blended with the R-wave to avoid notching.
    5. Widened QRS complex (fusion of pre-excitation and normal conduction).
    6. Secondary ST-T wave abnormalities (strictly discordant to the main QRS vector).
    7. Intermittent WPW with beat-to-beat variability and axis normalization.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Heart rate and HRV
    hr = rng.uniform(55, 95)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with slight natural variability
    rr_intervals = rng.normal(rr_mean, 0.02, int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Bazett's formula for approximate T-wave placement based on heart rate
    qt_mean = 0.38 * np.sqrt(rr_mean)
    t_mu_base = qt_mean - 0.12 # Target ~0.25s for 60bpm
    
    # Global physiological parameters defining the severity of the condition
    pre_excitation_base = rng.uniform(0.3, 1.0)
    is_intermittent = rng.random() < 0.15 # 15% chance of intermittent WPW
    
    # Vector axes (in radians) for continuous morphological diversity
    # Delta axis: -90 to 150 degrees covers various accessory pathway locations
    delta_axis = np.deg2rad(rng.uniform(-90, 150))
    
    # QRS (R and S) axes are influenced by the pre-excitation vector
    wpw_r_axis = delta_axis + np.deg2rad(rng.uniform(-30, 30))
    wpw_s_axis = wpw_r_axis + np.pi + np.deg2rad(rng.uniform(-60, 60))
    
    # T-wave axis is strictly discordant (opposite) to the main QRS vector in WPW
    wpw_t_axis = wpw_r_axis + np.pi + np.deg2rad(rng.uniform(-30, 30))
    
    # Normal axes for when pre-excitation is absent (intermittent WPW)
    normal_r_axis = np.deg2rad(rng.uniform(-30, 90))
    normal_s_axis = normal_r_axis + np.pi + np.deg2rad(rng.uniform(-45, 45))
    normal_t_axis = normal_r_axis + np.deg2rad(rng.uniform(-20, 20))
    
    # P-wave axis is typically normal (15 to 75 degrees)
    p_axis = np.deg2rad(rng.uniform(15, 75))
    
    # Global vector magnitudes
    delta_mag = rng.uniform(0.3, 0.7)
    qrs_mag = rng.uniform(0.8, 2.2)
    s_mag = qrs_mag * rng.uniform(0.1, 0.5)
    t_mag = rng.uniform(0.2, 0.6)
    p_mag = rng.uniform(0.1, 0.25)
    
    # Temporal width parameters
    p_sigma = rng.uniform(0.020, 0.025)
    r_mu = 0.0
    r_sigma_base = rng.uniform(0.010, 0.015)
    s_sigma = rng.uniform(0.012, 0.020)
    t_mu = t_mu_base + rng.uniform(-0.02, 0.02)
    t_sigma = rng.uniform(0.040, 0.060)
    st_sigma = rng.uniform(0.04, 0.06)
    
    # Pre-calculate synchronized beat parameters across leads to ensure physiological validity
    beat_params = []
    for _ in r_peaks:
        if is_intermittent and rng.random() < 0.3:
            # Sudden drop in pre-excitation (normal or near-normal conduction)
            beat_pre_ex = rng.uniform(0.0, 0.1)
        else:
            # Normal variation around base severity
            beat_pre_ex = np.clip(pre_excitation_base + rng.uniform(-0.05, 0.05), 0.0, 1.0)
            
        # High pre-excitation -> shorter PR, wider delta, wider R, wider QRS
        b_pr_interval = rng.uniform(0.08, 0.11) + 0.04 * (1 - beat_pre_ex) # 80-110ms in WPW, up to 150ms normal
        
        # Delta mu is positioned close to R mu (0.0) to ensure a smooth slur and avoid notching
        b_delta_mu = rng.uniform(-0.045, -0.030)
        b_delta_sigma = rng.uniform(0.020, 0.030)
        b_r_sigma = r_sigma_base + 0.005 * beat_pre_ex
        b_s_mu = rng.uniform(0.02, 0.035)
        
        # Calculate P wave position to strictly maintain the PR interval (onset to onset)
        b_p_mu = b_delta_mu - 2*b_delta_sigma + 2*p_sigma - b_pr_interval
        
        # Interpolate axes based on the degree of pre-excitation (for intermittent WPW)
        b_r_axis = normal_r_axis * (1 - beat_pre_ex) + wpw_r_axis * beat_pre_ex
        b_s_axis = normal_s_axis * (1 - beat_pre_ex) + wpw_s_axis * beat_pre_ex
        b_t_axis = normal_t_axis * (1 - beat_pre_ex) + wpw_t_axis * beat_pre_ex
        
        beat_params.append({
            'pre_ex': beat_pre_ex,
            'p_mu': b_p_mu,
            'delta_mu': b_delta_mu,
            'delta_sigma': b_delta_sigma,
            'r_sigma': b_r_sigma,
            's_mu': b_s_mu,
            'r_axis': b_r_axis,
            's_axis': b_s_axis,
            't_axis': b_t_axis,
            'beat_var': rng.uniform(0.95, 1.05)
        })
        
    for lead_idx in range(2):
        # Lead I is at 0 degrees, Lead II is at 60 degrees
        lead_angle = 0.0 if lead_idx == 0 else np.deg2rad(60.0)
        
        # Project P-wave vector to the current lead
        p_a = p_mag * np.cos(p_axis - lead_angle)
        
        # Respiratory modulation (amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.05, 0.12) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for beat_idx, r_time in enumerate(r_peaks):
            bp = beat_params[beat_idx]
            
            # Project Delta, R, S, and T vectors to the current lead
            delta_a = delta_mag * np.cos(delta_axis - lead_angle) * bp['pre_ex']
            r_a = qrs_mag * np.cos(bp['r_axis'] - lead_angle)
            s_a = s_mag * np.cos(bp['s_axis'] - lead_angle)
            t_a = t_mag * np.cos(bp['t_axis'] - lead_angle)
            
            # ST segment shift (discordant to main QRS vector)
            st_shift = -0.12 * r_a * bp['pre_ex']
            st_mu = bp['s_mu'] + 0.05
            
            # Construct the waveform
            lead_signal += gaussian(t, p_a * bp['beat_var'], r_time + bp['p_mu'], p_sigma)
            lead_signal += gaussian(t, delta_a * bp['beat_var'], r_time + bp['delta_mu'], bp['delta_sigma'])
            lead_signal += gaussian(t, r_a * bp['beat_var'], r_time + r_mu, bp['r_sigma'])
            lead_signal += gaussian(t, s_a * bp['beat_var'], r_time + bp['s_mu'], s_sigma)
            lead_signal += gaussian(t, st_shift * bp['beat_var'], r_time + st_mu, st_sigma)
            lead_signal += gaussian(t, t_a * bp['beat_var'], r_time + t_mu, t_sigma)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add realistic organic baseline wander (sum of low-frequency sines)
        baseline = np.zeros_like(t)
        for _ in range(rng.integers(3, 6)):
            freq = rng.uniform(0.02, 0.4)
            amp = rng.uniform(0.02, 0.1)
            phase = rng.uniform(0, 2*np.pi)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add powerline interference and EMG noise
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