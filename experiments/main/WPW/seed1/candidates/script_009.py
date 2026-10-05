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
    1. Continuous 2D vector projection for infinite morphological diversity.
    2. Variable degree of pre-excitation coupling PR interval, delta wave, and QRS width.
    3. Shortened PR interval (< 120 ms) due to accessory pathway bypass.
    4. Delta wave (slurred upstroke of QRS) seamlessly blended with the R-wave.
    5. Widened QRS complex (> 120 ms) due to fusion of pre-excitation and normal conduction.
    6. Secondary ST-T wave abnormalities (strictly discordant to the QRS vector).
    7. Intermittent WPW with beat-to-beat variability.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Heart rate and HRV
    hr = rng.uniform(55, 105)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with slight natural variability (RSA)
    rr_intervals = rng.normal(rr_mean, 0.02, int(duration / rr_mean) + 5)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Bazett's formula for approximate T-wave placement based on heart rate
    qt_mean = 0.38 * np.sqrt(rr_mean)
    t_mu_base = qt_mean - 0.08
    
    # Global physiological parameters defining the severity of the condition
    pre_excitation_base = rng.uniform(0.4, 1.0)
    is_intermittent = rng.random() < 0.15 # 15% chance of intermittent WPW
    
    # Vector axes (in radians) for continuous morphological diversity
    # Delta axis: -30 to 90 degrees ensures positive/flat delta in Leads I and II (typical WPW)
    delta_axis = np.deg2rad(rng.uniform(-30, 90))
    
    # QRS axis is influenced by the pre-excitation vector
    wpw_qrs_axis = delta_axis + np.deg2rad(rng.uniform(-15, 15))
    
    # T-wave axis is strictly discordant (opposite) to the QRS vector in WPW
    wpw_t_axis = wpw_qrs_axis + np.pi + np.deg2rad(rng.uniform(-20, 20))
    
    # Normal axes for when pre-excitation is absent (intermittent WPW)
    normal_qrs_axis = np.deg2rad(rng.uniform(10, 70))
    normal_t_axis = normal_qrs_axis + np.deg2rad(rng.uniform(-15, 15))
    
    # P-wave axis is typically normal (30 to 60 degrees)
    p_axis = np.deg2rad(rng.uniform(30, 60))
    
    # Global vector magnitudes
    delta_mag = rng.uniform(0.2, 0.5)
    qrs_mag = rng.uniform(1.0, 2.2)
    t_mag = rng.uniform(0.2, 0.5)
    p_mag = rng.uniform(0.1, 0.2)
    
    # Temporal width parameters
    p_sigma = rng.uniform(0.020, 0.025)
    r_mu = 0.0
    r_sigma_base = rng.uniform(0.012, 0.016)
    s_mu_base = rng.uniform(0.020, 0.035)
    s_sigma = rng.uniform(0.015, 0.020)
    t_mu = t_mu_base + rng.uniform(-0.02, 0.02)
    t_sigma = rng.uniform(0.040, 0.060)
    st_sigma = rng.uniform(0.04, 0.06)
    
    # Pre-calculate synchronized beat parameters across leads to ensure physiological validity
    beat_params = []
    for _ in r_peaks:
        if is_intermittent and rng.random() < 0.2:
            # Sudden drop in pre-excitation (normal or near-normal conduction)
            beat_pre_ex = rng.uniform(0.0, 0.1)
        else:
            # Normal variation around base severity
            beat_pre_ex = np.clip(pre_excitation_base + rng.uniform(-0.1, 0.1), 0.0, 1.0)
            
        # High pre-excitation -> shorter PR, wider delta, wider R, wider QRS
        b_pr_interval = rng.uniform(0.12, 0.16) - 0.05 * beat_pre_ex
        
        # Delta wave positioning to create a slurred upstroke that fuses with the R wave
        b_delta_mu = rng.uniform(-0.03, -0.02) - 0.03 * beat_pre_ex
        b_delta_sigma = rng.uniform(0.010, 0.015) + 0.015 * beat_pre_ex
        
        b_r_sigma = r_sigma_base + 0.005 * beat_pre_ex
        b_s_mu = s_mu_base + 0.010 * beat_pre_ex
        
        # Calculate P wave position to strictly maintain the PR interval (onset to onset)
        b_p_mu = b_delta_mu - 2*b_delta_sigma + 2*p_sigma - b_pr_interval
        
        # Interpolate axes based on the degree of pre-excitation (for intermittent WPW)
        b_qrs_axis = normal_qrs_axis * (1 - beat_pre_ex) + wpw_qrs_axis * beat_pre_ex
        b_t_axis = normal_t_axis * (1 - beat_pre_ex) + wpw_t_axis * beat_pre_ex
        
        beat_params.append({
            'pre_ex': beat_pre_ex,
            'p_mu': b_p_mu,
            'delta_mu': b_delta_mu,
            'delta_sigma': b_delta_sigma,
            'r_sigma': b_r_sigma,
            's_mu': b_s_mu,
            'qrs_axis': b_qrs_axis,
            't_axis': b_t_axis,
            'beat_var': rng.uniform(0.95, 1.05)
        })
        
    for lead_idx in range(2):
        # Lead I is at 0 degrees, Lead II is at 60 degrees
        lead_angle = 0.0 if lead_idx == 0 else np.deg2rad(60.0)
        
        # Project P-wave vector to the current lead
        p_a = p_mag * np.cos(p_axis - lead_angle)
        
        # Define minor QRS components for this lead
        small_s_ratio = rng.uniform(-0.15, -0.02)
        small_r_ratio = rng.uniform(0.05, 0.15)
        
        # Respiratory modulation (amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.05, 0.10) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for beat_idx, r_time in enumerate(r_peaks):
            bp = beat_params[beat_idx]
            
            # Project Delta, QRS, and T vectors to the current lead
            delta_a = delta_mag * np.cos(delta_axis - lead_angle) * bp['pre_ex']
            qrs_proj = qrs_mag * np.cos(bp['qrs_axis'] - lead_angle)
            t_a = t_mag * np.cos(bp['t_axis'] - lead_angle)
            
            # Determine main QRS morphology based on projection
            if qrs_proj > 0:
                r_a = qrs_proj
                s_a = small_s_ratio * qrs_mag
            else:
                r_a = small_r_ratio * qrs_mag
                s_a = qrs_proj
            
            # ST segment shift (discordant to QRS vector)
            st_shift = -0.08 * qrs_proj * bp['pre_ex']
            st_mu = bp['s_mu'] + 0.06
            
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
        for _ in range(rng.integers(2, 5)):
            freq = rng.uniform(0.05, 0.3)
            amp = rng.uniform(0.02, 0.10)
            phase = rng.uniform(0, 2*np.pi)
            baseline += amp * np.sin(2 * np.pi * freq * t + phase)
        
        # Add powerline interference and EMG noise (kept low for clean clinical look)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.001, 0.005) * np.sin(2 * np.pi * powerline_freq * t)
        emg = rng.normal(0, rng.uniform(0.002, 0.008), len(t))
        
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