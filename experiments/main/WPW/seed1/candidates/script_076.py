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
    1. Constrained vector axes to typical left/right free wall pathways to ensure positive 
       delta and R waves in Leads I and II, matching clinical blueprints and real samples.
    2. Improved delta wave blending parameters to create a smoother, more realistic slur.
    3. Dynamically calculated apparent QRS onset to strictly maintain the correct PR interval 
       regardless of the degree of pre-excitation (manifest vs. concealed/intermittent).
    4. Adjusted T-wave discordance angles to allow for realistic flat or moderately inverted 
       T-waves rather than strictly opposite (180-degree) vectors.
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
    pre_excitation_base = rng.uniform(0.4, 1.0)
    is_intermittent = rng.random() < 0.15 # 15% chance of intermittent WPW
    
    # Vector axes (in radians) for continuous morphological diversity
    # Constrain to typical pathways yielding positive delta/R in Leads I and II (10 to 70 deg)
    delta_axis = np.deg2rad(rng.uniform(10, 70))
    
    # QRS (R and S) axes are influenced by the pre-excitation vector
    wpw_r_axis = delta_axis + np.deg2rad(rng.uniform(-15, 15))
    wpw_s_axis = wpw_r_axis + np.pi + np.deg2rad(rng.uniform(-20, 20))
    
    # T-wave axis: discordant but not necessarily strictly inverted. 
    # 90 to 140 degrees away creates flat or moderately inverted T waves.
    discordance_angle = np.deg2rad(rng.uniform(90, 140)) * rng.choice([-1, 1])
    wpw_t_axis = wpw_r_axis + discordance_angle
    
    # Normal axes for when pre-excitation is absent (intermittent WPW)
    normal_r_axis = np.deg2rad(rng.uniform(10, 70))
    normal_s_axis = normal_r_axis + np.pi + np.deg2rad(rng.uniform(-20, 20))
    normal_t_axis = normal_r_axis + np.deg2rad(rng.uniform(-20, 20))
    
    # P-wave axis is typically normal (15 to 75 degrees)
    p_axis = np.deg2rad(rng.uniform(15, 75))
    
    # Global vector magnitudes
    delta_mag = rng.uniform(0.2, 0.5)
    qrs_mag = rng.uniform(1.2, 2.5)
    s_mag = qrs_mag * rng.uniform(0.05, 0.15)
    t_mag = rng.uniform(0.15, 0.35)
    p_mag = rng.uniform(0.1, 0.2)
    
    # Temporal width parameters
    p_sigma = rng.uniform(0.020, 0.025)
    r_mu = 0.0
    r_sigma_base = rng.uniform(0.010, 0.015)
    s_sigma = rng.uniform(0.012, 0.020)
    t_mu = t_mu_base + rng.uniform(-0.02, 0.02)
    t_sigma = rng.uniform(0.040, 0.050)
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
        b_delta_mu = rng.uniform(-0.035, -0.020)
        b_delta_sigma = rng.uniform(0.025, 0.035)
        b_r_sigma = r_sigma_base + 0.002 * beat_pre_ex
        b_s_mu = rng.uniform(0.02, 0.035)
        
        # Apparent QRS onset blends from R onset (normal) to Delta onset (WPW)
        normal_qrs_onset = -2 * b_r_sigma
        wpw_qrs_onset = b_delta_mu - 2 * b_delta_sigma
        apparent_qrs_onset = normal_qrs_onset * (1 - beat_pre_ex) + wpw_qrs_onset * beat_pre_ex
        
        # Calculate P wave position to strictly maintain the PR interval (onset to onset)
        b_p_mu = apparent_qrs_onset + 2*p_sigma - b_pr_interval
        
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
            st_shift = -0.05 * r_a * bp['pre_ex']
            st_mu = bp['s_mu'] + 0.04
            
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