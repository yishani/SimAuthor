import os
import numpy as np
from scipy.signal import resample_poly

def gaussian(x, a, mu, sigma):
    """Evaluates a standard Gaussian function."""
    return a * np.exp(-0.5 * ((x - mu) / sigma)**2)

def asym_gaussian(x, a, mu, sigma_left, sigma_right):
    """Evaluates an asymmetric Gaussian function to model skewed waves (P, T)."""
    sigma_x = np.where(x < mu, sigma_left, sigma_right)
    return a * np.exp(-0.5 * ((x - mu) / sigma_x)**2)

def generate_wpw_ecg(seed, fs_internal=44100, fs_out=16000, duration=10.0):
    """
    Generates a synthetic 2-lead ECG signal exhibiting Wolff-Parkinson-White (WPW) Syndrome.
    Synthesizes at a high internal sampling rate to capture sharp R-wave peaks, then downsamples.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs_internal)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to create diverse manifestations
    # 0: Left lateral (Positive delta in I and II)
    # 1: Posteroseptal (Positive in I, Negative delta in II - pseudo-inferior infarct)
    # 2: Right lateral (Negative/flat delta in I, Positive in II)
    pathway = rng.choice([0, 1, 2])
    
    # Heart rate and HRV (broadened for more diversity)
    hr = rng.uniform(50, 110)
    rr_mean = 60.0 / hr
    
    # Generate RR intervals with natural variability (RSA)
    n_beats = int(duration / rr_mean) + 5
    rr_intervals = rng.normal(rr_mean, rng.uniform(0.02, 0.08), n_beats)
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # WPW Hallmark: Short PR interval
    pr_interval = rng.uniform(0.08, 0.12)
    p_mu = -pr_interval
    
    for lead_idx in range(2):
        # Base parameters for normal-ish conduction (Amplitude, Center, Width)
        # Centers are relative to the R peak (0.0)
        p_a = rng.uniform(0.05, 0.20)
        p_sigma_l = rng.uniform(0.015, 0.025)
        p_sigma_r = rng.uniform(0.015, 0.03)
        
        q_a = rng.uniform(-0.2, 0.0)
        q_mu = rng.uniform(-0.02, -0.01)
        q_sigma = rng.uniform(0.005, 0.01)
        
        r_a = rng.uniform(0.6, 2.2)
        r_mu = 0.0
        r_sigma = rng.uniform(0.008, 0.014) # Sharper R wave
        
        s_a = rng.uniform(-0.5, -0.05)
        s_mu = rng.uniform(0.02, 0.04)
        s_sigma = rng.uniform(0.01, 0.02)
        
        # ST segment to model discordance smoothly
        st_a = rng.uniform(-0.05, 0.05)
        st_mu = rng.uniform(0.08, 0.12)
        st_sigma = rng.uniform(0.03, 0.06)
        
        t_a = rng.uniform(0.1, 0.5)
        t_mu = rng.uniform(0.20, 0.28)
        t_sigma_l = rng.uniform(0.03, 0.05)
        t_sigma_r = rng.uniform(0.04, 0.07)
        
        # WPW Hallmark: Delta wave
        # Positioned to overlap heavily with the R wave for a slurred upstroke
        delta_mu = rng.uniform(-0.04, -0.02)
        delta_sigma = rng.uniform(0.015, 0.03)
        
        # Adjust morphology based on the accessory pathway location
        if pathway == 0: # Left lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.15, 0.45)
                t_a = rng.uniform(-0.4, -0.05) # Discordant T wave inversion
                st_a = rng.uniform(-0.15, -0.02)
            else: # Lead II
                delta_a = rng.uniform(0.2, 0.5)
                t_a = rng.uniform(-0.25, 0.1)
                st_a = rng.uniform(-0.1, 0.02)
        elif pathway == 1: # Posteroseptal
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(0.1, 0.3)
                st_a = rng.uniform(-0.05, 0.05)
                t_a = rng.uniform(0.1, 0.3)
            else: # Lead II
                delta_a = rng.uniform(-0.6, -0.2) # Negative delta wave
                r_a = rng.uniform(0.1, 0.6)       # Attenuated R wave
                s_a = rng.uniform(-1.0, -0.3)     # Deep S wave
                t_a = rng.uniform(0.2, 0.6)       # Discordant positive T wave
                st_a = rng.uniform(0.05, 0.2)
        elif pathway == 2: # Right lateral
            if lead_idx == 0: # Lead I
                delta_a = rng.uniform(-0.3, -0.05)
                r_a = rng.uniform(0.3, 1.0)
                t_a = rng.uniform(0.1, 0.4)
                st_a = rng.uniform(0.02, 0.1)
            else: # Lead II
                delta_a = rng.uniform(0.15, 0.4)
                t_a = rng.uniform(-0.2, 0.1)
                st_a = rng.uniform(-0.08, 0.02)
                
        # Respiratory modulation (baseline wander and amplitude modulation)
        resp_freq = rng.uniform(0.2, 0.35)
        resp_mod = 1.0 + rng.uniform(0.02, 0.1) * np.sin(2 * np.pi * resp_freq * t)
        
        lead_signal = np.zeros_like(t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Add beat-to-beat morphological variation
            beat_var = rng.uniform(0.9, 1.1)
            
            lead_signal += asym_gaussian(t, p_a * beat_var, r_time + p_mu, p_sigma_l, p_sigma_r)
            lead_signal += gaussian(t, delta_a * beat_var, r_time + delta_mu, delta_sigma)
            
            # If delta is positive, it typically obscures the Q wave
            if not (delta_a > 0 and q_a < 0):
                lead_signal += gaussian(t, q_a * beat_var, r_time + q_mu, q_sigma)
                
            lead_signal += gaussian(t, r_a * beat_var, r_time + r_mu, r_sigma)
            lead_signal += gaussian(t, s_a * beat_var, r_time + s_mu, s_sigma)
            lead_signal += gaussian(t, st_a * beat_var, r_time + st_mu, st_sigma)
            lead_signal += asym_gaussian(t, t_a * beat_var, r_time + t_mu, t_sigma_l, t_sigma_r)
            
        # Apply amplitude modulation from respiration
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Complex baseline wander
        baseline = rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_freq * t)
        for _ in range(4):
            f = rng.uniform(0.01, 0.3)
            a = rng.uniform(0.02, 0.15)
            phase = rng.uniform(0, 2 * np.pi)
            baseline += a * np.sin(2 * np.pi * f * t + phase)
            
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.002, 0.01) * np.sin(2 * np.pi * powerline_freq * t)
        
        # 3. High frequency muscle artifact (EMG)
        emg = rng.normal(0, rng.uniform(0.005, 0.02), len(t))
        
        lead_signal += baseline + powerline + emg
        signal[:, lead_idx] = lead_signal
        
    # Resample to output sampling rate (16000 Hz)
    # 44100 to 16000 is a ratio of 160 / 441
    num_samples_out = int(duration * fs_out)
    signal_out = np.zeros((num_samples_out, 2))
    signal_out[:, 0] = resample_poly(signal[:, 0], 160, 441)
    signal_out[:, 1] = resample_poly(signal[:, 1], 160, 441)
    
    return signal_out

def main():
    # Configuration
    output_dir = "artifacts/wpw_ecg/visual_representation_100iter/generated/"
    n_samples = 100
    fs_out = 16000
    fs_internal = 44100
    duration = 10.0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate and save samples
    for i in range(n_samples):
        # Use a deterministic seed for each sample for reproducibility
        seed = 42 + i
        
        # Generate WPW ECG signal
        signal = generate_wpw_ecg(seed=seed, fs_internal=fs_internal, fs_out=fs_out, duration=duration)
        
        # Save to .npy file
        filename = f"wpw_ecg_{i:03d}.npy"
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, signal)

if __name__ == "__main__":
    main()