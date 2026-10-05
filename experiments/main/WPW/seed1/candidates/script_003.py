import os
import numpy as np

def gaussian(x, a, mu, sigma):
    """Evaluates a standard Gaussian function."""
    return a * np.exp(-0.5 * ((x - mu) / max(sigma, 1e-5))**2)

def asym_gaussian(x, a, mu, sigma1, sigma2):
    """Evaluates an asymmetric Gaussian function (different left/right widths)."""
    y = np.zeros_like(x)
    mask1 = x < mu
    mask2 = ~mask1
    y[mask1] = a * np.exp(-0.5 * ((x[mask1] - mu) / max(sigma1, 1e-5))**2)
    y[mask2] = a * np.exp(-0.5 * ((x[mask2] - mu) / max(sigma2, 1e-5))**2)
    return y

def generate_wpw_ecg(seed, fs=500, duration=10.0):
    """
    Generates a synthetic 2-lead ECG signal exhibiting Wolff-Parkinson-White (WPW) Syndrome.
    
    Refinement: Replaced discrete pathway categories with a continuous 2D vector-based 
    cardiac axis model. This massively increases morphological diversity (addressing low 
    spread ratio) while guaranteeing physiological consistency between Lead I and Lead II.
    Also strictly enforces the onset-to-onset PR interval and adds ST-segment discordance.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Heart rate and HRV with Respiratory Sinus Arrhythmia (RSA)
    hr = rng.uniform(60, 95)
    rr_mean = 60.0 / hr
    rsa_freq = rng.uniform(0.2, 0.35)
    rsa_amp = rng.uniform(0.02, 0.08)
    
    # Generate RR intervals
    n_beats = int(duration / rr_mean) + 10
    rr_intervals = rr_mean + rsa_amp * np.sin(2 * np.pi * rsa_freq * np.arange(n_beats))
    rr_intervals += rng.normal(0, 0.015, n_beats) # Random jitter
    
    r_peaks = np.cumsum(rr_intervals)
    r_peaks -= rng.uniform(0.2, rr_mean - 0.2)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Temporal parameters (consistent across leads for physiological synchrony)
    # Delta wave overlaps heavily with R-wave for the classic "slurred upstroke" fusion
    delta_mu = rng.uniform(-0.04, -0.02)
    delta_sigma = rng.uniform(0.02, 0.03)
    
    r_mu = 0.0
    r_sigma = rng.uniform(0.01, 0.015)
    
    s_mu = rng.uniform(0.02, 0.04)
    s_sigma = rng.uniform(0.015, 0.025)
    
    p_sigma = rng.uniform(0.015, 0.02)
    # Strictly enforce PR interval (onset to onset) between 80 and 110 ms
    pr_interval = rng.uniform(0.08, 0.11)
    p_mu = (delta_mu - 2*delta_sigma) - pr_interval + 2*p_sigma
    
    st_mu = rng.uniform(0.08, 0.12)
    st_sigma = rng.uniform(0.03, 0.05)
    
    t_mu = rng.uniform(0.20, 0.28)
    t_sigma1 = rng.uniform(0.04, 0.06) # Slower upstroke
    t_sigma2 = rng.uniform(0.02, 0.04) # Faster downstroke
    
    # Vector-based amplitude generation for continuous diversity
    p_axis = np.radians(rng.uniform(0, 75))
    delta_axis = np.radians(rng.uniform(-60, 120))
    r_axis = np.radians(rng.uniform(-15, 75))
    s_axis = r_axis + np.radians(rng.uniform(120, 240))
    # ST and T wave axes discordant to delta/R vector (Rule of Discordance)
    st_axis = r_axis + np.radians(rng.uniform(150, 210))
    t_axis = r_axis + np.radians(rng.uniform(135, 225))
    
    p_mag = rng.uniform(0.1, 0.2)
    delta_mag = rng.uniform(0.2, 0.7)
    r_mag = rng.uniform(1.0, 2.5)
    s_mag = rng.uniform(0.1, 0.8)
    st_mag = rng.uniform(0.05, 0.15)
    t_mag = rng.uniform(0.2, 0.6)
    
    # Lead projections (Lead I = 0 deg, Lead II = 60 deg)
    lead_II_angle = np.radians(60)
    
    p_a_I = p_mag * np.cos(p_axis)
    p_a_II = p_mag * np.cos(p_axis - lead_II_angle)
    
    delta_a_I = delta_mag * np.cos(delta_axis)
    delta_a_II = delta_mag * np.cos(delta_axis - lead_II_angle)
    
    r_a_I = r_mag * np.cos(r_axis)
    r_a_II = r_mag * np.cos(r_axis - lead_II_angle)
    
    s_a_I = s_mag * np.cos(s_axis)
    s_a_II = s_mag * np.cos(s_axis - lead_II_angle)
    
    st_a_I = st_mag * np.cos(st_axis)
    st_a_II = st_mag * np.cos(st_axis - lead_II_angle)
    
    t_a_I = t_mag * np.cos(t_axis)
    t_a_II = t_mag * np.cos(t_axis - lead_II_angle)
    
    # Respiratory amplitude modulation
    resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * rsa_freq * t)
    
    lead_I = np.zeros_like(t)
    lead_II = np.zeros_like(t)
    
    for r_time in r_peaks:
        beat_var = rng.uniform(0.95, 1.05)
        
        # Lead I
        lead_I += gaussian(t, p_a_I * beat_var, r_time + p_mu, p_sigma)
        lead_I += gaussian(t, delta_a_I * beat_var, r_time + delta_mu, delta_sigma)
        lead_I += gaussian(t, r_a_I * beat_var, r_time + r_mu, r_sigma)
        lead_I += gaussian(t, s_a_I * beat_var, r_time + s_mu, s_sigma)
        lead_I += gaussian(t, st_a_I * beat_var, r_time + st_mu, st_sigma)
        lead_I += asym_gaussian(t, t_a_I * beat_var, r_time + t_mu, t_sigma1, t_sigma2)
        
        # Lead II
        lead_II += gaussian(t, p_a_II * beat_var, r_time + p_mu, p_sigma)
        lead_II += gaussian(t, delta_a_II * beat_var, r_time + delta_mu, delta_sigma)
        lead_II += gaussian(t, r_a_II * beat_var, r_time + r_mu, r_sigma)
        lead_II += gaussian(t, s_a_II * beat_var, r_time + s_mu, s_sigma)
        lead_II += gaussian(t, st_a_II * beat_var, r_time + st_mu, st_sigma)
        lead_II += asym_gaussian(t, t_a_II * beat_var, r_time + t_mu, t_sigma1, t_sigma2)
        
    lead_I *= resp_mod
    lead_II *= resp_mod
    
    # Add noise and realistic baseline wander
    for _ in range(4):
        f = rng.uniform(0.05, 0.3)
        a_I = rng.uniform(0.02, 0.1)
        a_II = rng.uniform(0.02, 0.1)
        phase_I = rng.uniform(0, 2*np.pi)
        phase_II = phase_I + rng.uniform(-0.5, 0.5)
        lead_I += a_I * np.sin(2 * np.pi * f * t + phase_I)
        lead_II += a_II * np.sin(2 * np.pi * f * t + phase_II)
        
    emg_I = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
    emg_II = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
    
    powerline_freq = rng.choice([50.0, 60.0])
    powerline_I = rng.uniform(0.001, 0.005) * np.sin(2 * np.pi * powerline_freq * t)
    powerline_II = rng.uniform(0.001, 0.005) * np.sin(2 * np.pi * powerline_freq * t + rng.uniform(0, 0.1))
    
    signal[:, 0] = lead_I + emg_I + powerline_I
    signal[:, 1] = lead_II + emg_II + powerline_II
    
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