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
    
    Refinements in this version:
    1. Simplified fusion complex: Removed the independent Q-wave Gaussian to prevent artificial 
       notching. The WPW QRS is now a true fusion of a widened Delta, R, and S wave.
    2. Widened QRS: Increased sigmas for Delta, R, and S to ensure total QRS duration > 120ms.
    3. Strict ST-T Discordance: T-wave polarity is strictly forced to be opposite to the main QRS vector.
    4. Enhanced Variability: Added Respiratory Sinus Arrhythmia (RSA) and multi-frequency baseline 
       wander to improve inter-sample spread and physiological realism.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0, duration, 1/fs)
    signal = np.zeros((len(t), 2))
    
    # Randomize WPW accessory pathway location to create diverse manifestations
    # 0: Left lateral (Positive delta in I and II, tall R)
    # 1: Posteroseptal (Positive in I, Negative delta/deep S in II - pseudo-inferior infarct)
    # 2: Right lateral (Negative/flat delta in I, Positive in II)
    pathway = rng.choice([0, 1, 2])
    
    # Heart rate and Respiratory Sinus Arrhythmia (RSA)
    hr = rng.uniform(60, 95)
    rr_mean = 60.0 / hr
    
    n_beats = int(duration / rr_mean) + 10
    resp_rate = rng.uniform(0.2, 0.35) # Normal respiratory rate (12-21 breaths/min)
    t_beats = np.arange(n_beats) * rr_mean
    
    # RSA modulates the RR interval
    rsa_mod = rng.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_rate * t_beats)
    rr_intervals = rr_mean + rsa_mod + rng.normal(0, 0.01, n_beats)
    
    r_peaks = np.cumsum(rr_intervals)
    # Shift first peak to start early in the recording
    r_peaks -= rng.uniform(0.2, rr_mean)
    r_peaks = r_peaks[(r_peaks > 0) & (r_peaks < duration)]
    
    # Global morphological parameters for this patient (ensures intra-record consistency)
    # Delta wave is placed just before the R wave and is wide to create the slurred upstroke
    delta_mu = rng.uniform(-0.05, -0.035)
    delta_sigma = rng.uniform(0.022, 0.03)
    
    r_mu = 0.0
    r_sigma = rng.uniform(0.018, 0.025)
    
    s_mu = rng.uniform(0.03, 0.05)
    s_sigma = rng.uniform(0.02, 0.03)
    
    # P wave is placed relative to the Delta wave to ensure a short PR interval (< 120ms)
    p_mu = delta_mu - rng.uniform(0.06, 0.10)
    p_sigma = rng.uniform(0.015, 0.022)
    
    t_mu = rng.uniform(0.22, 0.28)
    t_sigma = rng.uniform(0.035, 0.05)
    
    # Define amplitudes based on the accessory pathway location (enforcing ST-T discordance)
    if pathway == 0: # Left lateral
        amps = {
            0: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(0.15, 0.35), 'r': rng.uniform(0.8, 1.5), 's': rng.uniform(-0.2, 0.0), 't': rng.uniform(-0.3, -0.1)},
            1: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(0.2, 0.4), 'r': rng.uniform(1.0, 1.8), 's': rng.uniform(-0.2, 0.0), 't': rng.uniform(-0.3, -0.05)}
        }
    elif pathway == 1: # Posteroseptal
        amps = {
            0: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(0.1, 0.25), 'r': rng.uniform(0.6, 1.2), 's': rng.uniform(-0.3, -0.1), 't': rng.uniform(-0.2, -0.05)},
            1: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(-0.4, -0.2), 'r': rng.uniform(0.1, 0.4), 's': rng.uniform(-1.0, -0.5), 't': rng.uniform(0.2, 0.5)} # Discordant to negative QRS
        }
    else: # Right lateral
        amps = {
            0: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(-0.2, -0.05), 'r': rng.uniform(0.2, 0.6), 's': rng.uniform(-0.8, -0.4), 't': rng.uniform(0.1, 0.3)}, # Discordant to negative QRS
            1: {'p': rng.uniform(0.08, 0.18), 'd': rng.uniform(0.15, 0.35), 'r': rng.uniform(0.8, 1.5), 's': rng.uniform(-0.3, -0.1), 't': rng.uniform(-0.3, -0.1)}
        }
        
    for lead_idx in range(2):
        lead_signal = np.zeros_like(t)
        a = amps[lead_idx]
        
        # Respiration amplitude modulation (ECG axis shifts slightly with breathing)
        resp_mod = 1.0 + rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * resp_rate * t)
        
        # Synthesize beats
        for r_time in r_peaks:
            # Slight beat-to-beat morphological variation
            b_var = rng.uniform(0.95, 1.05)
            t_var = rng.uniform(-0.002, 0.002) # Tiny conduction jitter
            
            # Construct the WPW complex (P + fused Delta/R/S + T)
            lead_signal += gaussian(t, a['p'] * b_var, r_time + p_mu + t_var, p_sigma)
            lead_signal += gaussian(t, a['d'] * b_var, r_time + delta_mu + t_var, delta_sigma)
            lead_signal += gaussian(t, a['r'] * b_var, r_time + r_mu + t_var, r_sigma)
            lead_signal += gaussian(t, a['s'] * b_var, r_time + s_mu + t_var, s_sigma)
            lead_signal += gaussian(t, a['t'] * b_var, r_time + t_mu + t_var, t_sigma)
            
        # Apply amplitude modulation
        lead_signal *= resp_mod
        
        # Add physiological and environmental noise
        # 1. Multi-frequency baseline wander (more organic than a single sine wave)
        bw_freq1 = rng.uniform(0.01, 0.05)
        bw_freq2 = rng.uniform(0.1, 0.3)
        baseline = rng.uniform(0.1, 0.3) * np.sin(2 * np.pi * bw_freq1 * t + rng.uniform(0, 2*np.pi))
        baseline += rng.uniform(0.05, 0.15) * np.sin(2 * np.pi * bw_freq2 * t + rng.uniform(0, 2*np.pi))
        
        # 2. Powerline interference (50/60 Hz)
        powerline_freq = rng.choice([50.0, 60.0])
        powerline = rng.uniform(0.005, 0.02) * np.sin(2 * np.pi * powerline_freq * t + rng.uniform(0, 2*np.pi))
        
        # 3. High frequency muscle artifact (EMG) - slightly low-passed for realism
        emg = rng.normal(0, rng.uniform(0.005, 0.015), len(t))
        emg = np.convolve(emg, np.ones(3)/3, mode='same') 
        
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