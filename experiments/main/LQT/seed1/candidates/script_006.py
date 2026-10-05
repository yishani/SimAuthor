import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. 
#
# Refinement: Introduced morphological diversity and physiological realism by 
# implementing asymmetric T-waves, probabilistic Q/S waves, variable QRS widths, 
# and 1/f-characteristic baseline wander. This addresses the mode collapse 
# (high inter-sample similarity) and synthetic appearance identified in the 
# discrepancy report, better matching the empirical distribution of real ECGs.
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "artifacts/lngqt_ecg/visual_representation_100iter/generated/"
RANDOM_SEED = 42

def gaussian_wave(t, center, sigma, amplitude):
    """Generates a symmetric Gaussian-shaped wave for ECG components."""
    return amplitude * np.exp(-0.5 * ((t - center) / sigma)**2)

def asymmetric_gaussian(t, center, sigma_left, sigma_right, amplitude):
    """Generates an asymmetric Gaussian-shaped wave for more realistic repolarization."""
    return np.where(t < center,
                    amplitude * np.exp(-0.5 * ((t - center) / sigma_left)**2),
                    amplitude * np.exp(-0.5 * ((t - center) / sigma_right)**2))

def generate_lqts_ecg(sample_idx, seed):
    """
    Generates a single 10-second ECG segment with Long QT Syndrome.
    """
    np.random.seed(seed)
    
    # Time vector
    t = np.arange(0, DURATION, 1.0 / FS)
    lead_I = np.zeros_like(t)
    lead_II = np.zeros_like(t)
    
    # -------------------------------------------------------------------------
    # 1. Subject-Specific Physiological Parameters
    # -------------------------------------------------------------------------
    # Heart rate and rhythm
    hr_mean = np.random.uniform(50.0, 90.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged; we simulate 460-600 ms)
    qtc = np.random.uniform(0.46, 0.60)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Vector Projection Model (Frontal Plane) ---
    deg2rad = np.pi / 180.0
    
    # Randomize electrical axes for this subject
    theta_R = np.random.uniform(20, 70) * deg2rad
    theta_T = theta_R + np.random.uniform(-15, 15) * deg2rad # T-wave axis usually concordant with QRS
    theta_P = np.random.uniform(40, 60) * deg2rad
    theta_Q = np.random.uniform(240, 300) * deg2rad # Q wave points away from main vector
    theta_S = np.random.uniform(200, 250) * deg2rad # S wave points up and right
    
    # Vector magnitudes (mV) and probabilistic presence to increase diversity
    m_P = np.random.uniform(0.05, 0.12)
    m_Q = np.random.uniform(0.05, 0.15) if np.random.rand() > 0.3 else 0.0
    m_R = np.random.uniform(0.8, 1.6)
    m_S = np.random.uniform(0.1, 0.3) if np.random.rand() > 0.2 else 0.0
    
    # Component widths (sigmas)
    sigma_P = np.random.uniform(0.02, 0.03)
    sigma_Q = np.random.uniform(0.01, 0.015)
    sigma_R = np.random.uniform(0.012, 0.018)
    sigma_S = np.random.uniform(0.012, 0.018)
    
    # Subject-specific T-wave parameters based on LQT type
    if lqt_type == 1:
        m_T = np.random.uniform(0.3, 0.6)
    elif lqt_type == 2:
        m_T1 = np.random.uniform(0.15, 0.25)
        m_T2 = np.random.uniform(0.10, 0.20)
        t_width = np.random.uniform(0.03, 0.04)
        notch_dist = np.random.uniform(0.08, 0.12)
    else:
        m_T = np.random.uniform(0.2, 0.4)
        t_width = np.random.uniform(0.02, 0.03) # Narrower for LQT3
        
    # T-Wave Alternans (TWA) for severe cases
    twa_severity = np.random.uniform(0.05, 0.15) if qtc > 0.50 and np.random.rand() > 0.5 else 0.0
    
    # Standard Lead Angles
    angle_I = 0.0
    angle_II = 60.0 * deg2rad
    
    # Pre-calculate static projections for P, Q, R, S
    p_I = m_P * np.cos(theta_P - angle_I)
    p_II = m_P * np.cos(theta_P - angle_II)
    
    q_I = m_Q * np.cos(theta_Q - angle_I)
    q_II = m_Q * np.cos(theta_Q - angle_II)
    
    r_I = m_R * np.cos(theta_R - angle_I)
    r_II = m_R * np.cos(theta_R - angle_II)
    
    s_I = m_S * np.cos(theta_S - angle_I)
    s_II = m_S * np.cos(theta_S - angle_II)
    
    # -------------------------------------------------------------------------
    # 2. Beat Generation (with Heart Rate Variability)
    # -------------------------------------------------------------------------
    num_beats = int(DURATION / (rr_mean * 0.7)) + 5
    
    # Respiratory Sinus Arrhythmia (RSA) + random noise for HRV
    resp_rate = np.random.uniform(0.2, 0.35) # Hz
    rr_intervals = rr_mean + 0.05 * np.sin(2 * np.pi * resp_rate * np.arange(num_beats)) 
    rr_intervals += np.random.normal(0, 0.015, num_beats)
    
    # Calculate R-peak times
    beat_times = np.cumsum(rr_intervals)
    beat_times -= (beat_times[0] - np.random.uniform(0.1, rr_mean))
    
    # -------------------------------------------------------------------------
    # 3. ECG Waveform Synthesis
    # -------------------------------------------------------------------------
    for i, beat_t in enumerate(beat_times):
        if beat_t > DURATION + 1.0:
            break
        if beat_t < -1.0:
            continue
            
        rr = rr_intervals[i]
        # Bazett's formula: QT = QTc * sqrt(RR)
        qt = qtc * np.sqrt(rr)
        
        # --- P, Q, R, S Waves ---
        lead_I += gaussian_wave(t, beat_t - 0.16, sigma_P, p_I)
        lead_II += gaussian_wave(t, beat_t - 0.16, sigma_P, p_II)
        
        if m_Q > 0:
            lead_I += gaussian_wave(t, beat_t - 0.02, sigma_Q, q_I)
            lead_II += gaussian_wave(t, beat_t - 0.02, sigma_Q, q_II)
        
        lead_I += gaussian_wave(t, beat_t, sigma_R, r_I)
        lead_II += gaussian_wave(t, beat_t, sigma_R, r_II)
        
        if m_S > 0:
            lead_I += gaussian_wave(t, beat_t + 0.03, sigma_S, s_I)
            lead_II += gaussian_wave(t, beat_t + 0.03, sigma_S, s_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (approx beat_t - 0.02)
        t_end = -0.02 + qt
        
        # T-Wave Alternans modulation applied to the vector magnitude
        twa_mod = twa_severity * (-1)**i
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            st_start = 0.06
            center_t = (st_start + t_end) / 2.0
            sigma_t = (t_end - st_start) / 4.0
            
            cur_m_T = m_T + twa_mod
            t_I = cur_m_T * np.cos(theta_T - angle_I)
            t_II = cur_m_T * np.cos(theta_T - angle_II)
            
            # Asymmetric: slower upstroke, faster downstroke
            lead_I += asymmetric_gaussian(t, beat_t + center_t, sigma_t * 1.2, sigma_t * 0.8, t_I)
            lead_II += asymmetric_gaussian(t, beat_t + center_t, sigma_t * 1.2, sigma_t * 0.8, t_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t = t_width
            center_t2 = t_end - 2 * sigma_t
            center_t1 = center_t2 - notch_dist
            
            cur_m_T1 = m_T1 + twa_mod
            cur_m_T2 = m_T2 + twa_mod
            
            t1_I = cur_m_T1 * np.cos(theta_T - angle_I)
            t1_II = cur_m_T1 * np.cos(theta_T - angle_II)
            t2_I = cur_m_T2 * np.cos(theta_T - angle_I)
            t2_II = cur_m_T2 * np.cos(theta_T - angle_II)
            
            lead_I += asymmetric_gaussian(t, beat_t + center_t1, sigma_t * 1.1, sigma_t * 0.9, t1_I)
            lead_II += asymmetric_gaussian(t, beat_t + center_t1, sigma_t * 1.1, sigma_t * 0.9, t1_II)
            lead_I += asymmetric_gaussian(t, beat_t + center_t2, sigma_t * 1.1, sigma_t * 0.9, t2_I)
            lead_II += asymmetric_gaussian(t, beat_t + center_t2, sigma_t * 1.1, sigma_t * 0.9, t2_II)
            
        else:
            # LQT3: Late-onset T wave
            sigma_t = t_width
            center_t = t_end - 2 * sigma_t
            
            cur_m_T = m_T + twa_mod
            t_I = cur_m_T * np.cos(theta_T - angle_I)
            t_II = cur_m_T * np.cos(theta_T - angle_II)
            
            # Asymmetric: slower upstroke, faster downstroke
            lead_I += asymmetric_gaussian(t, beat_t + center_t, sigma_t * 1.3, sigma_t * 0.7, t_I)
            lead_II += asymmetric_gaussian(t, beat_t + center_t, sigma_t * 1.3, sigma_t * 0.7, t_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # 1/f-characteristic baseline wander for more realistic organic drift
    baseline_wander_I = np.zeros_like(t)
    baseline_wander_II = np.zeros_like(t)
    for _ in range(6):
        bw_freq = np.random.uniform(0.05, 0.5)
        bw_amp = np.random.uniform(0.02, 0.08) * (0.2 / bw_freq) # 1/f scaling
        bw_phase_I = np.random.uniform(0, 2 * np.pi)
        bw_phase_II = bw_phase_I + np.random.uniform(-0.2, 0.2)
        baseline_wander_I += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase_I)
        baseline_wander_II += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase_II)
    
    # Powerline interference
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.002, 0.01)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact with amplitude modulation (bursts)
    noise_level = np.random.uniform(0.005, 0.015)
    noise_mod = 1.0 + 0.5 * np.sin(2 * np.pi * np.random.uniform(0.1, 0.5) * t)
    hf_noise_I = np.random.normal(0, noise_level, len(t)) * noise_mod
    hf_noise_II = np.random.normal(0, noise_level, len(t)) * noise_mod
    
    # Combine signal and noise
    lead_I += baseline_wander_I + powerline + hf_noise_I
    lead_II += baseline_wander_II + powerline + hf_noise_II

    # Stack leads: Lead I at col 0, Lead II at col 1. Shape: (5000, 2)
    ecg_signal = np.stack([lead_I, lead_II], axis=1)
    
    return ecg_signal

def main():
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Set global seed for reproducibility of the dataset generation
    np.random.seed(RANDOM_SEED)
    
    # Generate 100 diverse samples
    for i in range(N_SAMPLES):
        # Use a deterministic seed for each sample based on the global seed
        sample_seed = RANDOM_SEED + i
        
        ecg_sample = generate_lqts_ecg(sample_idx=i, seed=sample_seed)
        
        # Save to .npy file
        filename = f"lngqt_sample_{i:03d}.npy"
        filepath = os.path.join(OUTPUT_DIR, filename)
        np.save(filepath, ecg_sample)
        
if __name__ == "__main__":
    main()