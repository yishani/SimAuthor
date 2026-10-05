import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. 
#
# Refinement: Constrained the frontal plane electrical axes (theta_R, theta_T, 
# theta_P) to a narrower, more typical physiological range (e.g., theta_R 
# between 40 and 75 degrees). This eliminates unrealistic lead dominance 
# (such as Lead I R-wave exceeding Lead II or being near zero) observed in the 
# previous iteration, improving morphological similarity to the reference set 
# and increasing centroid similarity.
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "artifacts/lngqt_ecg/visual_representation_100iter/generated/"
RANDOM_SEED = 42

def gaussian_wave(t, center, sigma, amplitude):
    """Generates a Gaussian-shaped wave for ECG components."""
    return amplitude * np.exp(-0.5 * ((t - center) / sigma)**2)

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
    hr_mean = np.random.uniform(45.0, 100.0) # bpm
    rr_mean = 60.0 / hr_mean                 # seconds
    
    # LQTS Severity (450 ms to 650 ms)
    qtc = np.random.uniform(0.45, 0.65)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Vector Projection Model (Frontal Plane) ---
    deg2rad = np.pi / 180.0
    
    # Randomize electrical axes (Constrained to physiological normal range)
    theta_R = np.random.uniform(40, 75) * deg2rad
    theta_T = theta_R + np.random.uniform(-20, 30) * deg2rad 
    theta_P = np.random.uniform(45, 70) * deg2rad
    theta_Q = np.random.uniform(210, 260) * deg2rad 
    theta_S = np.random.uniform(210, 260) * deg2rad 
    
    # Vector magnitudes (mV) - Q and S are proportional to R
    m_P = np.random.uniform(0.05, 0.2)
    m_R = np.random.uniform(0.5, 2.0)
    m_Q = np.random.uniform(0.0, 0.15) * m_R
    m_S = np.random.uniform(0.05, 0.3) * m_R
    
    # Subject-specific structural timing
    t_p = np.random.uniform(-0.20, -0.12)
    sigma_p = np.random.uniform(0.015, 0.03)
    
    t_q = np.random.uniform(-0.025, -0.015)
    sigma_q = np.random.uniform(0.008, 0.012)
    
    sigma_r = np.random.uniform(0.01, 0.018)
    
    t_s = np.random.uniform(0.02, 0.04)
    sigma_s = np.random.uniform(0.01, 0.018)
    
    # Subject-specific T-wave morphological parameters based on LQT type
    if lqt_type == 1:
        m_T = np.random.uniform(0.3, 0.8)
        st_start_frac = np.random.uniform(0.02, 0.06)
    elif lqt_type == 2:
        m_T1 = np.random.uniform(0.15, 0.4)
        m_T2 = np.random.uniform(0.1, 0.3)
        st_start_frac = np.random.uniform(0.04, 0.08)
    else:
        m_T = np.random.uniform(0.25, 0.6)
        t_width_frac = np.random.uniform(0.15, 0.3)
        
    # T-Wave Alternans (TWA) for severe cases
    twa_severity = np.random.uniform(0.05, 0.2) if qtc > 0.50 and np.random.rand() > 0.5 else 0.0
    
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
        
        # Ensure T-wave doesn't unrealistically overlap with the next P-wave in tachycardia
        max_qt = rr + t_p - 0.05 - t_q
        qt = min(qt, max_qt)
        
        # --- P, Q, R, S Waves ---
        lead_I += gaussian_wave(t, beat_t + t_p, sigma_p, p_I)
        lead_II += gaussian_wave(t, beat_t + t_p, sigma_p, p_II)
        
        lead_I += gaussian_wave(t, beat_t + t_q, sigma_q, q_I)
        lead_II += gaussian_wave(t, beat_t + t_q, sigma_q, q_II)
        
        lead_I += gaussian_wave(t, beat_t, sigma_r, r_I)
        lead_II += gaussian_wave(t, beat_t, sigma_r, r_II)
        
        lead_I += gaussian_wave(t, beat_t + t_s, sigma_s, s_I)
        lead_II += gaussian_wave(t, beat_t + t_s, sigma_s, s_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset
        qrs_onset = t_q - 2.0 * sigma_q
        t_end = qrs_onset + qt
        
        # T-Wave Alternans modulation applied to the vector magnitude
        twa_mod = twa_severity * (-1)**i
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            st_start = st_start_frac
            t_avail = t_end - st_start
            center_t = st_start + t_avail * 0.5
            sigma_t = t_avail / 5.0 # 2.5 sigma on each side fits exactly
            
            cur_m_T = m_T + twa_mod
            t_I = cur_m_T * np.cos(theta_T - angle_I)
            t_II = cur_m_T * np.cos(theta_T - angle_II)
            
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, t_I)
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, t_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            st_start = st_start_frac
            t_avail = t_end - st_start
            
            sigma_t_frac = np.random.uniform(0.08, 0.12)
            sigma_t = t_avail * sigma_t_frac
            
            # Ensure the second peak's tail ends exactly at t_end
            pos2 = 1.0 - 2.5 * sigma_t_frac
            pos1 = pos2 - np.random.uniform(0.2, 0.35)
            pos1 = max(pos1, 0.15) # Prevent it from being too early
            
            center_t1 = st_start + t_avail * pos1
            center_t2 = st_start + t_avail * pos2
            
            cur_m_T1 = m_T1 + twa_mod
            cur_m_T2 = m_T2 + twa_mod
            
            t1_I = cur_m_T1 * np.cos(theta_T - angle_I)
            t1_II = cur_m_T1 * np.cos(theta_T - angle_II)
            t2_I = cur_m_T2 * np.cos(theta_T - angle_I)
            t2_II = cur_m_T2 * np.cos(theta_T - angle_II)
            
            lead_I += gaussian_wave(t, beat_t + center_t1, sigma_t, t1_I)
            lead_II += gaussian_wave(t, beat_t + center_t1, sigma_t, t1_II)
            lead_I += gaussian_wave(t, beat_t + center_t2, sigma_t, t2_I)
            lead_II += gaussian_wave(t, beat_t + center_t2, sigma_t, t2_II)
            
        else:
            # LQT3: Late-onset T wave with long isoelectric ST segment
            st_start = 0.05
            t_avail = t_end - st_start
            sigma_t = (t_avail * t_width_frac) / 5.0
            center_t = t_end - 2.5 * sigma_t
            
            cur_m_T = m_T + twa_mod
            t_I = cur_m_T * np.cos(theta_T - angle_I)
            t_II = cur_m_T * np.cos(theta_T - angle_II)
            
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, t_I)
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, t_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Organic baseline wander (sum of low frequency sines)
    baseline_wander_I = np.zeros_like(t)
    baseline_wander_II = np.zeros_like(t)
    for _ in range(5):
        bw_freq = np.random.uniform(0.02, 0.5)
        bw_amp = np.random.uniform(0.02, 0.08)
        bw_phase_I = np.random.uniform(0, 2 * np.pi)
        bw_phase_II = bw_phase_I + np.random.uniform(-0.5, 0.5) # Slightly correlated across leads
        baseline_wander_I += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase_I)
        baseline_wander_II += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase_II)
    
    # Powerline interference
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.002, 0.008)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact
    noise_level = np.random.uniform(0.01, 0.03)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
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