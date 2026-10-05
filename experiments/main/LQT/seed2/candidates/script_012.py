import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. It models three primary physiological 
# variants of LQTS:
#   - LQT1: Broad-based, asymmetric, prolonged T wave occupying the ST segment.
#   - LQT2: Low amplitude, notched (bifid) T wave.
#   - LQT3: Late-onset T wave with a long isoelectric ST segment.
#
# Hypothesis for Refinement:
# The previous simulator used a single, perfectly symmetric Gaussian for LQT1, 
# resulting in an artificial, overly smooth "hump" that dominated the ST segment. 
# Furthermore, the LQT2 notches were spaced too far apart, appearing as separate 
# waves rather than a single notched T-wave. By implementing a composite asymmetric 
# Gaussian for LQT1 (slower upstroke, faster downstroke) and reducing the peak 
# separation for LQT2, we will significantly improve the morphological realism of 
# the repolarization phase and reduce the distribution discrepancy.
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts_3.1/lngqt_ecg/runs/rep/seed2/generated/"
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
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # Heart rate and rhythm
    if lqt_type == 1:
        hr_mean = np.random.uniform(45.0, 80.0) # LQT1 often exhibits mild bradycardia
    else:
        hr_mean = np.random.uniform(55.0, 95.0)
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged; we simulate 460-650 ms)
    qtc = np.random.uniform(0.46, 0.65)
    
    # --- Electrical Axis Model ---
    # Expanded QRS axis for more diversity (-30 to 90 degrees)
    qrs_axis = np.random.uniform(-30, 90) * np.pi / 180.0
    # T-wave axis is usually concordant with QRS, within -15 to +60 degrees
    t_axis = qrs_axis + np.random.uniform(-15, 60) * np.pi / 180.0
    # P-wave axis is typically around 30-70 degrees
    p_axis = np.random.uniform(30, 70) * np.pi / 180.0
    
    # Base magnitudes (vector lengths in mV)
    p_mag = np.random.uniform(0.08, 0.18)
    
    has_q = np.random.rand() > 0.3
    q_mag = np.random.uniform(-0.3, -0.05) if has_q else 0.0
    
    # R-wave amplitude aligned with blueprint (1.0 to 2.5 mV)
    r_mag = np.random.uniform(1.0, 2.5)
    
    has_s = np.random.rand() > 0.2
    s_mag = np.random.uniform(-0.5, -0.05) if has_s else 0.0
    
    # Projections onto Lead I (0 degrees) and Lead II (60 degrees)
    lead_I_angle = 0.0
    lead_II_angle = np.pi / 3.0
    
    p_amp_I = p_mag * np.cos(p_axis - lead_I_angle)
    p_amp_II = p_mag * np.cos(p_axis - lead_II_angle)
    
    q_amp_I = q_mag * np.cos(qrs_axis - lead_I_angle)
    q_amp_II = q_mag * np.cos(qrs_axis - lead_II_angle)
    
    r_amp_I = r_mag * np.cos(qrs_axis - lead_I_angle)
    r_amp_II = r_mag * np.cos(qrs_axis - lead_II_angle)
    
    s_amp_I = s_mag * np.cos(qrs_axis - lead_I_angle)
    s_amp_II = s_mag * np.cos(qrs_axis - lead_II_angle)
    
    t_proj_I = np.cos(t_axis - lead_I_angle)
    t_proj_II = np.cos(t_axis - lead_II_angle)
    
    # Component Widths
    p_width = np.random.uniform(0.015, 0.022)
    q_width = np.random.uniform(0.008, 0.012)
    r_width = np.random.uniform(0.012, 0.018)
    s_width = np.random.uniform(0.012, 0.020)
    
    # T-Wave Alternans (TWA) for severe cases
    has_twa = (qtc > 0.50) and (np.random.rand() > 0.5)
    twa_amp = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
    # -------------------------------------------------------------------------
    # 2. Beat Generation (with Heart Rate Variability)
    # -------------------------------------------------------------------------
    # Estimate max beats needed
    num_beats = int(DURATION / (rr_mean * 0.7)) + 5
    
    # Respiratory Sinus Arrhythmia (RSA) + random noise for HRV
    resp_rate = np.random.uniform(0.2, 0.35) # Hz
    rr_intervals = rr_mean + 0.05 * np.sin(2 * np.pi * resp_rate * np.arange(num_beats)) 
    rr_intervals += np.random.normal(0, 0.015, num_beats)
    
    # Calculate R-peak times
    beat_times = np.cumsum(rr_intervals)
    # Randomize the start time so ECGs don't all start at the same phase
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
        
        # --- P, Q, R, S Waves (Standard Morphology) ---
        # P wave
        lead_II += gaussian_wave(t, beat_t - 0.16, p_width, p_amp_II)
        lead_I += gaussian_wave(t, beat_t - 0.16, p_width, p_amp_I)
        
        # Q wave
        lead_II += gaussian_wave(t, beat_t - 0.02, q_width, q_amp_II)
        lead_I += gaussian_wave(t, beat_t - 0.02, q_width, q_amp_I)
        
        # R wave
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        
        # S wave
        lead_II += gaussian_wave(t, beat_t + 0.03, s_width, s_amp_II)
        lead_I += gaussian_wave(t, beat_t + 0.03, s_width, s_amp_I)
        
        # J-point / early ST segment to smooth the transition from S-wave
        j_amp_I = s_amp_I * 0.2
        j_amp_II = s_amp_II * 0.2
        lead_II += gaussian_wave(t, beat_t + 0.05, 0.02, j_amp_II)
        lead_I += gaussian_wave(t, beat_t + 0.05, 0.02, j_amp_I)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (approx beat_t - 0.04) to T wave end.
        qrs_onset = -0.04
        s_end = 0.03 + 2 * s_width
        
        # Apply T-Wave Alternans modulation to the magnitude
        twa_mod = twa_amp * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based, asymmetric T-wave (slower upstroke, faster downstroke)
            sigma_t1 = np.random.uniform(0.06, 0.09)
            sigma_t2 = np.random.uniform(0.04, 0.06)
            
            center_t1 = qrs_onset + qt * 0.40
            center_t2 = qrs_onset + qt * 0.55
            
            # Ensure it doesn't overlap QRS too much, shift both components together
            min_center_t1 = s_end + 1.5 * sigma_t1
            if center_t1 < min_center_t1:
                shift = min_center_t1 - center_t1
                center_t1 += shift
                center_t2 += shift
            
            amp_mag = np.random.uniform(0.3, 0.55) + twa_mod
            
            amp_I = amp_mag * t_proj_I
            amp_II = amp_mag * t_proj_II
            
            # First component (slower upstroke, lower amplitude shoulder)
            lead_II += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_II * 0.6)
            lead_I += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_I * 0.6)
            
            # Second component (main peak and faster downstroke)
            lead_II += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_II)
            lead_I += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_I)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.025, 0.035)
            sigma_t2 = np.random.uniform(0.025, 0.035)
            
            # Centers placed closer together to form a notch rather than separate waves
            center_t1 = qrs_onset + qt * 0.50
            center_t2 = qrs_onset + qt * 0.65
            
            # Ensure T1 doesn't overlap QRS too much
            min_center_t1 = s_end + 1.5 * sigma_t1
            if center_t1 < min_center_t1:
                shift = min_center_t1 - center_t1
                center_t1 += shift
                center_t2 += shift
            
            amp_mag_1 = np.random.uniform(0.15, 0.25) + twa_mod
            amp_mag_2 = np.random.uniform(0.10, 0.20) + twa_mod
            
            amp_I_1 = amp_mag_1 * t_proj_I
            amp_II_1 = amp_mag_1 * t_proj_II
            
            amp_I_2 = amp_mag_2 * t_proj_I
            amp_II_2 = amp_mag_2 * t_proj_II
            
            # T1 (First peak)
            lead_II += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_II_1)
            lead_I += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_I_1)
            # T2 (Second peak / notch)
            lead_II += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_II_2)
            lead_I += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_I_2)
            
        else:
            # LQT3: Late-onset T wave (narrow, peaked, delayed)
            sigma_t = np.random.uniform(0.025, 0.035)
            center_t = qrs_onset + qt * 0.80
            
            amp_mag = np.random.uniform(0.25, 0.45) + twa_mod
            amp_I = amp_mag * t_proj_I
            amp_II = amp_mag * t_proj_II
            
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, amp_II)
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, amp_I)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Complex baseline wander (multiple low frequencies for realism)
    baseline_wander_I = np.zeros_like(t)
    baseline_wander_II = np.zeros_like(t)
    for _ in range(4):
        freq = np.random.uniform(0.01, 0.2)
        amp = np.random.uniform(0.02, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander_I += amp * np.sin(2 * np.pi * freq * t + phase)
        
        freq = np.random.uniform(0.01, 0.2)
        amp = np.random.uniform(0.02, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander_II += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.001, 0.005)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.01, 0.025)
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
        
    print(f"Successfully generated {N_SAMPLES} LQTS ECG samples in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()