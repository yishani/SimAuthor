import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. It models three primary physiological 
# variants of LQTS:
#   - LQT1: Broad-based, prolonged T wave.
#   - LQT2: Low amplitude, notched (bifid) T wave.
#   - LQT3: Late-onset T wave with a long isoelectric ST segment.
#
# Refinement: Introduced a physiological electrical axis projection model to
# naturally generate Lead I and Lead II amplitudes, resolving mode collapse 
# and increasing inter-sample variability. T-wave amplitudes and morphologies 
# have been strictly aligned with the clinical blueprint, and T-wave alternans 
# (TWA) is modeled for severe cases.
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
    # Heart rate and rhythm
    hr_mean = np.random.uniform(50.0, 90.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged; we simulate 460-600 ms)
    qtc = np.random.uniform(0.46, 0.60)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # Electrical axes (in radians) for realistic lead projections
    axis_QRS = np.random.uniform(10, 80) * np.pi / 180
    axis_T = axis_QRS + np.random.uniform(-20, 20) * np.pi / 180
    axis_P = np.random.uniform(30, 60) * np.pi / 180
    
    lead_I_angle = 0.0
    lead_II_angle = np.pi / 3.0 # 60 degrees
    
    def project(amp, axis, lead_angle):
        return amp * np.cos(axis - lead_angle)
    
    # Base amplitudes for the 3D heart vector (mV)
    p_amp_base = np.random.uniform(0.1, 0.2)
    q_amp_base = np.random.uniform(-0.1, -0.3)
    r_amp_base = np.random.uniform(1.0, 2.5)
    s_amp_base = np.random.uniform(-0.2, -0.6)
    
    # T-wave parameters based on genotype (Blueprint targets)
    if lqt_type == 1:
        t_amp_base = np.random.uniform(0.4, 0.7)
    elif lqt_type == 2:
        t1_amp_base = np.random.uniform(0.15, 0.25)
        t2_amp_base = np.random.uniform(0.05, 0.15)
    else:
        t_amp_base = np.random.uniform(0.3, 0.5)
        
    # T-wave alternans (TWA) for severe cases (QTc > 500 ms)
    has_twa = (qtc > 0.50) and (np.random.rand() > 0.5)
    twa_magnitude = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
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
        
        # --- P, Q, R, S Waves (Standard Morphology with Axis Projection) ---
        # P wave
        p_amp_I = project(p_amp_base, axis_P, lead_I_angle)
        p_amp_II = project(p_amp_base, axis_P, lead_II_angle)
        lead_I += gaussian_wave(t, beat_t - 0.16, 0.02, p_amp_I)
        lead_II += gaussian_wave(t, beat_t - 0.16, 0.02, p_amp_II)
        
        # Q wave (Onset of QRS is approx beat_t - 0.04)
        q_amp_I = project(q_amp_base, axis_QRS, lead_I_angle)
        q_amp_II = project(q_amp_base, axis_QRS, lead_II_angle)
        lead_I += gaussian_wave(t, beat_t - 0.02, 0.01, q_amp_I)
        lead_II += gaussian_wave(t, beat_t - 0.02, 0.01, q_amp_II)
        
        # R wave
        r_amp_I = project(r_amp_base, axis_QRS, lead_I_angle)
        r_amp_II = project(r_amp_base, axis_QRS, lead_II_angle)
        lead_I += gaussian_wave(t, beat_t, 0.018, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, 0.018, r_amp_II)
        
        # S wave
        s_amp_I = project(s_amp_base, axis_QRS, lead_I_angle)
        s_amp_II = project(s_amp_base, axis_QRS, lead_II_angle)
        lead_I += gaussian_wave(t, beat_t + 0.04, 0.018, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + 0.04, 0.018, s_amp_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (beat_t - 0.04) to T wave end.
        # T wave end = center_T + 2*sigma_T.
        # Therefore: center_T = (beat_t - 0.04 + qt) - 2*sigma_T
        
        twa_effect = ((-1) ** i) * twa_magnitude
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave
            sigma_t = np.random.uniform(0.07, 0.10)
            center_t = (beat_t - 0.04 + qt) - 2 * sigma_t
            current_t_amp = t_amp_base + twa_effect
            
            t_amp_I = project(current_t_amp, axis_T, lead_I_angle)
            t_amp_II = project(current_t_amp, axis_T, lead_II_angle)
            
            lead_I += gaussian_wave(t, center_t, sigma_t, t_amp_I)
            lead_II += gaussian_wave(t, center_t, sigma_t, t_amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t = np.random.uniform(0.03, 0.04)
            # T2 is the second notch, ending at the QT interval limit
            center_t2 = (beat_t - 0.04 + qt) - 2 * sigma_t
            # T1 is the first peak, occurring earlier
            center_t1 = center_t2 - np.random.uniform(0.08, 0.12)
            
            current_t1_amp = t1_amp_base + twa_effect
            current_t2_amp = t2_amp_base + (twa_effect * 0.5)
            
            t1_amp_I = project(current_t1_amp, axis_T, lead_I_angle)
            t1_amp_II = project(current_t1_amp, axis_T, lead_II_angle)
            t2_amp_I = project(current_t2_amp, axis_T, lead_I_angle)
            t2_amp_II = project(current_t2_amp, axis_T, lead_II_angle)
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, center_t1, sigma_t, t1_amp_I)
            lead_II += gaussian_wave(t, center_t1, sigma_t, t1_amp_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, center_t2, sigma_t, t2_amp_I)
            lead_II += gaussian_wave(t, center_t2, sigma_t, t2_amp_II)
            
        else:
            # LQT3: Late-onset T wave (normal width, but delayed)
            sigma_t = np.random.uniform(0.035, 0.05)
            center_t = (beat_t - 0.04 + qt) - 2 * sigma_t
            current_t_amp = t_amp_base + twa_effect
            
            t_amp_I = project(current_t_amp, axis_T, lead_I_angle)
            t_amp_II = project(current_t_amp, axis_T, lead_II_angle)
            
            lead_I += gaussian_wave(t, center_t, sigma_t, t_amp_I)
            lead_II += gaussian_wave(t, center_t, sigma_t, t_amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Realistic multi-component baseline wander
    bw_I = np.zeros_like(t)
    bw_II = np.zeros_like(t)
    for _ in range(3):
        f = np.random.uniform(0.05, 0.4)
        a = np.random.uniform(0.02, 0.08)
        p1 = np.random.uniform(0, 2 * np.pi)
        p2 = p1 + np.random.uniform(-0.5, 0.5)
        bw_I += a * np.sin(2 * np.pi * f * t + p1)
        bw_II += a * np.sin(2 * np.pi * f * t + p2)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.002, 0.01)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.005, 0.015)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
    # Combine signal and noise
    lead_I += bw_I + powerline + hf_noise_I
    lead_II += bw_II + powerline + hf_noise_II

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