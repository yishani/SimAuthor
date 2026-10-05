import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. It models three primary physiological 
# variants of LQTS:
#   - LQT1: Broad-based, prolonged T wave occupying the ST segment.
#   - LQT2: Low amplitude, notched (bifid) T wave.
#   - LQT3: Late-onset T wave with a long isoelectric ST segment.
#
# The QT interval is dynamically calculated using Bazett's formula (QTc),
# ensuring physiological scaling with heart rate variations.
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
    
    # LQTS Severity (QTc > 460 ms is typically prolonged; we simulate 460-650 ms)
    qtc = np.random.uniform(0.46, 0.65)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # Global amplitude scaling for Lead I relative to Lead II (Electrical Axis)
    # Widened range to increase inter-sample variability and prevent mode collapse
    scale_I = np.random.uniform(0.3, 1.2)
    
    # T-wave axis can sometimes differ slightly from QRS axis
    t_scale_I = scale_I * np.random.uniform(0.7, 1.3)
    
    # QRS-T morphology parameters (Base amplitudes for Lead II)
    # Adjusted to more realistic physiological ranges
    p_amp_II = np.random.uniform(0.08, 0.18)
    p_width = np.random.uniform(0.015, 0.022)
    
    q_amp_II = np.random.uniform(-0.15, 0.0)
    q_width = np.random.uniform(0.008, 0.012)
    
    r_amp_II = np.random.uniform(0.8, 1.6)
    r_width = np.random.uniform(0.012, 0.018)
    
    s_amp_II = np.random.uniform(-0.25, -0.05)
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
        lead_I += gaussian_wave(t, beat_t - 0.16, p_width, p_amp_II * scale_I)
        
        # Q wave (Onset of QRS is approx beat_t - 0.015)
        lead_II += gaussian_wave(t, beat_t - 0.015, q_width, q_amp_II)
        lead_I += gaussian_wave(t, beat_t - 0.015, q_width, q_amp_II * scale_I)
        
        # R wave
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_II * scale_I)
        
        # S wave
        lead_II += gaussian_wave(t, beat_t + 0.025, s_width, s_amp_II)
        lead_I += gaussian_wave(t, beat_t + 0.025, s_width, s_amp_II * scale_I)
        
        # J-point / early ST segment to smooth the transition from S-wave
        j_amp_II = s_amp_II * 0.2
        lead_II += gaussian_wave(t, beat_t + 0.04, 0.02, j_amp_II)
        lead_I += gaussian_wave(t, beat_t + 0.04, 0.02, j_amp_II * scale_I)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (beat_t - 0.015) to T wave end.
        t_end = -0.015 + qt
        s_end = 0.025 + 2 * s_width
        
        # Apply T-Wave Alternans modulation
        twa_mod = twa_amp * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            center_t = s_end + (t_end - s_end) * 0.45
            sigma_t = (t_end - s_end) / 3.5
            amp_II = np.random.uniform(0.3, 0.6) + twa_mod
            
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, amp_II)
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, amp_II * t_scale_I)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.025, 0.04)
            sigma_t2 = np.random.uniform(0.025, 0.04)
            
            center_t2 = t_end - 2.0 * sigma_t2
            center_t1 = max(s_end + 0.06, center_t2 - np.random.uniform(0.07, 0.12))
            
            amp_II_1 = np.random.uniform(0.15, 0.3) + twa_mod
            amp_II_2 = np.random.uniform(0.1, 0.25) + twa_mod
            
            # T1 (First peak)
            lead_II += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_II_1)
            lead_I += gaussian_wave(t, beat_t + center_t1, sigma_t1, amp_II_1 * t_scale_I)
            # T2 (Second peak / notch)
            lead_II += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_II_2)
            lead_I += gaussian_wave(t, beat_t + center_t2, sigma_t2, amp_II_2 * t_scale_I)
            
            # Slight ST segment elevation to connect S and T1 realistically
            st_center = (s_end + center_t1) / 2.0
            st_sigma = (center_t1 - s_end) / 2.5
            st_amp = np.random.uniform(0.0, 0.03)
            if st_sigma > 0:
                lead_II += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp)
                lead_I += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp * t_scale_I)
            
        else:
            # LQT3: Late-onset T wave (narrow, peaked, delayed)
            sigma_t = np.random.uniform(0.02, 0.035)
            center_t = t_end - 2.0 * sigma_t
            
            amp_II = np.random.uniform(0.25, 0.5) + twa_mod
            
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, amp_II)
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, amp_II * t_scale_I)
            
            # Long isoelectric ST segment with slight physiological drift
            st_center = (s_end + center_t) / 2.0
            st_sigma = (center_t - s_end) / 2.5
            st_amp = np.random.uniform(0.01, 0.04)
            if st_sigma > 0:
                lead_II += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp)
                lead_I += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp * t_scale_I)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Complex baseline wander (multiple low frequencies for realism)
    baseline_wander_I = np.zeros_like(t)
    baseline_wander_II = np.zeros_like(t)
    for _ in range(3):
        freq = np.random.uniform(0.05, 0.4)
        amp = np.random.uniform(0.02, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander_I += amp * np.sin(2 * np.pi * freq * t + phase)
        
        freq = np.random.uniform(0.05, 0.4)
        amp = np.random.uniform(0.02, 0.1)
        phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander_II += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.001, 0.005)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.002, 0.01)
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