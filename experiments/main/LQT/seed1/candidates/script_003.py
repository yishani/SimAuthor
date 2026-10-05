import os
import numpy as np

# =============================================================================
# Long QT Syndrome (LQTS) ECG Simulator
# =============================================================================
# This simulator generates synthetic 2-lead ECG signals (Lead I, Lead II) 
# demonstrating Long QT Syndrome. It models three primary physiological 
# variants of LQTS:
#   - LQT1: Broad-based, prolonged T wave occupying the entire ST segment.
#   - LQT2: Low amplitude, notched (bifid) T wave.
#   - LQT3: Late-onset T wave with a long isoelectric ST segment.
#
# The QT interval is dynamically calculated using Bazett's formula (QTc),
# ensuring physiological scaling with heart rate variations.
# =============================================================================

# Simulation Parameters
FS_INTERNAL = 44100             # Internal synthesis sampling rate
FS_OUT = 16000                  # Output sampling rate
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
    
    # Time vector at internal sampling rate
    num_samples_internal = int(DURATION * FS_INTERNAL)
    t = np.linspace(0, DURATION, num_samples_internal, endpoint=False)
    lead_I = np.zeros_like(t)
    lead_II = np.zeros_like(t)
    
    # -------------------------------------------------------------------------
    # 1. Subject-Specific Physiological Parameters
    # -------------------------------------------------------------------------
    # Heart rate and rhythm
    hr_mean = np.random.uniform(45.0, 95.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged; we simulate 460-600 ms)
    qtc = np.random.uniform(0.46, 0.60)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # Global amplitude scaling for leads to simulate anatomical differences (axis)
    scale_II = np.random.uniform(0.8, 1.3)
    scale_I = scale_II * np.random.uniform(0.4, 0.9) # Lead I is typically lower
    
    # Subject-specific QRS morphology to increase inter-sample diversity
    p_amp = np.random.uniform(0.08, 0.18)
    p_width = np.random.uniform(0.015, 0.025)
    p_offset = np.random.uniform(-0.18, -0.14)
    
    q_amp = np.random.uniform(-0.05, -0.20)
    q_width = np.random.uniform(0.006, 0.012)
    q_offset = np.random.uniform(-0.025, -0.015)
    
    r_amp = np.random.uniform(1.0, 2.2)
    r_width = np.random.uniform(0.010, 0.016)
    
    s_amp = np.random.uniform(-0.1, -0.4)
    s_width = np.random.uniform(0.010, 0.018)
    s_offset = np.random.uniform(0.025, 0.040)
    
    # Subject-specific T-wave parameters based on LQT type
    if lqt_type == 1:
        t_amp = np.random.uniform(0.4, 0.7)
    elif lqt_type == 2:
        t1_amp = np.random.uniform(0.15, 0.25)
        t2_amp = np.random.uniform(0.10, 0.20)
    else:
        t_amp = np.random.uniform(0.3, 0.5)
        
    # T-Wave Alternans (TWA) for severe cases
    twa_severity = np.random.uniform(0.05, 0.15) if qtc > 0.50 and np.random.rand() > 0.5 else 0.0
    
    # -------------------------------------------------------------------------
    # 2. Beat Generation (with Heart Rate Variability)
    # -------------------------------------------------------------------------
    # Estimate max beats needed
    num_beats = int(DURATION / (rr_mean * 0.7)) + 5
    
    # Respiratory Sinus Arrhythmia (RSA) + random noise for HRV
    resp_rate = np.random.uniform(0.2, 0.35) # Hz
    rr_intervals = rr_mean + 0.06 * np.sin(2 * np.pi * resp_rate * np.arange(num_beats)) 
    rr_intervals += np.random.normal(0, 0.02, num_beats)
    
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
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp * scale_II)
        
        # Q wave
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp * scale_II)
        
        # R wave
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp * scale_II)
        
        # S wave
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp * scale_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # T-Wave Alternans modulation
        twa_mod = twa_severity * (-1)**i
        
        # QRS onset is approximately at -0.04s relative to R-peak.
        # The QT interval is measured from QRS onset to T wave end.
        qrs_onset = -0.04
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            sigma_t = qt / 6.0
            center_t = qrs_onset + (qt / 2.0)
            
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, (t_amp + twa_mod) * scale_I)
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, (t_amp + twa_mod) * scale_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t = qt / 15.0
            center_t1 = qrs_onset + (qt * 0.55)
            center_t2 = qrs_onset + (qt * 0.80)
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + center_t1, sigma_t, (t1_amp + twa_mod) * scale_I)
            lead_II += gaussian_wave(t, beat_t + center_t1, sigma_t, (t1_amp + twa_mod) * scale_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + center_t2, sigma_t, (t2_amp + twa_mod) * scale_I)
            lead_II += gaussian_wave(t, beat_t + center_t2, sigma_t, (t2_amp + twa_mod) * scale_II)
            
        else:
            # LQT3: Late-onset T wave (normal width, but delayed)
            sigma_t = qt / 12.0
            center_t = qrs_onset + (qt * 0.85)
            
            lead_I += gaussian_wave(t, beat_t + center_t, sigma_t, (t_amp + twa_mod) * scale_I)
            lead_II += gaussian_wave(t, beat_t + center_t, sigma_t, (t_amp + twa_mod) * scale_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (sum of low frequency sines for more realism)
    baseline_wander = np.zeros_like(t)
    for _ in range(3):
        bw_freq = np.random.uniform(0.05, 0.4)
        bw_amp = np.random.uniform(0.02, 0.08)
        bw_phase = np.random.uniform(0, 2 * np.pi)
        baseline_wander += bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.02)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.01, 0.025)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
    # Combine signal and noise
    lead_I += baseline_wander + powerline + hf_noise_I
    lead_II += (baseline_wander * 1.2) + (powerline * 1.5) + hf_noise_II

    # -------------------------------------------------------------------------
    # 5. Resampling to Output Format
    # -------------------------------------------------------------------------
    num_samples_out = int(DURATION * FS_OUT)
    t_out = np.linspace(0, DURATION, num_samples_out, endpoint=False)
    
    lead_I_out = np.interp(t_out, t, lead_I)
    lead_II_out = np.interp(t_out, t, lead_II)

    # Stack leads: Lead I at col 0, Lead II at col 1. Shape: (160000, 2)
    ecg_signal = np.stack([lead_I_out, lead_II_out], axis=1)
    
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