# SimAuthor released simulator - LQT (root).
# Source: experiments/main/LQT/seed2/candidates/script_000.py (search score 0.286).
# Only OUTPUT_DIR was changed. Run: python root.py  or  simauthor generate LQT --which root

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
# The QT interval is dynamically calculated using Bazett's formula (QTc),
# ensuring physiological scaling with heart rate variations.
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "samples/LQT_root/"
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
    
    # Global amplitude scaling for leads to simulate anatomical differences
    scale_I = np.random.uniform(0.7, 1.1)
    scale_II = np.random.uniform(0.9, 1.3)
    
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
        lead_I += gaussian_wave(t, beat_t - 0.16, 0.02, 0.10 * scale_I)
        lead_II += gaussian_wave(t, beat_t - 0.16, 0.02, 0.15 * scale_II)
        
        # Q wave (Onset of QRS is approx beat_t - 0.04)
        lead_I += gaussian_wave(t, beat_t - 0.02, 0.01, -0.10 * scale_I)
        lead_II += gaussian_wave(t, beat_t - 0.02, 0.01, -0.15 * scale_II)
        
        # R wave
        lead_I += gaussian_wave(t, beat_t, 0.015, 0.80 * scale_I)
        lead_II += gaussian_wave(t, beat_t, 0.015, 1.20 * scale_II)
        
        # S wave
        lead_I += gaussian_wave(t, beat_t + 0.04, 0.015, -0.20 * scale_I)
        lead_II += gaussian_wave(t, beat_t + 0.04, 0.015, -0.30 * scale_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (beat_t - 0.04) to T wave end.
        # T wave end = center_T + 2*sigma_T.
        # Therefore: center_T = QT - 0.04 - 2*sigma_T
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave
            sigma_t = np.random.uniform(0.06, 0.08)
            dt_t = qt - 0.04 - 2 * sigma_t
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, 0.15 * scale_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, 0.25 * scale_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t = 0.03
            # T2 is the second notch, ending at the QT interval limit
            dt_t2 = qt - 0.04 - 2 * sigma_t
            # T1 is the first peak, occurring earlier
            dt_t1 = dt_t2 - 0.06
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t, 0.10 * scale_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t, 0.15 * scale_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t, 0.05 * scale_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t, 0.08 * scale_II)
            
        else:
            # LQT3: Late-onset T wave (normal width, but delayed)
            sigma_t = np.random.uniform(0.035, 0.05)
            dt_t = qt - 0.04 - 2 * sigma_t
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, 0.20 * scale_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, 0.30 * scale_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency)
    bw_freq = np.random.uniform(0.1, 0.5)
    bw_amp = np.random.uniform(0.05, 0.15)
    bw_phase = np.random.uniform(0, 2 * np.pi)
    baseline_wander = bw_amp * np.sin(2 * np.pi * bw_freq * t + bw_phase)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.02)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.005, 0.015)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
    # Combine signal and noise
    lead_I += baseline_wander + powerline + hf_noise_I
    lead_II += (baseline_wander * 1.2) + (powerline * 1.5) + hf_noise_II

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