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
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts_3.1/lngqt_ecg/runs/rep/seed3/generated/"
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
    
    # Global amplitude scaling for leads to simulate anatomical differences
    scale_I = np.random.uniform(0.6, 1.1)
    scale_II = np.random.uniform(0.8, 1.3)
    
    # Subject-specific P-QRS morphology (adds inter-sample variance to combat mode collapse)
    p_amp = np.random.uniform(0.08, 0.15)
    p_width = np.random.uniform(0.015, 0.025)
    
    q_amp = np.random.uniform(-0.05, -0.20)
    q_width = np.random.uniform(0.008, 0.012)
    
    r_amp = np.random.uniform(0.8, 1.5)
    r_width = np.random.uniform(0.01, 0.015)
    
    s_amp = np.random.uniform(-0.1, -0.4)
    s_width = np.random.uniform(0.012, 0.018)
    
    # T-wave Alternans (TWA) for severe cases
    has_twa = (qtc > 0.52) and (np.random.rand() > 0.5)
    twa_delta = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
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
        lead_I += gaussian_wave(t, beat_t - 0.16, p_width, p_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t - 0.16, p_width, p_amp * scale_II * 1.2)
        
        # Q wave (Onset of QRS is approx beat_t - 0.035)
        lead_I += gaussian_wave(t, beat_t - 0.015, q_width, q_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t - 0.015, q_width, q_amp * scale_II)
        
        # R wave
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp * scale_II)
        
        # S wave
        lead_I += gaussian_wave(t, beat_t + 0.03, s_width, s_amp * scale_I)
        lead_II += gaussian_wave(t, beat_t + 0.03, s_width, s_amp * scale_II)
        
        # --- T Wave (LQTS Specific Morphology) ---
        # The QT interval is measured from QRS onset (approx beat_t - 0.035) to T wave end.
        # T wave end = center_T + 2*sigma_T.
        # Therefore: center_T = qt - 0.035 - 2*sigma_T
        
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave
            sigma_t = np.random.uniform(0.06, 0.09)
            dt_t = qt - 0.035 - 2 * sigma_t
            
            base_amp_II = np.random.uniform(0.4, 0.7)
            amp_II = (base_amp_II + current_twa) * scale_II
            amp_I = (base_amp_II + current_twa) * 0.7 * scale_I
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t = np.random.uniform(0.02, 0.03)
            # T2 is the second notch, ending at the QT interval limit
            dt_t2 = qt - 0.035 - 2 * sigma_t
            # T1 is the first peak, occurring earlier
            dt_t1 = dt_t2 - np.random.uniform(0.06, 0.09)
            
            base_amp1_II = np.random.uniform(0.15, 0.25)
            base_amp2_II = np.random.uniform(0.10, 0.20)
            
            amp1_II = (base_amp1_II + current_twa) * scale_II
            amp2_II = (base_amp2_II + current_twa) * scale_II
            
            amp1_I = (base_amp1_II + current_twa) * 0.75 * scale_I
            amp2_I = (base_amp2_II + current_twa) * 0.75 * scale_I
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t, amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t, amp1_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t, amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t, amp2_II)
            
        else:
            # LQT3: Late-onset T wave (normal width, but delayed)
            sigma_t = np.random.uniform(0.03, 0.045)
            dt_t = qt - 0.035 - 2 * sigma_t
            
            base_amp_II = np.random.uniform(0.3, 0.5)
            amp_II = (base_amp_II + current_twa) * scale_II
            amp_I = (base_amp_II + current_twa) * 0.7 * scale_I
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.3)
    bw_freq2 = np.random.uniform(0.3, 0.6)
    bw_amp1 = np.random.uniform(0.05, 0.15)
    bw_amp2 = np.random.uniform(0.02, 0.08)
    baseline_wander = bw_amp1 * np.sin(2 * np.pi * bw_freq1 * t + np.random.uniform(0, 2 * np.pi)) + \
                      bw_amp2 * np.sin(2 * np.pi * bw_freq2 * t + np.random.uniform(0, 2 * np.pi))
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.02)
    powerline = pl_amp * np.sin(2 * np.pi * pl_freq * t + np.random.uniform(0, 2 * np.pi))
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.005, 0.02)
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