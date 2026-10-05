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
# REFINEMENT: Replaced the random electrical axis projection model with 
# explicitly constrained, typical Lead I and II amplitudes for the P-QRS-T 
# complex. The previous axis model frequently generated atypical or inverted 
# QRS morphologies that deviated significantly from standard clinical 
# presentations. Additionally, implemented precise QT interval mapping by 
# measuring exactly from the calculated QRS onset to the T-wave end, ensuring 
# the ST segment and T-wave widths perfectly reflect the target QTc.
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
    
    # LQTS Severity (QTc > 460 ms is typically prolonged)
    qtc = np.random.uniform(0.46, 0.60)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Explicit Lead I and II Amplitudes (mV) ---
    # Constraining to typical upright morphologies to avoid unrealistic inversions
    p_amp_I = np.random.uniform(0.05, 0.12)
    p_amp_II = np.random.uniform(0.08, 0.18)
    
    q_amp_I = -np.random.uniform(0.02, 0.10)
    q_amp_II = -np.random.uniform(0.02, 0.15)
    
    r_amp_I = np.random.uniform(0.4, 1.2)
    r_amp_II = np.random.uniform(0.8, 2.0)
    
    s_amp_I = -np.random.uniform(0.05, 0.20)
    s_amp_II = -np.random.uniform(0.05, 0.30)
    
    # Wave widths (seconds)
    p_width = np.random.uniform(0.02, 0.03)
    q_width = np.random.uniform(0.01, 0.015)
    r_width = np.random.uniform(0.015, 0.022)
    s_width = np.random.uniform(0.015, 0.022)
    
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
        
        # --- P, Q, R, S Waves ---
        # P wave
        p_offset = np.random.uniform(-0.20, -0.15)
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_II)
        
        # Q wave
        q_offset = np.random.uniform(-0.025, -0.015)
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_II)
        
        # R wave
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        
        # S wave
        s_offset = np.random.uniform(0.025, 0.04)
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_II)
        
        # --- Precise QT Interval Mapping ---
        # QRS onset is approximately 2 sigmas before the Q wave center
        qrs_onset = q_offset - 2.0 * q_width
        # QRS end is approximately 2 sigmas after the S wave center
        qrs_end = s_offset + 2.0 * s_width
        
        # T wave must end exactly at QRS onset + QT
        t_end = qrs_onset + qt
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            st_start = qrs_end + 0.02 # Small isoelectric delay before rise
            t_duration = t_end - st_start
            if t_duration < 0.1: 
                t_duration = 0.1 # Safety bound
                
            # Calculate sigma so the Gaussian spans the duration (using 6 sigma spread)
            sigma_t = t_duration / 6.0
            dt_t = st_start + 3.0 * sigma_t
            
            t_mag = np.random.uniform(0.3, 0.55) + current_twa
            amp_I = t_mag * np.random.uniform(0.6, 0.85)
            amp_II = t_mag
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.03, 0.045)
            sigma_t2 = np.random.uniform(0.025, 0.04)
            
            # T2 is the second notch, ending at the QT interval limit
            dt_t2 = t_end - 3.0 * sigma_t2
            # T1 is the first peak, occurring earlier
            notch_dist = np.random.uniform(0.08, 0.14)
            dt_t1 = dt_t2 - notch_dist
            
            # Ensure T1 doesn't overlap excessively with QRS
            if dt_t1 < qrs_end + 3.0 * sigma_t1:
                dt_t1 = qrs_end + 3.0 * sigma_t1
            
            t_mag1 = np.random.uniform(0.15, 0.25) + current_twa
            t_mag2 = np.random.uniform(0.10, 0.20) + current_twa
            
            amp1_I = t_mag1 * np.random.uniform(0.6, 0.85)
            amp1_II = t_mag1
            amp2_I = t_mag2 * np.random.uniform(0.6, 0.85)
            amp2_II = t_mag2
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_II)
            
        else:
            # LQT3: Late-onset T wave (long isoelectric ST segment)
            sigma_t = np.random.uniform(0.025, 0.04)
            # Push the narrow T wave to the very end of the QT interval
            dt_t = t_end - 3.0 * sigma_t
            
            t_mag = np.random.uniform(0.25, 0.45) + current_twa
            amp_I = t_mag * np.random.uniform(0.6, 0.85)
            amp_II = t_mag
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.25)
    bw_freq2 = np.random.uniform(0.25, 0.5)
    bw_freq3 = np.random.uniform(0.02, 0.08) # Very slow drift
    
    bw_amp1 = np.random.uniform(0.02, 0.08)
    bw_amp2 = np.random.uniform(0.01, 0.03)
    bw_amp3 = np.random.uniform(0.05, 0.15)
    
    # Shared phases to keep baseline wander correlated across leads
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    phase3 = np.random.uniform(0, 2 * np.pi)
    
    bw_I = bw_amp1 * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
           bw_amp2 * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
           bw_amp3 * np.sin(2 * np.pi * bw_freq3 * t + phase3)
           
    bw_II = (bw_amp1 * 1.2) * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
            (bw_amp2 * 0.9) * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
            (bw_amp3 * 1.4) * np.sin(2 * np.pi * bw_freq3 * t + phase3)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.015)
    pl_phase = np.random.uniform(0, 2 * np.pi)
    powerline_I = pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    powerline_II = pl_amp * 1.2 * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.005, 0.015)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
    # Combine signal and noise
    lead_I += bw_I + powerline_I + hf_noise_I
    lead_II += bw_II + powerline_II + hf_noise_II

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