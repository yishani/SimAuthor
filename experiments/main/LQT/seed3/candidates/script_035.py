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
# REFINEMENT: 
# 1. Replaced the complex 3D vector projection with direct amplitude 
#    assignments based on the blueprint's Lead II specifications, ensuring 
#    T-wave amplitudes perfectly match the clinical targets (e.g., 0.4-0.7 mV 
#    for LQT1). Lead I is scaled proportionally to simulate a typical +60 deg axis.
# 2. Implemented asymmetric Gaussians for T-waves to create a more realistic 
#    morphology (slower ascent, steeper descent), especially critical for 
#    the broad-based LQT1 T-wave.
# 3. Adjusted QRS widths and offsets to ensure a normal, sharp QRS complex 
#    (80-100 ms) and prevent base widening.
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts_3.1/lngqt_ecg/runs/rep/seed3/generated/"
RANDOM_SEED = 42

def gaussian_wave(t, center, sigma, amplitude):
    """Generates a symmetric Gaussian-shaped wave."""
    return amplitude * np.exp(-0.5 * ((t - center) / sigma)**2)

def asymmetric_gaussian(t, center, sigma_left, sigma_right, amplitude):
    """Generates an asymmetric Gaussian-shaped wave."""
    wave = np.zeros_like(t)
    left_mask = t <= center
    right_mask = t > center
    wave[left_mask] = amplitude * np.exp(-0.5 * ((t[left_mask] - center) / sigma_left)**2)
    wave[right_mask] = amplitude * np.exp(-0.5 * ((t[right_mask] - center) / sigma_right)**2)
    return wave

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
    
    # Heart rate and rhythm (LQT1 often has mild bradycardia)
    if lqt_type == 1:
        hr_mean = np.random.uniform(45.0, 80.0) # bpm
    else:
        hr_mean = np.random.uniform(55.0, 95.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc target: 460 ms to 600 ms)
    qtc = np.random.uniform(0.46, 0.60)
    
    # T-wave Alternans (TWA) for severe cases (QTc > 500 ms)
    has_twa = (qtc > 0.50) and (np.random.rand() > 0.5)
    twa_delta = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
    # Wave widths (narrow QRS: 80-100 ms total)
    p_width = np.random.uniform(0.015, 0.022)
    q_width = np.random.uniform(0.008, 0.012)
    r_width = np.random.uniform(0.012, 0.016)
    s_width = np.random.uniform(0.012, 0.016)
    
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
        
        # --- P, Q, R, S Amplitudes (Direct assignment for typical +60 deg axis) ---
        # Lead II is the primary diagnostic lead. Lead I mirrors it at ~70-80% amplitude.
        p_amp_II = np.random.uniform(0.1, 0.2)
        p_amp_I = p_amp_II * np.random.uniform(0.6, 0.8)
        
        q_amp_II = -np.random.uniform(0.0, 0.15)
        q_amp_I = q_amp_II * np.random.uniform(0.6, 0.8)
        
        r_amp_II = np.random.uniform(1.0, 2.5)
        r_amp_I = r_amp_II * np.random.uniform(0.6, 0.8)
        
        s_amp_II = -np.random.uniform(0.0, 0.25)
        s_amp_I = s_amp_II * np.random.uniform(0.6, 0.8)
        
        # --- P, Q, R, S Offsets ---
        p_offset = np.random.uniform(-0.16, -0.13)
        q_offset = np.random.uniform(-0.025, -0.015)
        s_offset = np.random.uniform(0.025, 0.035)
        
        # Add waves to leads
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_II)
        
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_II)
        
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_II)
        
        # --- T Wave (Dynamic ST-segment aware positioning) ---
        # QT interval is measured from QRS onset (approx beat_t + q_offset - 2*q_width)
        qrs_onset = q_offset - 2.0 * q_width
        # T wave end relative to R-peak
        t_end = qrs_onset + qt
        
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            st_start = s_offset + 2.0 * s_width # Approx end of QRS complex
            t_duration = t_end - st_start
            
            # Asymmetric T-wave: slower ascent, faster descent
            # Peak at ~60% of the duration
            dt_t = st_start + 0.6 * t_duration
            # 3 sigma covers the distance from peak to start/end
            sigma_left = (0.6 * t_duration) / 3.0
            sigma_right = (0.4 * t_duration) / 3.0
            
            t_amp_II = np.random.uniform(0.4, 0.7) + current_twa
            t_amp_I = t_amp_II * np.random.uniform(0.7, 0.8)
            
            lead_I += asymmetric_gaussian(t, beat_t + dt_t, sigma_left, sigma_right, t_amp_I)
            lead_II += asymmetric_gaussian(t, beat_t + dt_t, sigma_left, sigma_right, t_amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.025, 0.04)
            sigma_t2 = np.random.uniform(0.02, 0.035)
            
            # T2 ends at t_end
            dt_t2 = t_end - 3.0 * sigma_t2
            notch_dist = np.random.uniform(0.08, 0.14)
            dt_t1 = dt_t2 - notch_dist
            
            t_amp1_II = np.random.uniform(0.1, 0.25) + current_twa
            t_amp2_II = np.random.uniform(0.05, 0.2) + current_twa
            
            t_amp1_I = t_amp1_II * np.random.uniform(0.7, 0.8)
            t_amp2_I = t_amp2_II * np.random.uniform(0.7, 0.8)
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t1, t_amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t1, t_amp1_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t2, t_amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t2, t_amp2_II)
            
        else:
            # LQT3: Late-onset T wave (long isoelectric ST segment)
            sigma_left = np.random.uniform(0.02, 0.035)
            sigma_right = np.random.uniform(0.015, 0.025)
            
            # Push the narrow T wave to the very end of the QT interval
            dt_t = t_end - 3.0 * sigma_right
            
            t_amp_II = np.random.uniform(0.3, 0.5) + current_twa
            t_amp_I = t_amp_II * np.random.uniform(0.7, 0.8)
            
            lead_I += asymmetric_gaussian(t, beat_t + dt_t, sigma_left, sigma_right, t_amp_I)
            lead_II += asymmetric_gaussian(t, beat_t + dt_t, sigma_left, sigma_right, t_amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.3)
    bw_freq2 = np.random.uniform(0.3, 0.6)
    bw_freq3 = np.random.uniform(0.02, 0.08) # Very slow drift
    
    bw_amp1 = np.random.uniform(0.02, 0.1)
    bw_amp2 = np.random.uniform(0.01, 0.05)
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