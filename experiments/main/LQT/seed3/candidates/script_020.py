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
# REFINEMENT: Implemented asymmetric Gaussian modeling for T-waves to 
# significantly improve morphological realism. Real T-waves exhibit a slower 
# upstroke and a faster downstroke. This asymmetry naturally fills the ST 
# segment in LQT1 and creates a more organic late-onset peak in LQT3. 
# Additionally, introduced a subtle ST-segment bridge for LQT3 to prevent 
# unnatural mathematical flatness, and slightly widened the QRS complex to 
# reduce synthetic "spikiness".
# =============================================================================

# Simulation Parameters
FS = 500                        # Sampling rate in Hz
DURATION = 10.0                 # Duration in seconds
N_SAMPLES = 100                 # Number of samples to generate
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts_3.1/lngqt_ecg/runs/rep/seed3/generated/"
RANDOM_SEED = 42

def gaussian_wave(t, center, sigma, amplitude):
    """Generates a standard symmetric Gaussian-shaped wave."""
    return amplitude * np.exp(-0.5 * ((t - center) / sigma)**2)

def asymmetric_gaussian_wave(t, center, sigma_up, sigma_down, amplitude):
    """Generates an asymmetric Gaussian wave (slower upstroke, faster downstroke)."""
    wave = np.zeros_like(t)
    left = t < center
    right = t >= center
    wave[left] = amplitude * np.exp(-0.5 * ((t[left] - center) / sigma_up)**2)
    wave[right] = amplitude * np.exp(-0.5 * ((t[right] - center) / sigma_down)**2)
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
    # Heart rate and rhythm
    hr_mean = np.random.uniform(50.0, 90.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged)
    qtc = np.random.uniform(0.46, 0.62)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Electrical Axis and Vector Magnitudes ---
    # Normal QRS axis is -30 to +90 degrees.
    r_axis = np.random.uniform(-15, 90) * np.pi / 180.0
    q_axis = r_axis + np.random.uniform(130, 170) * np.pi / 180.0
    s_axis = r_axis - np.random.uniform(130, 170) * np.pi / 180.0
    # T wave axis is usually concordant with QRS
    t_axis = r_axis + np.random.uniform(-30, 45) * np.pi / 180.0
    p_axis = np.random.uniform(15, 75) * np.pi / 180.0
    
    # Vector magnitudes (mV)
    p_mag = np.random.uniform(0.1, 0.2)
    q_mag = np.random.uniform(0.0, 0.2)
    r_mag = np.random.uniform(0.8, 1.8) # Reduced max amplitude for realism
    s_mag = np.random.uniform(0.0, 0.35)
    
    # Lead angles in radians
    lead_I_angle = 0.0
    lead_II_angle = 60.0 * np.pi / 180.0
    
    # Projections onto Lead I and Lead II
    p_amp_I = p_mag * np.cos(p_axis - lead_I_angle)
    p_amp_II = p_mag * np.cos(p_axis - lead_II_angle)
    
    q_amp_I = q_mag * np.cos(q_axis - lead_I_angle)
    q_amp_II = q_mag * np.cos(q_axis - lead_II_angle)
    
    r_amp_I = r_mag * np.cos(r_axis - lead_I_angle)
    r_amp_II = r_mag * np.cos(r_axis - lead_II_angle)
    
    s_amp_I = s_mag * np.cos(s_axis - lead_I_angle)
    s_amp_II = s_mag * np.cos(s_axis - lead_II_angle)
    
    # Wave widths (slightly widened QRS for less synthetic spikiness)
    p_width = np.random.uniform(0.018, 0.025)
    q_width = np.random.uniform(0.008, 0.012)
    r_width = np.random.uniform(0.012, 0.018) 
    s_width = np.random.uniform(0.012, 0.018)
    
    # T-wave Alternans (TWA) for severe cases
    has_twa = (qtc > 0.52) and (np.random.rand() > 0.5)
    twa_delta = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
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
        p_offset = np.random.uniform(-0.18, -0.14)
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_II)
        
        q_offset = np.random.uniform(-0.025, -0.015)
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_II)
        
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        
        s_offset = np.random.uniform(0.025, 0.04)
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_II)
        
        # --- T Wave (Asymmetric and ST-segment aware) ---
        # T wave end relative to R-peak is qt - 0.035 (accounting for QRS onset)
        t_end = qt - 0.035
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based, asymmetric T wave occupying the ST segment
            sigma_up = np.random.uniform(0.07, 0.11)   # Slow upstroke
            sigma_down = np.random.uniform(0.04, 0.07) # Faster downstroke
            dt_t = t_end - 2.5 * sigma_down            # Align end of T-wave with QT interval
            
            t_mag = np.random.uniform(0.4, 0.65) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.03, 0.045)
            sigma_t2 = np.random.uniform(0.03, 0.045)
            
            dt_t2 = t_end - 2.5 * sigma_t2
            notch_dist = np.random.uniform(0.07, 0.11)
            dt_t1 = dt_t2 - notch_dist
            
            t_mag1 = np.random.uniform(0.1, 0.25) + current_twa
            t_mag2 = np.random.uniform(0.05, 0.2) + current_twa
            
            amp1_I = t_mag1 * np.cos(t_axis - lead_I_angle)
            amp1_II = t_mag1 * np.cos(t_axis - lead_II_angle)
            
            amp2_I = t_mag2 * np.cos(t_axis - lead_I_angle)
            amp2_II = t_mag2 * np.cos(t_axis - lead_II_angle)
            
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_II)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_II)
            
        else:
            # LQT3: Late-onset T wave with a long isoelectric ST segment
            sigma_up = np.random.uniform(0.03, 0.05)
            sigma_down = np.random.uniform(0.02, 0.04)
            dt_t = t_end - 2.5 * sigma_down
            
            t_mag = np.random.uniform(0.3, 0.5) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_II)
            
            # Add a very subtle, wide ST segment bridge to prevent unnatural mathematical flatness
            st_center = 0.05 + (dt_t - 0.05) / 2.0
            st_sigma = (dt_t - 0.05) / 3.0
            st_amp = np.random.uniform(0.01, 0.04)
            
            st_amp_I = st_amp * np.cos(t_axis - lead_I_angle)
            st_amp_II = st_amp * np.cos(t_axis - lead_II_angle)
            
            lead_I += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp_I)
            lead_II += gaussian_wave(t, beat_t + st_center, st_sigma, st_amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.3)
    bw_freq2 = np.random.uniform(0.3, 0.6)
    bw_freq3 = np.random.uniform(0.02, 0.08) 
    bw_freq4 = np.random.uniform(0.005, 0.02) # Very slow drift
    
    bw_amp1 = np.random.uniform(0.02, 0.1)
    bw_amp2 = np.random.uniform(0.01, 0.05)
    bw_amp3 = np.random.uniform(0.05, 0.15)
    bw_amp4 = np.random.uniform(0.1, 0.3)
    
    # Shared phases to keep baseline wander correlated across leads
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    phase3 = np.random.uniform(0, 2 * np.pi)
    phase4 = np.random.uniform(0, 2 * np.pi)
    
    bw_I = bw_amp1 * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
           bw_amp2 * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
           bw_amp3 * np.sin(2 * np.pi * bw_freq3 * t + phase3) + \
           bw_amp4 * np.sin(2 * np.pi * bw_freq4 * t + phase4)
           
    bw_II = (bw_amp1 * 1.2) * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
            (bw_amp2 * 0.9) * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
            (bw_amp3 * 1.4) * np.sin(2 * np.pi * bw_freq3 * t + phase3) + \
            (bw_amp4 * 1.1) * np.sin(2 * np.pi * bw_freq4 * t + phase4)
    
    # Powerline interference (50 Hz or 60 Hz)
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.02)
    pl_phase = np.random.uniform(0, 2 * np.pi)
    powerline_I = pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    powerline_II = pl_amp * 1.2 * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.005, 0.02)
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