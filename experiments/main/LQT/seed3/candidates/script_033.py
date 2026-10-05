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
# 1. Constrained the electrical axis to a normal physiological range (30-60 deg) 
#    to ensure consistent, positive QRS and T-wave morphologies in Leads I and II, 
#    resolving the discrepancy of missing or inverted waves.
# 2. Improved the dynamic T-wave positioning by explicitly linking the QT interval 
#    measurement to the actual generated Q-wave onset and S-wave end of each beat, 
#    ensuring perfectly adapted ST segments.
# 3. Reduced excessive baseline wander to prevent obscuring low-amplitude LQT2 T-waves.
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
    # Heart rate and rhythm (LQT1 often has mild bradycardia)
    hr_mean = np.random.uniform(45.0, 95.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged)
    qtc = np.random.uniform(0.46, 0.62)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Electrical Axis and Vector Magnitudes ---
    # Constrain axis to typical normal range to ensure consistent morphology across leads
    r_axis = np.random.uniform(30, 60) * np.pi / 180.0
    # Q and S axes opposite to R to create typical qRs/RS complexes
    q_axis = r_axis + np.pi
    s_axis = r_axis + np.pi
    # T wave axis concordant with QRS
    t_axis = r_axis + np.random.uniform(-10, 15) * np.pi / 180.0
    # P wave axis
    p_axis = np.random.uniform(40, 60) * np.pi / 180.0
    
    # Vector magnitudes (mV)
    p_mag = np.random.uniform(0.1, 0.2)
    q_mag = np.random.uniform(0.05, 0.15)
    r_mag = np.random.uniform(1.2, 2.5)
    s_mag = np.random.uniform(0.1, 0.3)
    
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
    
    # Wave widths
    p_width = np.random.uniform(0.015, 0.022)
    q_width = np.random.uniform(0.008, 0.012)
    r_width = np.random.uniform(0.012, 0.018)
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
        
        # --- P, Q, R, S Waves ---
        # P wave
        p_offset = -np.random.uniform(0.14, 0.18)
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_II)
        
        # Q wave
        q_offset = -np.random.uniform(0.015, 0.025)
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_II)
        
        # R wave
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        
        # S wave
        s_offset = np.random.uniform(0.02, 0.035)
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_II)
        
        # --- T Wave (Dynamic ST-segment aware positioning) ---
        # The QT interval is measured from QRS onset (q_offset) to T wave end.
        t_end = q_offset + qt
        # ST segment starts approximately after the S wave completes
        st_start = s_offset + 2.5 * s_width
        
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            t_duration = t_end - st_start
            if t_duration < 0.1: 
                t_duration = 0.1
            
            # Calculate sigma so the Gaussian spans from st_start to t_end (using 6 sigma spread)
            sigma_t = t_duration / 6.0
            dt_t = st_start + 3.0 * sigma_t
            
            t_mag = np.random.uniform(0.4, 0.7) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t1 = np.random.uniform(0.025, 0.04)
            sigma_t2 = np.random.uniform(0.02, 0.035)
            
            # T2 is the second notch, ending at the QT interval limit
            dt_t2 = t_end - 3.0 * sigma_t2
            # T1 is the first peak, occurring earlier
            notch_dist = np.random.uniform(0.08, 0.14)
            dt_t1 = dt_t2 - notch_dist
            
            t_mag1 = np.random.uniform(0.15, 0.25) + current_twa
            t_mag2 = np.random.uniform(0.1, 0.2) + current_twa
            
            amp1_I = t_mag1 * np.cos(t_axis - lead_I_angle)
            amp1_II = t_mag1 * np.cos(t_axis - lead_II_angle)
            
            amp2_I = t_mag2 * np.cos(t_axis - lead_I_angle)
            amp2_II = t_mag2 * np.cos(t_axis - lead_II_angle)
            
            # T1 (First peak)
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_II)
            # T2 (Second peak / notch)
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_II)
            
        else:
            # LQT3: Late-onset T wave (long isoelectric ST segment)
            sigma_t = np.random.uniform(0.02, 0.035)
            # Push the narrow T wave to the very end of the QT interval
            dt_t = t_end - 3.0 * sigma_t
            
            t_mag = np.random.uniform(0.3, 0.5) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.3)
    bw_freq2 = np.random.uniform(0.3, 0.6)
    bw_freq3 = np.random.uniform(0.02, 0.08) # Very slow drift (respiration)
    
    bw_amp1 = np.random.uniform(0.01, 0.05)
    bw_amp2 = np.random.uniform(0.005, 0.02)
    bw_amp3 = np.random.uniform(0.02, 0.08)
    
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
    pl_amp = np.random.uniform(0.002, 0.01)
    pl_phase = np.random.uniform(0, 2 * np.pi)
    powerline_I = pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    powerline_II = pl_amp * 1.2 * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    
    # High-frequency muscle artifact (Gaussian white noise)
    noise_level = np.random.uniform(0.002, 0.01)
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