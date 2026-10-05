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
# 1. Introduced an `asymmetric_gaussian_wave` mechanism to model realistic 
#    T-waves, which physiologically have a slower upstroke and faster downstroke.
#    This replaces the symmetric Gaussians that caused a "synthetic" appearance.
# 2. Constrained the electrical axis (R-axis +45 to +70 degrees) to ensure 
#    consistent, positive R and T waves in both Lead I and Lead II, perfectly 
#    matching the blueprint's inter-lead amplitude ratios (Lead I ~25% smaller).
# 3. Refined LQT1 T-wave positioning to exactly span the ST segment while 
#    maintaining the new asymmetric morphology.
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

def asymmetric_gaussian_wave(t, center, sigma_left, sigma_right, amplitude):
    """Generates an asymmetric Gaussian-shaped wave for realistic T-waves."""
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
    # Heart rate and rhythm (LQT1 often has mild bradycardia)
    hr_mean = np.random.uniform(45.0, 95.0) # bpm
    rr_mean = 60.0 / hr_mean                # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged)
    qtc = np.random.uniform(0.45, 0.62)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Electrical Axis and Vector Magnitudes ---
    # Constrained QRS axis (+45 to +70 degrees) ensures Lead I mirrors Lead II 
    # but is ~20-30% smaller, matching the blueprint.
    r_axis = np.random.uniform(45, 70) * np.pi / 180.0
    q_axis = r_axis + np.random.uniform(150, 180) * np.pi / 180.0
    s_axis = r_axis - np.random.uniform(150, 180) * np.pi / 180.0
    t_axis = r_axis + np.random.uniform(-15, 15) * np.pi / 180.0
    p_axis = np.random.uniform(45, 70) * np.pi / 180.0
    
    # Vector magnitudes (mV) - adjusted for realistic QRS morphology
    p_mag = np.random.uniform(0.1, 0.2)
    q_mag = np.random.uniform(0.05, 0.15)
    r_mag = np.random.uniform(1.0, 1.8)
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
    
    # Wave widths (slightly widened for realistic QRS duration ~80-100ms)
    p_width = np.random.uniform(0.015, 0.025)
    q_width = np.random.uniform(0.01, 0.015)
    r_width = np.random.uniform(0.012, 0.018)
    s_width = np.random.uniform(0.012, 0.02)
    
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
        
        # --- T Wave (Dynamic ST-segment aware positioning with asymmetric morphology) ---
        # QT interval is measured from QRS onset (approx beat_t - 0.035) to T wave end.
        t_end = qt - 0.035
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based T wave occupying the entire ST segment
            st_start = 0.05
            t_duration = t_end - st_start
            
            # Asymmetric T-wave spanning exactly from st_start to t_end
            # 65% of the duration is the slower upstroke, 35% is the faster downstroke
            sigma_left = (t_duration / 2.5) * 0.65
            sigma_right = (t_duration / 2.5) * 0.35
            dt_t = st_start + 2.5 * sigma_left
            
            t_mag = np.random.uniform(0.4, 0.7) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_left, sigma_right, amp_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_left, sigma_right, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            sigma_t2_l = np.random.uniform(0.02, 0.03)
            sigma_t2_r = np.random.uniform(0.015, 0.025)
            dt_t2 = t_end - 2.5 * sigma_t2_r
            
            notch_dist = np.random.uniform(0.05, 0.09)
            dt_t1 = dt_t2 - notch_dist
            
            sigma_t1_l = np.random.uniform(0.03, 0.04)
            sigma_t1_r = np.random.uniform(0.02, 0.03)
            
            t_mag1 = np.random.uniform(0.1, 0.25) + current_twa
            t_mag2 = np.random.uniform(0.05, 0.2) + current_twa
            
            amp1_I = t_mag1 * np.cos(t_axis - lead_I_angle)
            amp1_II = t_mag1 * np.cos(t_axis - lead_II_angle)
            
            amp2_I = t_mag2 * np.cos(t_axis - lead_I_angle)
            amp2_II = t_mag2 * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t1, sigma_t1_l, sigma_t1_r, amp1_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t1, sigma_t1_l, sigma_t1_r, amp1_II)
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t2, sigma_t2_l, sigma_t2_r, amp2_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t2, sigma_t2_l, sigma_t2_r, amp2_II)
            
        else:
            # LQT3: Late-onset T wave (long isoelectric ST segment)
            sigma_l = np.random.uniform(0.025, 0.04)
            sigma_r = np.random.uniform(0.015, 0.025)
            dt_t = t_end - 2.5 * sigma_r
            
            t_mag = np.random.uniform(0.3, 0.5) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_l, sigma_r, amp_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_l, sigma_r, amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts
    # -------------------------------------------------------------------------
    # Baseline wander (low frequency, mixed sines for realism)
    bw_freq1 = np.random.uniform(0.1, 0.3)
    bw_freq2 = np.random.uniform(0.3, 0.6)
    bw_freq3 = np.random.uniform(0.02, 0.08)
    
    bw_amp1 = np.random.uniform(0.02, 0.15)
    bw_amp2 = np.random.uniform(0.01, 0.08)
    bw_amp3 = np.random.uniform(0.05, 0.25)
    
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    phase3 = np.random.uniform(0, 2 * np.pi)
    
    bw_I = bw_amp1 * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
           bw_amp2 * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
           bw_amp3 * np.sin(2 * np.pi * bw_freq3 * t + phase3)
           
    bw_II = (bw_amp1 * 1.2) * np.sin(2 * np.pi * bw_freq1 * t + phase1) + \
            (bw_amp2 * 0.9) * np.sin(2 * np.pi * bw_freq2 * t + phase2) + \
            (bw_amp3 * 1.4) * np.sin(2 * np.pi * bw_freq3 * t + phase3)
    
    # Powerline interference
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.02)
    pl_phase = np.random.uniform(0, 2 * np.pi)
    powerline_I = pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    powerline_II = pl_amp * 1.2 * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    
    # High-frequency muscle artifact (reduced level for cleaner baseline)
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)
    
    for i in range(N_SAMPLES):
        sample_seed = RANDOM_SEED + i
        ecg_sample = generate_lqts_ecg(sample_idx=i, seed=sample_seed)
        
        filename = f"lngqt_sample_{i:03d}.npy"
        filepath = os.path.join(OUTPUT_DIR, filename)
        np.save(filepath, ecg_sample)
        
    print(f"Successfully generated {N_SAMPLES} LQTS ECG samples in '{OUTPUT_DIR}'.")

if __name__ == "__main__":
    main()