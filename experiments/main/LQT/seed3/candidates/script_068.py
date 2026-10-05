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
# REFINEMENT: Constrained electrical axes (P, Q, R, S, T) to physiological normal 
# ranges (e.g., R-axis 30-60 degrees) to prevent unrealistic inverted or missing 
# waves in Leads I and II. Scaled LQT2 notch positioning dynamically with the 
# available ST segment duration to prevent unnatural bunching. Reduced the 
# amplitude of baseline wander (brown noise and respiratory) to prevent severe 
# morphological distortion and better match the clean baselines of clinical recordings.
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
    """
    Generates an asymmetric Gaussian-shaped wave, typical for physiological T-waves
    which have a slower upstroke and a faster downstroke.
    """
    sigma = np.where(t <= center, sigma_up, sigma_down)
    return amplitude * np.exp(-0.5 * ((t - center) / sigma)**2)

def generate_brown_noise(N, fs):
    """Generates 1/f^2 brown noise for realistic smooth baseline wander."""
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(N, d=1/fs)
    f[0] = 1.0  # Avoid division by zero
    X = X / f   # 1/f amplitude spectrum -> 1/f^2 power spectrum
    X[0] = 0.0  # Explicitly remove DC component to prevent massive offsets
    brown = np.fft.irfft(X, n=N)
    return brown / np.std(brown)

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
    # Heart rate and rhythm (Expanded variance for diversity)
    hr_mean = np.random.uniform(40.0, 100.0) # bpm
    rr_mean = 60.0 / hr_mean                 # seconds
    
    # LQTS Severity (QTc > 460 ms is typically prolonged)
    qtc = np.random.uniform(0.46, 0.60)
    
    # LQTS Genotype/Morphology (1=LQT1, 2=LQT2, 3=LQT3)
    lqt_type = np.random.choice([1, 2, 3])
    
    # --- Electrical Axis and Vector Magnitudes ---
    # Constrain axes to physiological normal ranges to ensure positive projections in Leads I & II
    r_axis = np.random.uniform(30, 60) * np.pi / 180.0
    q_axis = r_axis + np.random.uniform(160, 200) * np.pi / 180.0
    s_axis = r_axis - np.random.uniform(160, 200) * np.pi / 180.0
    t_axis = r_axis + np.random.uniform(-15, 15) * np.pi / 180.0 
    p_axis = np.random.uniform(30, 60) * np.pi / 180.0
    
    # Vector magnitudes (mV)
    p_mag = np.random.uniform(0.12, 0.25)
    q_mag = np.random.uniform(0.05, 0.20)
    r_mag = np.random.uniform(1.0, 1.8)
    s_mag = np.random.uniform(0.15, 0.40)
    
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
    
    # Wave widths (adjusted for physiological diversity)
    p_width = np.random.uniform(0.015, 0.025)
    q_width = np.random.uniform(0.008, 0.015)
    r_width = np.random.uniform(0.010, 0.018)
    s_width = np.random.uniform(0.012, 0.020)
    
    # T-wave Alternans (TWA) for severe cases
    has_twa = (qtc > 0.52) and (np.random.rand() > 0.5)
    twa_delta = np.random.uniform(0.05, 0.15) if has_twa else 0.0
    
    # -------------------------------------------------------------------------
    # 2. Beat Generation (with Heart Rate Variability)
    # -------------------------------------------------------------------------
    num_beats = int(DURATION / (rr_mean * 0.7)) + 5
    
    resp_rate = np.random.uniform(0.15, 0.35) # Hz
    rr_intervals = rr_mean + 0.05 * np.sin(2 * np.pi * resp_rate * np.arange(num_beats)) 
    rr_intervals += np.random.normal(0, 0.015, num_beats)
    
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
        p_offset = np.random.uniform(-0.20, -0.12)
        lead_I += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_I)
        lead_II += gaussian_wave(t, beat_t + p_offset, p_width, p_amp_II)
        
        q_offset = np.random.uniform(-0.030, -0.015)
        lead_I += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_I)
        lead_II += gaussian_wave(t, beat_t + q_offset, q_width, q_amp_II)
        
        lead_I += gaussian_wave(t, beat_t, r_width, r_amp_I)
        lead_II += gaussian_wave(t, beat_t, r_width, r_amp_II)
        
        s_offset = np.random.uniform(0.020, 0.040)
        lead_I += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_I)
        lead_II += gaussian_wave(t, beat_t + s_offset, s_width, s_amp_II)
        
        # --- T Wave (Exact Physiological QT Mapping) ---
        q_onset = q_offset - 2.5 * q_width
        t_end = q_onset + qt
        current_twa = twa_delta * (1 if i % 2 == 0 else -1)
        
        if lqt_type == 1:
            # LQT1: Broad-based, symmetrical T wave occupying the entire ST segment
            st_start = s_offset + 2.5 * s_width
            t_duration = t_end - st_start
            
            # Symmetrical shaping spanning exactly from S-end to QT-end
            sigma_t = t_duration / 5.0
            dt_t = st_start + 2.5 * sigma_t
            
            t_mag = np.random.uniform(0.4, 0.7) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_I)
            lead_II += gaussian_wave(t, beat_t + dt_t, sigma_t, amp_II)
            
        elif lqt_type == 2:
            # LQT2: Low amplitude, notched (bifid) T wave
            st_start = s_offset + 2.5 * s_width
            t_duration = t_end - st_start
            
            # Scale notch parameters dynamically with the prolonged ST segment
            sigma_t2 = t_duration * np.random.uniform(0.10, 0.15)
            dt_t2 = t_end - 2.5 * sigma_t2
            
            notch_dist = t_duration * np.random.uniform(0.20, 0.35)
            dt_t1 = dt_t2 - notch_dist
            sigma_t1 = t_duration * np.random.uniform(0.10, 0.15)
            
            t_mag1 = np.random.uniform(0.10, 0.25) + current_twa
            t_mag2 = np.random.uniform(0.10, 0.25) + current_twa
            
            amp1_I = t_mag1 * np.cos(t_axis - lead_I_angle)
            amp1_II = t_mag1 * np.cos(t_axis - lead_II_angle)
            
            amp2_I = t_mag2 * np.cos(t_axis - lead_I_angle)
            amp2_II = t_mag2 * np.cos(t_axis - lead_II_angle)
            
            # Correctly apply both peaks to both leads
            lead_I += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_I)
            lead_II += gaussian_wave(t, beat_t + dt_t1, sigma_t1, amp1_II)
            
            lead_I += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_I)
            lead_II += gaussian_wave(t, beat_t + dt_t2, sigma_t2, amp2_II)
            
        else:
            # LQT3: Late-onset T wave (long isoelectric ST segment)
            sigma_up = np.random.uniform(0.02, 0.03)
            sigma_down = np.random.uniform(0.015, 0.025)
            dt_t = t_end - 2.5 * sigma_down
            
            t_mag = np.random.uniform(0.30, 0.50) + current_twa
            amp_I = t_mag * np.cos(t_axis - lead_I_angle)
            amp_II = t_mag * np.cos(t_axis - lead_II_angle)
            
            lead_I += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_I)
            lead_II += asymmetric_gaussian_wave(t, beat_t + dt_t, sigma_up, sigma_down, amp_II)

    # -------------------------------------------------------------------------
    # 4. Noise and Artifacts (Physiological Baseline Wander)
    # -------------------------------------------------------------------------
    # Respiratory baseline wander (reduced amplitude for cleaner baseline)
    resp_phase = np.random.uniform(0, 2 * np.pi)
    resp_wander_I = np.random.uniform(0.01, 0.03) * np.sin(2 * np.pi * resp_rate * t + resp_phase)
    resp_wander_II = np.random.uniform(0.01, 0.03) * np.sin(2 * np.pi * resp_rate * t + resp_phase + np.random.uniform(-0.2, 0.2))
    
    # Brown noise for realistic smooth low-frequency drift (reduced amplitude)
    brown_I = generate_brown_noise(len(t), FS) * np.random.uniform(0.01, 0.04)
    brown_II = generate_brown_noise(len(t), FS) * np.random.uniform(0.01, 0.04)
    
    # Powerline interference
    pl_freq = np.random.choice([50.0, 60.0])
    pl_amp = np.random.uniform(0.005, 0.015)
    pl_phase = np.random.uniform(0, 2 * np.pi)
    powerline_I = pl_amp * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    powerline_II = pl_amp * 1.2 * np.sin(2 * np.pi * pl_freq * t + pl_phase)
    
    # High-frequency muscle (EMG) noise - kept minimal for clean baseline
    noise_level = np.random.uniform(0.002, 0.008)
    hf_noise_I = np.random.normal(0, noise_level, len(t))
    hf_noise_II = np.random.normal(0, noise_level, len(t))
    
    lead_I += resp_wander_I + brown_I + powerline_I + hf_noise_I
    lead_II += resp_wander_II + brown_II + powerline_II + hf_noise_II

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