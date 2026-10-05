"""
COPD Audio Simulator

This script generates synthetic audio samples representing lung sounds of patients 
with Chronic Obstructive Pulmonary Disease (COPD).

Physiological and Acoustic Mechanisms Modeled:
1. Altered Respiratory Cycle: COPD is characterized by airflow limitation, leading to 
   a significantly prolonged expiratory phase. The Inspiration:Expiration (I:E) ratio 
   shifts from a normal ~1:2 to 1:3, 1:4, or even higher.
2. Diminished Breath Sounds: Hyperinflation of the lungs in COPD increases the amount 
   of air between the airways and the chest wall, muffling the normal vesicular breath sounds.
   Modeled via steep low-pass filtering (24 dB/octave > 150-250 Hz).
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration. Modeled via AM/FM modulated sine waves with harmonics 
   and narrowband noise to sound organic.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (coarse crackles).
   Modeled via bandpass-filtered noise bursts (150-400 Hz) in early inspiration.
5. Heart Sounds & Sensor Noise: Realistic low-frequency heart thumps and band-limited 
   sensor friction are included to provide an accurate acoustic noise floor and stabilize ZCR.

The simulator generates 100 diverse samples across mild, moderate, and severe presentations.
"""

import os
import math
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/copd_audio/runs/signal/seed2/generated/"
SR = 44100          # Internal sample rate in Hz
TARGET_SR = 16000   # Output sample rate in Hz
DURATION = 10.0     # Duration in seconds
N_SAMPLES = 100     # Number of samples to generate
SEED = 42           # Random seed for reproducibility

# =============================================================================
# Simulator Functions
# =============================================================================

def generate_copd_audio(severity_level, seed):
    """
    Generates a single COPD audio sample based on the specified severity.
    """
    np.random.seed(seed)
    t = np.arange(int(SR * DURATION)) / SR
    
    # -------------------------------------------------------------------------
    # 1. Parameter Sampling based on Severity
    # -------------------------------------------------------------------------
    if severity_level == 'mild':
        rr = np.random.uniform(14, 18)                # Respiratory rate (breaths/min)
        ie_ratio = np.random.uniform(2.5, 3.5)        # I:E ratio (normal is ~2.0)
        breath_amp = np.random.uniform(0.2, 0.3)      # Amplitude of base breath sounds
        wheeze_prob = 0.5                             # Probability of wheezing
        wheeze_intensity = np.random.uniform(0.4, 0.6)
        crackle_prob = 0.4                            # Probability of crackles
    elif severity_level == 'moderate':
        rr = np.random.uniform(16, 22)
        ie_ratio = np.random.uniform(3.5, 4.5)
        breath_amp = np.random.uniform(0.1, 0.2)      # More muffled due to hyperinflation
        wheeze_prob = 0.8
        wheeze_intensity = np.random.uniform(0.5, 0.8)
        crackle_prob = 0.7
    else: # severe
        rr = np.random.uniform(20, 26)                # Tachypnea
        ie_ratio = np.random.uniform(4.5, 6.0)        # Severely prolonged expiration
        breath_amp = np.random.uniform(0.05, 0.1)     # Significantly diminished sounds
        wheeze_prob = 1.0
        wheeze_intensity = np.random.uniform(0.6, 0.9)
        crackle_prob = 0.9
        
    # -------------------------------------------------------------------------
    # 2. Breathing Envelope Generation (Cycle-by-Cycle with Jitter)
    # -------------------------------------------------------------------------
    env = np.zeros_like(t)
    cycle_len_mean = 60.0 / rr
    
    current_time = 0.0
    breath_starts = []
    exp_starts = []
    
    while current_time < DURATION:
        # Jitter cycle length and I:E ratio by ±10%
        c_len = cycle_len_mean * np.random.uniform(0.9, 1.1)
        c_ie = ie_ratio * np.random.uniform(0.9, 1.1)
        
        t_i = c_len / (1.0 + c_ie)
        
        start_idx = int(current_time * SR)
        exp_idx = int((current_time + t_i) * SR)
        end_idx = int((current_time + c_len) * SR)
        
        if start_idx < len(t):
            breath_starts.append(current_time)
            
            # Inspiration: shorter, blunted attack
            i_end = min(exp_idx, len(t))
            if i_end > start_idx:
                phase_i = np.linspace(0, 1, i_end - start_idx)
                env[start_idx:i_end] = np.sin(np.pi * phase_i) ** 0.8
                
            # Expiration: highly asymmetrical, rapid initial rise, long flattened plateau
            e_end = min(end_idx, len(t))
            if e_end > exp_idx:
                exp_starts.append(current_time + t_i)
                phase_e = np.linspace(0, 1, e_end - exp_idx)
                env[exp_idx:e_end] = (np.sin(np.pi * phase_e) ** 0.5) * np.exp(-4.0 * phase_e)
                
        current_time += c_len
    
    # -------------------------------------------------------------------------
    # 3. Base Breath Sounds (Heavily Attenuated Noise)
    # -------------------------------------------------------------------------
    # Use pink-ish noise for a more natural breath sound body
    white_noise = np.random.normal(0, 1, len(t))
    b_pink, a_pink = signal.butter(1, 500, btype='lowpass', fs=SR)
    pink_noise = signal.filtfilt(b_pink, a_pink, white_noise)
    
    # 24 dB/octave lowpass (4th order via filtfilt) at 150-250 Hz
    lp_cutoff = np.random.uniform(150, 250)
    b_lp, a_lp = signal.butter(2, lp_cutoff, btype='lowpass', fs=SR)
    breath_sound = signal.filtfilt(b_lp, a_lp, pink_noise)
    
    # Normalize to peak of 1.0 before applying envelope and amplitude
    b_max = np.max(np.abs(breath_sound))
    if b_max > 0:
        breath_sound = breath_sound / b_max
        
    breath_sound = breath_sound * env * breath_amp
    
    # -------------------------------------------------------------------------
    # 4. Wheezes (Polyphonic Continuous Sounds)
    # -------------------------------------------------------------------------
    wheeze_sound = np.zeros_like(t)
    if np.random.rand() < wheeze_prob:
        n_wheezes = np.random.randint(2, 5) # 2 to 4 polyphonic wheezes
        for _ in range(n_wheezes):
            f0 = np.random.uniform(150, 600)
            fm_rate = np.random.uniform(5, 15)
            fm_depth = f0 * np.random.uniform(0.02, 0.05)
            
            # FM Jitter (Perlin-like noise to make vibrato organic)
            jitter_noise = np.random.normal(0, 1, len(t))
            b_j, a_j = signal.butter(1, 5, btype='lowpass', fs=SR)
            jitter = signal.filtfilt(b_j, a_j, jitter_noise)
            jitter = jitter / (np.std(jitter) + 1e-8)
            
            # Frequency Modulation (Vibrato + Jitter)
            fm = f0 + fm_depth * np.sin(2 * np.pi * fm_rate * t) + (f0 * 0.02) * jitter
            phase_acc = np.cumsum(fm) / SR
            
            # Amplitude Modulation (Organic chaotic fluttering)
            am_noise = np.random.normal(0, 1, len(t))
            b_am, a_am = signal.butter(2, 10, btype='lowpass', fs=SR)
            am_noise = signal.filtfilt(b_am, a_am, am_noise)
            am_noise = am_noise / (np.std(am_noise) + 1e-8)
            
            # Narrowband noise to reduce synthetic purity
            nb_noise = np.random.normal(0, 1, len(t))
            b_nb, a_nb = signal.butter(1, [max(20, f0-50), min(SR/2-1, f0+50)], btype='bandpass', fs=SR)
            nb_noise = signal.filtfilt(b_nb, a_nb, nb_noise)
            nb_noise = nb_noise / (np.std(nb_noise) + 1e-8)
            
            # Base sine + 1st harmonic for a richer, less synthetic timbre
            w = (np.sin(2 * np.pi * phase_acc) + 
                 0.2 * np.sin(2 * np.pi * 2 * phase_acc) + 
                 0.15 * nb_noise) * (1.0 + 0.3 * am_noise)
            
            # Envelope: Triggered ~500 ms after start of expiration
            w_env = np.zeros_like(t)
            for e_start in exp_starts:
                w_start_time = e_start + 0.5
                if w_start_time < DURATION:
                    idx_start = int(w_start_time * SR)
                    sustain_len = np.random.uniform(1.5, 2.5)
                    
                    attack_samples = int(0.3 * SR)
                    sustain_samples = int(sustain_len * SR)
                    release_samples = int(0.4 * SR)
                    
                    idx_end = idx_start + attack_samples + sustain_samples + release_samples
                    if idx_end > len(t):
                        idx_end = len(t)
                        
                    total_samples = idx_end - idx_start
                    if total_samples > 0:
                        local_env = np.ones(total_samples)
                        
                        a_end = min(attack_samples, total_samples)
                        local_env[:a_end] = np.linspace(0, 1, a_end)
                        
                        r_start = min(attack_samples + sustain_samples, total_samples)
                        if r_start < total_samples:
                            local_env[r_start:] = np.linspace(1, 0, total_samples - r_start)
                            
                        w_env[idx_start:idx_end] = np.maximum(w_env[idx_start:idx_end], local_env)
            
            wheeze_sound += w * w_env * (wheeze_intensity / n_wheezes)
            
    # -------------------------------------------------------------------------
    # 5. Crackles (Coarse, Early Inspiratory)
    # -------------------------------------------------------------------------
    crackle_sound = np.zeros_like(t)
    if np.random.rand() < crackle_prob:
        for b_start in breath_starts:
            n_crackles = np.random.randint(3, 9)
            # Clustered in the first 400 ms of inspiratory phase
            c_times = b_start + np.random.uniform(0.05, 0.4, n_crackles)
            for ct in c_times:
                if ct < DURATION:
                    idx = int(ct * SR)
                    burst_len = int(np.random.uniform(0.010, 0.015) * SR)
                    if idx < len(t):
                        end_idx = min(idx + burst_len, len(t))
                        actual_len = end_idx - idx
                        # Short burst of white noise with exponential decay
                        burst = np.random.normal(0, 1, actual_len)
                        decay = np.exp(-np.linspace(0, 3, actual_len))
                        crackle_sound[idx:end_idx] += burst * decay
                        
        # Bandpass filter crackles: 150 - 400 Hz (Coarse crackle profile)
        if np.any(crackle_sound):
            b_c, a_c = signal.butter(2, [150, 400], btype='bandpass', fs=SR)
            crackle_sound = signal.filtfilt(b_c, a_c, crackle_sound)
            # Normalize crackles to have a reasonable peak
            c_max = np.max(np.abs(crackle_sound))
            if c_max > 0:
                crackle_sound = (crackle_sound / c_max) * np.random.uniform(0.2, 0.5)
                
    # -------------------------------------------------------------------------
    # 6. Heart Sounds (Low frequency, helps lower ZCR and adds realism)
    # -------------------------------------------------------------------------
    heart_sound = np.zeros_like(t)
    hr = np.random.uniform(60, 95)
    beat_interval = 60.0 / hr
    
    h_time = np.random.uniform(0, beat_interval)
    while h_time < DURATION:
        # S1 (Lub)
        idx_s1 = int(h_time * SR)
        len_s1 = int(0.1 * SR)
        if idx_s1 + len_s1 < len(t):
            window = np.sin(np.pi * np.linspace(0, 1, len_s1)) ** 2
            noise = np.random.normal(0, 1, len_s1)
            heart_sound[idx_s1:idx_s1+len_s1] += window * noise
            
        # S2 (Dub)
        h_time_s2 = h_time + 0.3  # roughly 300ms after S1
        idx_s2 = int(h_time_s2 * SR)
        len_s2 = int(0.08 * SR)
        if idx_s2 + len_s2 < len(t):
            window = np.sin(np.pi * np.linspace(0, 1, len_s2)) ** 2
            noise = np.random.normal(0, 1, len_s2)
            heart_sound[idx_s2:idx_s2+len_s2] += window * noise * 0.7
            
        h_time += beat_interval + np.random.uniform(-0.05, 0.05)
        
    if np.any(heart_sound):
        b_h, a_h = signal.butter(2, [20, 100], btype='bandpass', fs=SR)
        heart_sound = signal.filtfilt(b_h, a_h, heart_sound)
        h_max = np.max(np.abs(heart_sound))
        if h_max > 0:
            heart_sound = (heart_sound / h_max) * np.random.uniform(0.02, 0.06)

    # -------------------------------------------------------------------------
    # 7. Background Noise (Room/Sensor)
    # -------------------------------------------------------------------------
    # Replaced pure white noise with band-limited noise to prevent artificially high ZCR
    white_bg = np.random.normal(0, 1, len(t))
    
    # Low-frequency rumble (brown noise)
    b_brown, a_brown = signal.butter(1, 100, btype='lowpass', fs=SR)
    brown_bg = signal.filtfilt(b_brown, a_brown, white_bg)
    brown_bg = brown_bg / (np.std(brown_bg) + 1e-8)
    
    # Muffled ambient/sensor noise (pink-ish, low-passed to avoid hiss)
    b_pink, a_pink = signal.butter(1, 1500, btype='lowpass', fs=SR)
    pink_bg = signal.filtfilt(b_pink, a_pink, white_bg)
    pink_bg = pink_bg / (np.std(pink_bg) + 1e-8)
    
    bg_noise = brown_bg * np.random.uniform(0.01, 0.03) + pink_bg * np.random.uniform(0.005, 0.015)
        
    # -------------------------------------------------------------------------
    # 8. Combine
    # -------------------------------------------------------------------------
    audio = breath_sound + wheeze_sound + crackle_sound + heart_sound + bg_noise
    return audio.astype(np.float32)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Severity distribution scaled to N_SAMPLES: 30% mild, 35% moderate, 35% severe.
    n_mild = N_SAMPLES * 3 // 10
    n_moderate = (N_SAMPLES - n_mild) // 2
    severities = (['mild'] * n_mild
                  + ['moderate'] * n_moderate
                  + ['severe'] * (N_SAMPLES - n_mild - n_moderate))

    print(f"Generating {N_SAMPLES} COPD audio samples...")

    # Calculate resampling factors
    gcd = math.gcd(TARGET_SR, SR)
    up = TARGET_SR // gcd
    down = SR // gcd

    for i, severity in enumerate(severities):
        sample_seed = SEED + i
        audio_signal = generate_copd_audio(severity_level=severity, seed=sample_seed)

        # Resample to 16000 Hz as per contract
        audio_resampled = signal.resample_poly(audio_signal, up, down)
        
        # Peak normalization to prevent clipping (target max amplitude = 0.95)
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_resampled = (audio_resampled / max_val) * 0.95

        # Save to WAV file
        filename = f"copd_sample_{i:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        wavfile.write(filepath, TARGET_SR, audio_resampled.astype(np.float32))
        print(f"Saved: {filename} (Severity: {severity})")
        
    print(f"All samples successfully saved to {OUTPUT_DIR}")