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
   Modeled via steep low-pass filtering (24 dB/octave) of pink noise.
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration. Modeled via AM/FM modulated sine waves with narrowband noise,
   generated per-breath with natural pitch drift.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (coarse crackles).
   Modeled via bandpass-filtered pink noise bursts (150-400 Hz) in early inspiration.

The simulator generates 100 diverse samples across mild, moderate, and severe presentations,
including the "Silent Chest" phenotype in severe cases.
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

def generate_pink_noise(N):
    """
    Generates pink noise by shaping white noise in the frequency domain (1/f power spectrum).
    This provides a natural broadband spectrum with lower ZCR than white noise.
    """
    white = np.random.normal(0, 1, N)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(N)
    f[0] = f[1]  # Avoid divide by zero
    X = X / np.sqrt(f)
    pink = np.fft.irfft(X, n=N)
    if np.std(pink) > 0:
        pink = pink / np.std(pink)
    return pink

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
        wheeze_prob = 0.5                             # Probability of patient having wheezes
        wheeze_intensity = np.random.uniform(0.3, 0.5)
        crackle_prob = 0.4                            # Probability of crackles
    elif severity_level == 'moderate':
        rr = np.random.uniform(16, 22)
        ie_ratio = np.random.uniform(3.5, 4.5)
        breath_amp = np.random.uniform(0.1, 0.2)      # More muffled due to hyperinflation
        wheeze_prob = 0.8
        wheeze_intensity = np.random.uniform(0.4, 0.7)
        crackle_prob = 0.7
    else: # severe
        rr = np.random.uniform(20, 26)                # Tachypnea
        ie_ratio = np.random.uniform(4.5, 6.0)        # Severely prolonged expiration
        
        # 20% chance of "Silent Chest" (GOLD 4) phenotype
        if np.random.rand() < 0.2:
            breath_amp = np.random.uniform(0.02, 0.05)
            wheeze_prob = 0.2
            wheeze_intensity = np.random.uniform(0.1, 0.3)
            crackle_prob = 0.3
        else:
            breath_amp = np.random.uniform(0.05, 0.1)     # Significantly diminished sounds
            wheeze_prob = 0.9
            wheeze_intensity = np.random.uniform(0.5, 0.8)
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
    # 3. Base Breath Sounds (Attenuated Pink Noise)
    # -------------------------------------------------------------------------
    pink_noise = generate_pink_noise(len(t))
    
    # COPD attenuation: 24 dB/octave slope via 2nd order filtfilt (matches blueprint)
    lp_cutoff = np.random.uniform(150, 250)
    b_copd, a_copd = signal.butter(2, lp_cutoff, btype='lowpass', fs=SR)
    breath_sound = signal.filtfilt(b_copd, a_copd, pink_noise)
    
    # Normalize to peak of 1.0 before applying envelope and amplitude
    b_max = np.max(np.abs(breath_sound))
    if b_max > 0:
        breath_sound = breath_sound / b_max
        
    breath_sound = breath_sound * env * breath_amp
    
    # -------------------------------------------------------------------------
    # 4. Wheezes (Polyphonic Continuous Sounds, Generated Per-Breath)
    # -------------------------------------------------------------------------
    wheeze_sound = np.zeros_like(t)
    if np.random.rand() < wheeze_prob:
        for e_start in exp_starts:
            # 80% chance to wheeze on this specific breath if patient is a wheezer
            if np.random.rand() < 0.8:
                # Find the start of the next inspiration to prevent wheeze bleed-over
                next_b_starts = [b for b in breath_starts if b > e_start]
                e_end = next_b_starts[0] if next_b_starts else DURATION
                
                w_start_time = e_start + np.random.uniform(0.2, 0.6)
                max_w_len = max(0, e_end - w_start_time - 0.1)
                
                if w_start_time < DURATION and max_w_len > 0.4:
                    n_wheezes = np.random.randint(1, 4) # 1 to 3 polyphonic wheezes per breath
                    for _ in range(n_wheezes):
                        f0_start = np.random.uniform(150, 600)
                        f0_end = f0_start * np.random.uniform(0.85, 1.15) # Natural pitch drift
                        fm_rate = np.random.uniform(5, 15)
                        fm_depth = f0_start * np.random.uniform(0.02, 0.05)
                        
                        attack_time = np.random.uniform(0.2, 0.4)
                        release_time = np.random.uniform(0.2, 0.4)
                        sustain_time = np.random.uniform(0.5, 2.0)
                        
                        if attack_time + sustain_time + release_time > max_w_len:
                            sustain_time = max(0.1, max_w_len - attack_time - release_time)
                            
                        idx_start = int(w_start_time * SR)
                        attack_samples = int(attack_time * SR)
                        sustain_samples = int(sustain_time * SR)
                        release_samples = int(release_time * SR)
                        
                        idx_end = idx_start + attack_samples + sustain_samples + release_samples
                        if idx_end > len(t):
                            idx_end = len(t)
                            
                        total_samples = idx_end - idx_start
                        if total_samples > 0:
                            t_local = np.arange(total_samples) / SR
                            
                            # Pitch contour
                            f0_contour = np.linspace(f0_start, f0_end, total_samples)
                            
                            # FM Jitter (Perlin-like noise to make vibrato organic)
                            jitter_noise = np.random.normal(0, 1, total_samples)
                            b_j, a_j = signal.butter(1, 5, btype='lowpass', fs=SR)
                            jitter = signal.filtfilt(b_j, a_j, jitter_noise)
                            if np.std(jitter) > 0: jitter = jitter / np.std(jitter)
                            
                            fm = f0_contour + fm_depth * np.sin(2 * np.pi * fm_rate * t_local) + (f0_start * 0.02) * jitter
                            phase_acc = np.cumsum(fm) / SR
                            
                            # Amplitude Modulation (Organic chaotic fluttering)
                            am_noise = np.random.normal(0, 1, total_samples)
                            b_am, a_am = signal.butter(2, 10, btype='lowpass', fs=SR)
                            am_noise = signal.filtfilt(b_am, a_am, am_noise)
                            if np.std(am_noise) > 0: am_noise = am_noise / np.std(am_noise)
                            
                            # Narrowband noise to reduce synthetic purity
                            nb_noise = np.random.normal(0, 1, total_samples)
                            low_f = max(20, min(f0_start, f0_end) - 50)
                            high_f = min(SR/2 - 1, max(f0_start, f0_end) + 50)
                            # Increased filter order to 3 (filtfilt makes it 6th order) to prevent high-frequency leakage
                            b_nb, a_nb = signal.butter(3, [low_f, high_f], btype='bandpass', fs=SR)
                            nb_noise = signal.filtfilt(b_nb, a_nb, nb_noise)
                            if np.std(nb_noise) > 0: nb_noise = nb_noise / np.std(nb_noise)
                            
                            w = (np.sin(2 * np.pi * phase_acc) + 0.1 * nb_noise) * (1.0 + 0.3 * am_noise)
                            
                            # Envelope
                            local_env = np.ones(total_samples)
                            a_end = min(attack_samples, total_samples)
                            local_env[:a_end] = np.linspace(0, 1, a_end)
                            r_start = min(attack_samples + sustain_samples, total_samples)
                            if r_start < total_samples:
                                local_env[r_start:] = np.linspace(1, 0, total_samples - r_start)
                                
                            wheeze_sound[idx_start:idx_end] += w * local_env * (wheeze_intensity / n_wheezes)
            
    # -------------------------------------------------------------------------
    # 5. Crackles (Coarse, Early Inspiratory)
    # -------------------------------------------------------------------------
    crackle_sound = np.zeros_like(t)
    if np.random.rand() < crackle_prob:
        # Pre-generate a buffer of pink noise for bursts to naturally reduce high-frequency transients
        pink_buffer = generate_pink_noise(SR)
        
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
                        
                        # Short burst of pink noise with exponential decay
                        start_p = np.random.randint(0, len(pink_buffer) - actual_len)
                        burst = pink_buffer[start_p:start_p + actual_len]
                        decay = np.exp(-np.linspace(0, 5, actual_len))
                        crackle_sound[idx:end_idx] += burst * decay
                        
        # Bandpass filter crackles: 150 - 400 Hz (Coarse crackle profile)
        if np.any(crackle_sound):
            # Increased filter order to 4 to ensure high frequencies are heavily attenuated
            b_c, a_c = signal.butter(4, [150, 400], btype='bandpass', fs=SR)
            crackle_sound = signal.lfilter(b_c, a_c, crackle_sound)
            # Normalize crackles to have a reasonable peak
            c_max = np.max(np.abs(crackle_sound))
            if c_max > 0:
                crackle_sound = (crackle_sound / c_max) * np.random.uniform(0.15, 0.4)
                
    # -------------------------------------------------------------------------
    # 6. Background Noise (Room/Sensor)
    # -------------------------------------------------------------------------
    # Mix of low-frequency rumble (brown noise) and very faint, low-passed sensor noise.
    # COPD recordings lack high-frequency energy due to hyperinflation acting as an acoustic insulator.
    
    b_brown, a_brown = signal.butter(1, 50, btype='lowpass', fs=SR)
    brown_bg = signal.filtfilt(b_brown, a_brown, np.random.normal(0, 1, len(t)))
    if np.std(brown_bg) > 0: brown_bg = brown_bg / np.std(brown_bg)
    
    # Sensor hiss: heavily low-passed to avoid unnatural high ZCR
    bg_pink = generate_pink_noise(len(t))
    b_hiss, a_hiss = signal.butter(2, 400, btype='lowpass', fs=SR)
    bg_pink_lp = signal.filtfilt(b_hiss, a_hiss, bg_pink)
    if np.std(bg_pink_lp) > 0: bg_pink_lp = bg_pink_lp / np.std(bg_pink_lp)
    
    # Faint broadband noise floor to prevent digital silence in high frequencies (improves MFCC realism)
    noise_floor = bg_pink * np.random.uniform(0.0005, 0.0015)
    
    bg_noise = bg_pink_lp * np.random.uniform(0.001, 0.005) + brown_bg * np.random.uniform(0.01, 0.04) + noise_floor
        
    # -------------------------------------------------------------------------
    # 7. Combine
    # -------------------------------------------------------------------------
    audio = breath_sound + wheeze_sound + crackle_sound + bg_noise
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