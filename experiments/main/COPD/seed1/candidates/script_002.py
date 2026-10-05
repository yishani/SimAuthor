"""
COPD Audio Simulator

This script generates synthetic audio samples representing lung sounds of patients 
with Chronic Obstructive Pulmonary Disease (COPD).

Physiological and Acoustic Mechanisms Modeled:
1. Altered Respiratory Cycle: COPD is characterized by airflow limitation, leading to 
   a significantly prolonged expiratory phase. The I:E ratio shifts to 1:3, 1:4, or higher.
2. Diminished Breath Sounds: Hyperinflation of the lungs acts as an acoustic insulator.
   Base breath sounds are simulated using pink noise with a steep dynamic low-pass filter.
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes).
   Simulated using additive synthesis with FM/AM jitter and narrowband noise for organic texture.
4. Crackles: Coarse crackles due to secretions, simulated as bandpass-filtered noise 
   bursts clustered in early inspiration.
5. Background Noise: A continuous low-level, low-pass filtered noise floor simulates the 
   recording environment and sensor noise, preventing unnatural absolute silence and 
   maintaining a realistic Zero Crossing Rate (ZCR).
"""

import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/copd_audio/runs/signal/seed1/generated/"
SR = 44100          # Internal synthesis sample rate in Hz
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
    Synthesizes at 44.1kHz, then resamples to 16kHz.
    """
    np.random.seed(seed)
    t = np.arange(int(SR * DURATION)) / SR
    audio = np.zeros_like(t)
    
    # -------------------------------------------------------------------------
    # 1. Continuous Background Noise Floor (Room/Sensor Noise)
    # -------------------------------------------------------------------------
    # Generate low-pass filtered noise for the background to avoid absolute silence.
    # This simulates body/sensor noise and prevents the unnaturally high ZCR 
    # caused by broadband pink noise.
    bg_noise = np.random.randn(len(t))
    b_bg, a_bg = signal.butter(2, 250, 'lowpass', fs=SR)
    bg_noise = signal.filtfilt(b_bg, a_bg, bg_noise)
    bg_noise = (bg_noise / (np.std(bg_noise) + 1e-6)) * 0.005
    audio += bg_noise
    
    # -------------------------------------------------------------------------
    # 2. Breath-by-Breath Generation
    # -------------------------------------------------------------------------
    idx = 0
    while idx < len(t):
        # Sample physiological parameters based on severity
        if severity_level == 'mild':
            t_i = np.random.normal(1.2, 0.1)
            t_e = np.random.normal(3.0, 0.3)
            t_p = np.random.uniform(0.2, 0.4)
            breath_amp = np.random.uniform(0.15, 0.25)
            wheeze_prob = 0.3
            crackle_prob = 0.4
        elif severity_level == 'moderate':
            t_i = np.random.normal(1.2, 0.1)
            t_e = np.random.normal(4.0, 0.4)
            t_p = np.random.uniform(0.2, 0.5)
            breath_amp = np.random.uniform(0.1, 0.15)
            wheeze_prob = 0.7
            crackle_prob = 0.6
        else: # severe
            t_i = np.random.normal(1.0, 0.1)
            t_e = np.random.normal(5.0, 0.5)
            t_p = np.random.uniform(0.3, 0.6)
            breath_amp = np.random.uniform(0.05, 0.1)
            wheeze_prob = 0.9
            crackle_prob = 0.8
            
        n_i = int(t_i * SR)
        n_e = int(t_e * SR)
        n_p = int(t_p * SR)
        n_breath = n_i + n_e
        
        if idx + n_breath > len(t):
            break # Not enough space for a full breath, stop generation
            
        # --- Base Breath Sound (Vesicular) ---
        # Generate pink noise for this specific breath
        X_b = np.fft.rfft(np.random.randn(n_breath))
        breath_noise = np.fft.irfft(X_b / np.sqrt(np.arange(1, len(X_b) + 1)), n=n_breath)
        
        # Apply steep low-pass filter to simulate hyperinflated lung attenuation
        # Using 3rd order (18 dB/octave) to match blueprint and retain some mid-freqs
        cutoff = np.random.uniform(150, 250)
        b_lp, a_lp = signal.butter(3, cutoff, btype='lowpass', fs=SR)
        breath_noise = signal.filtfilt(b_lp, a_lp, breath_noise)
        breath_noise = breath_noise / (np.std(breath_noise) + 1e-6)
        
        # Create respiratory envelope
        env_i = np.sin(np.pi * np.arange(n_i) / n_i)
        phase_e = np.arange(n_e) / n_e
        env_e = np.sin(np.pi * phase_e) * np.exp(-3.0 * phase_e) # Rapid rise, long plateau
        
        breath_env = np.concatenate([env_i, env_e])
        audio[idx:idx+n_breath] += breath_noise * breath_env * breath_amp
        
        # --- Coarse Crackles ---
        if np.random.rand() < crackle_prob:
            # 3 to 8 crackles clustered in the early inspiratory phase
            n_crackles = np.random.randint(3, 9)
            crackle_region = min(n_i, int(0.4 * SR))
            crackle_times = np.random.uniform(0, crackle_region / SR, n_crackles)
            
            for ct in crackle_times:
                c_idx = idx + int(ct * SR)
                c_len = int(np.random.uniform(0.010, 0.015) * SR) # 10-15 ms burst
                if c_idx + c_len < len(t):
                    # Generate padded noise to avoid filter edge transients
                    pad = int(0.05 * SR)
                    burst = np.random.randn(c_len + 2 * pad)
                    b_c, a_c = signal.butter(2, [150, 400], btype='bandpass', fs=SR)
                    burst = signal.filtfilt(b_c, a_c, burst)
                    burst = burst[pad:pad+c_len]
                    burst = burst / (np.std(burst) + 1e-6)
                    
                    burst_env = signal.windows.hann(c_len)
                    crackle_amp = np.random.uniform(0.2, 0.4)
                    audio[c_idx:c_idx+c_len] += burst * burst_env * crackle_amp
                    
        # --- Polyphonic Wheezes ---
        if np.random.rand() < wheeze_prob:
            n_wheezes = np.random.randint(2, 5)
            for _ in range(n_wheezes):
                f0 = np.random.uniform(150, 600)
                fm_rate = np.random.uniform(5, 15)
                fm_depth = f0 * np.random.uniform(0.02, 0.05)
                
                # Triggered ~500 ms after the start of expiration
                delay = np.random.uniform(0.3, 0.6)
                w_start = idx + n_i + int(delay * SR)
                w_dur = np.random.uniform(1.5, 2.5)
                w_len = int(w_dur * SR)
                
                if w_start + w_len < len(t) and w_start < idx + n_breath:
                    w_len = min(w_len, idx + n_breath - w_start)
                    t_w = np.arange(w_len) / SR
                    
                    # FM Jitter for organic vibrato
                    jitter_noise = np.random.randn(w_len)
                    b_j, a_j = signal.butter(2, 5, 'lowpass', fs=SR)
                    jitter = signal.filtfilt(b_j, a_j, jitter_noise)
                    jitter = (jitter / (np.std(jitter) + 1e-6)) * np.random.uniform(0.5, 1.5)
                    
                    fm = f0 + fm_depth * np.sin(2 * np.pi * fm_rate * t_w + jitter)
                    phase_acc = np.cumsum(fm) / SR
                    w = np.sin(2 * np.pi * phase_acc)
                    
                    # AM Noise to reduce synthetic purity
                    am_raw = np.random.randn(w_len)
                    b_am, a_am = signal.butter(2, 10, 'lowpass', fs=SR)
                    am_filtered = signal.filtfilt(b_am, a_am, am_raw)
                    am_filtered = am_filtered / (np.std(am_filtered) + 1e-6)
                    am_noise = 1.0 + 0.3 * am_filtered
                    w = w * am_noise
                    
                    # Narrowband noise to further reduce synthetic purity (Blueprint pitfall)
                    nb_noise = np.random.randn(w_len)
                    b_nb, a_nb = signal.butter(2, [f0 - 50, f0 + 50], btype='bandpass', fs=SR)
                    nb_noise = signal.filtfilt(b_nb, a_nb, nb_noise)
                    nb_noise = (nb_noise / (np.std(nb_noise) + 1e-6)) * 0.15
                    w = w + nb_noise
                    
                    # ADSR-like Envelope
                    att_len = min(int(0.3 * SR), w_len // 2)
                    rel_len = min(int(0.4 * SR), w_len // 2)
                    w_env = np.ones(w_len)
                    if att_len > 0:
                        w_env[:att_len] = np.linspace(0, 1, att_len)
                    if rel_len > 0:
                        w_env[-rel_len:] = np.linspace(1, 0, rel_len)
                        
                    wheeze_amp = np.random.uniform(0.2, 0.5)
                    audio[w_start:w_start+w_len] += w * w_env * wheeze_amp
                    
        # Advance to next breath cycle (including pause)
        idx += n_breath + n_p
        
    # -------------------------------------------------------------------------
    # 3. Resample and Normalize
    # -------------------------------------------------------------------------
    # Resample to target sample rate (16000 Hz)
    audio_16k = signal.resample_poly(audio, TARGET_SR, SR)
    
    # Peak normalization to prevent clipping (target max amplitude = 0.95)
    max_val = np.max(np.abs(audio_16k))
    if max_val > 0:
        audio_16k = (audio_16k / max_val) * 0.95
        
    return audio_16k.astype(np.float32)

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

    for i, severity in enumerate(severities):
        sample_seed = SEED + i
        audio_signal = generate_copd_audio(severity_level=severity, seed=sample_seed)

        # Save to WAV file
        filename = f"copd_sample_{i:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        wavfile.write(filepath, TARGET_SR, audio_signal)
        print(f"Saved: {filename} (Severity: {severity})")
        
    print(f"All samples successfully saved to {OUTPUT_DIR}")