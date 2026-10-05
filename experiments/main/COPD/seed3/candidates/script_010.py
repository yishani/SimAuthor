"""
COPD Audio Simulator

This script generates synthetic audio samples representing lung sounds of patients 
with Chronic Obstructive Pulmonary Disease (COPD).

Physiological and Acoustic Mechanisms Modeled:
1. Altered Respiratory Cycle: COPD is characterized by airflow limitation, leading to 
   a significantly prolonged expiratory phase. The Inspiration:Expiration (I:E) ratio 
   shifts from a normal ~1:2 to 1:3, 1:4, or even higher. Cycle-by-cycle jitter is applied.
2. Diminished Breath Sounds: Hyperinflation of the lungs in COPD increases the amount 
   of air between the airways and the chest wall, muffling the normal vesicular breath sounds.
   Modeled using pink noise passed through a 12 dB/octave lowpass filter (150-250 Hz).
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration. Modeled with FM/AM modulated sine waves mixed with narrowband noise.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (coarse crackles).
   Modeled as bandpassed bursts of white noise clustered in early inspiration.
5. Silent Chest: In very severe cases, airflow is too restricted to produce wheezes, 
   resulting in extreme attenuation.
6. Body Rumble: A continuous low-frequency rumble is included to simulate stethoscope 
   handling and body noise, anchoring the Zero Crossing Rate (ZCR) to realistic levels.

The simulator generates 100 diverse samples across mild, moderate, and severe presentations.
"""

import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "artifacts/copd_audio/visual_signal_100iter/generated/"
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
    """
    np.random.seed(seed)
    total_samples = int(SR * DURATION)
    t = np.arange(total_samples) / SR
    
    # -------------------------------------------------------------------------
    # 1. Parameter Sampling based on Severity
    # -------------------------------------------------------------------------
    is_silent_chest = False
    
    if severity_level == 'mild':
        breath_amp = np.random.uniform(0.1, 0.15)
        wheeze_prob = 0.5
        wheeze_intensity = np.random.uniform(0.1, 0.15)
        crackle_prob = 0.3
    elif severity_level == 'moderate':
        breath_amp = np.random.uniform(0.08, 0.12)
        wheeze_prob = 0.8
        wheeze_intensity = np.random.uniform(0.15, 0.25)
        crackle_prob = 0.7
    else: # severe
        if np.random.rand() < 0.3:
            is_silent_chest = True
            breath_amp = np.random.uniform(0.02, 0.04)
            wheeze_prob = 0.1
            wheeze_intensity = np.random.uniform(0.05, 0.1)
            crackle_prob = 0.2
        else:
            breath_amp = np.random.uniform(0.05, 0.08)
            wheeze_prob = 1.0
            wheeze_intensity = np.random.uniform(0.2, 0.3)
            crackle_prob = 0.9
            
    # -------------------------------------------------------------------------
    # 2. Breathing Envelope Generation (Cycle-by-Cycle Jitter)
    # -------------------------------------------------------------------------
    env = np.zeros(total_samples)
    cycle_starts = []
    exp_starts = []
    
    current_idx = 0
    
    while current_idx < total_samples:
        # Jittered durations
        insp_dur = np.random.normal(1.2, 0.1)
        if severity_level == 'mild':
            exp_dur = np.random.normal(3.0, 0.3)
        elif severity_level == 'severe':
            exp_dur = np.random.normal(5.0, 0.4)
        else:
            exp_dur = np.random.normal(4.0, 0.3)
        pause_dur = np.random.uniform(0.2, 0.5)
        
        insp_len = int(insp_dur * SR)
        exp_len = int(exp_dur * SR)
        pause_len = int(pause_dur * SR)
        
        cycle_starts.append(current_idx)
        exp_starts.append(current_idx + insp_len)
        
        # Inspiration envelope
        if current_idx + insp_len <= total_samples:
            env[current_idx:current_idx+insp_len] = np.sin(np.pi * np.arange(insp_len) / insp_len)
        else:
            rem = total_samples - current_idx
            env[current_idx:] = np.sin(np.pi * np.arange(rem) / insp_len)
            break
            
        current_idx += insp_len
        
        # Expiration envelope (prolonged, decaying exponential)
        if current_idx + exp_len <= total_samples:
            env[current_idx:current_idx+exp_len] = (
                np.sin(np.pi * np.arange(exp_len) / exp_len) * np.exp(-3.0 * np.arange(exp_len) / exp_len)
            )
        else:
            rem = total_samples - current_idx
            env[current_idx:] = (
                np.sin(np.pi * np.arange(rem) / exp_len) * np.exp(-3.0 * np.arange(rem) / exp_len)
            )
            break
            
        current_idx += exp_len + pause_len
        
    # -------------------------------------------------------------------------
    # 3. Base Breath Sounds (Muffled Pink Noise)
    # -------------------------------------------------------------------------
    white_noise = np.random.normal(0, 1, total_samples)
    # Simple pink noise approximation (-6 dB/octave)
    pink_noise = signal.lfilter([1.0], [1.0, -0.95], white_noise)
    
    # 2nd order lowpass filter (12 dB/octave) to simulate acoustic insulation
    # Reduced from 4th order to allow realistic mid-frequency breath sounds
    cutoff = np.random.uniform(150, 250)
    b, a = signal.butter(2, cutoff, btype='lowpass', fs=SR)
    breath_sound = signal.filtfilt(b, a, pink_noise)
    
    max_b = np.max(np.abs(breath_sound))
    if max_b > 0:
        breath_sound = breath_sound / max_b
        
    breath_sound = breath_sound * env * breath_amp
    
    # -------------------------------------------------------------------------
    # 4. Wheezes (Polyphonic Continuous Sounds)
    # -------------------------------------------------------------------------
    wheeze_sound = np.zeros(total_samples)
    if np.random.rand() < wheeze_prob:
        n_wheezes = np.random.randint(2, 5) # 2 to 4 polyphonic wheezes
        for _ in range(n_wheezes):
            f0 = np.random.uniform(150, 600)
            fm_rate = np.random.uniform(5, 15)
            fm_depth = f0 * np.random.uniform(0.02, 0.05)
            
            # FM modulation for organic fluttering + random wander
            b_w, a_w = signal.butter(2, 2.0, btype='lowpass', fs=SR)
            wander = signal.filtfilt(b_w, a_w, np.random.normal(0, 1, total_samples))
            wander = wander / (np.std(wander) + 1e-8)
            fm = f0 + fm_depth * np.sin(2 * np.pi * fm_rate * t) + (f0 * 0.03 * wander)
            
            phase_acc = np.cumsum(fm) / SR
            w = np.sin(2 * np.pi * phase_acc)
            
            # Add narrowband noise to avoid synthetic purity
            bw = 25.0
            b_nb, a_nb = signal.butter(2, [max(20, f0 - bw), min(SR/2 - 20, f0 + bw)], btype='bandpass', fs=SR)
            nb_noise = signal.filtfilt(b_nb, a_nb, np.random.normal(0, 1, total_samples))
            nb_noise = nb_noise / (np.std(nb_noise) + 1e-8) * 0.3
            
            w = w + nb_noise
            
            # AM modulation with normalized lowpass noise
            b_n, a_n = signal.butter(2, 50, btype='lowpass', fs=SR)
            noise_lp = signal.filtfilt(b_n, a_n, np.random.normal(0, 1, total_samples))
            noise_lp = noise_lp / (np.std(noise_lp) + 1e-8)
            w = w * (1.0 + 0.3 * noise_lp)
            
            w_env = np.zeros(total_samples)
            for i, exp_start in enumerate(exp_starts):
                # Trigger 500 ms after expiration starts
                trigger_idx = exp_start + int(0.5 * SR)
                if trigger_idx < total_samples:
                    attack_len = int(0.3 * SR)
                    sustain_len = int(np.random.uniform(1.5, 2.5) * SR)
                    release_len = int(0.4 * SR)
                    
                    if i + 1 < len(cycle_starts):
                        exp_end = cycle_starts[i+1]
                    else:
                        exp_end = total_samples
                        
                    max_len = exp_end - trigger_idx
                    if max_len < attack_len + release_len:
                        continue
                        
                    sustain_len = min(sustain_len, max_len - attack_len - release_len)
                    total_len = attack_len + sustain_len + release_len
                    
                    env_segment = np.concatenate([
                        np.linspace(0, 1, attack_len),
                        np.ones(sustain_len),
                        np.linspace(1, 0, release_len)
                    ])
                    w_env[trigger_idx:trigger_idx+total_len] = env_segment
                    
            wheeze_sound += w * w_env * (wheeze_intensity / n_wheezes)
            
    # -------------------------------------------------------------------------
    # 5. Crackles (Coarse, Early Inspiratory)
    # -------------------------------------------------------------------------
    crackle_sound = np.zeros(total_samples)
    if np.random.rand() < crackle_prob:
        for i, start_idx in enumerate(cycle_starts):
            n_crackles_cycle = np.random.randint(3, 9)
            # Clustered in the first 400 ms of inspiration
            max_offset = int(0.4 * SR)
            
            insp_end = exp_starts[i]
            insp_len = insp_end - start_idx
            max_offset = min(max_offset, insp_len)
            
            if max_offset > 0:
                offsets = np.random.randint(0, max_offset, n_crackles_cycle)
                for offset in offsets:
                    idx = start_idx + offset
                    burst_len = int(np.random.uniform(0.010, 0.015) * SR)
                    if idx + burst_len < total_samples:
                        burst = np.random.normal(0, 1, burst_len)
                        window = signal.windows.hann(burst_len)
                        burst = burst * window
                        crackle_sound[idx:idx+burst_len] += burst
                        
        if np.any(crackle_sound):
            # Bandpass filter to 150-400 Hz
            b_c, a_c = signal.butter(4, [150, 400], btype='bandpass', fs=SR)
            crackle_sound = signal.filtfilt(b_c, a_c, crackle_sound)
            max_c = np.max(np.abs(crackle_sound))
            if max_c > 0:
                crackle_sound = (crackle_sound / max_c) * np.random.uniform(0.05, 0.15)
                
    # -------------------------------------------------------------------------
    # 6. Noise Floor & Combination
    # -------------------------------------------------------------------------
    # Body rumble (stethoscope handling, heart, muscle noise) - anchors ZCR during pauses
    b_rumble, a_rumble = signal.butter(2, 40, btype='lowpass', fs=SR)
    rumble = signal.filtfilt(b_rumble, a_rumble, np.random.normal(0, 1, total_samples))
    rumble = (rumble / (np.std(rumble) + 1e-8)) * np.random.uniform(0.005, 0.015)
    
    # Ambient room noise (very quiet, gently lowpassed)
    b_amb, a_amb = signal.butter(2, 2000, btype='lowpass', fs=SR)
    ambient = signal.filtfilt(b_amb, a_amb, np.random.normal(0, 1, total_samples))
    ambient = (ambient / (np.std(ambient) + 1e-8)) * 0.0005
    
    noise_floor = rumble + ambient
    
    audio = breath_sound + wheeze_sound + crackle_sound + noise_floor
    
    # Peak normalization only if clipping occurs, preserving the quiet nature of COPD
    max_val = np.max(np.abs(audio))
    if max_val > 0.95:
        audio = (audio / max_val) * 0.95
        
    return audio.astype(np.float32)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Define severity distribution for exactly 100 samples
    severities = ['mild']*33 + ['moderate']*33 + ['severe']*34
    
    print(f"Generating {N_SAMPLES} COPD audio samples...")
    
    for i, severity in enumerate(severities):
        sample_seed = SEED + i
        audio_signal = generate_copd_audio(severity_level=severity, seed=sample_seed)
        
        # Resample to target sample rate (16000 Hz) before saving
        audio_16k = signal.resample_poly(audio_signal, TARGET_SR, SR)
        
        # Save to WAV file
        filename = f"copd_sample_{i:03d}.wav"
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        wavfile.write(filepath, TARGET_SR, audio_16k.astype(np.float32))
        
    print(f"All samples successfully saved to {OUTPUT_DIR}")