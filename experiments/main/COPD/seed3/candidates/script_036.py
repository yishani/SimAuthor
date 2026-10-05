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
   Modeled using pink noise passed through an 18 dB/octave lowpass filter (150-250 Hz).
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration. Modeled with FM/AM modulated sine waves (with harmonics) 
   mixed with narrowband noise for organic texture.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (coarse crackles).
   Modeled as broadband bursts of noise (100-600 Hz) clustered in early inspiration.
5. Body Rumble & Noise Floor: A dominant low-frequency body rumble (< 80 Hz) is included 
   to simulate the acoustic environment of an electronic stethoscope and anchor the zero-crossing 
   rate, along with a gentle broadband room noise to provide realistic high-frequency texture.
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
    
    # 3rd order lowpass filter (18 dB/octave) to simulate severe acoustic insulation
    cutoff = np.random.uniform(150, 250)
    b, a = signal.butter(3, cutoff, btype='lowpass', fs=SR)
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
            
            # FM modulation for organic fluttering
            fm = f0 + fm_depth * np.sin(2 * np.pi * fm_rate * t)
            phase_acc = np.cumsum(fm) / SR
            
            # Base sine wave + 2nd harmonic for a more musical, less synthetic tone
            w_sine = np.sin(2 * np.pi * phase_acc) + 0.25 * np.sin(4 * np.pi * phase_acc)
            
            # Add narrowband noise to avoid synthetic purity
            w_noise = np.random.normal(0, 1, total_samples)
            b_wn, a_wn = signal.butter(2, [max(50, f0 - 50), min(SR/2 - 1, f0 + 50)], btype='bandpass', fs=SR)
            w_noise = signal.filtfilt(b_wn, a_wn, w_noise)
            max_wn = np.max(np.abs(w_noise))
            if max_wn > 0:
                w_noise = w_noise / max_wn
                
            w = w_sine + 0.2 * w_noise
            
            # AM modulation with lowpass noise
            noise = np.random.normal(0, 1, total_samples)
            b_n, a_n = signal.butter(2, 50, btype='lowpass', fs=SR)
            noise_lp = signal.filtfilt(b_n, a_n, noise)
            w = w * (1.0 + 0.2 * noise_lp)
            
            w_env = np.zeros(total_samples)
            for i, exp_start in enumerate(exp_starts):
                # Trigger organically during expiration
                trigger_idx = exp_start + int(np.random.uniform(0.3, 0.6) * SR)
                if trigger_idx < total_samples:
                    total_len = int(np.random.uniform(1.0, 2.5) * SR)
                    
                    if i + 1 < len(cycle_starts):
                        exp_end = cycle_starts[i+1]
                    else:
                        exp_end = total_samples
                        
                    max_len = exp_end - trigger_idx
                    if max_len < int(0.5 * SR):
                        continue
                        
                    total_len = min(total_len, max_len)
                    
                    # Tukey window provides a smooth, organic envelope without blocky edges
                    env_segment = signal.windows.tukey(total_len, alpha=0.4)
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
            # Bandpass filter to 100-600 Hz for coarse broadband transients
            b_c, a_c = signal.butter(2, [100, 600], btype='bandpass', fs=SR)
            crackle_sound = signal.filtfilt(b_c, a_c, crackle_sound)
            max_c = np.max(np.abs(crackle_sound))
            if max_c > 0:
                crackle_sound = (crackle_sound / max_c) * np.random.uniform(0.05, 0.15)
                
    # -------------------------------------------------------------------------
    # 6. Noise Floor & Combination
    # -------------------------------------------------------------------------
    # Strong body rumble (< 80 Hz) dominates the zero-crossings and provides a realistic baseline
    rumble_white = np.random.normal(0, 1, total_samples)
    b_r, a_r = signal.butter(2, 80, btype='lowpass', fs=SR)
    body_rumble = signal.filtfilt(b_r, a_r, rumble_white)
    max_r = np.max(np.abs(body_rumble))
    if max_r > 0:
        body_rumble = (body_rumble / max_r) * np.random.uniform(0.05, 0.10)
        
    # Broadband ambient/sensor noise to provide realistic high-frequency texture
    room_white = np.random.normal(0, 1, total_samples)
    b_room, a_room = signal.butter(1, 2000, btype='lowpass', fs=SR)
    room_noise = signal.filtfilt(b_room, a_room, room_white)
    max_p = np.max(np.abs(room_noise))
    if max_p > 0:
        room_noise = (room_noise / max_p) * np.random.uniform(0.005, 0.015)
        
    noise_floor = body_rumble + room_noise
    
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