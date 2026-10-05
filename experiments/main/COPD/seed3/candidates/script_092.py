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
   Modeled using pink noise passed through a steep 24 dB/octave (4th order) lowpass filter, 
   with cutoff varying by severity to ensure realistic spectral diversity and severe high-frequency attenuation.
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration. Modeled with FM/AM modulated sine waves and wandering pitch.
   Artificial harmonics are avoided to allow organic polyphony via multiple independent wheezes.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (coarse crackles).
   Modeled as asymmetrically enveloped bursts of noise, bandpass-filtered (100-800 Hz) to preserve 
   broadband transient characteristics of coarse crackles.
5. Body Rumble & Noise Floor: A dominant low-frequency body rumble (< 150 Hz, 4th order lowpass) 
   and a dark room noise floor are included to simulate the acoustic environment of an electronic 
   stethoscope, which naturally anchors the zero-crossing rate and matches empirical spectral slopes.
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
        breath_amp = np.random.uniform(0.04, 0.10)
        wheeze_prob = 0.4
        wheeze_intensity = np.random.uniform(0.02, 0.06)
        crackle_prob = 0.4
        cutoff = np.random.uniform(150, 300)
    elif severity_level == 'moderate':
        breath_amp = np.random.uniform(0.03, 0.07)
        wheeze_prob = 0.7
        wheeze_intensity = np.random.uniform(0.04, 0.09)
        crackle_prob = 0.7
        cutoff = np.random.uniform(100, 200)
    else: # severe
        if np.random.rand() < 0.3:
            is_silent_chest = True
            breath_amp = np.random.uniform(0.01, 0.03)
            wheeze_prob = 0.2
            wheeze_intensity = np.random.uniform(0.01, 0.04)
            crackle_prob = 0.3
            cutoff = np.random.uniform(60, 120)
        else:
            breath_amp = np.random.uniform(0.02, 0.05)
            wheeze_prob = 0.9
            wheeze_intensity = np.random.uniform(0.06, 0.12)
            crackle_prob = 0.9
            cutoff = np.random.uniform(80, 150)
            
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
    
    # 4th order lowpass filter (24 dB/octave) to simulate heavy acoustic insulation
    # Characteristic of hyperinflated COPD lungs, steepening the spectral roll-off
    sos = signal.butter(4, cutoff, btype='lowpass', fs=SR, output='sos')
    breath_sound = signal.sosfiltfilt(sos, pink_noise)
    
    max_b = np.max(np.abs(breath_sound))
    if max_b > 0:
        breath_sound = breath_sound / max_b
        
    breath_sound = breath_sound * env * breath_amp
    
    # -------------------------------------------------------------------------
    # 4. Wheezes (Polyphonic Continuous Sounds)
    # -------------------------------------------------------------------------
    wheeze_sound = np.zeros(total_samples)
    if np.random.rand() < wheeze_prob:
        n_wheezes = np.random.randint(1, 5) # 1 to 4 polyphonic wheezes
        for _ in range(n_wheezes):
            f0 = np.random.uniform(150, 600)
            fm_rate = np.random.uniform(5, 15)
            fm_depth_vib = f0 * np.random.uniform(0.01, 0.03)
            fm_depth_wander = f0 * np.random.uniform(0.05, 0.15)
            
            # FM modulation with stochastic jitter for organic wandering pitch
            fm_jitter = signal.filtfilt(*signal.butter(2, 1, btype='lowpass', fs=SR), np.random.normal(0, 1, total_samples))
            max_j = np.max(np.abs(fm_jitter))
            if max_j > 0:
                fm_jitter = fm_jitter / max_j
                
            fm = f0 + fm_depth_vib * np.sin(2 * np.pi * fm_rate * t) + fm_depth_wander * fm_jitter
            phase_acc = np.cumsum(fm) / SR
            
            # Base sine (removed artificial harmonics for more organic polyphony via multiple independent wheezes)
            w_sine = np.sin(2 * np.pi * phase_acc)
            
            # Add narrowband noise to avoid synthetic purity
            w_noise = np.random.normal(0, 1, total_samples)
            b_wn, a_wn = signal.butter(2, [max(20, f0 - 30), min(SR/2 - 1, f0 + 30)], btype='bandpass', fs=SR)
            w_noise = signal.filtfilt(b_wn, a_wn, w_noise)
            max_wn = np.max(np.abs(w_noise))
            if max_wn > 0:
                w_noise = w_noise / max_wn
                
            w = w_sine + 0.2 * w_noise
            
            # AM modulation with lowpass noise for organic amplitude fluctuation
            noise = np.random.normal(0, 1, total_samples)
            b_n, a_n = signal.butter(2, 10, btype='lowpass', fs=SR)
            noise_lp = signal.filtfilt(b_n, a_n, noise)
            max_nlp = np.max(np.abs(noise_lp))
            if max_nlp > 0:
                noise_lp = noise_lp / max_nlp
            w = w * (1.0 + 0.5 * noise_lp)
            
            w_env = np.zeros(total_samples)
            for i, exp_start in enumerate(exp_starts):
                # Trigger 500 ms after expiration starts
                trigger_idx = exp_start + int(0.5 * SR)
                if trigger_idx < total_samples:
                    attack_len = int(0.3 * SR)
                    sustain_len = int(np.random.uniform(1.0, 2.5) * SR)
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
                    
                    # Multiply by the expiration envelope to give it a natural decay
                    exp_env = env[trigger_idx:trigger_idx+total_len]
                    if np.max(exp_env) > 0:
                        exp_env = exp_env / np.max(exp_env)
                        
                    w_env[trigger_idx:trigger_idx+total_len] = env_segment * exp_env
                    
            wheeze_sound += w * w_env * (wheeze_intensity / n_wheezes)
            
    # -------------------------------------------------------------------------
    # 5. Crackles (Coarse, Early Inspiratory)
    # -------------------------------------------------------------------------
    crackle_sound = np.zeros(total_samples)
    if np.random.rand() < crackle_prob:
        for i, start_idx in enumerate(cycle_starts):
            n_crackles_cycle = np.random.randint(3, 9) # 3 to 8 crackles per blueprint
            # Clustered in the first 400 ms of inspiration
            max_offset = int(0.4 * SR)
            
            insp_end = exp_starts[i]
            insp_len = insp_end - start_idx
            max_offset = min(max_offset, insp_len)
            
            if max_offset > 0:
                offsets = np.random.randint(0, max_offset, n_crackles_cycle)
                for offset in offsets:
                    idx = start_idx + offset
                    burst_len = int(np.random.uniform(0.010, 0.020) * SR) # 10-20 ms per blueprint
                    if idx + burst_len < total_samples:
                        burst = np.random.normal(0, 1, burst_len)
                        # Asymmetric window for explosive transient: fast attack, exponential decay
                        attack_len = int(0.1 * burst_len)
                        decay_len = burst_len - attack_len
                        window = np.concatenate([
                            np.linspace(0, 1, attack_len),
                            np.exp(-4.0 * np.arange(decay_len) / decay_len)
                        ])
                        burst = burst * window
                        crackle_sound[idx:idx+burst_len] += burst
                        
        if np.any(crackle_sound):
            # Bandpass filter 100-800 Hz for coarse crackles (COPD)
            # Wider bandpass preserves transient pop while keeping dominant energy low
            sos_c = signal.butter(2, [100, 800], btype='bandpass', fs=SR, output='sos')
            crackle_filtered = signal.sosfiltfilt(sos_c, crackle_sound)
            
            max_c = np.max(np.abs(crackle_filtered))
            if max_c > 0:
                crackle_sound = (crackle_filtered / max_c) * np.random.uniform(0.05, 0.25)
                
    # -------------------------------------------------------------------------
    # 6. Noise Floor & Combination
    # -------------------------------------------------------------------------
    # Add a realistic stethoscope noise floor (body rumble + room noise)
    
    # Body rumble (< 150 Hz) dominates the zero-crossings and provides a realistic baseline
    rumble_white = np.random.normal(0, 1, total_samples)
    # 4th order lowpass for steep roll-off, preventing HF leakage that inflates ZCR
    sos_r = signal.butter(4, np.random.uniform(80, 150), btype='lowpass', fs=SR, output='sos')
    body_rumble = signal.sosfiltfilt(sos_r, rumble_white)
    max_r = np.max(np.abs(body_rumble))
    if max_r > 0:
        # Strong low-frequency baseline to anchor ZCR against wheezes/crackles
        body_rumble = (body_rumble / max_r) * np.random.uniform(0.4, 0.9)
        
    # Subtle room/sensor noise (darker noise floor)
    room_white = np.random.normal(0, 1, total_samples)
    # 2nd order lowpass at 250 Hz to simulate a muffled acoustic environment and reduce HF noise
    sos_room = signal.butter(2, 250, btype='lowpass', fs=SR, output='sos')
    room_noise = signal.sosfiltfilt(sos_room, room_white)
    max_p = np.max(np.abs(room_noise))
    if max_p > 0:
        room_noise = (room_noise / max_p) * np.random.uniform(0.005, 0.03)
        
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