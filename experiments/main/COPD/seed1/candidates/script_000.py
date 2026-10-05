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
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes), 
   predominantly during expiration.
4. Crackles: Secretions in the airways can cause discontinuous popping sounds (crackles).

The simulator generates 20 diverse samples across mild, moderate, and severe presentations.
"""

import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/copd_audio/runs/signal/seed1/generated/"
SR = 44100          # Sample rate in Hz
DURATION = 20.0     # Duration in seconds
N_SAMPLES = 100      # Number of samples to generate
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
        breath_amp = np.random.uniform(0.7, 0.9)      # Amplitude of base breath sounds
        wheeze_prob = 0.5                             # Probability of wheezing
        wheeze_intensity = np.random.uniform(0.1, 0.3)
        crackle_density = np.random.uniform(0, 5)     # Crackles per second
    elif severity_level == 'moderate':
        rr = np.random.uniform(16, 22)
        ie_ratio = np.random.uniform(3.5, 4.5)
        breath_amp = np.random.uniform(0.5, 0.7)      # More muffled due to hyperinflation
        wheeze_prob = 0.8
        wheeze_intensity = np.random.uniform(0.3, 0.6)
        crackle_density = np.random.uniform(5, 15)
    else: # severe
        rr = np.random.uniform(20, 26)                # Tachypnea
        ie_ratio = np.random.uniform(4.5, 6.0)        # Severely prolonged expiration
        breath_amp = np.random.uniform(0.3, 0.5)      # Significantly diminished sounds
        wheeze_prob = 1.0
        wheeze_intensity = np.random.uniform(0.6, 0.9)
        crackle_density = np.random.uniform(10, 25)
        
    # -------------------------------------------------------------------------
    # 2. Breathing Envelope Generation
    # -------------------------------------------------------------------------
    cycle_len = 60.0 / rr
    phase = (t % cycle_len) / cycle_len
    p_i = 1.0 / (1.0 + ie_ratio) # Proportion of cycle spent in inspiration
    
    env = np.zeros_like(t)
    mask_i = phase < p_i
    mask_e = phase >= p_i
    
    # Inspiration: relatively sharp and short
    env[mask_i] = np.sin(np.pi * phase[mask_i] / p_i)
    
    # Expiration: prolonged, decaying exponential shape characteristic of obstruction
    env[mask_e] = (np.sin(np.pi * (phase[mask_e] - p_i) / (1.0 - p_i)) * 
                   np.exp(-3.0 * (phase[mask_e] - p_i) / (1.0 - p_i)))
    
    # -------------------------------------------------------------------------
    # 3. Base Breath Sounds (Vesicular/Bronchial Noise)
    # -------------------------------------------------------------------------
    # Start with white noise
    white_noise = np.random.normal(0, 1, len(t))
    
    # Bandpass filter to simulate airflow turbulence (100 - 800 Hz)
    b, a = signal.butter(4, [100, 800], btype='bandpass', fs=SR)
    breath_sound = signal.filtfilt(b, a, white_noise)
    
    # Lowpass filter to simulate tissue muffling (chest wall attenuation)
    b_lp, a_lp = signal.butter(2, 400, btype='lowpass', fs=SR)
    breath_sound = signal.filtfilt(b_lp, a_lp, breath_sound)
    
    # Apply envelope and amplitude
    breath_sound = breath_sound * env * breath_amp
    
    # -------------------------------------------------------------------------
    # 4. Wheezes (Polyphonic Continuous Sounds)
    # -------------------------------------------------------------------------
    wheeze_sound = np.zeros_like(t)
    if np.random.rand() < wheeze_prob:
        n_wheezes = np.random.randint(1, 4) # Polyphonic (multiple airways)
        for _ in range(n_wheezes):
            f0 = np.random.uniform(250, 600)
            # Add slight frequency drift (FM) to make it sound organic
            fm = f0 + np.random.uniform(10, 30) * np.sin(2 * np.pi * np.random.uniform(0.2, 0.8) * t)
            phase_acc = np.cumsum(fm) / SR
            
            # Base sine wave + first harmonic
            w = np.sin(2 * np.pi * phase_acc) + 0.3 * np.sin(2 * np.pi * 2 * phase_acc)
            
            # Wheezes in COPD are predominantly expiratory
            w_env = np.zeros_like(t)
            w_env[mask_e] = env[mask_e] ** 1.5 # Sharpen the envelope
            
            # Occasionally add inspiratory wheezing in more severe cases
            if np.random.rand() < 0.3:
                w_env[mask_i] = env[mask_i] ** 1.5
                
            wheeze_sound += w * w_env * (wheeze_intensity / n_wheezes)
            
    # -------------------------------------------------------------------------
    # 5. Crackles (Discontinuous Explosive Sounds)
    # -------------------------------------------------------------------------
    crackle_sound = np.zeros_like(t)
    n_crackles = int(DURATION * crackle_density)
    if n_crackles > 0:
        # Distribute crackles randomly in time
        times = np.random.uniform(0, DURATION, n_crackles)
        # Only keep crackles that occur during active airflow (envelope > threshold)
        valid_times = [tm for tm in times if env[int(tm * SR)] > 0.1]
        
        impulses = np.zeros_like(t)
        for tm in valid_times:
            idx = int(tm * SR)
            if idx < len(t):
                impulses[idx] = np.random.choice([-1.0, 1.0]) * np.random.uniform(0.5, 1.0)
                
        # Create a crackle kernel (short damped oscillation)
        k_t = np.arange(0, 0.02, 1/SR)
        crackle_freq = np.random.uniform(200, 600)
        decay = np.random.uniform(300, 600) # Higher decay = finer crackles
        kernel = np.sin(2 * np.pi * crackle_freq * k_t) * np.exp(-decay * k_t)
        
        crackle_sound = signal.convolve(impulses, kernel, mode='same')
        crackle_sound = crackle_sound * 0.5 # Scale crackle amplitude
        
    # -------------------------------------------------------------------------
    # 6. Combine and Normalize
    # -------------------------------------------------------------------------
    audio = breath_sound + wheeze_sound + crackle_sound
    
    # Peak normalization to prevent clipping (target max amplitude = 0.95)
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = (audio / max_val) * 0.95
        
    return audio.astype(np.float32)

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Severity distribution scaled to N_SAMPLES: 30% mild, 35% moderate,
    # 35% severe. At N_SAMPLES=100 this reproduces the original 6/7/7 split;
    # at N_SAMPLES=100 it scales to 30/35/35, so the generation contract can
    # be patched to 100 without a hardcoded file count.
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
        
        wavfile.write(filepath, SR, audio_signal)
        print(f"Saved: {filename} (Severity: {severity})")
        
    print(f"All samples successfully saved to {OUTPUT_DIR}")