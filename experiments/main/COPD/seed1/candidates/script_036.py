"""
COPD Audio Simulator

This script generates synthetic audio samples representing lung sounds of patients 
with Chronic Obstructive Pulmonary Disease (COPD).

Physiological and Acoustic Mechanisms Modeled:
1. Altered Respiratory Cycle: COPD is characterized by airflow limitation, leading to 
   a significantly prolonged expiratory phase. The I:E ratio shifts to 1:3, 1:4, or higher.
2. Diminished Breath Sounds: Hyperinflation of the lungs acts as an acoustic insulator.
   Base breath sounds are simulated using pink noise with a dynamic low-pass filter.
3. Wheezes: Narrowed airways cause continuous, musical sounds (polyphonic wheezes).
   Simulated using additive synthesis with FM/AM jitter, phase noise, and harmonics.
4. Crackles: Coarse crackles due to secretions, simulated as bandpass-filtered noise 
   bursts clustered in early inspiration.
5. Background Noise: A two-component noise floor simulates the recording environment:
   a low-frequency body/sensor rumble (5-50 Hz) and a broadband room noise floor.

Refinement:
- Changed the breath noise low-pass filter from `sosfiltfilt` (24 dB/octave) to `sosfilt` 
  (12 dB/octave). This better matches the blueprint's 12-18 dB/octave specification, 
  populating the mid-frequencies to fix the negative MFCC shifts.
- Changed the broadband room noise floor from pink noise to brown noise (1/f^2). This 
  removes unnatural continuous high-frequency energy, significantly lowering the ZCR.
- Added narrowband phase noise to the wheeze oscillators and slightly reduced their 
  amplitude/count. This makes the wheezes less synthetic and prevents pure high-frequency 
  tones from artificially inflating the ZCR, better matching the visual evidence of real samples.
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
# Helper Functions
# =============================================================================

def generate_low_freq_noise(length, sr, cutoff_hz):
    """
    Generates low-pass filtered white noise using FFT.
    This guarantees absolute numerical stability for extremely low cutoffs
    (e.g., 5 Hz) where IIR filters (b, a) at 44.1kHz would introduce 
    severe high-frequency quantization noise.
    """
    X = np.fft.rfft(np.random.randn(length))
    freqs = np.fft.rfftfreq(length, 1/sr)
    X[freqs > cutoff_hz] = 0
    return np.fft.irfft(X, n=length)

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
    # 1. Continuous Background Noise Floor (Room/Sensor/Body Noise)
    # -------------------------------------------------------------------------
    freqs_bg = np.fft.rfftfreq(len(t), 1/SR)
    
    # Component A: Low-frequency body rumble and stethoscope coupling (5 - 50 Hz)
    X_rumble = np.fft.rfft(np.random.randn(len(t)))
    S_pink = np.sqrt(np.arange(1, len(X_rumble) + 1))
    X_rumble = X_rumble / S_pink
    X_rumble[(freqs_bg < 5.0) | (freqs_bg > 50.0)] = 0
    rumble = np.fft.irfft(X_rumble, n=len(t))
    rumble_amp = np.random.uniform(0.015, 0.035)
    rumble = (rumble / (np.std(rumble) + 1e-6)) * rumble_amp
    
    # Component B: Broadband room/sensor noise floor
    # Changed to brown noise to reduce unnatural high-frequency zero-crossings
    X_room = np.fft.rfft(np.random.randn(len(t)))
    S_brown = np.arange(1, len(X_room) + 1)
    X_room = X_room / S_brown
    X_room[freqs_bg < 20.0] = 0  # High-pass to avoid adding to the rumble
    room = np.fft.irfft(X_room, n=len(t))
    room_amp = np.random.uniform(0.001, 0.003)
    room = (room / (np.std(room) + 1e-6)) * room_amp
    
    audio += rumble + room
    
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
        
        # Apply low-pass filter to simulate hyperinflated lung attenuation.
        # Using sosfilt (forward only) for a 12 dB/octave roll-off, which matches
        # the blueprint's 12-18 dB/octave specification and populates mid-frequencies.
        cutoff = np.random.uniform(150, 250)
        sos_lp = signal.butter(2, cutoff, btype='lowpass', fs=SR, output='sos')
        breath_noise = signal.sosfilt(sos_lp, breath_noise)
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
                    sos_c = signal.butter(2, [150, 400], btype='bandpass', fs=SR, output='sos')
                    burst = signal.sosfiltfilt(sos_c, burst)
                    burst = burst[pad:pad+c_len]
                    burst = burst / (np.std(burst) + 1e-6)
                    
                    burst_env = signal.windows.hann(c_len)
                    crackle_amp = np.random.uniform(0.2, 0.4)
                    audio[c_idx:c_idx+c_len] += burst * burst_env * crackle_amp
                    
        # --- Polyphonic Wheezes ---
        if np.random.rand() < wheeze_prob:
            n_wheezes = np.random.randint(1, 4)
            for _ in range(n_wheezes):
                f0 = np.random.uniform(150, 500)
                fm_rate = np.random.uniform(4, 10)
                fm_depth = f0 * np.random.uniform(0.01, 0.03)
                
                # Triggered ~500 ms after the start of expiration
                delay = np.random.uniform(0.2, 0.5)
                w_start = idx + n_i + int(delay * SR)
                w_dur = np.random.uniform(1.0, 2.0)
                w_len = int(w_dur * SR)
                
                if w_start + w_len < len(t) and w_start < idx + n_breath:
                    w_len = min(w_len, idx + n_breath - w_start)
                    t_w = np.arange(w_len) / SR
                    
                    # FM Jitter for organic vibrato
                    jitter = generate_low_freq_noise(w_len, SR, 5.0)
                    jitter = (jitter / (np.std(jitter) + 1e-6)) * np.random.uniform(0.5, 1.0)
                    
                    fm = f0 + fm_depth * np.sin(2 * np.pi * fm_rate * t_w + jitter)
                    phase_acc = np.cumsum(fm) / SR
                    
                    # Phase noise to add narrowband noise texture and reduce synthetic purity
                    phase_noise = generate_low_freq_noise(w_len, SR, 30.0)
                    phase_noise = (phase_noise / (np.std(phase_noise) + 1e-6)) * 0.2
                    
                    # Base sine wave + subtle first harmonic
                    w = np.sin(2 * np.pi * phase_acc + phase_noise) + 0.1 * np.sin(4 * np.pi * phase_acc + phase_noise * 2)
                    
                    # AM Noise to reduce synthetic purity
                    am_filtered = generate_low_freq_noise(w_len, SR, 10.0)
                    am_filtered = am_filtered / (np.std(am_filtered) + 1e-6)
                    am_noise = 1.0 + 0.4 * am_filtered
                    w = w * am_noise
                    
                    # ADSR-like Envelope
                    att_len = min(int(0.3 * SR), w_len // 2)
                    rel_len = min(int(0.4 * SR), w_len // 2)
                    w_env = np.ones(w_len)
                    if att_len > 0:
                        w_env[:att_len] = np.linspace(0, 1, att_len)
                    if rel_len > 0:
                        w_env[-rel_len:] = np.linspace(1, 0, rel_len)
                        
                    wheeze_amp = np.random.uniform(0.1, 0.3)
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