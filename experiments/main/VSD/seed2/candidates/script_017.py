"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur in membranous 
defects, or an early-to-mid systolic murmur in muscular defects. The murmur begins 
with the first heart sound (S1) and is typically high-pitched and harsh due to the 
high-pressure gradient between the left and right ventricles.

The simulator models:
- S1 and S2 heart sounds as broadband, low-frequency percussive transients.
- The VSD murmur as bandpass-filtered pink noise with a harsh resonance at ~350 Hz.
- Subtypes: Membranous (holosystolic plateau) and Muscular (decrescendo).
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary significantly with respiration).
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Realistic stethoscope acoustic transmission (bandpass filtering) to prevent 
  unrealistic high-frequency zero-crossing rates.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_pink_noise(num_samples):
    """
    Generates pink noise (1/f) using the frequency domain.
    Pink noise is preferable to white noise as it naturally models 
    the energy roll-off of fluid turbulence and prevents unrealistic ZCR.
    """
    if num_samples == 0:
        return np.array([])
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1]  # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    return np.fft.irfft(X_pink, n=num_samples)

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    tone with harmonics and noise, shaped by a percussive envelope.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack, slower decay
    attack_time = 0.015
    attack_samples = int(attack_time * fs)
    if attack_samples > len(t) // 2:
        attack_samples = len(t) // 2
        
    env = np.ones_like(t)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    env[attack_samples:] = np.exp(-5 * (t[attack_samples:] - attack_time) / (duration - attack_time))
    
    # Frequency modulation: rapid drop for a "thud" sound
    f_t = f0 * np.exp(-3 * t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base tone + harmonics for a broader spectrum
    tone = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase)
    
    # Add a little bit of noise for the transient impact
    noise = np.random.randn(len(t)) * 0.1
    
    sound = tone + noise
    
    # Low-pass filter to keep it sounding muffled like a real heart sound
    b, a = signal.butter(2, 200 / (0.5 * fs), btype='low')
    sound = signal.filtfilt(b, a, sound)
    
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, subtype='membranous'):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    Supports both membranous (holosystolic) and muscular (decrescendo) envelopes.
    """
    num_samples = int(duration * fs)
    if num_samples < 10:
        return np.zeros(num_samples)
        
    noise = generate_pink_noise(num_samples)
    
    nyq = 0.5 * fs
    # High-pass filter
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    # Low-pass filter
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance at ~350 Hz to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_peak, a_peak, murmur)
    
    murmur = murmur + 0.5 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    if subtype == 'membranous':
        # Holosystolic envelope: flat plateau spanning the entire duration
        env = signal.windows.tukey(num_samples, alpha=0.1)
    else:
        # Muscular envelope: decrescendo as the muscle contracts and closes the defect
        t = np.linspace(0, 1, num_samples)
        env = np.exp(-4 * t)
        attack_samples = int(0.02 * fs)
        if attack_samples > 0 and attack_samples < num_samples:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
            
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level, subtype):
    """
    Assembles a full PCG signal with VSD characteristics.
    """
    total_samples = int(duration_sec * fs)
    audio = np.zeros(total_samples)
    
    # Generate beat timestamps with Heart Rate Variability (HRV)
    beats = []
    current_time = np.random.uniform(0, 0.5)
    while current_time < duration_sec:
        beats.append(current_time)
        rr_mean = 60.0 / hr_mean
        rr_interval = np.random.normal(rr_mean, hrv_std)
        rr_interval = np.clip(rr_interval, 0.4, 1.5) # Constrain to physiological limits
        current_time += rr_interval
        
    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        s1_dur = np.random.uniform(0.05, 0.08)
        s2_dur = np.random.uniform(0.04, 0.06)
        
        # Generate components
        s1 = generate_heart_sound(f0=np.random.uniform(50, 80), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(70, 100), duration=s2_dur, fs=fs) * s2_amp
        
        if subtype == 'muscular':
            # Muscular VSD murmurs end before S2
            murmur_dur = systole_dur * np.random.uniform(0.5, 0.8)
            murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, subtype='muscular') * murmur_amp
        else:
            # Membranous VSD murmurs are holosystolic
            murmur = generate_murmur(duration=systole_dur, fs=fs, freq_band=murmur_band, subtype='membranous') * murmur_amp
        
        def add_to_audio(sig, start_time):
            start_idx = int(start_time * fs)
            end_idx = start_idx + len(sig)
            if start_idx < total_samples:
                if end_idx > total_samples:
                    sig = sig[:total_samples - start_idx]
                    end_idx = total_samples
                audio[start_idx:end_idx] += sig
                
        # Assemble the cycle
        add_to_audio(s1, beat_time)
        add_to_audio(murmur, beat_time) # Murmur starts synchronously with S1
        add_to_audio(s2, beat_time + systole_dur) # S2 marks the end of systole
        
    # Add low-frequency body/sensor noise
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient room noise (filtered to stethoscope range)
    ambient = generate_pink_noise(total_samples)
    b_amb, a_amb = signal.butter(2, 1000 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * np.random.uniform(0.005, 0.02)
    
    # Apply a master stethoscope filter (bandpass 30 - 2000 Hz)
    # This prevents unrealistic high-frequency energy and normalizes ZCR
    b_master, a_master = signal.butter(2, [30 / (0.5 * fs), 2000 / (0.5 * fs)], btype='band')
    audio = signal.filtfilt(b_master, a_master, audio)
    
    return audio

def main():
    # Set explicit random seed for reproducibility
    np.random.seed(42)
    
    output_dir = "[PROJECT_ROOT]/artifacts/vsd_audio/runs/signal/seed2/generated/"
    os.makedirs(output_dir, exist_ok=True)
    
    fs_internal = 44100
    fs_output = 16000
    duration_sec = 10.0
    num_samples = 100
    
    print(f"Generating {num_samples} VSD audio samples...")
    
    for i in range(num_samples):
        # Sample physiological parameters to ensure diversity
        hr_mean = np.random.uniform(65, 105)
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.3, 0.4)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        murmur_amp = np.random.uniform(0.6, 1.8) * (1.2 - 0.4 * severity)
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 900)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.4 * severity
        
        noise_level = np.random.uniform(0.05, 0.15)
        
        # VSD Subtype: 80% Membranous (holosystolic), 20% Muscular (early-to-mid systolic)
        subtype = np.random.choice(['membranous', 'muscular'], p=[0.8, 0.2])
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_ratio=systole_ratio,
            murmur_amp=murmur_amp,
            murmur_band=murmur_band,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            noise_level=noise_level,
            subtype=subtype
        )
        
        # Resample to target frequency (16000 Hz)
        audio_16k = signal.resample_poly(audio, fs_output, fs_internal)
        
        # Peak-normalize to prevent clipping and maximize dynamic range
        max_val = np.max(np.abs(audio_16k))
        if max_val > 0:
            audio_16k /= max_val
            
        # Convert to 16-bit PCM
        audio_16bit = np.int16(audio_16k * 32767)
        
        # Save to disk
        filename = os.path.join(output_dir, f"vsd_sample_{i:03d}.wav")
        wavfile.write(filename, fs_output, audio_16bit)
        
    print(f"Successfully generated {num_samples} samples in {output_dir}")

if __name__ == "__main__":
    main()