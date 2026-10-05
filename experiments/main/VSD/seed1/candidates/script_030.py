"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur. 
The murmur begins with the first heart sound (S1) and extends to the 
second heart sound (S2), obscuring the systolic pause. The murmur is 
typically high-pitched and harsh due to the high-pressure gradient 
between the left and right ventricles.

The simulator models:
- S1 and S2 heart sounds as broadband transients (swept tones + noise).
- The VSD murmur as bandpass-filtered pink noise with a 350Hz resonance 
  and a holosystolic envelope that slightly overlaps S2.
- Heart rate variability (HRV) and respiratory amplitude modulation.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise modeled as low-passed pink noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a realistic heart sound (S1 or S2) using a frequency-swept
    tone with harmonics and low-frequency noise, shaped by an asymmetric envelope.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack (10ms), slower exponential decay
    attack_time = 0.01
    decay_time = duration - attack_time
    env = np.zeros_like(t)
    attack_idx = int(attack_time * fs)
    if attack_idx > 0:
        env[:attack_idx] = np.linspace(0, 1, attack_idx)
    if len(t) > attack_idx:
        env[attack_idx:] = np.exp(-5 * (t[attack_idx:] - attack_time) / decay_time)
        
    # Base frequency with rapid downward sweep
    f_t = f0 * np.exp(-15 * t)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Main tone + harmonics for a richer sound
    tone = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    
    # Add low-frequency noise for the "thud" characteristic
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = env * (tone + 0.5 * noise)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with a resonant peak to model fluid turbulence.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    # Generate pink noise via FFT
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples, d=1/fs)
    freqs[0] = freqs[1] # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    noise = np.fft.irfft(X_pink, n=num_samples)
    
    # Bandpass filter to give the murmur its characteristic pitch
    nyq = 0.5 * fs
    b, a = signal.butter(4, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add resonance at ~350 Hz to emphasize the "harsh" quality
    f_res = 350.0
    if freq_band[0] < f_res < freq_band[1]:
        b_res, a_res = signal.iirpeak(f_res / nyq, Q=1.5)
        res_sig = signal.filtfilt(b_res, a_res, murmur)
        murmur = murmur + 1.0 * res_sig
        
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    env = signal.windows.tukey(num_samples, alpha=0.1)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level):
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
        
        # Generate components
        s1 = generate_heart_sound(f0=60, duration=0.06, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=90, duration=0.05, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic. Extend slightly (20ms) to overlap with S2
        # to ensure there is no unnatural gap before S2.
        murmur_dur = systole_dur + 0.02
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band) * murmur_amp
        
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
        add_to_audio(murmur, beat_time) # Murmur starts with S1
        add_to_audio(s2, beat_time + systole_dur) # S2 marks the end of systole
        
    # Apply respiratory modulation (low frequency amplitude modulation)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.15 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add broadband pink-ish background noise to model ambient/sensor noise
    bg_white = np.random.randn(total_samples)
    bg_freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    bg_freqs[0] = bg_freqs[1]
    bg_pink = np.fft.irfft(np.fft.rfft(bg_white) / np.sqrt(bg_freqs), n=total_samples)
    
    b_bg, a_bg = signal.butter(2, 800 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b_bg, a_bg, bg_pink)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Apply chest wall / stethoscope filtering
    # High-pass to remove sub-bass rumble, low-pass to simulate tissue attenuation
    b_hp, a_hp = signal.butter(2, 30 / (0.5 * fs), btype='high')
    audio = signal.filtfilt(b_hp, a_hp, audio)
    
    b_lp, a_lp = signal.butter(1, 1000 / (0.5 * fs), btype='low')
    audio = signal.filtfilt(b_lp, a_lp, audio)
    
    return audio

def main():
    # Set explicit random seed for reproducibility
    np.random.seed(42)
    
    output_dir = "[PROJECT_ROOT]/artifacts/vsd_audio/runs/signal/seed1/generated/"
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
        
        s1_amp = np.random.uniform(0.5, 0.8)
        
        if severity < 0.5:
            # Restrictive: louder murmur (+3 to +6 dB relative to S1)
            murmur_ratio = np.random.uniform(1.4, 2.0)
        else:
            # Unrestrictive: softer murmur
            murmur_ratio = np.random.uniform(0.5, 1.0)
            
        murmur_amp = s1_amp * murmur_ratio
        
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(500, 800)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 0.8) + 0.3 * severity
        
        noise_level = np.random.uniform(0.02, 0.08)
        
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
            noise_level=noise_level
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