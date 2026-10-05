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

Refinements in this version:
- Replaced strict bandpass filter with independent 4th-order HPF and LPF 
  plus a 350 Hz resonance to better match the "harsh" spectral roll-off.
- Improved S1/S2 to be broadband transients rather than pure sine waves.
- Added broadband pink ambient noise and deep rumble to fix missing mid/high 
  frequency energy and correct the Zero Crossing Rate (ZCR).
- Introduced a 20% probability for Muscular VSD (shorter, decrescendo murmur) 
  to increase physiological diversity.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) as a broadband transient.
    Combines a low-frequency sweep (thud) with band-limited noise (valve click).
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack, slower exponential decay
    env = np.exp(-t / (duration / 4)) * np.sin(np.pi * t / duration)
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Low-frequency sweep for the main "thud"
    f_t = f0 * np.exp(-3 * t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    thud = np.sin(phase)
    
    # Band-limited noise for the valve closure "click"
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, [50 / (0.5 * fs), 400 / (0.5 * fs)], btype='band')
    click = signal.filtfilt(b, a, noise)
    
    sound = env * (thud + 0.25 * click)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, vsd_type='membranous'):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    Uses pink noise with specific HPF/LPF and resonance to model fluid turbulence.
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
    
    nyq = 0.5 * fs
    
    # 4th-order Butterworth HPF and LPF as per scientific blueprint
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a slight resonance at 350 Hz to emphasize the "harsh" quality
    b_res, a_res = signal.iirpeak(350 / nyq, 1.5)
    resonance = signal.filtfilt(b_res, a_res, noise)
    
    murmur = murmur + 0.4 * resonance
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Apply envelope based on VSD subtype
    if vsd_type == 'muscular':
        # Decrescendo envelope for muscular VSD
        t = np.linspace(0, 1, num_samples)
        env = np.exp(-3 * t)
        # Smooth attack (15ms)
        attack_samples = int(0.015 * fs)
        if attack_samples > 0 and attack_samples < num_samples:
            env[:attack_samples] *= np.linspace(0, 1, attack_samples)
    else:
        # Holosystolic plateau for membranous VSD
        env = signal.windows.tukey(num_samples, alpha=0.1)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level, vsd_type):
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
        s1 = generate_heart_sound(f0=80, duration=0.06, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=120, duration=0.05, fs=fs) * s2_amp
        
        # Determine murmur duration based on subtype
        if vsd_type == 'muscular':
            # Muscular VSD murmur ends before S2
            murmur_dur = max(0.1, systole_dur - np.random.uniform(0.05, 0.10))
        else:
            # Membranous VSD murmur is holosystolic
            murmur_dur = systole_dur
            
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, vsd_type=vsd_type) * murmur_amp
        
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
        
    # Apply respiratory modulation (low frequency amplitude modulation)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.15 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add realistic ambient and sensor noise
    # 1. Low-frequency rumble (body/sensor movement)
    rumble = np.random.randn(total_samples)
    b_r, a_r = signal.butter(2, 50 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b_r, a_r, rumble)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    # 2. Broadband ambient noise (pinkish)
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    X_pink = X_white / np.sqrt(freqs)
    ambient = np.fft.irfft(X_pink, n=total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
    # Combine noises
    bg_noise = rumble * 0.7 + ambient * 0.3
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
        
    audio += bg_noise * noise_level
    
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
        
        # VSD subtype (80% membranous, 20% muscular)
        vsd_type = np.random.choice(['membranous', 'muscular'], p=[0.8, 0.2])
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
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
        
        # Increased noise level to provide a realistic noise floor
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
            noise_level=noise_level,
            vsd_type=vsd_type
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