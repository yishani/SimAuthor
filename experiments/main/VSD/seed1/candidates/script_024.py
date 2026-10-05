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
- S1 and S2 heart sounds as frequency-modulated low-frequency bursts with added transient noise.
- The VSD murmur as shaped pink noise with a holosystolic envelope and a harsh resonance.
- Heart rate variability (HRV) and respiratory amplitude modulation.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient broadband and low-frequency sensor noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave combined with low-pass filtered noise for a realistic "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    tone = np.sin(phase)
    
    # Add low-pass filtered noise for a broadband transient effect
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 250 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    if np.max(np.abs(noise)) > 0:
        noise /= np.max(np.abs(noise))
        
    sound = env * (tone + 0.5 * noise)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with independent HPF/LPF and a peaking resonance 
    to model fluid turbulence and avoid an artificial flat plateau.
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
    
    # High-pass filter
    b_h, a_h = signal.butter(4, freq_band[0]/nyq, btype='high')
    murmur = signal.filtfilt(b_h, a_h, noise)
    
    # Low-pass filter
    b_l, a_l = signal.butter(4, freq_band[1]/nyq, btype='low')
    murmur = signal.filtfilt(b_l, a_l, murmur)
    
    # Add a slight resonance to emphasize the "harsh" quality (around 300-400 Hz)
    res_freq = np.random.uniform(300, 400)
    b_peak, a_peak = signal.iirpeak(res_freq / nyq, 1.5)
    murmur_peak = signal.filtfilt(b_peak, a_peak, murmur)
    murmur = murmur + 1.5 * murmur_peak
    
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
        
        # Generate components with slight frequency variations
        s1_f0 = np.random.uniform(70, 90)
        s2_f0 = np.random.uniform(100, 140)
        s1 = generate_heart_sound(f0=s1_f0, duration=0.06, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=0.05, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic (starts with S1, extends slightly into S2)
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
    
    # Add low-frequency body/sensor noise
    bg_noise_lf = np.random.randn(total_samples)
    b, a = signal.butter(2, 100 / (0.5 * fs), btype='low')
    bg_noise_lf = signal.filtfilt(b, a, bg_noise_lf)
    if np.max(np.abs(bg_noise_lf)) > 0:
        bg_noise_lf /= np.max(np.abs(bg_noise_lf))
        
    # Add broadband pink noise to simulate ambient/stethoscope hiss
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    X_pink = X_white / np.sqrt(freqs)
    bg_noise_pink = np.fft.irfft(X_pink, n=total_samples)
    if np.max(np.abs(bg_noise_pink)) > 0:
        bg_noise_pink /= np.max(np.abs(bg_noise_pink))
        
    audio += bg_noise_lf * noise_level + bg_noise_pink * (noise_level * 0.3)
    
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
        # Sample physiological parameters with increased variance for diversity
        hr_mean = np.random.uniform(60, 110)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.3, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        s1_amp = np.random.uniform(0.4, 0.9)
        
        if severity < 0.5:
            # Restrictive: louder murmur (+3 to +6 dB relative to S1)
            murmur_ratio = np.random.uniform(1.2, 2.2)
        else:
            # Unrestrictive: softer murmur
            murmur_ratio = np.random.uniform(0.4, 1.2)
            
        murmur_amp = s1_amp * murmur_ratio
        
        # Lowered HPF to include more low-frequency energy, reducing ZCR
        murmur_low = np.random.uniform(100, 300)
        murmur_high = np.random.uniform(400, 900)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.4, 0.9) + 0.3 * severity
        
        noise_level = np.random.uniform(0.005, 0.08)
        
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