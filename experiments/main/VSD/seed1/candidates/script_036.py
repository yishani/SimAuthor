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
- S1 and S2 heart sounds as frequency-modulated bursts with harmonics.
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope.
- Physiological systole duration scaling with heart rate.
- Heart rate variability (HRV) and respiratory amplitude modulation.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise to match real clinical recordings.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs, is_s2=False):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    waveform with harmonics and a smooth envelope.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope with slightly sharper attack/decay than a pure sine squared
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 1.5
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a natural "thud"
    f_t = f0 - 15 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    if is_s2:
        # S2 is often "snappier" with higher frequency content
        sound = env * (np.sin(phase) + 0.3 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase))
    else:
        # S1 is more "booming"
        sound = env * (np.sin(phase) + 0.15 * np.sin(2 * phase))
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise to model fluid turbulence roll-off.
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
    # Using a 2nd order filter for a gentler, more natural roll-off
    nyq = 0.5 * fs
    b, a = signal.butter(2, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    # alpha=0.2 gives slightly softer edges to prevent high-frequency clicks
    env = signal.windows.tukey(num_samples, alpha=0.2)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, 
                        s1_f0, s2_f0, noise_level):
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
        # Physiological scaling of systole duration (approx 300ms at 60 BPM)
        systole_dur = 0.3 * np.sqrt(rr)
        
        # Generate components
        s1 = generate_heart_sound(f0=s1_f0, duration=0.08, fs=fs, is_s2=False) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=0.06, fs=fs, is_s2=True) * s2_amp
        
        # VSD murmur is holosystolic (lasts exactly from S1 to S2)
        murmur = generate_murmur(duration=systole_dur, fs=fs, freq_band=murmur_band) * murmur_amp
        
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
    
    # Add low-frequency body/sensor baseline wander
    bg_noise = np.random.randn(total_samples)
    b, a = signal.butter(2, 50 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient room/sensor noise (pinkish)
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    X_pink = X_white / (freqs ** 0.5)
    ambient = np.fft.irfft(X_pink, n=total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * (noise_level * 0.2)
    
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
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        s1_amp = np.random.uniform(0.7, 1.0)
        s1_f0 = np.random.uniform(40, 60)
        s2_f0 = np.random.uniform(60, 85)
        
        if severity < 0.5:
            # Restrictive: louder murmur, but balanced with S1
            murmur_ratio = np.random.uniform(0.6, 1.2)
        else:
            # Unrestrictive: softer murmur
            murmur_ratio = np.random.uniform(0.2, 0.6)
            
        murmur_amp = s1_amp * murmur_ratio
        
        # Lowered frequency bands for a more natural sound and better ZCR match
        murmur_low = np.random.uniform(100, 150)
        murmur_high = np.random.uniform(400, 600)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.6, 0.9) + 0.2 * severity
        
        noise_level = np.random.uniform(0.02, 0.08)
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            murmur_amp=murmur_amp,
            murmur_band=murmur_band,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            s1_f0=s1_f0,
            s2_f0=s2_f0,
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