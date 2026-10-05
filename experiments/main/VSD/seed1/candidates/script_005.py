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
- S1 and S2 heart sounds as frequency-modulated low-frequency bursts.
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope.
- Heart rate variability (HRV) and physiological systole scaling.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with an asymmetric envelope.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Asymmetric envelope: faster attack, slower decay
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a more natural "thud" sound
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    sound = env * np.sin(phase)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD using pink noise.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
    
    # Generate pink noise (1/f power spectrum) to model fluid turbulence
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(N, 1/fs)
    freqs[0] = freqs[1] # prevent division by zero
    X = X / np.sqrt(freqs)
    noise = np.fft.irfft(X, n=N)
    
    # Apply high-pass and low-pass filters
    nyq = 0.5 * fs
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    noise = signal.filtfilt(b_hp, a_hp, noise)
    
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    noise = signal.filtfilt(b_lp, a_lp, noise)
    
    # Add a slight resonance at 350 Hz to emphasize the "harsh" quality
    b_res, a_res = signal.iirpeak(350, 1.5, fs=fs)
    res_noise = signal.filtfilt(b_res, a_res, noise)
    
    murmur = noise + 0.5 * res_noise
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: flat plateau with rapid attack/decay
    # Tukey window with alpha=0.1 provides ~15ms attack/decay for a ~300ms duration
    env = signal.windows.tukey(N, alpha=0.1)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_factor, 
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
    for i, beat_time in enumerate(beats):
        if i < len(beats) - 1:
            rr = beats[i+1] - beats[i]
        else:
            rr = 60.0 / hr_mean
            
        # Systole duration scales with sqrt(RR) (Bazett's principle)
        systole_dur = systole_factor * np.sqrt(rr)
        
        # Generate components
        s1 = generate_heart_sound(f0=80, duration=0.07, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=100, duration=0.05, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic, extending slightly into S2
        murmur_dur = systole_dur + 0.03
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band) * murmur_amp
        
        def add_to_audio(sig, start_time):
            if len(sig) == 0:
                return
            start_idx = int(start_time * fs)
            end_idx = start_idx + len(sig)
            if start_idx < total_samples:
                if end_idx > total_samples:
                    sig = sig[:total_samples - start_idx]
                    end_idx = total_samples
                audio[start_idx:end_idx] += sig
                
        # Assemble the cycle
        add_to_audio(s1, beat_time)
        add_to_audio(murmur, beat_time) # Murmur starts exactly with S1
        add_to_audio(s2, beat_time + systole_dur) # S2 marks the end of systole
        
    # Add low-frequency body/sensor noise (broadband white noise removed to fix ZCR discrepancy)
    bg_noise = np.random.randn(total_samples)
    b, a = signal.butter(2, 100 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
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
        systole_factor = np.random.uniform(0.28, 0.34)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        murmur_amp = np.random.uniform(0.6, 1.8) * (1.2 - 0.4 * severity)
        murmur_low = np.random.uniform(180, 250)
        murmur_high = np.random.uniform(500, 800)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.4 * severity
        
        noise_level = np.random.uniform(0.01, 0.05)
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_factor=systole_factor,
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