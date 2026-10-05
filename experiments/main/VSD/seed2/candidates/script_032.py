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

Refinement:
Addressed spectral blockiness and artificially high ZCR by implementing 
the blueprint's recommended spectral shaping for the murmur (gentle HPF, 
steeper LPF, and a 350 Hz resonance peak). Additionally, introduced a 
physiological low-frequency rumble (<50 Hz) to better integrate the 
murmur into a realistic PCG noise floor, correcting the ZCR shift and 
improving mid-band MFCC representations.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_pink_noise(num_samples):
    """
    Generates pink noise (1/f) using the frequency domain.
    Pink noise is preferable to white noise as it naturally models 
    the energy roll-off of fluid turbulence.
    """
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1]  # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    return np.fft.irfft(X_pink, n=num_samples)

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with an asymmetric envelope and subtle harmonics for realism.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: faster attack, slower decay
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 2.5
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a natural "thud"
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Main tone + subtle second harmonic
    sound = env * (np.sin(phase) + 0.15 * np.sin(2 * phase))
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, murmur_low, murmur_high):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD,
    using independent filters and a resonance peak to avoid a blocky spectrum.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    
    # Gentle High-Pass Filter to allow some low-frequency integration
    b_hp, a_hp = signal.butter(2, murmur_low / nyq, btype='high')
    noise_hp = signal.filtfilt(b_hp, a_hp, noise)
    
    # Steeper Low-Pass Filter for the upper roll-off
    b_lp, a_lp = signal.butter(4, murmur_high / nyq, btype='low')
    murmur_base = signal.filtfilt(b_lp, a_lp, noise_hp)
    
    # Add a slight resonance (Q ≈ 1.5) at 350 Hz to emphasize the "harsh" quality
    b_res, a_res = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_res, a_res, noise)
    
    # Combine base murmur with resonance
    murmur = murmur_base + 0.5 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth ~15ms fade-in/fade-out
    env = signal.windows.tukey(num_samples, alpha=0.1)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_low, murmur_high, s1_amp, s2_amp, 
                        noise_level, rumble_level, ambient_level):
    """
    Assembles a full PCG signal with VSD characteristics and realistic noise floors.
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
        s1 = generate_heart_sound(f0=np.random.uniform(70, 90), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(100, 130), duration=s2_dur, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic (lasts exactly from S1 to S2)
        murmur = generate_murmur(duration=systole_dur, fs=fs, murmur_low=murmur_low, murmur_high=murmur_high) * murmur_amp
        
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
        
    # Add strong low-frequency rumble (0-50 Hz) to model physiological baseline and correct ZCR
    rumble = generate_pink_noise(total_samples)
    b_r, a_r = signal.butter(2, 50 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b_r, a_r, rumble)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
    audio += rumble * rumble_level
        
    # Add general low-frequency body/sensor noise (up to 200 Hz)
    bg_noise = generate_pink_noise(total_samples)
    b_bg, a_bg = signal.butter(2, 200 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b_bg, a_bg, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a tiny bit of ambient pink noise for realism
    ambient = generate_pink_noise(total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * ambient_level
    
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
        hr_mean = np.random.uniform(60, 110)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.3, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        murmur_amp = np.random.uniform(0.6, 1.6) * (1.5 - 0.7 * severity)
        murmur_low = np.random.uniform(120, 220)
        murmur_high = np.random.uniform(550, 850) - 150 * severity
        
        s1_amp = np.random.uniform(0.7, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.7, 1.2) + 0.5 * severity
        
        noise_level = np.random.uniform(0.05, 0.2)
        rumble_level = np.random.uniform(0.15, 0.4)
        ambient_level = np.random.uniform(0.005, 0.025)
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_ratio=systole_ratio,
            murmur_amp=murmur_amp,
            murmur_low=murmur_low,
            murmur_high=murmur_high,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            noise_level=noise_level,
            rumble_level=rumble_level,
            ambient_level=ambient_level
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