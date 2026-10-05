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
- S1 and S2 heart sounds as realistic transients with a low-frequency thud and broadband snap.
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope and harsh resonance.
- Heart rate variability (HRV).
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Realistic ambient (pink) and sensor (brown) background noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a realistic heart sound with a low-frequency thud and a 
    broadband transient snap (valve closure).
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack, slower decay
    attack_time = 0.01
    attack_samples = int(attack_time * fs)
    decay_samples = len(t) - attack_samples
    
    env = np.ones_like(t)
    if attack_samples > 0:
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
    if decay_samples > 0:
        env[attack_samples:] = np.exp(-np.linspace(0, 4, decay_samples))
        
    # Base low-frequency "thud"
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    thud = np.sin(phase)
    
    # Broadband "snap" (valve closure)
    snap = np.random.randn(len(t))
    nyq = 0.5 * fs
    b, a = signal.butter(2, [30 / nyq, 400 / nyq], btype='band')
    snap = signal.filtfilt(b, a, snap)
    
    # Combine
    sound = (thud * 0.7 + snap * 0.3) * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise to model fluid turbulence roll-off, with added resonance.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    # Generate pink noise via FFT
    X_white = np.fft.rfft(np.random.randn(num_samples))
    X_white[0] = 0  # Remove DC
    freqs = np.fft.rfftfreq(num_samples, d=1/fs)
    freqs[0] = 1.0  # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    noise = np.fft.irfft(X_pink, n=num_samples)
    
    # Bandpass filter to give the murmur its characteristic pitch
    nyq = 0.5 * fs
    b, a = signal.butter(4, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add resonance at ~350 Hz for the "harsh" quality described in blueprint
    b_peak, a_peak = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.4 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    # alpha=0.1 gives ~15ms attack/release for a 300ms duration
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
        s1 = generate_heart_sound(f0=80, duration=0.06, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=120, duration=0.05, fs=fs) * s2_amp
        
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
        
    # Add realistic background noise (pink noise for ambient, brown for rumble)
    bg_white = np.random.randn(total_samples)
    X_white = np.fft.rfft(bg_white)
    X_white[0] = 0  # Remove DC offset
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = 1.0  # Prevent division by zero
    
    # Pink noise (sensor/ambient)
    X_pink = X_white / np.sqrt(freqs)
    bg_pink = np.fft.irfft(X_pink, n=total_samples)
    b_pink, a_pink = signal.butter(2, 500 / (0.5 * fs), btype='low')
    bg_pink = signal.filtfilt(b_pink, a_pink, bg_pink)
    if np.max(np.abs(bg_pink)) > 0:
        bg_pink /= np.max(np.abs(bg_pink))
        
    # Brown noise (low frequency rumble / baseline wander)
    X_brown = X_white / freqs
    bg_brown = np.fft.irfft(X_brown, n=total_samples)
    b_brown, a_brown = signal.butter(2, 50 / (0.5 * fs), btype='low')
    bg_brown = signal.filtfilt(b_brown, a_brown, bg_brown)
    if np.max(np.abs(bg_brown)) > 0:
        bg_brown /= np.max(np.abs(bg_brown))
        
    # Combine noises
    audio += bg_pink * noise_level + bg_brown * (noise_level * 1.5)
    
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
        
        noise_level = np.random.uniform(0.01, 0.05)
        
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