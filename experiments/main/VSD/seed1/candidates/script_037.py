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
- S1 and S2 heart sounds as broadband transients with a low-frequency thud.
- The VSD murmur as bandpass-filtered pink noise with a 350 Hz resonance 
  and a holosystolic envelope.
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary with respiration).
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a realistic heart sound (S1 or S2) with a low-frequency thud
    and a broadband valve click.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack, slower exponential decay
    attack_time = 0.01
    env = np.zeros_like(t)
    attack_idx = int(attack_time * fs)
    if attack_idx > 0:
        env[:attack_idx] = np.sin(np.pi/2 * np.linspace(0, 1, attack_idx))
    env[attack_idx:] = np.exp(- (t[attack_idx:] - attack_time) / 0.015)
    
    # Frequency modulation for a natural "thud"
    f_t = f0 * np.exp(-t / 0.02)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    thud = env * np.sin(phase)
    
    # Add a broadband transient (valve click)
    click = np.random.randn(len(t))
    b, a = signal.butter(2, [100 / (0.5 * fs), 1000 / (0.5 * fs)], btype='band')
    click = signal.filtfilt(b, a, click)
    click_env = np.exp(-t / 0.005)
    click = click * click_env
    
    sound = thud + 0.15 * click
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with specific filtering and resonance.
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
    
    # High-pass filter (4th order)
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    # Low-pass filter (4th order)
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance at ~350 Hz to emphasize the "harsh" quality (Blueprint)
    b_res, a_res = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_res, a_res, murmur)
    
    if np.max(np.abs(murmur_res)) > 0:
        murmur_res /= np.max(np.abs(murmur_res))
    if np.max(np.abs(murmur)) > 0:
        murmur /= np.max(np.abs(murmur))
        
    murmur = murmur + 0.5 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # Tukey window provides a flat top with smooth ~15ms fade-in/fade-out
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
        
        # Generate components with slight random frequency variations
        s1_f0 = np.random.uniform(70, 90)
        s2_f0 = np.random.uniform(100, 130)
        
        s1 = generate_heart_sound(f0=s1_f0, duration=0.06, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=0.05, fs=fs) * s2_amp
        
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
        
    # Note: Respiratory modulation is intentionally omitted here because 
    # VSD murmurs are respiratory invariant (unlike right-sided murmurs).
    
    # Add low-frequency body/sensor noise
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
        systole_ratio = np.random.uniform(0.3, 0.4)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        s1_amp = np.random.uniform(0.5, 0.8)
        
        if severity < 0.5:
            # Restrictive: louder murmur (+3 to +6 dB relative to S1)
            murmur_ratio = np.random.uniform(1.2, 1.8)
        else:
            # Unrestrictive: softer murmur
            murmur_ratio = np.random.uniform(0.5, 1.0)
            
        murmur_amp = s1_amp * murmur_ratio
        
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 800)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 0.8) + 0.3 * severity
        
        # Realistic noise level to balance ZCR and provide natural baseline
        noise_level = np.random.uniform(0.05, 0.15)
        
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