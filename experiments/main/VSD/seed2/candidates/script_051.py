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
- S1 and S2 heart sounds with asymmetric envelopes and low-frequency thuds.
- The VSD murmur as shaped pink noise with a holosystolic envelope, 
  gentle spectral roll-offs, and a resonance peak at ~350 Hz for harshness.
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary significantly with respiration).
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise using pink noise to maintain realistic zero-crossing rates.
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
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1]  # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    return np.fft.irfft(X_pink, n=num_samples)

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) with a sharp attack and 
    broadband low-frequency 'thud' for realism.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope for a sharper attack
    attack_time = 0.15 * duration
    attack_idx = int(attack_time * fs)
    env = np.zeros_like(t)
    if attack_idx > 0:
        env[:attack_idx] = np.linspace(0, 1, attack_idx)
    env[attack_idx:] = np.exp(-5 * np.linspace(0, 1, len(t) - attack_idx))
    
    # Downward frequency modulation
    f_t = f0 - 40 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    sound = env * np.sin(phase)
    
    # Add low-frequency noise for the "thud"
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    thud = signal.filtfilt(b, a, noise)
    sound += 0.25 * env * thud
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses gentler roll-offs and a resonance peak to match real spectral profiles.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    
    # 2nd order filters for a more natural, broadband roll-off
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    b_lp, a_lp = signal.butter(2, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance at ~350 Hz for the characteristic "harsh" quality
    w0 = 350 / nyq
    b_peak, a_peak = signal.iirpeak(w0, 1.5)
    resonance = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.6 * resonance
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: Tukey window for a flat top with smooth fade-in/out
    env = signal.windows.tukey(num_samples, alpha=0.2)
    
    # Add slight amplitude modulation for turbulence realism
    mod_freq = np.random.uniform(20, 40)
    t = np.arange(num_samples) / fs
    am = 1.0 + 0.15 * np.sin(2 * np.pi * mod_freq * t + np.random.uniform(0, 2*np.pi))
    
    return murmur * env * am

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
        
        s1_dur = np.random.uniform(0.05, 0.08)
        s2_dur = np.random.uniform(0.04, 0.06)
        
        # Generate components
        s1 = generate_heart_sound(f0=np.random.uniform(60, 80), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(90, 120), duration=s2_dur, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic
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
        # Murmur starts slightly after S1 onset (0-10ms) to prevent complete masking
        add_to_audio(murmur, beat_time + 0.01) 
        add_to_audio(s2, beat_time + systole_dur) 
        
    # Add low-frequency body/sensor noise (using pink noise)
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 200 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level * 1.5
    
    # Add ambient pink noise for realism
    ambient = generate_pink_noise(total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * 0.005
    
    # Final high-pass filter to simulate stethoscope diaphragm (cutoff ~50 Hz)
    b_out, a_out = signal.butter(2, 50 / (0.5 * fs), btype='high')
    audio = signal.filtfilt(b_out, a_out, audio)
    
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
        severity = np.random.uniform(0, 1)
        
        # Adjust murmur amplitude to be more balanced with S1/S2
        murmur_amp = np.random.uniform(0.5, 1.2) * (1.2 - 0.4 * severity)
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(500, 700)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.3 * severity
        
        noise_level = np.random.uniform(0.03, 0.10)
        
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