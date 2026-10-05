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
- S1 and S2 heart sounds as frequency-modulated bursts with harmonics for a natural "thud".
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope.
- Severity-dependent spectral shaping and amplitude scaling.
- Muscular VSD subtypes (~20%) with early-to-mid systolic decrescendo murmurs.
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary significantly with respiration).
- Stethoscope acoustic transmission (global low-pass filtering).
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
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    base tone with harmonics for a more natural, broadband "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base tone + harmonics to fill mid-frequencies and sound more realistic
    sound = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    
    sound = env * sound
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, is_muscular=False):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    Uses a 4th-order HPF and a 2nd-order LPF for a natural upper roll-off.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    murmur_low, murmur_high = freq_band
    
    # 4th-order Butterworth High-Pass Filter (steep low cut)
    b_hp, a_hp = signal.butter(4, murmur_low / nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    # 2nd-order Butterworth Low-Pass Filter (gentler upper roll-off for natural spectrum)
    b_lp, a_lp = signal.butter(2, murmur_high / nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add slight resonance to emphasize the "harsh" quality
    center_freq = (murmur_low + murmur_high) / 2
    b_res, a_res = signal.iirpeak(center_freq / nyq, 1.5)
    res_sig = signal.filtfilt(b_res, a_res, noise)
    
    murmur = murmur + 0.3 * res_sig
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    if is_muscular:
        # Muscular VSD: early-to-mid systolic, decrescendo shape
        env = signal.windows.tukey(num_samples, alpha=0.4)
        decay = np.linspace(1.0, 0.0, num_samples)
        env = env * decay
    else:
        # Membranous VSD: Holosystolic envelope spanning S1 to S2
        # Alpha = 0.2 provides a flat plateau with slightly softer edges than 0.1
        env = signal.windows.tukey(num_samples, alpha=0.2)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level, is_muscular):
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
        s2 = generate_heart_sound(f0=np.random.uniform(70, 110), duration=s2_dur, fs=fs) * s2_amp
        
        if is_muscular:
            # Muscular VSD murmurs end before S2
            murmur_dur = systole_dur * np.random.uniform(0.6, 0.85)
        else:
            # Membranous VSD murmurs are holosystolic, often obscuring A2
            murmur_dur = systole_dur + s2_dur * np.random.uniform(0.3, 0.7)
            
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, is_muscular=is_muscular) * murmur_amp
        
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
        
    # Add low-frequency body/sensor noise
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient noise (low-passed to avoid unrealistic high ZCR)
    ambient = generate_pink_noise(total_samples)
    b_amb, a_amb = signal.butter(2, 800 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * np.random.uniform(0.002, 0.01)
    
    # Apply global stethoscope low-pass filter to simulate acoustic transmission
    b_steth, a_steth = signal.butter(2, 1000 / (0.5 * fs), btype='low')
    audio = signal.filtfilt(b_steth, a_steth, audio)
    
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
        hr_mean = np.random.uniform(60, 120)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.25, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        # Subtype selection (~20% muscular)
        is_muscular = np.random.rand() < 0.2
        
        # Severity dictates the frequency band:
        # Blueprint: Dominant band 200-600 Hz. 
        # Small VSD (severity=0): higher-pitched (e.g., 200-600 Hz)
        # Large VSD (severity=1): lower-pitched (e.g., 150-300 Hz)
        murmur_low = 200 - 50 * severity
        murmur_high = 600 - 300 * severity
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.5, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 1.2) + 0.5 * severity
        
        # Blueprint: Restrictive VSD (severity=0) murmur is +3 to +6 dB louder than S1 (1.4x to 2.0x)
        # Unrestrictive VSD (severity=1) murmur is softer
        murmur_ratio = np.random.uniform(1.4, 2.0) - severity * np.random.uniform(0.6, 1.2)
        murmur_amp = s1_amp * max(0.4, murmur_ratio)
        
        noise_level = np.random.uniform(0.02, 0.15)
        
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
            is_muscular=is_muscular
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