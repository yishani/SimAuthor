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
- S1 and S2 heart sounds as frequency-modulated low-frequency bursts with broadband clicks.
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope and 350Hz resonance.
- Heart rate variability (HRV) and minimal respiratory amplitude modulation.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Realistic ambient and body/sensor noise floors using pink noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) combining a low-frequency 
    thud, filtered noise, and a broadband transient click for realism.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope for the low-frequency thud
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a natural "thud"
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    tone = np.sin(phase)
    
    # Low-frequency noise component
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    filtered_noise = signal.filtfilt(b, a, noise)
    if np.max(np.abs(filtered_noise)) > 0:
        filtered_noise /= np.max(np.abs(filtered_noise))
        
    # Broadband transient (valve click) at the beginning
    click_env = np.exp(-t * 150) # Fast decay
    click = np.random.randn(len(t)) * click_env
    b_click, a_click = signal.butter(2, [100 / (0.5 * fs), 1000 / (0.5 * fs)], btype='band')
    click = signal.filtfilt(b_click, a_click, click)
    if np.max(np.abs(click)) > 0:
        click /= np.max(np.abs(click))
        
    # Combine components
    sound = (tone + 0.3 * filtered_noise) * env + 0.15 * click
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with a 350 Hz resonance and a gentle high-frequency roll-off.
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
    # 4th order HPF for sharp low cutoff, 2nd order LPF for natural high roll-off
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    b_lp, a_lp = signal.butter(2, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance at ~350 Hz to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_peak, a_peak, murmur)
    murmur = murmur + 0.4 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: flat top with smooth 15ms fade-in/fade-out
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
        
        # VSD murmur is holosystolic (starts just after S1 onset, ends at S2 onset)
        murmur_start = beat_time + 0.02 
        murmur_end = beat_time + systole_dur + 0.02
        murmur_dur = murmur_end - murmur_start
        
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
        add_to_audio(murmur, murmur_start)
        add_to_audio(s2, beat_time + systole_dur)
        
    # Apply minimal respiratory modulation (VSD murmurs are respiratory invariant)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.05 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Generate continuous pink noise for background
    bg_samples = total_samples
    X_white = np.fft.rfft(np.random.randn(bg_samples))
    freqs = np.fft.rfftfreq(bg_samples, d=1/fs)
    freqs[0] = freqs[1]
    X_pink = X_white / np.sqrt(freqs)
    
    # 1. Ambient/sensor noise (broadband up to 2000 Hz)
    ambient_noise = np.fft.irfft(X_pink, n=bg_samples)
    b_amb, a_amb = signal.butter(2, 2000 / (0.5 * fs), btype='low')
    ambient_noise = signal.filtfilt(b_amb, a_amb, ambient_noise)
    if np.max(np.abs(ambient_noise)) > 0:
        ambient_noise /= np.max(np.abs(ambient_noise))
        
    # 2. Body noise / rumble (low frequency)
    body_noise = np.fft.irfft(X_pink, n=bg_samples)
    b_body, a_body = signal.butter(2, 100 / (0.5 * fs), btype='low')
    body_noise = signal.filtfilt(b_body, a_body, body_noise)
    if np.max(np.abs(body_noise)) > 0:
        body_noise /= np.max(np.abs(body_noise))
        
    # Mix background noises to lower ZCR and improve spectral shape
    audio += ambient_noise * noise_level * 0.3 + body_noise * noise_level * 1.5
    
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
        
        # Slightly wider band to improve spectral realism
        murmur_low = np.random.uniform(100, 200)
        murmur_high = np.random.uniform(500, 800)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 0.8) + 0.3 * severity
        
        # Increased noise level for a more realistic clinical recording floor
        noise_level = np.random.uniform(0.02, 0.10)
        
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