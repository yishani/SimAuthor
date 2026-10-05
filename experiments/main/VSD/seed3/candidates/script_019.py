"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur. 
The murmur begins with the first heart sound (S1) and extends to the 
second heart sound (S2). The murmur is typically high-pitched and harsh 
due to the high-pressure gradient between the left and right ventricles.

Refinements:
- Hypothesis: The previous simulator produced excessively high Zero-Crossing 
  Rates (ZCR) and poor MFCC overlap because the murmur was modeled as a 
  strictly band-limited, overly loud block of high-frequency noise. 
- Action: Reduced the murmur's relative amplitude, lowered its high-frequency 
  cutoff, and replaced the 4th-order bandpass filter with gentler 2nd-order 
  high-pass and low-pass filters for a more natural spectral roll-off.
- Action: Enhanced heart sounds (S1/S2) with low-passed transient noise to 
  give them more realistic broadband energy.
- Action: Replaced white background noise with pink background noise to 
  better model the continuous low-frequency rumble of body sounds.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave combined with low-passed transient noise for a realistic "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope (fast attack, slower decay)
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    tone = np.sin(phase)
    
    # Add low-passed noise for the broadband valve closure snap
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = (tone + 0.5 * noise) * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with 2nd-order filters for a natural spectral roll-off.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
        
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Generate pink noise (1/f spectrum)
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    S = np.sqrt(np.arange(1, len(X) + 1))
    X_pink = X / S
    noise = np.fft.irfft(X_pink, n=N)
    
    # Use 2nd-order filters for a gentler, more natural spectral roll-off
    nyq = 0.5 * fs
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    b_lp, a_lp = signal.butter(2, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a slight resonance to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(300, 1.5, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.4 * murmur_peak
    
    # Re-apply low-pass to ensure high frequencies don't leak too much from the peak
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window with alpha=0.2 provides a flat top with slightly soft edges
    env = signal.windows.tukey(N, alpha=0.2)
    
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
        
        # Generate components (S1: ~65ms, S2: ~50ms)
        s1 = generate_heart_sound(f0=80, duration=0.065, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=120, duration=0.050, fs=fs) * s2_amp
        
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
    # VSD murmurs are largely invariant to respiration, so modulation is kept minimal
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.05 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add low-frequency body/sensor noise using pink noise for a natural spectrum
    white_bg = np.random.randn(total_samples)
    X_bg = np.fft.rfft(white_bg)
    S_bg = np.sqrt(np.arange(1, len(X_bg) + 1))
    X_bg_pink = X_bg / S_bg
    bg_noise = np.fft.irfft(X_bg_pink, n=total_samples)
    
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a tiny bit of ambient noise (low-passed to simulate stethoscope acoustic transmission)
    ambient = np.random.randn(total_samples)
    b_amb, a_amb = signal.butter(2, 500 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * 0.01
    
    return audio

def main():
    # Set explicit random seed for reproducibility
    np.random.seed(42)
    
    output_dir = "artifacts/vsd_audio/visual_signal_100iter/generated/"
    os.makedirs(output_dir, exist_ok=True)
    
    # Immutable Output Contract parameters
    fs_internal = 44100
    fs_output = 16000
    duration_sec = 10.0
    num_samples = 100
    
    print(f"Generating {num_samples} VSD audio samples...")
    
    for i in range(num_samples):
        # Sample physiological parameters to ensure diversity
        hr_mean = np.random.uniform(65, 105)
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.35, 0.42)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        # Murmur amplitude: reduced to be more realistic relative to S1 and lower ZCR
        murmur_amp = np.random.uniform(0.5, 1.2) * (1.0 - 0.3 * severity)
        
        # Lowered high-frequency cutoff to reduce excessive ZCR
        murmur_low = np.random.uniform(100, 200)
        murmur_high = np.random.uniform(350, 550)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.3 * severity
        
        # Increased background noise level for a more realistic low-frequency body rumble
        noise_level = np.random.uniform(0.05, 0.12)
        
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