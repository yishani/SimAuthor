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

Refinements:
- Replaced the "brick-wall" 4th-order double-pass bandpass filter with a gentler 
  2nd-order filter and a 1000 Hz upper roll-off to restore a natural spectral 
  envelope and fix negative mid-high MFCCs.
- Introduced a pink noise low-frequency baseline (rumble) to simulate realistic 
  body/sensor noise, which naturally reduces the excessively high Zero-Crossing Rate (ZCR).
- Enhanced S1 and S2 generation with harmonics and low-passed noise to create a 
  more realistic, broadband "thud" rather than a pure sine wave.
- Broadened the ambient noise floor to better match real stethoscope recordings.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    base wave with harmonics and a noise component for a realistic "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope (faster attack, slower decay)
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base sound with harmonics for a richer tone
    sound = env * (np.sin(phase) + 0.4 * np.sin(2 * phase) + 0.2 * np.sin(3 * phase))
    
    # Add a little bit of low-passed noise for the broadband "thud" impact
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound += 0.3 * env * noise
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise with a gentle bandpass and natural upper roll-off.
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
    
    nyq = 0.5 * fs
    
    # Gentle bandpass filter (2nd order) to avoid a "brick-wall" unnatural spectrum
    b_bp, a_bp = signal.butter(2, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b_bp, a_bp, noise)
    
    # Add a slight resonance at 350 Hz to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(350, 2.0, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, murmur)
    
    murmur = murmur + 0.6 * murmur_peak
    
    # Upper roll-off to simulate stethoscope and chest wall transmission
    b_lp, a_lp = signal.butter(2, 1000/nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth ~15ms fade-in/fade-out
    env = signal.windows.tukey(N, alpha=0.1)
    
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
    
    # Add low-frequency body/sensor noise (using pink noise for natural 1/f rumble)
    # This realistic baseline wander naturally reduces the Zero-Crossing Rate (ZCR)
    bg_white = np.random.randn(total_samples)
    X_bg = np.fft.rfft(bg_white)
    S_bg = np.sqrt(np.arange(1, len(X_bg) + 1))
    bg_pink = np.fft.irfft(X_bg / S_bg, n=total_samples)
    
    b, a = signal.butter(2, 100 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_pink)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient noise (low-passed to simulate stethoscope acoustic transmission)
    ambient = np.random.randn(total_samples)
    b_amb, a_amb = signal.butter(2, 1000 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * np.random.uniform(0.01, 0.03)
    
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
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        # Murmur amplitude: +3 dB to +6 dB relative to S1 (approx 1.4x to 2.0x linear)
        murmur_amp = np.random.uniform(1.2, 1.8) * (1.0 - 0.3 * severity)
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 900)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.4 * severity
        
        # Increased noise level for a more realistic baseline rumble
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