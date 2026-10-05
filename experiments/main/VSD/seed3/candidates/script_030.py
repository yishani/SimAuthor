"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur. 
The murmur begins with the first heart sound (S1) and extends to the 
second heart sound (S2). The murmur is typically high-pitched and harsh 
due to the high-pressure gradient between the left and right ventricles.

Refinements (Hypothesis-Driven):
- Addressed excessively high Zero-Crossing Rate (ZCR) by introducing a strong 
  low-frequency baseline rumble (<40 Hz) to simulate stethoscope/body acoustics, 
  and by lowering the fundamental frequencies (f0) of S1 and S2.
- Addressed negative mid-to-high MFCC shifts and under-diversity by replacing 
  the static pink noise with a variable spectral roll-off (between pink and brown noise), 
  using gentler 2nd-order filters instead of 4th-order to allow natural spectral leakage, 
  and introducing variable chest resonances (formants).
- Improved the murmur envelope by replacing the static Tukey window with a randomized 
  alpha parameter and adding slight amplitude modulation to simulate fluid turbulence 
  more naturally, reducing the "blocky" artificial appearance in the spectrogram.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with an asymmetric envelope and a broadband transient.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: faster attack, slower decay
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a natural "thud"
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base low-frequency thump
    sound = np.sin(phase)
    
    # Add a transient noise component for the mechanical valve closure
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 400 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = sound + np.random.uniform(0.05, 0.2) * noise
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, roll_off):
    """
    Generates a harsh holosystolic murmur typical of VSD.
    Uses noise with a variable spectral roll-off and amplitude modulation.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
        
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Generate noise with variable spectral roll-off (e.g., 1.0 = pink, 2.0 = brown)
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    S = np.arange(1, len(X) + 1) ** roll_off
    X_filtered = X / S
    noise = np.fft.irfft(X_filtered, n=N)
    
    nyq = 0.5 * fs
    
    # 2nd-order filters for a gentler, more natural spectral roll-off
    b_hp, a_hp = signal.butter(2, freq_band[0] / nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    b_lp, a_lp = signal.butter(2, freq_band[1] / nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a variable broad resonance to simulate chest cavity acoustics (formants)
    res_freq = np.random.uniform(200, 500)
    res_q = np.random.uniform(1.0, 3.0)
    b_peak, a_peak = signal.iirpeak(res_freq, res_q, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + np.random.uniform(0.2, 0.6) * murmur_peak
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # Variable alpha for diversity in attack/decay times
    alpha = np.random.uniform(0.1, 0.4)
    env = signal.windows.tukey(N, alpha=alpha)
    
    # Add slight amplitude modulation to simulate turbulent fluid dynamics
    mod_freq = np.random.uniform(10, 25)
    modulation = 1.0 - np.random.uniform(0.05, 0.2) * np.sin(2 * np.pi * mod_freq * t)
    env = env * modulation
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, roll_off, s1_amp, s2_amp, 
                        noise_level, rumble_level):
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
        
        # Generate components with variable fundamental frequencies
        f0_s1 = np.random.uniform(40, 70)
        f0_s2 = np.random.uniform(60, 100)
        
        s1 = generate_heart_sound(f0=f0_s1, duration=0.07, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=f0_s2, duration=0.05, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic (lasts exactly from S1 to S2)
        murmur = generate_murmur(duration=systole_dur, fs=fs, freq_band=murmur_band, roll_off=roll_off) * murmur_amp
        
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
    resp_mod = 1.0 + 0.05 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add strong low-frequency body/sensor noise (rumble) to match real recordings and lower ZCR
    rumble = np.random.randn(total_samples)
    b, a = signal.butter(2, 40 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b, a, rumble)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
    audio += rumble * rumble_level
    
    # Add general ambient noise
    ambient = np.random.randn(total_samples)
    b_amb, a_amb = signal.butter(2, 500 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * noise_level
    
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
        hr_mean = np.random.uniform(60, 110)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.32, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        # Murmur parameters
        murmur_low = np.random.uniform(50, 150)
        murmur_high = np.random.uniform(300, 900)
        murmur_band = (murmur_low, murmur_high)
        roll_off = np.random.uniform(0.8, 1.5) # Varies between pink and brown noise
        
        # Amplitudes
        murmur_amp = np.random.uniform(0.3, 1.2) * (1.0 - 0.3 * severity)
        s1_amp = np.random.uniform(0.7, 1.3)
        s2_amp = np.random.uniform(0.7, 1.3) + 0.3 * severity
        
        # Noise levels
        noise_level = np.random.uniform(0.01, 0.05)
        rumble_level = np.random.uniform(0.1, 0.3)
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_ratio=systole_ratio,
            murmur_amp=murmur_amp,
            murmur_band=murmur_band,
            roll_off=roll_off,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            noise_level=noise_level,
            rumble_level=rumble_level
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