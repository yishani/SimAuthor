"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur. 
The murmur begins with the first heart sound (S1) and extends to the 
second heart sound (S2). The murmur's characteristics (pitch, amplitude, 
and envelope) vary significantly based on the defect's size and location.

Refinements:
- Hypothesis: The low spread ratio across MFCCs and excessively high ZCR are 
  caused by the murmur being a strictly bandpassed, uniform block of noise 
  with unnatural silence between beats. Real PCGs have a continuous low-frequency 
  baseline that dominates the waveform and lowers the zero-crossing rate.
- Fix 1 (Baseline Noise): Significantly increased the continuous low-frequency 
  body/sensor noise to fill the silent gaps between beats. This provides a realistic 
  baseline drift that drastically lowers the ZCR and increases MFCC-0.
- Fix 2 (Spectral Leakage): Changed the murmur bandpass filter from 4th-order 
  to 2nd-order to allow more natural spectral leakage, reducing the artificial 
  "blocky" appearance in the spectrogram.
- Fix 3 (Diversity): Randomized the noise spectral roll-off (1/f^alpha) and 
  the Tukey window shape to increase the spread ratio across all MFCCs and ZCR, 
  fixing the under-diversity issue.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs, noise_cutoff=250):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with an asymmetric envelope and a broadband transient.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: faster attack, slower decay (peak at ~1/3 duration)
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a more natural "thud" sound
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base low-frequency thump
    sound = np.sin(phase)
    
    # Add a transient noise component for the mechanical valve closure
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, noise_cutoff / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = sound + 0.15 * noise
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, env_type='plateau', rolloff_exp=0.85, tukey_alpha=0.1):
    """
    Generates a harsh holosystolic murmur typical of VSD.
    Models both membranous (plateau) and muscular (decrescendo) envelopes.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
        
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Generate noise with variable roll-off to increase physiological diversity
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    # Avoid division by zero for DC
    S = np.power(np.arange(1, len(X) + 1), rolloff_exp)
    X_pink = X / S
    noise = np.fft.irfft(X_pink, n=N)
    
    # Bandpass filter to give the murmur its characteristic pitch
    # Using 2nd order (effective 4th order with filtfilt) for gentler, more natural spectral leakage
    nyq = 0.5 * fs
    b, a = signal.butter(2, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add a slight resonance in the lower third of the band to emphasize the "harsh" quality
    res_freq = freq_band[0] + (freq_band[1] - freq_band[0]) * 0.3
    b_peak, a_peak = signal.iirpeak(res_freq, 1.5, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.4 * murmur_peak
    
    # Re-filter to ensure band limits
    murmur = signal.filtfilt(b, a, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Apply envelope based on VSD subtype
    if env_type == 'decrescendo':
        # Muscular VSD: decrescendo envelope as the muscle contracts and closes the defect
        env = signal.windows.tukey(N, alpha=tukey_alpha)
        decay = np.linspace(1.0, 0.0, N)
        env = env * decay
    else:
        # Membranous VSD: classic holosystolic plateau envelope
        env = signal.windows.tukey(N, alpha=tukey_alpha)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, murmur_dur_ratio, env_type,
                        s1_amp, s2_amp, noise_level, rolloff_exp, tukey_alpha):
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
        
    def add_to_audio(sig, start_time):
        start_idx = int(start_time * fs)
        end_idx = start_idx + len(sig)
        if start_idx < total_samples:
            if end_idx > total_samples:
                sig = sig[:total_samples - start_idx]
                end_idx = total_samples
            audio[start_idx:end_idx] += sig

    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        # Generate components with variable noise cutoffs for diversity
        s1_noise_cutoff = np.random.uniform(150, 300)
        s2_noise_cutoff = np.random.uniform(200, 400)
        
        s1 = generate_heart_sound(f0=np.random.uniform(70, 90), duration=0.065, fs=fs, noise_cutoff=s1_noise_cutoff) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(100, 140), duration=0.050, fs=fs, noise_cutoff=s2_noise_cutoff) * s2_amp
        
        # VSD murmur duration depends on subtype (muscular is shorter)
        murmur_dur = systole_dur * murmur_dur_ratio
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, 
                                 env_type=env_type, rolloff_exp=rolloff_exp, tukey_alpha=tukey_alpha) * murmur_amp
        
        # Assemble the cycle
        add_to_audio(s1, beat_time)
        add_to_audio(murmur, beat_time) # Murmur starts synchronously with S1
        add_to_audio(s2, beat_time + systole_dur) # S2 marks the end of systole
        
    # Apply respiratory modulation (low frequency amplitude modulation)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.05 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add continuous low-frequency body/sensor noise to provide a realistic baseline
    # This fills the "silence" gaps and significantly lowers the ZCR
    bg_noise = np.random.randn(total_samples)
    bg_cutoff = np.random.uniform(30, 80)
    b, a = signal.butter(2, bg_cutoff / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a tiny bit of ambient broadband noise
    ambient = np.random.randn(total_samples)
    amb_cutoff = np.random.uniform(300, 1000)
    b_amb, a_amb = signal.butter(2, amb_cutoff / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * np.random.uniform(0.005, 0.03)
    
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
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.35, 0.45)
        
        # VSD Severity (0 = restrictive/small, 1 = unrestrictive/large)
        severity = np.random.uniform(0, 1)
        
        # VSD Subtype (~80% membranous, ~20% muscular)
        is_muscular = np.random.uniform(0, 1) > 0.8
        
        # Variable envelope shape and spectral roll-off for diversity
        rolloff_exp = np.random.uniform(0.5, 1.2)
        tukey_alpha = np.random.uniform(0.1, 0.25)
        
        if is_muscular:
            env_type = 'decrescendo'
            murmur_dur_ratio = np.random.uniform(0.5, 0.85) # Ends before S2
        else:
            env_type = 'plateau'
            murmur_dur_ratio = 1.0 # Holosystolic
            
        # Map severity to frequency and amplitude based on the blueprint:
        murmur_low = np.random.uniform(100, 250) * (1 - severity) + np.random.uniform(40, 100) * severity
        murmur_high = np.random.uniform(500, 1000) * (1 - severity) + np.random.uniform(200, 500) * severity
        murmur_band = (murmur_low, murmur_high)
        
        murmur_amp = np.random.uniform(0.8, 1.5) * (1 - severity) + np.random.uniform(0.15, 0.5) * severity
        
        s1_amp = np.random.uniform(0.7, 1.2)
        s2_amp = np.random.uniform(0.7, 1.1) + 0.5 * severity
        
        # Increased baseline noise level to match real PCG recordings and lower ZCR
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
            murmur_dur_ratio=murmur_dur_ratio,
            env_type=env_type,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            noise_level=noise_level,
            rolloff_exp=rolloff_exp,
            tukey_alpha=tukey_alpha
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