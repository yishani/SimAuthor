"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic murmur. However, the 
acoustic properties vary significantly based on the defect's size (severity) 
and anatomical location (membranous vs. muscular). 
- Small (restrictive) defects produce loud, high-pitched murmurs.
- Large (unrestrictive) defects produce softer, lower-pitched murmurs and 
  often accentuate the P2 component of the second heart sound.
- Muscular defects can close during late systole, creating a shorter, 
  decrescendo murmur rather than a strict holosystolic plateau.

Refinements:
- Introduced physiological subtypes (Membranous vs. Muscular) and severity 
  levels (Restrictive vs. Unrestrictive) to drive acoustic parameters. This 
  massively increases dataset diversity (addressing low spread ratios).
- Adjusted the murmur frequency bands based on severity. Restrictive VSDs now 
  extend up to 1000 Hz with a gentler 2nd-order low-pass roll-off, adding 
  missing mid-high frequency energy to correct the negative shifts in MFCC 6-9.
- Added a decrescendo envelope option for Muscular VSDs to break up the 
  unnatural "blocky" plateau seen in the spectrograms.
- Increased the amplitude and variance of continuous low-frequency background 
  noise (stethoscope rumble) and ambient pink noise to fill unnatural spectral 
  gaps and naturally lower the excessively high Zero-Crossing Rate (ZCR).
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
    b, a = signal.butter(2, 400 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = sound + 0.25 * noise
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, peak_freq, env_type='plateau'):
    """
    Generates a harsh murmur typical of VSD.
    Uses pink noise to naturally model the energy roll-off of fluid turbulence.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
        
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Generate pink noise (1/f power spectrum)
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    S = np.sqrt(np.arange(1, len(X) + 1))
    X_pink = X / S
    noise = np.fft.irfft(X_pink, n=N)
    
    nyq = 0.5 * fs
    # Separate HPF (steep) and LPF (gentle) for more natural roll-off
    b_hp, a_hp = signal.butter(4, freq_band[0] / nyq, btype='high')
    b_lp, a_lp = signal.butter(2, freq_band[1] / nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a resonance peak to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(peak_freq, 1.5, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.5 * murmur_peak
    
    # Re-filter to ensure strict band limits
    murmur = signal.filtfilt(b_hp, a_hp, murmur)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Apply envelope based on VSD subtype
    if env_type == 'decrescendo':
        # Fast attack, then linear decay (Muscular VSD)
        attack_len = int(0.05 * fs)
        if attack_len > N: attack_len = N
        env = np.ones(N)
        env[:attack_len] = np.linspace(0, 1, attack_len)
        env[attack_len:] = np.linspace(1, 0, N - attack_len)
    else:
        # Holosystolic plateau (Membranous VSD)
        env = signal.windows.tukey(N, alpha=0.1)
        
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, peak_freq, vsd_type, 
                        s1_amp, s2_amp, noise_level):
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
        s1 = generate_heart_sound(f0=np.random.uniform(70, 90), duration=0.065, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(100, 140), duration=0.050, fs=fs) * s2_amp
        
        # Determine murmur duration and envelope based on subtype
        if vsd_type == 'muscular':
            # Muscular VSDs often close in late systole
            murmur_dur = max(0.1, systole_dur - np.random.uniform(0.05, 0.12))
            env_type = 'decrescendo'
        else:
            # Membranous VSDs are strictly holosystolic
            murmur_dur = systole_dur
            env_type = 'plateau'
            
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, 
                                 peak_freq=peak_freq, env_type=env_type) * murmur_amp
        
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
    
    # Add continuous low-frequency body/sensor noise (stethoscope rumble)
    bg_noise = np.random.randn(total_samples)
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient room noise (pink noise, low-passed)
    white_amb = np.random.randn(total_samples)
    X_amb = np.fft.rfft(white_amb)
    S_amb = np.sqrt(np.arange(1, len(X_amb) + 1))
    X_pink_amb = X_amb / S_amb
    ambient = np.fft.irfft(X_pink_amb, n=total_samples)
    
    b_amb, a_amb = signal.butter(2, 800 / (0.5 * fs), btype='low')
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
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.32, 0.45)
        
        # Determine VSD subtype and severity
        vsd_type = np.random.choice(['membranous', 'muscular'], p=[0.8, 0.2])
        severity = np.random.uniform(0, 1) # 0 = small/restrictive, 1 = large/unrestrictive
        
        if severity < 0.5:
            # Small/restrictive VSD: louder, higher pitched, harsh
            murmur_low = np.random.uniform(150, 250)
            murmur_high = np.random.uniform(600, 1000)
            peak_freq = np.random.uniform(400, 700)
            murmur_amp = np.random.uniform(1.0, 2.0)
            s2_amp = np.random.uniform(0.8, 1.2)
        else:
            # Large/unrestrictive VSD: softer, lower pitched
            murmur_low = np.random.uniform(80, 150)
            murmur_high = np.random.uniform(300, 600)
            peak_freq = np.random.uniform(150, 300)
            murmur_amp = np.random.uniform(0.3, 1.0)
            # Accentuated P2 due to pulmonary hypertension in large VSDs
            s2_amp = np.random.uniform(1.2, 1.8)
            
        murmur_band = (murmur_low, murmur_high)
        s1_amp = np.random.uniform(0.8, 1.2)
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
            peak_freq=peak_freq,
            vsd_type=vsd_type,
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