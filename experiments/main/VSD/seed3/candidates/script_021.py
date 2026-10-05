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
- Fixed an overly steep (effective 16th-order) bandpass filter on the murmur 
  by reducing it to a single 2nd-order filtfilt pass (effective 4th-order). 
  This allows a natural spectral roll-off, addressing negative mid/high MFCC shifts.
- Increased the murmur's upper frequency limit to 600-1000 Hz to match the blueprint.
- Lowered S1 and S2 fundamental frequencies (to 50 Hz and 70 Hz) and increased 
  low-frequency background rumble to anchor zero-crossings, drastically reducing 
  the unnaturally high ZCR.
- Applied the 350 Hz resonance peak filter to the already bandpassed murmur to 
  prevent injecting out-of-band high-frequency noise.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with a smooth envelope.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a more natural "thud" sound
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    sound = env * np.sin(phase)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses pink noise to naturally model the energy roll-off of fluid turbulence.
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
    
    # 2nd-order Butterworth bandpass (becomes 4th-order zero-phase with filtfilt)
    # This provides a natural roll-off rather than a brick-wall cutoff.
    nyq = 0.5 * fs
    b, a = signal.butter(2, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add a slight resonance at 350 Hz to emphasize the "harsh" quality
    # Applied to the bandpassed signal to avoid amplifying out-of-band noise
    b_peak, a_peak = signal.iirpeak(350, 1.5, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, murmur)
    
    murmur = murmur + 0.5 * murmur_peak
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    # alpha=0.1 on a ~300ms signal gives ~15ms attack/decay
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
        # Lowered fundamental frequencies to provide realistic low-frequency energy
        s1 = generate_heart_sound(f0=50, duration=0.065, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=70, duration=0.050, fs=fs) * s2_amp
        
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
    
    # Add strong low-frequency body/sensor noise (rumble) to anchor zero-crossings
    bg_noise = np.random.randn(total_samples)
    b, a = signal.butter(2, 80 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a gentle ambient noise floor (low-passed to simulate acoustic transmission)
    ambient = np.random.randn(total_samples)
    b_amb, a_amb = signal.butter(1, 1000 / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * 0.015
    
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
        
        # Murmur amplitude balanced with S1/S2 to prevent overpowering the signal
        murmur_amp = np.random.uniform(1.2, 1.8) * (1.0 - 0.4 * severity)
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 1000)
        murmur_band = (murmur_low, murmur_high)
        
        # Stronger S1/S2 to provide realistic low-frequency energy
        s1_amp = np.random.uniform(1.0, 1.4)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(1.0, 1.4) + 0.4 * severity
        
        # Increased low-frequency noise level
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