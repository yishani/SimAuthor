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
- S1 and S2 heart sounds as frequency-modulated low-frequency bursts with harmonics.
- The VSD murmur as filtered pink noise with a holosystolic envelope, shaped 
  to have a harsh resonance and appropriate spectral roll-off.
- Heart rate variability (HRV) and respiratory amplitude modulation.
- Variations in severity (affecting murmur amplitude and P2 intensity).
- A composite background noise profile (rumble, ambient, and baseline wander) 
  to accurately reflect real-world sensor acoustics and zero-crossing rates.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with harmonics and a smooth envelope.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a more natural "thud" sound
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base sine wave + harmonics for a richer, more broadband transient
    sound = np.sin(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.sin(3 * phase)
    sound = env * sound
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, holosystolic murmur typical of VSD.
    Uses true pink noise and resonance to model fluid turbulence.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    # Generate true pink noise (1/f power spectrum)
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples, d=1/fs)
    freqs[0] = freqs[1] # Avoid division by zero
    X_noise = X_white / np.sqrt(freqs)
    noise = np.fft.irfft(X_noise, n=num_samples)
    
    nyq = 0.5 * fs
    
    # 4th-order HPF at ~200 Hz
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    # 4th-order LPF at ~600-900 Hz for natural high-end roll-off
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance to emphasize the "harsh" quality in the mid-frequencies
    w0 = np.random.uniform(300, 400) / nyq
    b_res, a_res = signal.iirpeak(w0, 1.5)
    murmur_res = signal.filtfilt(b_res, a_res, murmur)
    murmur = murmur + 0.6 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    # alpha=0.1 gives ~15ms attack/release for a 300ms duration
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
        
    # Patient-specific heart sound frequencies
    s1_f0 = np.random.uniform(50, 80)
    s2_f0 = np.random.uniform(80, 120)
        
    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        # Generate components
        s1 = generate_heart_sound(f0=s1_f0, duration=0.08, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=0.06, fs=fs) * s2_amp
        
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
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.15 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add realistic composite background noise profile
    # 1. Low-frequency rumble (body/sensor noise)
    rumble = np.random.randn(total_samples)
    b_r, a_r = signal.butter(2, 100 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b_r, a_r, rumble)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    # 2. Broadband ambient noise (pink noise to fill high-frequency spectrum)
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    ambient = np.fft.irfft(X_white / np.sqrt(freqs), n=total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
    # 3. Baseline wander (very low frequency, reduces artificial zero-crossings)
    wander = np.random.randn(total_samples)
    b_w, a_w = signal.butter(2, 2 / (0.5 * fs), btype='low')
    wander = signal.filtfilt(b_w, a_w, wander)
    if np.max(np.abs(wander)) > 0:
        wander /= np.max(np.abs(wander))

    # Combine noises with appropriate relative weightings
    noise_sig = rumble * (noise_level * 2.0) + \
                ambient * (noise_level * 0.5) + \
                wander * (noise_level * 4.0)
                
    audio += noise_sig
    
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
        
        s1_amp = np.random.uniform(0.6, 0.9)
        
        if severity < 0.5:
            # Restrictive: louder murmur, but balanced to not completely mask S1/S2
            murmur_ratio = np.random.uniform(1.0, 1.5)
        else:
            # Unrestrictive: softer murmur
            murmur_ratio = np.random.uniform(0.4, 0.8)
            
        murmur_amp = s1_amp * murmur_ratio
        
        # Adjusted frequency bands to better match empirical spectral profiles
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 900)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.6, 0.9) + 0.2 * severity
        
        noise_level = np.random.uniform(0.02, 0.08)
        
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