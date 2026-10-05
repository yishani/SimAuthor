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

Refinements in this version:
- Improved spectral profile of the murmur using a gentler 2nd-order bandpass 
  and a 350 Hz resonance to match the "harsh" quality and fix MFCC discrepancies.
- Added realistic low-frequency baseline rumble and broadband ambient noise 
  to correct the excessively high Zero Crossing Rate (ZCR).
- Enhanced S1 and S2 with broadband transient noise for a more realistic "thud".
- Introduced a 20% probability of Muscular VSD (decrescendo envelope, shorter duration) 
  to increase natural diversity and match clinical subtypes.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave combined with a low-passed noise transient for a realistic "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    tone = np.sin(phase)
    
    # Broadband transient noise for the valve snap
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 300 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    if np.max(np.abs(noise)) > 0:
        noise /= np.max(np.abs(noise))
        
    sound = env * (tone + 0.4 * noise)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, env_type='plateau'):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    Uses pink noise with a 2nd-order bandpass and a 350 Hz resonance.
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
    
    # Gentler bandpass filter (2nd order) for more natural roll-off
    nyq = 0.5 * fs
    b, a = signal.butter(2, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add resonance at 350 Hz to emphasize the "harsh" quality
    b_res, a_res = signal.iirpeak(350, 1.5, fs)
    murmur_res = signal.filtfilt(b_res, a_res, noise)
    
    murmur = murmur + 0.5 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Envelope shaping
    env = signal.windows.tukey(num_samples, alpha=0.1)
    if env_type == 'decrescendo':
        decrescendo = np.linspace(1.0, 0.0, num_samples)
        env = env * decrescendo
        
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level, vsd_type):
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
        
        # Split S2 (A2 and P2 components)
        s2_a2 = generate_heart_sound(f0=100, duration=0.04, fs=fs) * s2_amp
        s2_p2 = generate_heart_sound(f0=120, duration=0.04, fs=fs) * (s2_amp * np.random.uniform(0.7, 1.1))
        split_delay = np.random.uniform(0.01, 0.03)
        s2_length = int((0.04 + split_delay) * fs)
        s2 = np.zeros(s2_length)
        s2[:len(s2_a2)] += s2_a2
        s2[-len(s2_p2):] += s2_p2
        
        # VSD murmur timing and envelope based on subtype
        if vsd_type == 'muscular':
            murmur_dur = systole_dur * np.random.uniform(0.6, 0.8) # Terminates early
            murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, env_type='decrescendo') * murmur_amp
        else:
            murmur_dur = systole_dur + 0.02 # Holosystolic, extends slightly into S2 to obscure A2
            murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, env_type='plateau') * murmur_amp
        
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
        add_to_audio(murmur, beat_time) # Murmur starts synchronously with S1
        add_to_audio(s2, beat_time + systole_dur)
        
    # Apply respiratory modulation (low frequency amplitude modulation)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.15 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add realistic body/sensor noise (pink noise based)
    bg_noise_white = np.random.randn(total_samples)
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    bg_noise_pink = np.fft.irfft(np.fft.rfft(bg_noise_white) / np.sqrt(freqs), n=total_samples)
    
    # 1. Low-frequency rumble (body noise, < 100 Hz) to provide a realistic baseline
    b_low, a_low = signal.butter(2, 100 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b_low, a_low, bg_noise_pink)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    # 2. Broadband ambient/sensor noise
    ambient = bg_noise_pink
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
    audio += rumble * noise_level * 2.0
    audio += ambient * noise_level * 0.2
    
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
        
        # VSD Subtype: 80% Membranous (holosystolic), 20% Muscular (early termination)
        vsd_type = np.random.choice(['membranous', 'muscular'], p=[0.8, 0.2])
        
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
        
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(500, 700)
        murmur_band = (murmur_low, murmur_high)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 0.8) + 0.3 * severity
        
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
            noise_level=noise_level,
            vsd_type=vsd_type
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