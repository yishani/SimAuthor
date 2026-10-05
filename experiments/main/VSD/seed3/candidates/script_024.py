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
- Replaced the hard-low-passed ambient noise with a combination of brown noise (1/f^2) 
  and broadband pink noise (1/f). This provides a realistic, continuous spectral roll-off 
  that fixes the large negative shifts in higher MFCC bands, and establishes a strong 
  low-frequency baseline that naturally reduces the excessively high Zero-Crossing Rate (ZCR).
- Improved the heart sound (S1/S2) generator to use an asymmetric envelope with a fast 
  attack and added low-frequency rumble, replacing the pure sine waves with more realistic 
  broadband transients that blend naturally with the murmur.
- Maintained the core VSD murmur mechanics (pink noise, 200-600 Hz dominant band, 
  plateau envelope) as specified in the scientific blueprint.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with an asymmetric envelope and low-frequency rumble for a 
    realistic broadband transient.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: fast attack, slower exponential decay
    attack_time = 0.01
    env = (1 - np.exp(-t / attack_time)) * np.exp(-t / (duration / 3))
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Frequency modulation: starts higher, drops lower for a natural "thud"
    f_t = f0 + 20 * np.exp(-t * 30)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    sound = env * np.sin(phase)
    
    # Add low-frequency rumble to make the transient more broadband
    rumble = np.random.randn(len(t))
    b, a = signal.butter(2, 100 / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b, a, rumble)
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    sound += 0.2 * env * rumble
    
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
    
    # Bandpass filter to give the murmur its characteristic pitch
    nyq = 0.5 * fs
    b_hp, a_hp = signal.butter(4, freq_band[0]/nyq, btype='high')
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a slight resonance at 350 Hz to emphasize the "harsh" quality
    b_peak, a_peak = signal.iirpeak(350, 1.5, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur + 0.5 * murmur_peak
    
    # Re-filter to ensure strict band limits
    murmur = signal.filtfilt(b_hp, a_hp, murmur)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
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
                        murmur_amp, murmur_band, s1_amp, s2_amp):
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
    
    # Add brown noise (1/f^2) for realistic body/sensor background
    # This provides a natural continuous roll-off (fixing negative MFCCs)
    # and is dominated by low frequencies (lowering the overall ZCR)
    white_bg = np.random.randn(total_samples)
    X_bg = np.fft.rfft(white_bg)
    S_bg = np.arange(1, len(X_bg) + 1)
    brown = np.fft.irfft(X_bg / S_bg, n=total_samples)
    if np.max(np.abs(brown)) > 0:
        brown /= np.max(np.abs(brown))
    audio += brown * np.random.uniform(0.1, 0.25)
    
    # Add a tiny amount of broadband pink noise to represent the room/sensor noise floor
    white_pink = np.random.randn(total_samples)
    X_pink = np.fft.rfft(white_pink)
    S_pink = np.sqrt(np.arange(1, len(X_pink) + 1))
    pink = np.fft.irfft(X_pink / S_pink, n=total_samples)
    if np.max(np.abs(pink)) > 0:
        pink /= np.max(np.abs(pink))
    audio += pink * np.random.uniform(0.005, 0.015)
    
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
        murmur_amp = np.random.uniform(1.4, 2.0) * (1.0 - 0.4 * severity)
        murmur_low = np.random.uniform(150, 250)
        murmur_high = np.random.uniform(600, 800)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.4 * severity
        
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
            s2_amp=s2_amp
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