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
- S1 and S2 heart sounds as broadband bursts (frequency-modulated base + noise).
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope 
  and a mid-frequency resonance to emphasize the "harsh" quality.
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary significantly with respiration).
- Variations in severity (affecting murmur amplitude and P2 intensity).
- Ambient and sensor noise using pink noise to maintain realistic zero-crossing rates.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_pink_noise(num_samples):
    """
    Generates pink noise (1/f) using the frequency domain.
    Pink noise is preferable to white noise as it naturally models 
    the energy roll-off of fluid turbulence and prevents unrealistic ZCR.
    """
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples)
    freqs[0] = freqs[1]  # Avoid division by zero
    X_pink = X_white / np.sqrt(freqs)
    return np.fft.irfft(X_pink, n=num_samples)

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2).
    Uses a frequency-modulated sine wave with harmonics and a low-pass 
    filtered noise burst to create a realistic, broadband "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 20 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base tone and first harmonic
    sound = np.sin(phase) + 0.3 * np.sin(2 * phase)
    
    # Add transient low-frequency noise for broadband impact
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 200 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound += 0.5 * noise
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, res_freq):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Uses separate HPF and LPF for a natural roll-off, and adds a resonance peak.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    
    # Apply 2nd order filters with filtfilt to achieve effective 4th-order zero-phase response.
    # This provides a gentler, more natural spectral slope than an 8th-order effective bandpass.
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    b_lp, a_lp = signal.butter(2, freq_band[1]/nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a resonance (Q=1.5) to emphasize the "harsh" quality and break up the flat spectrum
    b_res, a_res = signal.iirpeak(res_freq / nyq, 1.5)
    resonance = signal.filtfilt(b_res, a_res, noise)
    
    murmur = murmur + 0.4 * resonance
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # alpha=0.15 ensures a rapid attack/decay with a long, flat plateau
    env = signal.windows.tukey(num_samples, alpha=0.15)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, res_freq, s1_amp, s2_amp, noise_level):
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
        
        s1_dur = np.random.uniform(0.05, 0.08)
        s2_dur = np.random.uniform(0.04, 0.06)
        
        # Generate components
        s1 = generate_heart_sound(f0=np.random.uniform(70, 90), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(100, 130), duration=s2_dur, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic (lasts exactly from S1 to S2)
        murmur = generate_murmur(duration=systole_dur, fs=fs, freq_band=murmur_band, res_freq=res_freq) * murmur_amp
        
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
        
    # Note: Respiratory modulation is intentionally omitted to respect VSD respiratory invariance
        
    # Add low-frequency body/sensor noise (cutoff at 250 Hz to provide a realistic baseline)
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 250 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a tiny bit of ambient pink noise for realism
    ambient = generate_pink_noise(total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * 0.005
    
    return audio

def main():
    # Set explicit random seed for reproducibility
    np.random.seed(42)
    
    output_dir = "[PROJECT_ROOT]/artifacts/vsd_audio/runs/signal/seed2/generated/"
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
        
        murmur_amp = np.random.uniform(0.4, 2.2) * (1.2 - 0.4 * severity)
        murmur_low = np.random.uniform(150, 300)
        murmur_high = np.random.uniform(500, 900)
        murmur_band = (murmur_low, murmur_high)
        res_freq = np.random.uniform(300, 450)
        
        s1_amp = np.random.uniform(0.6, 1.4)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.6, 1.4) + 0.4 * severity
        
        noise_level = np.random.uniform(0.03, 0.15)
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_ratio=systole_ratio,
            murmur_amp=murmur_amp,
            murmur_band=murmur_band,
            res_freq=res_freq,
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