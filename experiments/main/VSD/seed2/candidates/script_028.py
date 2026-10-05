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
- S1 and S2 heart sounds as frequency-modulated bursts with harmonics for realistic timbre.
- The VSD murmur as bandpass-filtered pink noise with a holosystolic envelope and a 350 Hz resonance.
- Heart rate variability (HRV).
- Respiratory invariance (VSD murmurs do not vary significantly with respiration).
- Variations in severity (affecting murmur amplitude, pitch, and P2 intensity).
- Ambient and body noise using low-passed pink noise to maintain realistic zero-crossing rates.
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
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    base wave with harmonics to create a realistic, broadband "thud".
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Asymmetric envelope: faster attack, slower decay
    env = (t / duration) ** 1.5 * (1 - t / duration) ** 3
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Base wave with harmonics for realistic spectral richness
    sound = (np.sin(phase) + 
             0.4 * np.sin(2 * phase) + 
             0.2 * np.sin(3 * phase) + 
             0.1 * np.sin(4 * phase))
    
    # Add a little bit of noise to simulate valve closure turbulence
    noise = np.random.randn(len(t)) * 0.05
    sound += noise
    
    # Low pass filter to remove harsh high frequencies from the noise
    nyq = 0.5 * fs
    b, a = signal.butter(2, 400 / nyq, btype='low')
    sound = signal.filtfilt(b, a, sound)
    
    sound = env * sound
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band):
    """
    Generates a harsh, high-pitched holosystolic murmur typical of VSD.
    Includes a resonance peak to emphasize the 'harsh' quality.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    
    # Main bandpass filter to give the murmur its characteristic pitch
    b, a = signal.butter(4, [freq_band[0]/nyq, freq_band[1]/nyq], btype='band')
    murmur = signal.filtfilt(b, a, noise)
    
    # Add a resonance at 350 Hz to emphasize the "harsh" quality (Q ~ 1.5)
    b_peak, a_peak = signal.iirpeak(350 / nyq, 1.5)
    murmur_res = signal.filtfilt(b_peak, a_peak, noise)
    
    # Combine broadband murmur with the resonant peak
    murmur = murmur + 0.4 * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration (S1 to S2)
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    # alpha=0.1 ensures a rapid attack/decay with a long, flat plateau
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
        
    # Note: Respiratory modulation is intentionally omitted to respect VSD respiratory invariance
        
    # Add low-frequency body/sensor noise (using pink noise low-passed at 250 Hz)
    # This fills the low-frequency spectrum and maintains realistic ZCR
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 250 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a tiny bit of ambient pink noise for realism (prevents digital silence)
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
        hr_mean = np.random.uniform(60, 110)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.3, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        murmur_amp = np.random.uniform(0.5, 2.0) * (1.2 - 0.4 * severity)
        murmur_low = np.random.uniform(120, 250)
        murmur_high = np.random.uniform(500, 900)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.6, 1.4)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.6, 1.4) + 0.4 * severity
        
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