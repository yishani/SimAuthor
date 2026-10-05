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

Refinement:
Addressed severe negative shifts in mid-to-high MFCCs and unrealistic visual 
morphology by introducing broadband transients (valve snaps) to S1/S2 and 
restoring a natural high-frequency noise floor. The ambient noise is now added 
after the global stethoscope filter with a gentler roll-off, and a strong 
low-frequency body rumble is included to correct the ZCR and spectral balance.
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
    Generates a single heart sound (S1 or S2) combining a low-frequency 
    thump and a broadband transient snap (valve closure).
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Low-frequency thump with asymmetric envelope (fast attack, slower decay)
    env = (t / duration) * np.exp(-6 * t / duration)
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Slight downward frequency modulation for a natural "thud"
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    thump = env * np.sin(phase)
    
    # Broadband snap (valve closure) to provide realistic vertical transients
    snap_env = np.exp(-80 * t) # Very fast decay (~12ms)
    snap_noise = generate_pink_noise(len(t))
    
    # High-pass the snap to remove low-frequency mud, allowing it to extend high
    b_snap, a_snap = signal.butter(1, 100 / (0.5 * fs), btype='high')
    snap = signal.filtfilt(b_snap, a_snap, snap_noise) * snap_env
    
    if np.max(np.abs(snap)) > 0:
        snap /= np.max(np.abs(snap))
        
    # Combine thump and snap
    sound = thump + 0.4 * snap
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, is_muscular=False):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    murmur_low, murmur_high = freq_band
    
    # 2nd-order Butterworth High-Pass Filter
    b_hp, a_hp = signal.butter(2, murmur_low / nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    # 4th-order Butterworth Low-Pass Filter
    b_lp, a_lp = signal.butter(4, murmur_high / nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    if is_muscular:
        # Muscular VSD: early-to-mid systolic, decrescendo shape
        env = signal.windows.tukey(num_samples, alpha=0.4)
        decay = np.linspace(1.0, 0.0, num_samples)
        env = env * decay
    else:
        # Membranous VSD: Holosystolic envelope spanning S1 to S2
        # Alpha = 0.2 ensures a rapid attack/decay with a flat plateau
        env = signal.windows.tukey(num_samples, alpha=0.2)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, rumble_level, 
                        ambient_level, steth_cutoff, amb_cutoff, is_muscular):
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
        s1 = generate_heart_sound(f0=np.random.uniform(50, 80), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(70, 110), duration=s2_dur, fs=fs) * s2_amp
        
        if is_muscular:
            # Muscular VSD murmurs end before S2
            murmur_dur = systole_dur * np.random.uniform(0.6, 0.85)
        else:
            # Membranous VSD murmurs are holosystolic
            murmur_dur = systole_dur
            
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, is_muscular=is_muscular) * murmur_amp
        
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
        
    # Apply global stethoscope low-pass filter to simulate acoustic transmission
    # 1st order provides a gentler roll-off, preserving some transient energy
    b_steth, a_steth = signal.butter(1, steth_cutoff / (0.5 * fs), btype='low')
    audio = signal.filtfilt(b_steth, a_steth, audio)
    
    # Add low-frequency body rumble (internal) to balance ZCR
    bg_noise = generate_pink_noise(total_samples)
    b_rumble, a_rumble = signal.butter(2, 50 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b_rumble, a_rumble, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * rumble_level
    
    # Add ambient/sensor noise AFTER stethoscope filter to provide a natural noise floor
    ambient = generate_pink_noise(total_samples)
    b_amb, a_amb = signal.butter(1, amb_cutoff / (0.5 * fs), btype='low')
    ambient = signal.filtfilt(b_amb, a_amb, ambient)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * ambient_level
    
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
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.30, 0.45)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        # Subtype selection (~20% muscular)
        is_muscular = np.random.rand() < 0.2
        
        # Murmur amplitude is slightly lower than S1/S2 to match visual evidence (haze vs bright lines)
        murmur_amp = np.random.uniform(0.3, 0.8) * (1.2 - 0.4 * severity)
        
        # Severity dictates the frequency band:
        # Small VSD (severity=0): loud, high-pitched (peak energy ~200-600 Hz)
        # Large VSD (severity=1): softer, lower-pitched (peak energy ~150-400 Hz)
        murmur_low = np.random.uniform(150, 250) - 50 * severity
        murmur_high = np.random.uniform(550, 750) - 200 * severity
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.6, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.6, 1.2) + 0.3 * severity
        
        rumble_level = np.random.uniform(0.05, 0.25)
        ambient_level = np.random.uniform(0.01, 0.06)
        
        steth_cutoff = np.random.uniform(600, 1000)
        amb_cutoff = np.random.uniform(1500, 3000)
        
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
            rumble_level=rumble_level,
            ambient_level=ambient_level,
            steth_cutoff=steth_cutoff,
            amb_cutoff=amb_cutoff,
            is_muscular=is_muscular
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