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
- Turbulent envelope variations to model realistic fluid dynamics.
- Minimal respiratory amplitude modulation (VSD is respiratory invariant).
- Variations in severity (affecting murmur amplitude, pitch, and P2 intensity).
- Ambient and sensor noise (low-frequency rumble + brown noise).
- Stethoscope acoustic low-pass filtering to remove unnatural high frequencies.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with a smooth envelope and harmonics for realism.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a more natural "thud" sound
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    # Add a harmonic to increase low-frequency complexity
    sound = env * (np.sin(phase) + 0.3 * np.sin(2 * phase))
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, res_freq, res_q, alpha):
    """
    Generates a harsh, holosystolic murmur typical of VSD.
    Uses a modified noise spectrum and resonance to model fluid turbulence.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    # Generate pink noise to better model the spectral roll-off of fluid turbulence
    X_white = np.fft.rfft(np.random.randn(num_samples))
    freqs = np.fft.rfftfreq(num_samples, d=1/fs)
    freqs[0] = freqs[1] # Avoid division by zero
    X_noise = X_white / (freqs ** 0.5)
    noise = np.fft.irfft(X_noise, n=num_samples)
    
    nyq = 0.5 * fs
    
    # 2nd-order HPF for a gentler low-end roll-off, allowing blending with S1/S2
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    # 4th-order LPF for a steeper high-end roll-off
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance to emphasize the "harsh" quality in the mid-frequencies
    b_res, a_res = signal.iirpeak(res_freq / nyq, res_q)
    murmur_res = signal.filtfilt(b_res, a_res, murmur)
    murmur = murmur + np.random.uniform(0.4, 0.8) * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Holosystolic envelope: spans the entire duration
    # A Tukey window provides a flat top with smooth fade-in/fade-out
    env = signal.windows.tukey(num_samples, alpha=alpha)
    
    # Add low-frequency amplitude modulation to simulate turbulent flow variations
    mod = np.random.randn(num_samples)
    b_mod, a_mod = signal.butter(2, 25 / nyq, btype='low')
    mod = signal.filtfilt(b_mod, a_mod, mod)
    if np.max(np.abs(mod)) > 0:
        mod /= np.max(np.abs(mod))
    
    env = env * (1.0 + 0.2 * mod)
    env = np.clip(env, 0, None)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, res_freq, res_q, alpha,
                        s1_amp, s2_amp, s1_f0, s2_f0, noise_level, ambient_level, steth_cutoff):
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
        rr_interval = np.clip(rr_interval, 0.3, 1.5) # Constrain to physiological limits
        current_time += rr_interval
        
    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        # Scale heart sound duration slightly with heart rate
        s1_dur = np.clip(0.08 * np.sqrt(60.0 / hr_mean), 0.05, 0.1)
        s2_dur = np.clip(0.06 * np.sqrt(60.0 / hr_mean), 0.04, 0.08)
        
        # Generate components
        s1 = generate_heart_sound(f0=s1_f0, duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=s2_dur, fs=fs) * s2_amp
        
        # VSD murmur is holosystolic (lasts from S1 to slightly after S2 onset to obscure A2)
        murmur_dur = systole_dur + np.random.uniform(0.01, 0.04)
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, 
                                 res_freq=res_freq, res_q=res_q, alpha=alpha) * murmur_amp
        
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
        
    # Apply very subtle respiratory modulation (VSD murmurs are largely respiratory invariant)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.4) # 12 to 24 breaths per minute
    resp_mod = 1.0 + np.random.uniform(0.02, 0.08) * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    nyq = 0.5 * fs
    
    # Add low-frequency body/sensor noise (rumble)
    rumble = np.random.randn(total_samples)
    b, a = signal.butter(2, np.random.uniform(50, 120) / nyq, btype='low')
    rumble = signal.filtfilt(b, a, rumble)
    rumble = signal.filtfilt(b, a, rumble) # Apply twice for steeper roll-off
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    # Add broadband ambient noise (brown noise) to fill out the spectrum realistically
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    X_brown = X_white / freqs
    ambient = np.fft.irfft(X_brown, n=total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
    # Combine noises (rumble dominates, ambient is subtle)
    bg_noise = rumble * noise_level + ambient * ambient_level
    audio += bg_noise
    
    # Apply stethoscope acoustic low-pass filter (tubing acts as a strong LPF)
    # 4th order filter used to steeply roll off high frequencies and reduce ZCR
    b_steth, a_steth = signal.butter(4, steth_cutoff / nyq, btype='low')
    audio = signal.filtfilt(b_steth, a_steth, audio)
    
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
        # Sample physiological parameters to ensure diversity (covering pediatric and adult)
        hr_mean = np.random.uniform(70, 140)
        hrv_std = np.random.uniform(0.01, 0.06)
        
        # Systole ratio roughly scales with sqrt(RR)
        rr_mean = 60.0 / hr_mean
        base_systole_ratio = 0.35 / np.sqrt(rr_mean)
        systole_ratio = np.clip(base_systole_ratio + np.random.uniform(-0.05, 0.05), 0.3, 0.55)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        # Paradoxically, smaller defects often produce louder murmurs (Maladie de Roger)
        severity = np.random.uniform(0, 1)
        
        s1_amp = np.random.uniform(0.5, 1.0)
        s1_f0 = np.random.uniform(40, 90)
        s2_f0 = np.random.uniform(70, 130)
        
        if severity < 0.5:
            # Restrictive: louder, higher-pitched murmur
            murmur_ratio = np.random.uniform(0.4, 0.9)
            murmur_low = np.random.uniform(100, 250)
            murmur_high = np.random.uniform(400, 900)
            res_freq = np.random.uniform(300, 600)
        else:
            # Unrestrictive: softer, lower-pitched murmur
            murmur_ratio = np.random.uniform(0.15, 0.5)
            murmur_low = np.random.uniform(50, 150)
            murmur_high = np.random.uniform(200, 500)
            res_freq = np.random.uniform(150, 300)
            
        murmur_amp = s1_amp * murmur_ratio
        murmur_band = (murmur_low, murmur_high)
        res_q = np.random.uniform(1.0, 2.5)
        alpha = np.random.uniform(0.05, 0.15) # Ensures rapid attack/decay for plateau
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 1.0) + 0.2 * severity
        
        noise_level = np.random.uniform(0.05, 0.20)
        ambient_level = np.random.uniform(0.01, 0.05)
        steth_cutoff = np.random.uniform(500, 1200)
        
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
            res_q=res_q,
            alpha=alpha,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            s1_f0=s1_f0,
            s2_f0=s2_f0,
            noise_level=noise_level,
            ambient_level=ambient_level,
            steth_cutoff=steth_cutoff
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