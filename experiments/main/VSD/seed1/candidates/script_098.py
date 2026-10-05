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
- Addressed under-diverse spectral profiles (MFCC-0) by randomizing the 
  ambient noise color (from pink to red noise).
- Addressed negative mean shifts in mid-high MFCCs by significantly raising 
  the upper bound of the stethoscope low-pass filter and murmur frequency bands.
- Increased ZCR diversity and realism by introducing occasional handling noise 
  (clicks) and a subtle high-frequency valve closure transient to S1/S2.
- Softened the murmur's "blocky" plateau envelope with increased amplitude 
  modulation depth and wider Tukey window alpha ranges to better blend with S1/S2.
"""

import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import os

def generate_heart_sound(f0, duration, fs):
    """
    Generates a single heart sound (S1 or S2) using a frequency-modulated
    sine wave with a smooth envelope, plus a subtle high-frequency valve click.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope for the main low-frequency thump
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a natural "thud"
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    thump = env * np.sin(phase)
    
    # Add a subtle high-frequency click for valve closure realism
    click_dur = min(0.02, duration)
    t_click = t[:int(click_dur * fs)]
    env_click = np.exp(-200 * t_click)
    click_f = np.random.uniform(150, 400)
    click = env_click * np.sin(2 * np.pi * click_f * t_click)
    
    sound = thump
    sound[:len(click)] += click * np.random.uniform(0.05, 0.2)
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, res_freq, alpha, envelope_type='plateau'):
    """
    Generates a harsh murmur typical of VSD.
    Uses filtered pink noise and resonance to model fluid turbulence.
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
    
    # 2nd-order HPF for a gentler low-end roll-off
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    # 4th-order LPF for a steeper high-end roll-off
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add resonance to emphasize the "harsh" quality in the mid-frequencies
    q_val = np.random.uniform(0.5, 2.0)
    b_res, a_res = signal.iirpeak(res_freq / nyq, q_val)
    murmur_res = signal.filtfilt(b_res, a_res, murmur)
    murmur = murmur + np.random.uniform(0.5, 1.2) * murmur_res
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Apply envelope based on VSD subtype
    if envelope_type == 'plateau':
        # Holosystolic envelope: spans the entire duration
        env = signal.windows.tukey(num_samples, alpha=alpha)
        # Add amplitude modulation for turbulence realism and to soften the "blocky" shape
        t = np.linspace(0, duration, num_samples, endpoint=False)
        am_depth = np.random.uniform(0.1, 0.3)
        am_rate = np.random.uniform(15, 50)
        am = 1.0 - am_depth * np.sin(2 * np.pi * am_rate * t)
        env = env * am
    elif envelope_type == 'decrescendo':
        # Muscular VSD envelope: early-to-mid systolic with exponential decay
        attack_samples = int(0.02 * fs)
        if attack_samples > num_samples:
            attack_samples = num_samples // 2
        if attack_samples == 0:
            attack_samples = 1
        attack = np.linspace(0, 1, attack_samples)
        decay_samples = num_samples - attack_samples
        t_decay = np.linspace(0, 4, decay_samples) # 4 time constants
        decay = np.exp(-t_decay)
        env = np.concatenate((attack, decay))
    else:
        env = np.ones(num_samples)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, res_freq, murmur_alpha, murmur_subtype,
                        s1_amp, s2_amp, noise_level, rumble_cutoff, steth_cutoff):
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
    s1_f0 = np.random.uniform(40, 90)
    s2_f0 = np.random.uniform(70, 130)
        
    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        # Variable heart sound durations for diversity
        s1_dur = np.random.uniform(0.06, 0.10)
        s2_dur = np.random.uniform(0.04, 0.08)
        
        # Generate components
        s1 = generate_heart_sound(f0=s1_f0, duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=s2_dur, fs=fs) * s2_amp
        
        # Determine murmur duration and envelope based on subtype
        if murmur_subtype == 'muscular':
            # Muscular VSD closes during late systole
            murmur_dur = systole_dur * np.random.uniform(0.4, 0.7)
            env_type = 'decrescendo'
        else:
            # Membranous VSD is holosystolic, slightly overlapping S2
            murmur_dur = systole_dur + np.random.uniform(0.0, 0.05)
            env_type = 'plateau'
            
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, 
                                 res_freq=res_freq, alpha=murmur_alpha, envelope_type=env_type) * murmur_amp
        
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
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.03 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add low-frequency body/sensor noise (rumble)
    rumble = np.random.randn(total_samples)
    b, a = signal.butter(2, rumble_cutoff / (0.5 * fs), btype='low')
    rumble = signal.filtfilt(b, a, rumble)
    rumble = signal.filtfilt(b, a, rumble) # Apply twice for steeper roll-off
    if np.max(np.abs(rumble)) > 0:
        rumble /= np.max(np.abs(rumble))
        
    # Add broadband ambient noise with variable spectral slope (pink to red) to increase MFCC-0 diversity
    X_white = np.fft.rfft(np.random.randn(total_samples))
    freqs = np.fft.rfftfreq(total_samples, d=1/fs)
    freqs[0] = freqs[1]
    noise_power = np.random.uniform(0.5, 1.5) 
    X_ambient = X_white / (freqs ** noise_power)
    ambient = np.fft.irfft(X_ambient, n=total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
    # Combine noises
    ambient_level = noise_level * np.random.uniform(0.1, 0.8)
    bg_noise = rumble * noise_level + ambient * ambient_level
    audio += bg_noise
    
    # Add occasional random clicks/artifacts (handling noise) to increase ZCR diversity
    num_clicks = np.random.randint(0, 6)
    for _ in range(num_clicks):
        click_idx = np.random.randint(0, total_samples - int(0.01 * fs))
        click_dur = int(np.random.uniform(0.002, 0.01) * fs)
        click = np.random.randn(click_dur) * np.random.uniform(0.5, 2.0) * noise_level
        click *= np.exp(-np.linspace(0, 5, click_dur))
        audio[click_idx:click_idx+click_dur] += click
    
    # Apply stethoscope acoustic low-pass filter
    b_steth, a_steth = signal.butter(2, steth_cutoff / (0.5 * fs), btype='low')
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
        # Sample physiological parameters to ensure diversity
        hr_mean = np.random.uniform(60, 110)
        hrv_std = np.random.uniform(0.01, 0.06)
        systole_ratio = np.random.uniform(0.3, 0.45)
        
        # Subtype: 80% Membranous (holosystolic), 20% Muscular (early-to-mid systolic)
        subtype = np.random.choice(['membranous', 'muscular'], p=[0.8, 0.2])
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        s1_amp = np.random.uniform(0.5, 1.0)
        
        if severity < 0.5:
            # Restrictive: louder, higher-pitched murmur
            murmur_ratio = np.random.uniform(0.5, 1.2)
            murmur_low = np.random.uniform(80, 200)
            murmur_high = np.random.uniform(500, 1200)
            res_freq = np.random.uniform(400, 800)
        else:
            # Unrestrictive: softer, lower-pitched murmur
            murmur_ratio = np.random.uniform(0.15, 0.6)
            murmur_low = np.random.uniform(40, 120)
            murmur_high = np.random.uniform(250, 600)
            res_freq = np.random.uniform(200, 400)
            
        murmur_amp = s1_amp * murmur_ratio
        murmur_band = (murmur_low, murmur_high)
        
        # Variable Tukey window alpha to increase envelope diversity
        murmur_alpha = np.random.uniform(0.2, 0.8)
        
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.5, 1.0) + 0.2 * severity
        
        # Noise level and wide variable cutoffs to improve ZCR and MFCC spread
        noise_level = np.random.uniform(0.02, 0.3)
        rumble_cutoff = np.random.uniform(20, 150)
        # Significantly widened stethoscope cutoff to allow high-frequency murmur/noise components
        steth_cutoff = np.random.uniform(800, 5000)
        
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
            murmur_alpha=murmur_alpha,
            murmur_subtype=subtype,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            noise_level=noise_level,
            rumble_cutoff=rumble_cutoff,
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