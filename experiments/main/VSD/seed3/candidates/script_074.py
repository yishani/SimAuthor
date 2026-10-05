"""
Ventricular Septal Defect (VSD) Audio Simulator

This script generates 100 synthetic phonocardiogram (PCG) audio samples 
representing Ventricular Septal Defect (VSD). 

Physiological basis:
VSD is classically characterized by a holosystolic (pansystolic) murmur. 
The murmur begins with the first heart sound (S1) and extends to the 
second heart sound (S2). The murmur's characteristics (pitch, amplitude, 
and envelope) vary significantly based on the defect's size and location.

Refinements:
- Hypothesis: The generated signals exhibit under-diverse MFCCs and a visually "blocky", artificial 
  spectrogram appearance because the murmur and global low-pass filters are too steep (4th order applied 
  bidirectionally), creating unnatural hard frequency cutoffs. Additionally, the murmur's spectral roll-off is fixed.
- Fix: Reduced the order of the murmur's upper low-pass filter and the global low-pass filter 
  from 4th to 2nd order to create a gentler, more natural high-frequency roll-off. Introduced a 
  variable spectral exponent for the murmur noise (0.8 to 1.6) to increase spectral diversity, 
  widened the global low-pass cutoff range (800-2500 Hz) to allow realistic high-frequency leakage, 
  and increased the transient noise cutoff for S1/S2 to better match real broadband clicks.
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
    # Increased cutoff to 800 Hz to provide a more realistic broadband "click"
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 800 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    
    sound = sound + 0.15 * noise
    sound = sound * env
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, env_type='plateau', alpha=0.1, murmur_noise_color=1.0):
    """
    Generates a harsh holosystolic murmur typical of VSD.
    Models both membranous (plateau) and muscular (decrescendo) envelopes,
    with added turbulence modulation for realism.
    """
    N = int(duration * fs)
    if N == 0:
        return np.array([])
        
    t = np.linspace(0, duration, N, endpoint=False)
    
    # Generate colored noise with variable spectral roll-off
    white = np.random.randn(N)
    X = np.fft.rfft(white)
    S = np.power(np.arange(1, len(X) + 1, dtype=float), murmur_noise_color)
    X_colored = X / S
    noise = np.fft.irfft(X_colored, n=N)
    
    nyq = 0.5 * fs
    
    # Gentle HPF (2nd order) to keep some low-frequency rumble
    sos_hp = signal.butter(2, freq_band[0]/nyq, btype='high', output='sos')
    # Gentle LPF (2nd order) for a more natural, tapering upper roll-off
    sos_lp = signal.butter(2, freq_band[1]/nyq, btype='low', output='sos')
    
    murmur_base = signal.sosfiltfilt(sos_hp, noise)
    
    # Add a slight resonance in the lower-mid band to emphasize the "harsh" quality
    res_freq = freq_band[0] + (freq_band[1] - freq_band[0]) * 0.4
    b_peak, a_peak = signal.iirpeak(res_freq, 2.0, fs)
    murmur_peak = signal.filtfilt(b_peak, a_peak, noise)
    
    murmur = murmur_base + 0.5 * murmur_peak
    
    # Apply LPF once to the combined signal to ensure upper band limits without over-steepening
    murmur = signal.sosfiltfilt(sos_lp, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    # Apply base envelope based on VSD subtype
    if env_type == 'decrescendo':
        env = signal.windows.tukey(N, alpha=alpha)
        decay = np.linspace(1.0, 0.0, N)
        env = env * decay
    else:
        env = signal.windows.tukey(N, alpha=alpha)
        
    # Add low-frequency turbulence modulation to break up the artificial plateau
    turbulence = np.random.randn(N)
    b_t, a_t = signal.butter(2, 10 / nyq, btype='low')
    turbulence = signal.filtfilt(b_t, a_t, turbulence)
    if np.max(np.abs(turbulence)) > 0:
        turbulence /= np.max(np.abs(turbulence))
    
    # Modulate envelope between 0.6 and 1.4 for more natural variation
    env = env * (1.0 + 0.4 * turbulence)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, murmur_dur_ratio, env_type,
                        s1_amp, s2_amp, s1_f0, s1_dur, s2_f0, s2_dur,
                        noise_level, ambient_noise_level, ambient_noise_color, 
                        murmur_noise_color, alpha, global_lp_cutoff):
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
        
    def add_to_audio(sig, start_time):
        start_idx = int(start_time * fs)
        end_idx = start_idx + len(sig)
        if start_idx < total_samples:
            if end_idx > total_samples:
                sig = sig[:total_samples - start_idx]
                end_idx = total_samples
            audio[start_idx:end_idx] += sig

    # Synthesize cardiac cycles
    for beat_time in beats:
        rr = 60.0 / hr_mean
        systole_dur = rr * systole_ratio
        
        # Generate components
        s1 = generate_heart_sound(f0=s1_f0, duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=s2_f0, duration=s2_dur, fs=fs) * s2_amp
        
        # VSD murmur duration depends on subtype (muscular is shorter)
        murmur_dur = systole_dur * murmur_dur_ratio
        murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, 
                                 env_type=env_type, alpha=alpha, 
                                 murmur_noise_color=murmur_noise_color) * murmur_amp
        
        # Assemble the cycle
        add_to_audio(s1, beat_time)
        add_to_audio(murmur, beat_time) # Murmur starts synchronously with S1
        add_to_audio(s2, beat_time + systole_dur) # S2 marks the end of systole
        
    # Apply respiratory modulation (low frequency amplitude modulation)
    t = np.linspace(0, duration_sec, total_samples, endpoint=False)
    resp_rate = np.random.uniform(0.2, 0.35) # 12 to 21 breaths per minute
    resp_mod = 1.0 + 0.05 * np.sin(2 * np.pi * resp_rate * t)
    audio *= resp_mod
    
    # Add low-frequency body/sensor noise (rumble)
    bg_noise = np.random.randn(total_samples)
    sos_bg = signal.butter(2, np.random.uniform(40, 120) / (0.5 * fs), btype='low', output='sos')
    bg_noise = signal.sosfiltfilt(sos_bg, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add a continuous ambient noise floor to fill spectral gaps
    white = np.random.randn(total_samples)
    X = np.fft.rfft(white)
    S = np.power(np.arange(1, len(X) + 1, dtype=float), ambient_noise_color)
    X_colored = X / S
    ambient_noise = np.fft.irfft(X_colored, n=total_samples)
    if np.max(np.abs(ambient_noise)) > 0:
        ambient_noise /= np.max(np.abs(ambient_noise))
    audio += ambient_noise * ambient_noise_level
    
    # Apply global low-pass filter to simulate body tissue and stethoscope attenuation
    # Reduced to 2nd order to allow natural high-frequency leakage
    sos_global = signal.butter(2, global_lp_cutoff / (0.5 * fs), btype='low', output='sos')
    audio = signal.sosfiltfilt(sos_global, audio)
    
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
        hr_mean = np.random.uniform(55, 120)
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.30, 0.50)
        
        # VSD Severity (0 = restrictive/small, 1 = unrestrictive/large)
        severity = np.random.uniform(0, 1)
        
        # VSD Subtype (~80% membranous, ~20% muscular)
        is_muscular = np.random.uniform(0, 1) > 0.8
        
        if is_muscular:
            env_type = 'decrescendo'
            murmur_dur_ratio = np.random.uniform(0.5, 0.85) # Ends before S2
        else:
            env_type = 'plateau'
            murmur_dur_ratio = 1.0 # Holosystolic
            
        # Map severity to frequency and amplitude based on the blueprint:
        if severity < 0.5:
            # Small VSD: high pressure gradient -> loud, high-pitched
            murmur_low = np.random.uniform(60, 120)
            murmur_high = np.random.uniform(400, 800)
            murmur_amp = np.random.uniform(0.6, 1.8)
        else:
            # Large VSD: low pressure gradient -> soft, low-pitched
            murmur_low = np.random.uniform(30, 80)
            murmur_high = np.random.uniform(200, 400)
            murmur_amp = np.random.uniform(0.1, 0.8)
            
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.7, 1.2)
        # P2 accentuation in severe VSD due to pulmonary hypertension
        s2_amp = np.random.uniform(0.7, 1.1) + 0.5 * severity
        
        # Diverse heart sound properties
        s1_f0 = np.random.uniform(60, 100)
        s1_dur = np.random.uniform(0.07, 0.12)
        s2_f0 = np.random.uniform(90, 140)
        s2_dur = np.random.uniform(0.05, 0.09)
        
        # Diverse noise levels, colors, and envelope shapes to improve spread ratios
        noise_level = np.random.uniform(0.02, 0.35)
        ambient_noise_level = np.random.uniform(0.01, 0.15)
        ambient_noise_color = np.random.uniform(0.8, 2.2) 
        murmur_noise_color = np.random.uniform(0.8, 1.6)
        alpha = np.random.uniform(0.2, 0.6) # Softer envelope edges
        global_lp_cutoff = np.random.uniform(800, 2500) # Widened to allow more high-frequency leakage
        
        # Generate the high-resolution signal
        audio = generate_vsd_signal(
            duration_sec=duration_sec,
            fs=fs_internal,
            hr_mean=hr_mean,
            hrv_std=hrv_std,
            systole_ratio=systole_ratio,
            murmur_amp=murmur_amp,
            murmur_band=murmur_band,
            murmur_dur_ratio=murmur_dur_ratio,
            env_type=env_type,
            s1_amp=s1_amp,
            s2_amp=s2_amp,
            s1_f0=s1_f0,
            s1_dur=s1_dur,
            s2_f0=s2_f0,
            s2_dur=s2_dur,
            noise_level=noise_level,
            ambient_noise_level=ambient_noise_level,
            ambient_noise_color=ambient_noise_color,
            murmur_noise_color=murmur_noise_color,
            alpha=alpha,
            global_lp_cutoff=global_lp_cutoff
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