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
    sine wave with a smooth envelope and added texture.
    """
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)
    
    # Smooth envelope (beta-like distribution)
    env = (t / duration) ** 2 * (1 - t / duration) ** 2
    if np.max(env) > 0:
        env /= np.max(env)
        
    # Downward frequency modulation for a natural "thud" sound
    f_t = f0 - 30 * (t / duration)
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    sound = env * np.sin(phase)
    
    # Add low-pass filtered noise to make it sound less like a pure synthetic tone
    noise = np.random.randn(len(t))
    b, a = signal.butter(2, 150 / (0.5 * fs), btype='low')
    noise = signal.filtfilt(b, a, noise)
    if np.max(np.abs(noise)) > 0:
        noise /= np.max(np.abs(noise))
    sound += 0.15 * env * noise
    
    # Peak normalize
    max_val = np.max(np.abs(sound))
    if max_val > 0:
        sound /= max_val
        
    return sound

def generate_murmur(duration, fs, freq_band, is_muscular=False):
    """
    Generates a harsh, high-pitched murmur typical of VSD.
    Supports both holosystolic (membranous) and decrescendo (muscular) envelopes.
    """
    num_samples = int(duration * fs)
    if num_samples == 0:
        return np.array([])
        
    noise = generate_pink_noise(num_samples)
    nyq = 0.5 * fs
    
    # 2nd order HPF for a gentler roll-off at the bottom. This allows more low-frequency 
    # bleed from the pink noise, filling the spectral gap between heart sounds and the murmur,
    # which significantly improves MFCC realism and lowers artificially high ZCR.
    b_hp, a_hp = signal.butter(2, freq_band[0]/nyq, btype='high')
    murmur = signal.filtfilt(b_hp, a_hp, noise)
    
    # 4th order LPF for the upper roll-off
    b_lp, a_lp = signal.butter(4, freq_band[1]/nyq, btype='low')
    murmur = signal.filtfilt(b_lp, a_lp, murmur)
    
    # Add a resonance peak to emphasize the "harsh" quality of the VSD murmur
    res_freq = np.random.uniform(300, 450)
    b_res, a_res = signal.iirpeak(res_freq / nyq, np.random.uniform(1.0, 2.0))
    murmur = murmur + 0.5 * signal.filtfilt(b_res, a_res, murmur)
    
    # Peak normalize before applying envelope
    max_val = np.max(np.abs(murmur))
    if max_val > 0:
        murmur /= max_val
        
    if is_muscular:
        # Muscular VSD: Decrescendo envelope as the contracting muscle closes the defect
        attack_samples = int(0.015 * fs)
        env = np.zeros(num_samples)
        if attack_samples > 0 and attack_samples < num_samples:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
            decay_length = num_samples - attack_samples
            env[attack_samples:] = np.exp(-3 * np.linspace(0, 1, decay_length))
        else:
            env = np.exp(-3 * np.linspace(0, 1, num_samples))
    else:
        # Membranous VSD: Holosystolic plateau
        # A Tukey window provides a flat top with smooth fade-in/fade-out
        env = signal.windows.tukey(num_samples, alpha=0.1)
    
    return murmur * env

def generate_vsd_signal(duration_sec, fs, hr_mean, hrv_std, systole_ratio, 
                        murmur_amp, murmur_band, s1_amp, s2_amp, noise_level, is_muscular):
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
        
        s1_dur = np.random.uniform(0.06, 0.10)
        s2_dur = np.random.uniform(0.05, 0.08)
        
        # Generate components (lowered f0 to provide more realistic low-frequency energy)
        s1 = generate_heart_sound(f0=np.random.uniform(50, 80), duration=s1_dur, fs=fs) * s1_amp
        s2 = generate_heart_sound(f0=np.random.uniform(80, 110), duration=s2_dur, fs=fs) * s2_amp
        
        if is_muscular:
            # Muscular VSD murmur is shorter, ending before S2
            murmur_dur = max(0.1, systole_dur - np.random.uniform(0.05, 0.12))
            murmur = generate_murmur(duration=murmur_dur, fs=fs, freq_band=murmur_band, is_muscular=True) * murmur_amp
        else:
            # Membranous VSD murmur is holosystolic (lasts exactly from S1 to S2)
            murmur = generate_murmur(duration=systole_dur, fs=fs, freq_band=murmur_band, is_muscular=False) * murmur_amp
        
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
        
    # Add low-frequency body/sensor noise (cutoff increased to 200Hz for more realistic body)
    bg_noise = generate_pink_noise(total_samples)
    b, a = signal.butter(2, 200 / (0.5 * fs), btype='low')
    bg_noise = signal.filtfilt(b, a, bg_noise)
    if np.max(np.abs(bg_noise)) > 0:
        bg_noise /= np.max(np.abs(bg_noise))
    audio += bg_noise * noise_level
    
    # Add ambient pink noise to prevent digital silence in high frequencies
    ambient = generate_pink_noise(total_samples)
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
    audio += ambient * np.random.uniform(0.01, 0.03)
    
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
    
    for i in range(num_samples):
        # Sample physiological parameters to ensure diversity
        hr_mean = np.random.uniform(65, 105)
        hrv_std = np.random.uniform(0.01, 0.05)
        systole_ratio = np.random.uniform(0.3, 0.4)
        
        # VSD severity proxy (0 = small/restrictive, 1 = large/unrestrictive)
        severity = np.random.uniform(0, 1)
        
        # Subtype: ~80% membranous, ~20% muscular
        is_muscular = np.random.rand() < 0.2
        
        murmur_amp = np.random.uniform(0.6, 1.8) * (1.2 - 0.4 * severity)
        
        # Lowered HPF cutoff to allow more low-frequency energy, reducing ZCR
        murmur_low = np.random.uniform(100, 180)
        murmur_high = np.random.uniform(600, 900)
        murmur_band = (murmur_low, murmur_high)
        
        s1_amp = np.random.uniform(0.8, 1.2)
        # In severe VSD with pulmonary hypertension, P2 (part of S2) can be accentuated
        s2_amp = np.random.uniform(0.8, 1.2) + 0.4 * severity
        
        noise_level = np.random.uniform(0.05, 0.15)
        
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

if __name__ == "__main__":
    main()