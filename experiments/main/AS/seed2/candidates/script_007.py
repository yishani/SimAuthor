import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f) using the FFT method.
    This provides a more realistic spectral roll-off for biological sounds
    compared to white noise, reducing unnatural high-frequency zero crossings.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n_samples)
    f[0] = f[1]  # Avoid division by zero
    X /= np.sqrt(f)
    pink = np.fft.irfft(X, n=n_samples)
    return pink / (np.std(pink) + 1e-9)

def generate_gabor_wavelet(freq, duration, fs, phase=0.0):
    """
    Generates a Gabor wavelet to simulate heart sounds (S1, S2, S4).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sigma = duration / 6.0
    mu = duration / 2.0
    envelope = np.exp(-0.5 * ((t - mu) / sigma)**2)
    wave = np.sin(2 * np.pi * freq * t + phase) * envelope
    return wave

def generate_as_murmur(duration, peak_fraction, fs):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using pink noise.
    """
    n_samples = int(fs * duration)
    if n_samples == 0:
        return np.array([])
        
    t_norm = np.linspace(0, 1, n_samples, endpoint=False)
    envelope = np.zeros_like(t_norm)
    
    peak_idx = int(peak_fraction * n_samples)
    
    if peak_idx > 0:
        envelope[:peak_idx] = np.sin(0.5 * np.pi * t_norm[:peak_idx] / peak_fraction)**2
        
    if peak_idx < n_samples:
        envelope[peak_idx:] = np.cos(0.5 * np.pi * (t_norm[peak_idx:] - peak_fraction) / (1.0 - peak_fraction))**2
        
    # Use pink noise for a more realistic turbulent fluid sound
    noise = generate_pink_noise(n_samples)
    
    # Gentler bandpass filter for murmur frequencies (150-500 Hz for AS)
    b, a = signal.butter(2, [150, 500], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # Normalize so the peak amplitude is roughly 1.0 before envelope
    filtered_noise = filtered_noise / (np.std(filtered_noise) + 1e-9) * 0.3
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise using pink noise.
    """
    n_samples = int(fs * duration)
    
    # Base ambient noise
    noise = generate_pink_noise(n_samples)
    b, a = signal.butter(2, 100, btype='lowpass', fs=fs)
    ambient = signal.filtfilt(b, a, noise)
    
    # Stronger low-frequency rumble (body sounds, sensor coupling)
    rumble = generate_pink_noise(n_samples)
    b_r, a_r = signal.butter(2, 30, btype='lowpass', fs=fs)
    rumble = signal.filtfilt(b_r, a_r, rumble)
    
    return (ambient + 2.0 * rumble) * noise_level

def simulate_aortic_stenosis_pcg():
    # Configuration
    FS_INTERNAL = 44100
    FS_OUTPUT = 16000
    DURATION = 10.0
    NUM_SAMPLES = 100
    OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/aortic_stenosis_audio/runs/signal/seed3/generated/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    for i in range(NUM_SAMPLES):
        heart_rate = np.random.uniform(60, 100)
        rr_interval = 60.0 / heart_rate
        
        severity = np.random.uniform(0.1, 1.0)
        resp_rate = np.random.uniform(12, 20) / 60.0
        noise_level = np.random.uniform(0.02, 0.08) # Slightly increased for better noise floor
        
        # Adjusted systole duration to better fit physiological norms
        systole_duration = 0.35 * np.sqrt(rr_interval)
        
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity # Matches blueprint ratios
        
        s2_intensity = 1.0 - 0.9 * severity # A2 becomes very soft in severe AS
        
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        current_time = np.random.uniform(0, rr_interval)
        
        while current_time < DURATION:
            s1_duration = 0.1
            t_s1 = current_time
            # Murmur starts 30ms after S1 ends
            t_murmur_start = current_time + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            
            s1 = generate_gabor_wavelet(freq=50, duration=s1_duration, fs=FS_INTERNAL)
            
            def add_to_buffer(signal_array, start_time):
                start_idx = int(start_time * FS_INTERNAL)
                end_idx = start_idx + len(signal_array)
                
                if start_idx >= total_samples or start_idx < 0:
                    return
                    
                if end_idx > total_samples:
                    signal_array = signal_array[:total_samples - start_idx]
                    end_idx = total_samples
                    
                audio_buffer[start_idx:end_idx] += signal_array

            add_to_buffer(s1, t_s1)
            
            # Add S4 for severe AS (occurs 80ms before S1)
            if severity > 0.7:
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= 0.3
                t_s4 = t_s1 - 0.08
                if t_s4 >= 0:
                    add_to_buffer(s4, t_s4)
            
            # Murmur ends 20ms before S2 begins
            murmur_duration = (t_s2 - 0.02) - t_murmur_start
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            s2 = generate_gabor_wavelet(freq=70, duration=0.08, fs=FS_INTERNAL)
            s2 *= s2_intensity
            add_to_buffer(s2, t_s2)
            
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level)
        
        audio_resampled = signal.resample_poly(audio_buffer, FS_OUTPUT, FS_INTERNAL)
        
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_resampled /= (max_val + 1e-9)
            
        audio_pcm = (audio_resampled * 32767.0).astype(np.int16)
        filename = os.path.join(OUTPUT_DIR, f"as_sample_{i:02d}.wav")
        wavfile.write(filename, FS_OUTPUT, audio_pcm)

if __name__ == "__main__":
    simulate_aortic_stenosis_pcg()