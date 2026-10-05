import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f spectrum) using FFT.
    This provides a more realistic spectral tilt for turbulent murmurs 
    and body acoustics compared to white noise, reducing unrealistic high-frequency ZCR.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # Prevent division by zero
    X /= np.sqrt(freqs)
    X[0] = 0.0  # Remove DC component
    return np.fft.irfft(X, n=n_samples)

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
    characteristic of Aortic Stenosis using filtered pink noise.
    """
    n_samples = int(fs * duration)
    if n_samples == 0:
        return np.array([])
        
    t_norm = np.linspace(0, 1, n_samples, endpoint=False)
    envelope = np.zeros_like(t_norm)
    
    # Create the diamond-shaped envelope using two half-Hann windows
    peak_idx = int(peak_fraction * n_samples)
    
    # Crescendo phase
    if peak_idx > 0:
        envelope[:peak_idx] = np.sin(0.5 * np.pi * t_norm[:peak_idx] / peak_fraction)**2
        
    # Decrescendo phase
    if peak_idx < n_samples:
        envelope[peak_idx:] = np.cos(0.5 * np.pi * (t_norm[peak_idx:] - peak_fraction) / (1.0 - peak_fraction))**2
        
    # Generate turbulent pink noise
    noise = generate_pink_noise(n_samples)
    
    # Bandpass filter for murmur frequencies (150-400 Hz for AS)
    # Using SOS (Second-Order Sections) to prevent severe numerical instability at high fs
    sos = signal.butter(2, [150, 400], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, noise)
    
    # Normalize filtered noise to maintain consistent amplitude before envelope
    max_val = np.max(np.abs(filtered_noise))
    if max_val > 0:
        filtered_noise /= max_val
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise using pink noise.
    """
    n_samples = int(fs * duration)
    noise = generate_pink_noise(n_samples)
    
    # Low-pass filter to simulate body/sensor noise
    # Using SOS to prevent high-frequency artifacts from unstable b,a coefficients
    sos = signal.butter(2, 100, btype='lowpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos, noise)
    
    max_val = np.max(np.abs(ambient))
    if max_val > 0:
        ambient /= max_val
        
    return ambient * noise_level

def simulate_aortic_stenosis_pcg():
    # Configuration
    FS_INTERNAL = 44100
    FS_OUTPUT = 16000
    DURATION = 10.0
    NUM_SAMPLES = 100
    OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/aortic_stenosis_audio/runs/signal/seed4/generated/"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(42)
    
    for i in range(NUM_SAMPLES):
        # 1. Sample physiological parameters for diversity
        heart_rate = np.random.uniform(60, 100) # bpm
        rr_interval = 60.0 / heart_rate
        
        # Severity from 0.1 (mild) to 1.0 (severe)
        severity = np.random.uniform(0.1, 1.0)
        
        # Respiratory rate for amplitude modulation
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level
        noise_level = np.random.uniform(0.02, 0.08)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity
        
        # S2 parameters based on severity (A2 is delayed and soft/absent in severe AS)
        s2_intensity = 1.0 - 1.0 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        def add_to_buffer(signal_array, start_time):
            start_idx = int(start_time * FS_INTERNAL)
            
            if start_idx < 0:
                crop_samples = -start_idx
                if crop_samples >= len(signal_array):
                    return
                signal_array = signal_array[crop_samples:]
                start_idx = 0
                
            end_idx = start_idx + len(signal_array)
            
            if start_idx >= total_samples:
                return
                
            if end_idx > total_samples:
                signal_array = signal_array[:total_samples - start_idx]
                end_idx = total_samples
                
            audio_buffer[start_idx:end_idx] += signal_array

        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            t_murmur_start = current_time + 0.09 # 30ms after 60ms S1 ends
            t_s2 = current_time + systole_duration
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=40, duration=0.06, fs=FS_INTERNAL)
            
            # Murmur ends 20ms before S2
            murmur_duration = (t_s2 - 0.02) - t_murmur_start
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            s2 = generate_gabor_wavelet(freq=50, duration=0.06, fs=FS_INTERNAL)
            s2 *= max(0.0, s2_intensity)
            
            add_to_buffer(s1, t_s1)
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
            # S4 generation for severe AS (LVH compensation)
            if severity > 0.7:
                s4_intensity = 0.3 * (severity - 0.7) / 0.3
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                t_s4 = t_s1 - 0.08
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level)
        
        # 7. Resample to 16000 Hz
        audio_resampled = signal.resample_poly(audio_buffer, FS_OUTPUT, FS_INTERNAL)
        
        # 8. Peak Normalization
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_resampled /= (max_val + 1e-9)
            
        # 9. Save to WAV
        audio_pcm = (audio_resampled * 32767.0).astype(np.int16)
        filename = os.path.join(OUTPUT_DIR, f"as_sample_{i:02d}.wav")
        wavfile.write(filename, FS_OUTPUT, audio_pcm)

if __name__ == "__main__":
    simulate_aortic_stenosis_pcg()