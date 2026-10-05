import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f spectrum) using FFT.
    This provides a more realistic, low-frequency dominant source 
    for both turbulent blood flow and ambient body noise.
    """
    if n_samples == 0:
        return np.array([])
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    # 1/f amplitude scaling
    scaling = np.sqrt(np.arange(1, len(X) + 1))
    X_pink = X / scaling
    pink = np.fft.irfft(X_pink, n=n_samples)
    # Normalize variance
    std = np.std(pink)
    if std > 0:
        pink = pink / std
    return pink

def generate_gabor_wavelet(freq, duration, fs, phase=0.0):
    """
    Generates a Gabor wavelet to simulate heart sounds (S1, S2).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Gaussian envelope centered in the duration
    sigma = duration / 6.0
    mu = duration / 2.0
    envelope = np.exp(-0.5 * ((t - mu) / sigma)**2)
    wave = np.sin(2 * np.pi * freq * t + phase) * envelope
    return wave

def generate_as_murmur(duration, peak_fraction, fs):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis, using pink noise for realistic turbulence.
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
    
    # Bandpass filter for murmur frequencies (typically 150-400 Hz for AS)
    # Using 2nd order for a lower Q-factor, making it less tonal and more harsh
    b, a = signal.butter(2, [150, 400], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise using pink noise.
    """
    n_samples = int(fs * duration)
    noise = generate_pink_noise(n_samples)
    # Low-pass filter to simulate body/sensor noise
    b, a = signal.butter(2, 100, btype='lowpass', fs=fs)
    ambient = signal.filtfilt(b, a, noise)
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
        # Affects murmur intensity, murmur peak timing, and S2 intensity
        severity = np.random.uniform(0.1, 1.0)
        
        # Respiratory rate for amplitude modulation
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level (increased slightly to provide a better low-frequency floor)
        noise_level = np.random.uniform(0.02, 0.08)
        
        # 2. Derived cardiac timing parameters
        # AS prolongs ventricular ejection time, so systole duration increases with severity
        systole_duration = (0.30 + 0.10 * severity) * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        # In severe AS, the murmur peaks later in systole
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.0 * severity
        
        # S2 parameters based on severity
        # In severe AS, A2 is delayed and soft (or absent)
        s2_intensity = 1.0 - 0.9 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        s1_duration = 0.08
        s2_duration = 0.08
        
        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            # Murmur starts ~30ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            
            # Murmur ends ~20ms before S2 begins
            murmur_duration = t_s2 - 0.02 - t_murmur_start
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=50, duration=s1_duration, fs=FS_INTERNAL)
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            s2 = generate_gabor_wavelet(freq=70, duration=s2_duration, fs=FS_INTERNAL)
            s2 *= s2_intensity
            
            # Add to buffer
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
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level)
        
        # 7. Resample to 16000 Hz
        # Using resample_poly for high-quality anti-aliased resampling
        audio_resampled = signal.resample_poly(audio_buffer, FS_OUTPUT, FS_INTERNAL)
        
        # 8. Peak Normalization
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_resampled /= (max_val + 1e-9)
            
        # 9. Save to WAV
        # Convert to 16-bit PCM
        audio_pcm = (audio_resampled * 32767.0).astype(np.int16)
        filename = os.path.join(OUTPUT_DIR, f"as_sample_{i:02d}.wav")
        wavfile.write(filename, FS_OUTPUT, audio_pcm)

if __name__ == "__main__":
    simulate_aortic_stenosis_pcg()