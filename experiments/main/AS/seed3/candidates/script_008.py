import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_gabor_wavelet(freq, duration, fs, phase=0.0):
    """
    Generates a Gabor wavelet to simulate heart sounds (S1, S2, S4).
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    # Gaussian envelope centered in the duration
    sigma = duration / 6.0
    mu = duration / 2.0
    envelope = np.exp(-0.5 * ((t - mu) / sigma)**2)
    wave = np.sin(2 * np.pi * freq * t + phase) * envelope
    return wave

def generate_as_murmur(duration, peak_fraction, fs, f_low=150, f_high=400):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using Pink Noise.
    """
    n_samples = int(fs * duration)
    if n_samples <= 0:
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
        
    # Generate Pink Noise (1/f power spectrum) for realistic fluid turbulence
    X = np.fft.rfft(np.random.randn(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1e-9 # avoid divide by zero
    X /= np.sqrt(f)
    noise = np.fft.irfft(X, n=n_samples)
    
    # Normalize noise before filtering
    noise /= np.std(noise) + 1e-9
    
    # Bandpass filter for murmur frequencies (low-Q to avoid tonal whistling)
    b, a = signal.butter(2, [f_low, f_high], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates realistic low-frequency ambient and sensor noise using Brown and Pink noise.
    """
    n_samples = int(fs * duration)
    
    # Brown noise (1/f^2 power) for low-frequency stethoscope/body rumble
    X_brown = np.fft.rfft(np.random.randn(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1e-9
    X_brown /= f
    brown_noise = np.fft.irfft(X_brown, n=n_samples)
    brown_noise /= np.std(brown_noise) + 1e-9
    
    # Pink noise (1/f power) for general room tone
    X_pink = np.fft.rfft(np.random.randn(n_samples))
    X_pink /= np.sqrt(f)
    pink_noise = np.fft.irfft(X_pink, n=n_samples)
    pink_noise /= np.std(pink_noise) + 1e-9
    
    # Combine and scale
    ambient = (brown_noise * 0.7 + pink_noise * 0.3) * noise_level
    
    # Low-pass filter to simulate stethoscope bell/diaphragm attenuation of high frequencies
    b, a = signal.butter(2, 150, btype='lowpass', fs=fs)
    ambient = signal.filtfilt(b, a, ambient)
    
    return ambient

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
        
        # Noise level (increased to provide realistic low-frequency baseline)
        noise_level = np.random.uniform(0.02, 0.08)
        
        # 2. Derived cardiac timing parameters
        # Systole duration approximates Bazett's formula
        systole_duration = 0.35 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.4 + 1.2 * severity
        
        # Randomize murmur frequency band slightly per patient
        f_low = np.random.uniform(130, 170)
        f_high = np.random.uniform(350, 450)
        
        # S2 parameters based on severity (A2 is delayed and soft/absent in severe AS)
        s2_intensity = 1.0 - 0.9 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        def add_to_buffer(signal_array, start_time):
            start_idx = int(start_time * FS_INTERNAL)
            end_idx = start_idx + len(signal_array)
            
            if start_idx >= total_samples or end_idx <= 0:
                return
                
            if start_idx < 0:
                signal_array = signal_array[-start_idx:]
                start_idx = 0
                
            if end_idx > total_samples:
                signal_array = signal_array[:total_samples - start_idx]
                end_idx = total_samples
                
            audio_buffer[start_idx:end_idx] += signal_array

        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            s1_duration = 0.08
            
            # Murmur starts ~30ms after S1 ends
            t_murmur_start = current_time + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=50, duration=s1_duration, fs=FS_INTERNAL)
            
            # Murmur ends ~20ms before S2 begins
            murmur_duration = (t_s2 - 0.02) - t_murmur_start
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL, f_low, f_high)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            s2 = generate_gabor_wavelet(freq=70, duration=0.06, fs=FS_INTERNAL)
            s2 *= s2_intensity
            
            add_to_buffer(s1, t_s1)
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
            # Add S4 for severe AS (occurs ~80ms before S1)
            if severity > 0.7:
                s4_intensity = 0.3 * (severity - 0.7) / 0.3
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                t_s4 = t_s1 - 0.08
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation to heart sounds
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise (not modulated by respiration in the same way)
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