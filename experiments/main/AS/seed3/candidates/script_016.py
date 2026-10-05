import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f spectrum) using the FFT method.
    This provides a more realistic, naturally rolling-off frequency 
    profile for both murmurs and ambient body sounds compared to white noise.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    # Create 1/f amplitude multiplier (avoiding division by zero at DC)
    S = np.sqrt(np.arange(1, len(X) + 1))
    X_pink = X / S
    return np.fft.irfft(X_pink, n=n_samples)

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
    
    # Bandpass filter for murmur frequencies (150-500 Hz for AS)
    b, a = signal.butter(2, [150, 500], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # Normalize filtered noise so the envelope dictates the exact peak amplitude
    max_val = np.max(np.abs(filtered_noise))
    if max_val > 0:
        filtered_noise /= max_val
        
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates ambient noise, sensor hiss, and low-frequency baseline wander.
    The baseline wander drastically reduces spurious zero-crossings (ZCR),
    while the broadband hiss populates upper MFCC bands.
    """
    n_samples = int(fs * duration)
    
    # 1. Ambient body/room noise (Pink noise, low-passed)
    pink = generate_pink_noise(n_samples)
    b, a = signal.butter(2, 250, btype='lowpass', fs=fs)
    ambient = signal.filtfilt(b, a, pink)
    rms = np.sqrt(np.mean(ambient**2))
    if rms > 0:
        ambient /= rms
        
    # 2. Baseline wander (Very low frequency physiological movement)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wander = 0.5 * np.sin(2 * np.pi * 0.5 * t) + 0.3 * np.sin(2 * np.pi * 1.2 * t + np.random.uniform(0, 2*np.pi))
    
    # 3. Sensor hiss (Broadband noise floor to prevent log(0) in MFCCs)
    hiss = np.random.randn(n_samples) * 0.05
    
    # Combine components
    return (ambient + hiss) * noise_level + wander * (noise_level * 5)

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
        noise_level = np.random.uniform(0.01, 0.05)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity
        
        # S2 parameters based on severity (A2 is delayed and soft/absent in severe AS)
        s2_intensity = 1.0 - 0.9 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # Helper to safely add signals to the buffer
        def add_to_buffer(signal_array, start_time):
            if start_time < 0:
                crop_samples = int(-start_time * FS_INTERNAL)
                if crop_samples >= len(signal_array):
                    return
                signal_array = signal_array[crop_samples:]
                start_idx = 0
            else:
                start_idx = int(start_time * FS_INTERNAL)
                
            end_idx = start_idx + len(signal_array)
            
            if start_idx >= total_samples:
                return
                
            if end_idx > total_samples:
                signal_array = signal_array[:total_samples - start_idx]
                end_idx = total_samples
                
            audio_buffer[start_idx:end_idx] += signal_array

        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            t_murmur_start = current_time + 0.09
            t_s2 = current_time + systole_duration
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=40, duration=0.06, fs=FS_INTERNAL)
            
            murmur_duration = systole_duration - 0.11
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            s2 = generate_gabor_wavelet(freq=50, duration=0.06, fs=FS_INTERNAL)
            s2 *= s2_intensity
            
            add_to_buffer(s1, t_s1)
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
            # Add S4 for severe AS
            if severity > 0.7:
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= 0.3 * (severity - 0.7) / 0.3
                t_s4 = t_s1 - 0.08
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise (ambient, hiss, and baseline wander)
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