import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

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

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f) using FFT to simulate turbulent fluid dynamics
    and natural body sounds, avoiding the harshness of white noise.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    f = np.arange(len(X))
    f[0] = 1.0  # Avoid division by zero at DC
    X = X / np.sqrt(f)
    X[0] = 0.0  # Remove DC offset
    pink = np.fft.irfft(X, n=n_samples)
    return pink

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
    # CRITICAL FIX: Use SOS (Second-Order Sections) to prevent severe numerical 
    # instability that occurs with b, a forms at low normalized frequencies.
    sos = signal.butter(4, [150, 400], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, noise)
    
    # Normalize to peak 1.0 before applying envelope for consistent intensity scaling
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
    
    # Low-pass filter to simulate body/sensor noise (SOS for stability)
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
    OUTPUT_DIR = "[PROJECT_ROOT]/artifacts/aortic_stenosis_audio/runs/signal/seed3/generated/"
    
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
        
        # Increased noise level to provide a realistic continuous low-frequency rumble
        noise_level = np.random.uniform(0.05, 0.15)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        # Murmur intensity can exceed S1 in severe cases (up to 2.0x S1)
        murmur_intensity = 0.5 + 1.5 * severity
        
        # S2 parameters based on severity (A2 is delayed and soft/absent in severe AS)
        s2_intensity = 1.0 - 0.7 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        def add_to_buffer(signal_array, start_time):
            if start_time < 0:
                start_idx = int(start_time * FS_INTERNAL)
                if -start_idx >= len(signal_array):
                    return
                signal_array = signal_array[-start_idx:]
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

        while current_time < DURATION:
            # Timing for this specific cycle
            s1_duration = 0.06
            s2_duration = 0.06
            
            t_s1 = current_time
            # Murmur starts ~30 ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            
            # Murmur ends ~20 ms before S2 begins
            murmur_duration = t_s2 - 0.02 - t_murmur_start
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=40, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            if murmur_duration > 0.05:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            s2 = generate_gabor_wavelet(freq=50, duration=s2_duration, fs=FS_INTERNAL)
            s2 *= s2_intensity
            add_to_buffer(s2, t_s2)
            
            # Add S4 for severe AS (~80 ms before S1)
            if severity > 0.7:
                t_s4 = t_s1 - 0.08
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= 0.3
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