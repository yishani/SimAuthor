import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f spectrum) to simulate turbulent blood flow
    and realistic body/sensor noise.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    
    # Create 1/f amplitude multiplier
    f = np.fft.rfftfreq(n_samples)
    f[0] = f[1] if n_samples > 1 else 1.0  # Avoid division by zero at DC
    
    # Apply 1/sqrt(f) to amplitude (which gives 1/f power spectrum)
    X /= np.sqrt(f)
    
    pink = np.fft.irfft(X, n=n_samples)
    
    # Normalize to zero mean and unit variance
    pink -= np.mean(pink)
    pink /= (np.std(pink) + 1e-9)
    
    return pink

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
        
    # Generate turbulent pink noise (more realistic for fluid dynamics than white noise)
    noise = generate_pink_noise(n_samples)
    
    # Bandpass filter for murmur frequencies (typically 150-400 Hz for AS)
    # Using 2nd order (lower Q) and SOS to avoid tonal artifacts and numerical instability
    sos = signal.butter(2, [150, 400], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, noise)
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise.
    """
    n_samples = int(fs * duration)
    # Use pink noise for more natural ambient/body noise
    noise = generate_pink_noise(n_samples)
    
    # Bandpass filter to simulate body/sensor noise and remove extreme baseline wander
    sos = signal.butter(2, [20, 150], btype='bandpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos, noise)
    
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
        noise_level = np.random.uniform(0.01, 0.05)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity  # Mild: ~0.65, Severe: 2.0
        
        # S2 parameters based on severity (A2 is delayed and soft/absent in severe AS)
        s2_intensity = max(0.0, 1.0 - 0.9 * severity)  # Mild: ~0.9, Severe: 0.1
        
        # S4 parameters (prominent in severe AS due to LVH)
        s4_intensity = 0.0
        if severity > 0.7:
            s4_intensity = (severity - 0.7) / 0.3 * 0.3
            
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # Helper to add signals safely
        def add_to_buffer(signal_array, start_time):
            start_idx = int(start_time * FS_INTERNAL)
            if start_idx < 0:
                crop = -start_idx
                if crop >= len(signal_array):
                    return
                signal_array = signal_array[crop:]
                start_idx = 0
                
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
            s1_duration = 0.06
            
            # Murmur starts ~30 ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            
            t_s2 = current_time + systole_duration
            
            # Murmur ends ~20 ms before S2 begins
            t_murmur_end = t_s2 - 0.02
            murmur_duration = t_murmur_end - t_murmur_start
            
            # Generate components
            if s4_intensity > 0:
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                add_to_buffer(s4 * s4_intensity, t_s1 - 0.08)
                
            s1 = generate_gabor_wavelet(freq=40, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                add_to_buffer(murmur * murmur_intensity, t_murmur_start)
                
            s2 = generate_gabor_wavelet(freq=40, duration=0.06, fs=FS_INTERNAL)
            add_to_buffer(s2 * s2_intensity, t_s2)
            
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