import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """Generates pink noise using FFT for 1/f spectral shaping."""
    if n_samples == 0:
        return np.array([])
    X_white = np.fft.rfft(np.random.randn(n_samples))
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = freqs[1]  # Avoid divide by zero
    X_pink = X_white / np.sqrt(freqs)
    pink = np.fft.irfft(X_pink, n=n_samples)
    return pink / (np.std(pink) + 1e-9)

def generate_gabor_wavelet(freq, duration, fs, phase=0.0):
    """Generates a Gabor wavelet to simulate heart sounds (S1, S2, S4, EC)."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sigma = duration / 6.0
    mu = duration / 2.0
    envelope = np.exp(-0.5 * ((t - mu) / sigma)**2)
    wave = np.sin(2 * np.pi * freq * t + phase) * envelope
    return wave

def generate_as_murmur(duration, peak_fraction, severity, fs):
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
        
    # Generate turbulent pink noise (avoids tonal whistling and reduces unnatural ZCR)
    noise = generate_pink_noise(n_samples)
    
    # Bandpass filter for murmur frequencies
    # Upper frequency extends higher in severe AS due to high-velocity jets
    upper_freq = 400 + 400 * severity 
    b, a = signal.butter(2, [150, upper_freq], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates realistic background noise combining low-frequency body noise 
    and broadband ambient room/sensor noise.
    """
    n_samples = int(fs * duration)
    
    # Ambient noise (broadband pink noise)
    ambient = generate_pink_noise(n_samples)
    
    # Body/stethoscope noise (low-passed pink noise)
    body = generate_pink_noise(n_samples)
    b, a = signal.butter(2, 100, btype='lowpass', fs=fs)
    body = signal.filtfilt(b, a, body)
    
    # Combine them: predominantly low-frequency body noise
    noise = (body * 0.85) + (ambient * 0.15)
    
    return noise * noise_level

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
        
        # Noise level
        noise_level = np.random.uniform(0.02, 0.08)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity # Louder in severe AS
        
        # S2 parameters based on severity
        s2_intensity = 1.0 - 0.8 * severity # Soft or absent in severe AS
        
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
        current_time = np.random.uniform(0.1, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            s1_duration = 0.08
            
            # Murmur onset: ~30 ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            
            # S2 timing
            t_s2 = current_time + systole_duration
            
            # Murmur offset: ~20 ms before S2 begins
            t_murmur_end = t_s2 - 0.02
            murmur_duration = t_murmur_end - t_murmur_start
            
            # Generate S1
            s1_freq = np.random.uniform(40, 60)
            s1 = generate_gabor_wavelet(freq=s1_freq, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            # Generate Ejection Click (EC) for mild/moderate AS
            if severity < 0.5 and np.random.rand() > 0.5:
                ec = generate_gabor_wavelet(freq=250, duration=0.02, fs=FS_INTERNAL)
                ec *= 0.4
                t_ec = t_s1 + 0.05
                add_to_buffer(ec, t_ec)
            
            # Generate Murmur
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, severity, FS_INTERNAL)
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            # Generate S2
            s2_freq = np.random.uniform(60, 80)
            s2 = generate_gabor_wavelet(freq=s2_freq, duration=0.08, fs=FS_INTERNAL)
            s2 *= s2_intensity
            add_to_buffer(s2, t_s2)
            
            # Generate S4 for severe AS
            if severity > 0.7:
                s4 = generate_gabor_wavelet(freq=30, duration=0.06, fs=FS_INTERNAL)
                s4 *= 0.3
                t_s4 = current_time - 0.08
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation
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