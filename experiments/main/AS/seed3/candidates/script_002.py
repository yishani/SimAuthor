import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f spectrum) using FFT.
    Pink noise provides a more realistic, less harsh broadband source 
    than white noise, naturally reducing unrealistic high-frequency zero crossings.
    """
    x = np.random.randn(n_samples)
    X = np.fft.rfft(x)
    freqs = np.arange(1, len(X) + 1)
    X = X / np.sqrt(freqs)
    pink = np.fft.irfft(X, n_samples)
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
    characteristic of Aortic Stenosis using a pink noise source.
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
        
    # Generate turbulent pink noise instead of white noise to prevent tonal ringing
    # and to lower the artificially high Zero Crossing Rate (ZCR).
    noise = generate_pink_noise(n_samples)
    
    # Broad bandpass filter for murmur frequencies (150-500 Hz)
    # Using a 2nd order filter (lower Q) to keep it harsh and broadband, filling out MFCCs
    b, a = signal.butter(2, [150, 500], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # Normalize to maintain consistent intensity
    max_val = np.max(np.abs(filtered_noise))
    if max_val > 0:
        filtered_noise /= max_val
        
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates broadband pink ambient noise and low-frequency rumble.
    """
    n_samples = int(fs * duration)
    
    # Pink noise for broadband ambient sounds (fills in the spectrogram background)
    ambient = generate_pink_noise(n_samples)
    max_amb = np.max(np.abs(ambient))
    if max_amb > 0:
        ambient /= max_amb
        
    # Low-frequency rumble (body/sensor noise)
    white = np.random.randn(n_samples)
    b, a = signal.butter(2, 100, btype='lowpass', fs=fs)
    rumble = signal.filtfilt(b, a, white)
    max_rumble = np.max(np.abs(rumble))
    if max_rumble > 0:
        rumble /= max_rumble
        
    # Mix ambient and rumble
    mixed_noise = 0.3 * ambient + 0.7 * rumble
    
    return mixed_noise * noise_level

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
        heart_rate = np.random.uniform(60, 90) # bpm
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
        
        # S2 parameters based on severity
        s2_intensity = 1.0 - 0.9 * severity
        
        # S4 parameters (prominent in severe AS)
        s4_intensity = 0.0
        if severity > 0.6:
            s4_intensity = 0.3 * ((severity - 0.6) / 0.4)
            
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            t_s4 = current_time - 0.08
            t_s1 = current_time
            t_murmur_start = current_time + 0.08 # 20ms gap after 60ms S1
            t_s2 = current_time + systole_duration
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=40, duration=0.06, fs=FS_INTERNAL)
            
            murmur_duration = (t_s2 - 0.02) - t_murmur_start # Ends 20ms before S2
            if murmur_duration > 0.05:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            s2 = generate_gabor_wavelet(freq=50, duration=0.06, fs=FS_INTERNAL)
            s2 *= s2_intensity
            
            # Add to buffer
            def add_to_buffer(signal_array, start_time):
                if start_time < 0:
                    # Handle negative start times by slicing the signal
                    start_idx = 0
                    signal_offset = int(-start_time * FS_INTERNAL)
                    if signal_offset >= len(signal_array):
                        return
                    signal_array = signal_array[signal_offset:]
                else:
                    start_idx = int(start_time * FS_INTERNAL)
                    
                end_idx = start_idx + len(signal_array)
                
                if start_idx >= total_samples:
                    return
                    
                if end_idx > total_samples:
                    signal_array = signal_array[:total_samples - start_idx]
                    end_idx = total_samples
                    
                audio_buffer[start_idx:end_idx] += signal_array

            if s4_intensity > 0:
                s4 = generate_gabor_wavelet(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                add_to_buffer(s4, t_s4)
                
            add_to_buffer(s1, t_s1)
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
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