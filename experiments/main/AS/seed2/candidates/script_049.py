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
    Generates pink noise (1/f) in the frequency domain.
    Used for turbulent fluid dynamics (murmur).
    """
    if n_samples <= 0:
        return np.array([])
    X = np.fft.rfft(np.random.randn(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1.0 # avoid divide by zero
    X /= np.sqrt(f)
    pink = np.fft.irfft(X, n=n_samples)
    if np.std(pink) > 0:
        pink /= np.std(pink)
    return pink

def generate_brown_noise(n_samples):
    """
    Generates brown noise (1/f^2) via integration.
    Used for low-frequency body/sensor rumble.
    """
    if n_samples <= 0:
        return np.array([])
    white = np.random.randn(n_samples)
    brown = np.cumsum(white)
    if np.std(brown) > 0:
        brown /= np.std(brown)
    return brown

def generate_as_murmur(duration, peak_fraction, fs):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using filtered pink noise.
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
        
    # Generate turbulent pink noise
    pink_noise = generate_pink_noise(n_samples)
    
    # Bandpass filter for murmur frequencies
    # Widened f0 and Q ranges to match blueprint and increase diversity
    f0 = np.random.uniform(150, 300)
    Q = np.random.uniform(1.0, 2.0)
    bw = f0 / Q
    low = max(20.0, f0 - bw/2)
    high = min(fs/2 - 1.0, f0 + bw/2)
    
    # Use SOS for numerical stability
    sos = signal.butter(2, [low, high], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, pink_noise)
    
    if np.max(np.abs(filtered_noise)) > 0:
        filtered_noise /= np.max(np.abs(filtered_noise))
        
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise using brown noise
    to provide a realistic low-frequency baseline rumble and wander.
    """
    n_samples = int(fs * duration)
    brown = generate_brown_noise(n_samples)
    
    # Bandpass to simulate body/sensor noise
    sos = signal.butter(2, [1.0, 100.0], btype='bandpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos, brown)
    
    if np.std(ambient) > 0:
        ambient /= np.std(ambient)
        
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
        heart_rate = np.random.uniform(55, 95) # bpm
        rr_interval = 60.0 / heart_rate
        
        # Severity from 0.1 (mild) to 1.0 (severe)
        severity = np.random.uniform(0.1, 1.0)
        
        # Respiratory rate for amplitude modulation
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level (widened range for MFCC diversity)
        noise_level = np.random.uniform(0.05, 0.35)
        
        # 2. Derived cardiac timing parameters
        # Systole duration is slightly prolonged in AS
        systole_duration = 0.35 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = np.random.uniform(0.15, 0.35) + 0.4 * severity 
        
        # S2 parameters based on severity (A2 is soft/absent in severe AS)
        s2_intensity = max(0.0, np.random.uniform(0.8, 1.2) - 1.2 * severity)
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            s1_duration = np.random.uniform(0.06, 0.10)
            t_s1 = current_time
            
            # Murmur starts ~20ms after S1 ends, ends ~20ms before S2
            t_murmur_start = t_s1 + s1_duration + np.random.uniform(0.01, 0.03)
            t_s2 = t_s1 + systole_duration
            t_murmur_end = t_s2 - np.random.uniform(0.01, 0.03)
            murmur_duration = t_murmur_end - t_murmur_start
            
            # Generate S1
            s1_freq = np.random.uniform(30, 55)
            s1 = generate_gabor_wavelet(freq=s1_freq, duration=s1_duration, fs=FS_INTERNAL)
            
            # Generate Murmur
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
            else:
                murmur = np.array([])
                
            # Generate S2
            s2_freq = np.random.uniform(45, 75)
            s2_duration = np.random.uniform(0.05, 0.08)
            s2 = generate_gabor_wavelet(freq=s2_freq, duration=s2_duration, fs=FS_INTERNAL)
            s2 *= s2_intensity
            
            # Generate S4 (prominent in severe AS due to LVH)
            t_s4 = t_s1 - np.random.uniform(0.07, 0.09)
            s4_intensity = 0.0
            if severity > 0.6:
                s4_intensity = np.random.uniform(0.1, 0.3) * (severity - 0.6) / 0.4
                
            # Add to buffer
            def add_to_buffer(signal_array, start_time):
                if start_time < 0:
                    start_idx = 0
                    offset = int(-start_time * FS_INTERNAL)
                    if offset >= len(signal_array):
                        return
                    signal_array = signal_array[offset:]
                else:
                    start_idx = int(start_time * FS_INTERNAL)
                    
                end_idx = start_idx + len(signal_array)
                
                if start_idx >= total_samples:
                    return
                    
                if end_idx > total_samples:
                    signal_array = signal_array[:total_samples - start_idx]
                    end_idx = total_samples
                    
                audio_buffer[start_idx:end_idx] += signal_array

            add_to_buffer(s1, t_s1)
            add_to_buffer(murmur, t_murmur_start)
            add_to_buffer(s2, t_s2)
            
            if s4_intensity > 0:
                s4 = generate_gabor_wavelet(freq=np.random.uniform(25, 35), duration=0.06, fs=FS_INTERNAL)
                s4 *= s4_intensity
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - np.random.uniform(0.1, 0.2) * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise (brown noise rumble)
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level)
        
        # 6.5 Apply body/chest acoustic low-pass filter
        # Increased filter order to 4 and widened cutoff range to better simulate heavy high-frequency 
        # attenuation by body tissues, which reduces artificially high ZCR and improves MFCC diversity.
        body_lpf_cutoff = np.random.uniform(120.0, 300.0)
        sos_body = signal.butter(4, body_lpf_cutoff, btype='lowpass', fs=FS_INTERNAL, output='sos')
        audio_buffer = signal.sosfiltfilt(sos_body, audio_buffer)
        
        # 6.6 Apply stethoscope diaphragm high-pass filter
        # Lowered cutoff from 25 Hz to 5.0 Hz to preserve the natural low-frequency baseline rumble.
        # This baseline offset prevents the high-frequency murmur from constantly crossing zero, 
        # drastically reducing the ZCR to match empirical physiological recordings.
        sos_hpf = signal.butter(2, 5.0, btype='highpass', fs=FS_INTERNAL, output='sos')
        audio_buffer = signal.sosfiltfilt(sos_hpf, audio_buffer)
        
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