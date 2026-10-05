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

def generate_as_murmur(duration, peak_fraction, fs, low_cut=150, high_cut=500):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using a Beta distribution envelope
    and bandpassed pink-ish noise.
    """
    n_samples = int(fs * duration)
    if n_samples == 0:
        return np.array([])
        
    t_norm = np.linspace(0, 1, n_samples, endpoint=False)
    
    # Create the diamond-shaped envelope using a Beta distribution curve
    # Map peak_fraction to alpha and beta parameters
    alpha_param = 1.0 + 6.0 * peak_fraction
    beta_param = 7.0 - 6.0 * peak_fraction
    
    envelope = (t_norm**(alpha_param - 1)) * ((1.0 - t_norm)**(beta_param - 1))
    max_env = np.max(envelope)
    if max_env > 0:
        envelope /= max_env
        
    # Generate turbulent noise (Pink-ish profile)
    white_noise = np.random.randn(n_samples)
    
    # 1st order lowpass to soften the white noise (tilt spectrum)
    sos_soften = signal.butter(1, 200, btype='lowpass', fs=fs, output='sos')
    soft_noise = signal.sosfilt(sos_soften, white_noise)
    
    # Bandpass filter for murmur frequencies using stable SOS format
    sos_bp = signal.butter(2, [low_cut, high_cut], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos_bp, soft_noise)
    
    # Normalize noise to have consistent peak amplitude before applying envelope
    max_noise = np.max(np.abs(filtered_noise))
    if max_noise > 0:
        filtered_noise /= max_noise
        
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient rumble and sensor noise using stable SOS filters.
    """
    n_samples = int(fs * duration)
    
    # Ambient sensor noise
    noise = np.random.randn(n_samples)
    sos_ambient = signal.butter(2, 150, btype='lowpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos_ambient, noise)
    
    # Deep physiological rumble (very low frequency)
    rumble_noise = np.random.randn(n_samples)
    sos_rumble = signal.butter(2, 30, btype='lowpass', fs=fs, output='sos')
    rumble = signal.sosfiltfilt(sos_rumble, rumble_noise)
    
    # Combine and normalize to a stable RMS
    combined = ambient * 0.5 + rumble * 2.0
    rms = np.sqrt(np.mean(combined**2))
    if rms > 0:
        combined /= rms
        
    return combined * noise_level * 0.2

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
        heart_rate = np.random.uniform(50, 110) # bpm
        rr_interval = 60.0 / heart_rate
        
        # Severity from 0.1 (mild) to 1.0 (severe)
        severity = np.random.uniform(0.1, 1.0)
        
        # Respiratory rate for amplitude modulation
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level
        noise_level = np.random.uniform(0.01, 0.05)
        
        # Patient-specific spectral parameters to increase dataset diversity
        s1_freq = np.random.uniform(40, 60)
        s2_freq = np.random.uniform(60, 85)
        s4_freq = np.random.uniform(25, 40)
        murmur_low_cut = np.random.uniform(120, 180)
        murmur_high_cut = np.random.uniform(400, 600)
        
        # 2. Derived cardiac timing and intensity parameters
        # In severe AS, the murmur peaks later in systole
        murmur_peak_fraction = 0.35 + 0.35 * severity + np.random.uniform(-0.05, 0.05)
        murmur_peak_fraction = np.clip(murmur_peak_fraction, 0.2, 0.8)
        
        # Murmur intensity increases with severity
        murmur_intensity = 0.5 + 1.5 * severity + np.random.uniform(-0.2, 0.2)
        murmur_intensity = np.clip(murmur_intensity, 0.3, 2.5)
        
        # In severe AS, A2 is delayed and soft (or absent)
        s2_intensity = 1.0 - 0.9 * severity + np.random.uniform(-0.1, 0.1)
        s2_intensity = np.clip(s2_intensity, 0.0, 1.0)
        
        # S4 is prominent in severe AS
        s4_intensity = 0.0
        if severity > 0.5:
            s4_intensity = 0.3 * (severity - 0.5) / 0.5 + np.random.uniform(0, 0.1)
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # Helper to add signals to the main buffer
        def add_to_buffer(signal_array, start_time):
            start_idx = int(start_time * FS_INTERNAL)
            end_idx = start_idx + len(signal_array)
            
            if start_idx >= total_samples or start_idx < 0:
                return
                
            if end_idx > total_samples:
                signal_array = signal_array[:total_samples - start_idx]
                end_idx = total_samples
                
            audio_buffer[start_idx:end_idx] += signal_array

        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
        while current_time < DURATION:
            # Timing for this specific cycle
            systole_duration = 0.35 * np.sqrt(rr_interval)
            
            t_s1 = current_time
            s1_duration = 0.08
            
            # S4 occurs ~70-100 ms before S1
            t_s4 = t_s1 - np.random.uniform(0.07, 0.10)
            if t_s4 >= 0 and s4_intensity > 0:
                s4 = generate_gabor_wavelet(freq=s4_freq, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                add_to_buffer(s4, t_s4)
            
            # S1
            s1 = generate_gabor_wavelet(freq=s1_freq, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            # Murmur timing: starts 30ms after S1, ends 20ms before S2
            t_murmur_start = t_s1 + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            murmur_duration = (t_s2 - 0.02) - t_murmur_start
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(
                    murmur_duration, 
                    murmur_peak_fraction, 
                    FS_INTERNAL, 
                    low_cut=murmur_low_cut, 
                    high_cut=murmur_high_cut
                )
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            # S2
            s2 = generate_gabor_wavelet(freq=s2_freq, duration=0.08, fs=FS_INTERNAL)
            s2 *= s2_intensity
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