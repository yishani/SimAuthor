import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_heart_sound(freq, duration, fs, phase=0.0):
    """
    Generates a heart sound (S1, S2, S4, EC) using a fast-attack, exponential-decay envelope.
    This provides a more realistic, asymmetric 'thump' compared to a symmetric Gabor wavelet.
    """
    n_samples = int(fs * duration)
    if n_samples == 0:
        return np.array([])
        
    t = np.linspace(0, duration, n_samples, endpoint=False)
    attack_time = 0.015
    attack_idx = int(attack_time * fs)
    envelope = np.zeros_like(t)
    
    if attack_idx > 0 and attack_idx < n_samples:
        envelope[:attack_idx] = np.sin(0.5 * np.pi * t[:attack_idx] / attack_time)**2
        decay_time = duration - attack_time
        envelope[attack_idx:] = np.exp(-5.0 * (t[attack_idx:] - attack_time) / decay_time)
    else:
        # Fallback if duration is very short
        envelope = np.ones_like(t)
        
    wave = np.sin(2 * np.pi * freq * t + phase) * envelope
    return wave

def generate_as_murmur(duration, peak_fraction, fs):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using Pink Noise to prevent unnatural high-frequency ZCR.
    """
    n_samples = int(fs * duration)
    if n_samples == 0:
        return np.array([])
        
    t_norm = np.linspace(0, 1, n_samples, endpoint=False)
    envelope = np.zeros_like(t_norm)
    
    # Create the diamond-shaped envelope
    peak_idx = int(peak_fraction * n_samples)
    
    # Crescendo phase
    if peak_idx > 0:
        envelope[:peak_idx] = np.sin(0.5 * np.pi * t_norm[:peak_idx] / peak_fraction)**2
        
    # Decrescendo phase
    if peak_idx < n_samples:
        envelope[peak_idx:] = np.cos(0.5 * np.pi * (t_norm[peak_idx:] - peak_fraction) / (1.0 - peak_fraction))**2
        
    # Generate Pink Noise via FFT (1/f power spectrum)
    X = np.fft.rfft(np.random.randn(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1.0 # Avoid division by zero
    X /= np.sqrt(f)
    noise = np.fft.irfft(X, n=n_samples)
    
    # Bandpass filter for murmur frequencies (150-400 Hz)
    # Using 2nd order for a broader, less tonal (harsh) sound
    b, a = signal.butter(2, [150, 400], btype='bandpass', fs=fs)
    filtered_noise = signal.filtfilt(b, a, noise)
    
    # Normalize noise before applying envelope
    max_val = np.max(np.abs(filtered_noise))
    if max_val > 0:
        filtered_noise /= max_val
        
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise with a 1/f^1.5 characteristic.
    This provides the necessary low-frequency baseline seen in real PCGs, fixing ZCR anomalies.
    """
    n_samples = int(fs * duration)
    
    # Generate noise between pink and brown
    X = np.fft.rfft(np.random.randn(n_samples))
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1.0
    X /= (f ** 0.75)
    noise = np.fft.irfft(X, n=n_samples)
    
    if np.max(np.abs(noise)) > 0:
        noise /= np.max(np.abs(noise))
        
    # Low-pass filter to simulate body/sensor acoustic response
    b, a = signal.butter(2, 250, btype='lowpass', fs=fs)
    ambient = signal.filtfilt(b, a, noise)
    
    if np.max(np.abs(ambient)) > 0:
        ambient /= np.max(np.abs(ambient))
        
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
        heart_rate = np.random.uniform(60, 90) # bpm
        rr_interval = 60.0 / heart_rate
        
        # Severity from 0.1 (mild) to 1.0 (severe)
        severity = np.random.uniform(0.1, 1.0)
        
        # Respiratory rate for amplitude modulation
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level (increased to provide realistic low-frequency baseline)
        noise_level = np.random.uniform(0.05, 0.15)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        murmur_intensity = 0.5 + 1.5 * severity
        
        # S2 parameters based on severity
        s2_intensity = 1.0 - 0.9 * severity
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
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
            s1_duration = 0.06
            s2_duration = 0.06
            
            t_s1 = current_time
            # Blueprint: Murmur onset 30 ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            t_s2 = current_time + systole_duration
            
            # Blueprint: Murmur offset 20 ms before S2 begins
            murmur_duration = t_s2 - 0.02 - t_murmur_start
            
            # Generate components
            s1 = generate_heart_sound(freq=40, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            s2 = generate_heart_sound(freq=50, duration=s2_duration, fs=FS_INTERNAL)
            s2 *= s2_intensity
            add_to_buffer(s2, t_s2)
            
            # Add S4 for severe AS
            if severity > 0.7:
                s4_intensity = 0.3 * (severity - 0.7) / 0.3
                s4 = generate_heart_sound(freq=30, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                t_s4 = t_s1 - 0.08
                add_to_buffer(s4, t_s4)
                
            # Add Ejection Click for mild/moderate AS (50% chance)
            if severity < 0.5 and np.random.rand() > 0.5:
                ec_intensity = 0.3 * (1.0 - severity)
                ec = generate_heart_sound(freq=250, duration=0.02, fs=FS_INTERNAL)
                ec *= ec_intensity
                t_ec = t_s1 + s1_duration + 0.01
                add_to_buffer(ec, t_ec)
            
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