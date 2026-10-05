import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f) to simulate realistic physiological acoustics 
    and turbulent blood flow, avoiding the harsh, tonal qualities of filtered white noise.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_samples)
    freqs[0] = 1.0  # Prevent division by zero
    X = X / np.sqrt(freqs)
    X[0] = 0.0  # Remove DC offset
    pink = np.fft.irfft(X, n=n_samples)
    return pink

def generate_gabor_wavelet(freq, duration, fs, phase=0.0):
    """
    Generates a Gabor wavelet to simulate heart sounds (S1, S2, S4, EC).
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
    characteristic of Aortic Stenosis using pink noise and stable SOS filtering.
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
    pink = generate_pink_noise(n_samples)
    
    # Bandpass filter for murmur frequencies (100-500 Hz for AS)
    # Broadened slightly to 100 Hz to add more low-mid body to the harsh murmur
    sos = signal.butter(2, [100, 500], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, pink)
    
    # Normalize to ensure consistent amplitude before applying the envelope
    if np.max(np.abs(filtered_noise)) > 0:
        filtered_noise = filtered_noise / np.max(np.abs(filtered_noise))
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level):
    """
    Generates low-frequency ambient and sensor noise using pink noise,
    plus baseline wander to simulate respiration and sensor movement.
    This drastically reduces unnatural zero-crossing rates (ZCR) by providing
    a realistic physiological low-frequency baseline.
    """
    n_samples = int(fs * duration)
    pink = generate_pink_noise(n_samples)
    
    # Gentle low-pass filter to simulate body/sensor noise
    sos = signal.butter(1, 100, btype='lowpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos, pink)
    
    if np.max(np.abs(ambient)) > 0:
        ambient = ambient / np.max(np.abs(ambient))
        
    # Baseline wander (respiratory and movement artifacts)
    t = np.linspace(0, duration, n_samples, endpoint=False)
    wander = 0.6 * np.sin(2 * np.pi * 0.25 * t + np.random.uniform(0, 2*np.pi)) + \
             0.4 * np.sin(2 * np.pi * 0.6 * t + np.random.uniform(0, 2*np.pi)) + \
             0.2 * np.sin(2 * np.pi * 1.1 * t + np.random.uniform(0, 2*np.pi))
             
    return ambient * noise_level + wander * (noise_level * 2.0)

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
        
        # Respiratory rate for amplitude modulation and S2 splitting
        resp_rate = np.random.uniform(12, 20) / 60.0 # Hz
        
        # Noise level (increased to provide a realistic continuous low-frequency baseline)
        noise_level = np.random.uniform(0.15, 0.35)
        
        # 2. Derived cardiac timing parameters
        systole_duration = 0.30 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        murmur_peak_fraction = 0.31 + 0.39 * severity # Peaks later in severe AS (0.35 to 0.70)
        murmur_intensity = 0.3 + 1.2 * severity       # Louder in severe AS (0.3 to 1.5 relative to S1)
        
        # S2 (A2) parameters based on severity
        a2_intensity = np.clip(1.1 - 1.0 * severity, 0.0, 1.0) # A2 becomes soft/absent in severe AS
        
        # Optional Ejection Click for mild AS
        has_ec = (severity < 0.5) and (np.random.rand() > 0.5)
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
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
            s1_duration = 0.07
            t_s1 = current_time
            t_s2 = current_time + systole_duration
            
            # Murmur leaves a brief gap after S1 and ends before S2
            t_murmur_start = current_time + s1_duration + 0.02
            murmur_duration = t_s2 - 0.02 - t_murmur_start
            
            # Generate components
            s1 = generate_gabor_wavelet(freq=50, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            if has_ec:
                ec = generate_gabor_wavelet(freq=250, duration=0.02, fs=FS_INTERNAL)
                add_to_buffer(ec * 0.4, t_s1 + 0.05)
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL)
                add_to_buffer(murmur * murmur_intensity, t_murmur_start)
                
            # S2 components (A2 and P2)
            a2 = generate_gabor_wavelet(freq=70, duration=0.06, fs=FS_INTERNAL)
            p2 = generate_gabor_wavelet(freq=60, duration=0.06, fs=FS_INTERNAL)
            
            # Respiratory phase for S2 splitting
            resp_phase = 2 * np.pi * resp_rate * current_time
            insp_factor = (np.sin(resp_phase) + 1) / 2.0 
            
            if severity < 0.5:
                # Normal splitting: A2 precedes P2, widens on inspiration
                split = 0.01 + 0.03 * insp_factor
                add_to_buffer(a2 * a2_intensity, t_s2)
                add_to_buffer(p2 * 0.4, t_s2 + split)
            else:
                # Paradoxical splitting: P2 precedes A2, widens on expiration
                split = 0.01 + 0.03 * (1.0 - insp_factor)
                add_to_buffer(p2 * 0.4, t_s2)
                add_to_buffer(a2 * a2_intensity, t_s2 + split)
                
            # S4 (prominent in severe AS due to LVH)
            if severity > 0.7:
                s4_intensity = 0.3 * (severity - 0.7) / 0.3
                s4 = generate_gabor_wavelet(freq=30, duration=0.06, fs=FS_INTERNAL)
                t_s4 = t_s1 - 0.08
                if t_s4 >= 0:
                    add_to_buffer(s4 * s4_intensity, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise and baseline wander
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level)
        
        # 6.5 Apply chest wall acoustic attenuation (gentle low-pass filter)
        # A 1st-order butterworth with filtfilt results in a 2nd-order effective roll-off (-12 dB/oct).
        # This is gentler than the previous 4th-order effective filter, preserving some high-frequency 
        # complexity while still attenuating it, which fixes the negative MFCC shifts.
        sos_chest = signal.butter(1, 250, btype='lowpass', fs=FS_INTERNAL, output='sos')
        audio_buffer = signal.sosfiltfilt(sos_chest, audio_buffer)
        
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