import os
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile

def generate_heart_sound(freq, duration, fs):
    """
    Generates a physiologically realistic heart sound (S1, S2, S4, EC) 
    using a fast-attack, exponential-decay envelope and slight frequency modulation.
    """
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Fast attack (10ms or half duration)
    attack_time = min(0.01, duration / 2.0)
    attack_samples = int(attack_time * fs)
    
    envelope = np.zeros_like(t)
    if attack_samples > 0:
        # Smooth quarter-sine attack to prevent clicks
        envelope[:attack_samples] = np.sin(np.linspace(0, np.pi/2, attack_samples))
    
    decay_time = duration - attack_time
    if decay_time > 0:
        decay_samples = len(t) - attack_samples
        tau = decay_time / 3.0  # 3 time constants for decay
        decay_curve = np.exp(-np.linspace(0, decay_time, decay_samples) / tau)
        # Force the very end to exactly 0 to prevent boundary clicks
        decay_curve = (decay_curve - decay_curve[-1]) / (1.0 - decay_curve[-1] + 1e-9)
        envelope[attack_samples:] = decay_curve
        
    # Frequency modulation: slight pitch drop for realism
    f_t = np.linspace(freq + 15, freq - 5, len(t))
    phase = 2 * np.pi * np.cumsum(f_t) / fs
    
    wave = np.sin(phase) * envelope
    return wave

def generate_pink_noise(n_samples):
    """
    Generates pink noise (1/f) for realistic murmurs and body rumble.
    """
    white = np.random.randn(n_samples)
    X = np.fft.rfft(white)
    f = np.fft.rfftfreq(n_samples)
    f[0] = 1.0  # Avoid divide by zero
    X /= np.sqrt(f)
    pink = np.fft.irfft(X, n=n_samples)
    pink -= np.mean(pink)
    pink /= (np.std(pink) + 1e-9)
    return pink

def generate_as_murmur(duration, peak_fraction, fs, f_low, f_high):
    """
    Generates a crescendo-decrescendo (diamond-shaped) systolic murmur 
    characteristic of Aortic Stenosis using pink noise.
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
    
    # Bandpass filter for murmur frequencies.
    # Reduced to 2nd order (effective 4th with filtfilt) for a less resonant, more natural broadband sound
    sos = signal.butter(2, [f_low, f_high], btype='bandpass', fs=fs, output='sos')
    filtered_noise = signal.sosfiltfilt(sos, noise)
    
    # Normalize filtered noise to have a consistent amplitude scale
    filtered_noise /= (np.std(filtered_noise) + 1e-9)
    # Scale down so that a murmur_intensity of 1.0 roughly matches the peak of S1
    filtered_noise *= 0.33
    
    return filtered_noise * envelope

def generate_background_noise(duration, fs, noise_level, cutoff):
    """
    Generates low-frequency ambient and sensor noise using pink noise
    to provide a realistic baseline wander, plus a trace of wideband sensor hiss.
    """
    n_samples = int(fs * duration)
    pink = generate_pink_noise(n_samples)
    # Low-pass filter to simulate body/sensor noise 
    sos = signal.butter(4, cutoff, btype='lowpass', fs=fs, output='sos')
    ambient = signal.sosfiltfilt(sos, pink)
    
    ambient /= (np.std(ambient) + 1e-9)
    
    # Add a very small amount of wideband pink noise for sensor hiss (improves ZCR realism)
    hiss = generate_pink_noise(n_samples)
    
    return (ambient + 0.05 * hiss) * noise_level

def simulate_aortic_stenosis_pcg():
    # Configuration
    FS_INTERNAL = 44100
    FS_OUTPUT = 16000
    DURATION = 10.0
    NUM_SAMPLES = 100
    OUTPUT_DIR = "artifacts/aortic_stenosis_audio/visual_signal_100iter/generated/"
    
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
        
        # Noise level for realistic continuous low-frequency energy
        noise_level = np.random.uniform(0.2, 0.5)
        
        # 2. Derived cardiac timing parameters
        # Systole duration approximates electromechanical systole
        systole_duration = 0.35 * np.sqrt(rr_interval)
        
        # Murmur parameters based on severity
        # Mild AS peaks at ~35%, Severe AS peaks at ~70%
        murmur_peak_fraction = 0.35 + 0.35 * severity 
        
        # Increased intensity to better match the blueprint (0.5 to 2.0 relative to S1)
        murmur_intensity = np.random.uniform(0.5, 1.5) + 1.0 * severity
        
        # S2 parameters based on severity
        # In severe AS, A2 is delayed and soft (or absent)
        s2_intensity = 1.0 - 0.7 * severity
        
        # Constrain murmur frequencies to match empirical stethoscope recordings
        # Shifted higher to restore missing mid-to-high frequency energy
        murmur_f_low = np.random.uniform(150, 250)
        murmur_f_high = np.random.uniform(400, 700)
        bg_cutoff = np.random.uniform(50, 150)
        
        # 3. Initialize audio buffer
        total_samples = int(FS_INTERNAL * DURATION)
        audio_buffer = np.zeros(total_samples)
        
        # 4. Generate cardiac cycles
        current_time = np.random.uniform(0, rr_interval) # Random phase start
        
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

        while current_time < DURATION:
            # Timing for this specific cycle
            t_s1 = current_time
            s1_duration = 0.06
            
            # Murmur starts 30 ms after S1 ends
            t_murmur_start = t_s1 + s1_duration + 0.03
            
            t_s2 = current_time + systole_duration
            s2_duration = 0.06
            
            # Murmur ends 20 ms before S2 begins
            murmur_duration = t_s2 - 0.02 - t_murmur_start
            
            # Generate components with widened frequency randomization for diversity
            s1_freq = np.random.uniform(30, 60)
            s1 = generate_heart_sound(freq=s1_freq, duration=s1_duration, fs=FS_INTERNAL)
            add_to_buffer(s1, t_s1)
            
            # Ejection Click (EC) for mild/moderate AS
            if severity < 0.6:
                ec_intensity = 0.5 * (1.0 - severity / 0.6)
                ec_freq = np.random.uniform(200, 300)
                ec = generate_heart_sound(freq=ec_freq, duration=0.02, fs=FS_INTERNAL)
                ec *= ec_intensity
                t_ec = t_s1 + s1_duration + 0.04
                if t_ec < t_s2:
                    add_to_buffer(ec, t_ec)
            
            if murmur_duration > 0:
                murmur = generate_as_murmur(murmur_duration, murmur_peak_fraction, FS_INTERNAL, murmur_f_low, murmur_f_high)
                murmur *= murmur_intensity
                add_to_buffer(murmur, t_murmur_start)
                
            s2_freq = np.random.uniform(30, 60)
            s2 = generate_heart_sound(freq=s2_freq, duration=s2_duration, fs=FS_INTERNAL)
            s2 *= s2_intensity
            add_to_buffer(s2, t_s2)
            
            # S4 generation for severe AS (occurs 80 ms before S1)
            if severity > 0.5:
                s4_intensity = 0.2 * ((severity - 0.5) / 0.5)
                s4_freq = np.random.uniform(25, 35)
                s4 = generate_heart_sound(freq=s4_freq, duration=0.05, fs=FS_INTERNAL)
                s4 *= s4_intensity
                t_s4 = t_s1 - 0.08
                add_to_buffer(s4, t_s4)
            
            # Advance to next cycle (add slight HRV)
            current_time += rr_interval + np.random.normal(0, 0.02)
            
        # 5. Apply respiratory modulation (simulating chest cavity volume changes)
        t_array = np.linspace(0, DURATION, total_samples, endpoint=False)
        resp_modulation = 1.0 - 0.15 * np.sin(2 * np.pi * resp_rate * t_array)
        audio_buffer *= resp_modulation
        
        # 6. Add background noise
        audio_buffer += generate_background_noise(DURATION, FS_INTERNAL, noise_level, bg_cutoff)
        
        # 7. Apply global acoustic low-pass filter (simulates stethoscope and body tissue)
        # Relaxed cutoff to preserve murmur harshness and increase ZCR
        lp_cutoff = np.random.uniform(800, 1500)
        sos_body = signal.butter(2, lp_cutoff, btype='lowpass', fs=FS_INTERNAL, output='sos')
        audio_buffer = signal.sosfiltfilt(sos_body, audio_buffer)
        
        # 8. Resample to 16000 Hz
        audio_resampled = signal.resample_poly(audio_buffer, FS_OUTPUT, FS_INTERNAL)
        
        # 9. Peak Normalization
        max_val = np.max(np.abs(audio_resampled))
        if max_val > 0:
            audio_resampled /= (max_val + 1e-9)
            
        # 10. Save to WAV
        audio_pcm = (audio_resampled * 32767.0).astype(np.int16)
        filename = os.path.join(OUTPUT_DIR, f"as_sample_{i:02d}.wav")
        wavfile.write(filename, FS_OUTPUT, audio_pcm)

if __name__ == "__main__":
    simulate_aortic_stenosis_pcg()