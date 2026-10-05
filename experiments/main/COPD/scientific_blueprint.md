# Scientific Blueprint: COPD (Audio Signal)

## 1. Condition Overview

**Definition & Pathophysiology:**
Chronic Obstructive Pulmonary Disease (COPD) is a chronic, progressive inflammatory lung disease characterized by largely irreversible airflow limitation. It encompasses two primary phenotypes: *emphysema* (destruction of alveolar walls leading to hyperinflation and loss of elastic recoil) and *chronic bronchitis* (chronic airway inflammation and mucus hypersecretion). Acoustically, the hyperinflated lungs act as an acoustic insulator, severely attenuating sound transmission to the chest wall, while narrowed airways generate turbulent airflow and adventitious (abnormal) sounds.

**Subtypes & Severity:**
Severity is typically graded using the GOLD criteria (Stages 1–4: Mild, Moderate, Severe, Very Severe) based on forced expiratory volume (FEV1). 

**Demographics:**
Prevalence is approximately 10% in adults over 40 globally. It is predominantly found in individuals with a history of tobacco smoking or long-term exposure to biomass smoke and occupational dust.

---

## 2. Signal Manifestation

In the audio domain (lung auscultation), COPD manifests through distinct alterations in timing, spectral power, and the presence of superimposed transient and continuous acoustic events.

**Acoustic Event Structure:**
*   **Base Breath Sounds (Vesicular):** Markedly diminished in amplitude.
*   **Wheezes:** Continuous, musical, sinusoidal sounds caused by airway narrowing. In COPD, these are typically *polyphonic* (multiple simultaneous frequencies) and predominantly expiratory.
*   **Crackles (Rales):** Discontinuous, explosive transient sounds. COPD typically presents with *coarse crackles* (early inspiratory or expiratory) due to air bubbling through mucus in larger airways.

**Temporal Organization and Timing Patterns:**
*   **I:E Ratio (Inspiration to Expiration):** A healthy resting I:E ratio is approximately 1:1.5 to 1:2. In COPD, loss of elastic recoil necessitates active, prolonged expiration. The I:E ratio shifts to **1:3, 1:4, or even 1:5**.
*   **Phase Durations:** Inspiration typically lasts 1.0–1.5 seconds; Expiration is stretched to 3.0–6.0 seconds (est.).

**Spectral Characteristics & Dominant Frequency Bands:**
*   **Diminished Breath Sounds:** Normal vesicular sounds span 100–1000 Hz. In COPD, frequencies above 200 Hz are severely attenuated. The spectral roll-off is much steeper than in healthy lungs (e.g., -12 to -18 dB/octave above 150 Hz).
*   **Wheezes:** Narrowband spectral peaks. COPD wheezes generally occupy the **100–800 Hz** range (lower pitch than asthma). Duration is > 250 ms.
*   **Coarse Crackles:** Broadband transients with a dominant frequency between **100–500 Hz**. Duration is relatively long for a crackle, typically **10–20 ms** per pop.

**Amplitude Relationships:**
*   Overall signal power is reduced by **10 to 20 dB** (est.) compared to a healthy adult of similar BMI.
*   Wheeze amplitude often exceeds the base expiratory breath sound by 6–12 dB, making the wheeze the dominant acoustic feature during the expiratory phase.

**Envelope / Amplitude-Modulation Behavior:**
*   **Inspiratory Envelope:** Shorter, with a blunted attack and lower peak amplitude.
*   **Expiratory Envelope:** Highly asymmetrical. It features a rapid initial rise followed by a long, flattened, low-amplitude plateau (the prolonged expiratory phase), often modulated by the sinusoidal envelope of a wheeze.

**Recording / Acquisition Variability:**
*   **Sensor Placement:** Attenuation is most severe at the lung bases (due to gravity-dependent hyperinflation). Wheezes and coarse crackles are more prominent over the upper anterior chest and trachea.
*   **Body Habitus:** The "barrel chest" (increased anterior-posterior diameter) common in emphysema adds physical distance between the lung and the microphone, further acting as a low-pass filter and attenuator.

---

## 3. Severity and Variability

**Changes with Condition Severity:**
*   **Mild (GOLD 1):** Breath sounds slightly diminished. I:E ratio ~1:2.5. Wheezing may only be present during *forced* expiration.
*   **Moderate to Severe (GOLD 2–3):** Markedly diminished breath sounds. Resting I:E ratio > 1:3. Resting expiratory polyphonic wheezes and early inspiratory coarse crackles are prominent.
*   **Very Severe (GOLD 4):** The "Silent Chest." Airflow is so severely restricted that it cannot generate enough turbulence to produce wheezes. The audio signal is characterized by extreme attenuation (near silence) with a massively prolonged expiratory phase.

**Sources of Natural Variability:**
*   **Beat-to-Beat (Breath-to-Breath):** Crackles are highly variable and may disappear or change characteristics after a patient coughs (which clears mucus). Wheeze frequencies may drift by ±50 Hz between breaths depending on airway tone.
*   **Phenotype Variance:** Emphysema-dominant patients exhibit maximum attenuation and prolonged expiration with fewer adventitious sounds. Bronchitis-dominant patients exhibit more coarse crackles and rhonchi (low-pitched wheezes < 200 Hz).

---

## 4. Key Discriminators

The following features are critical for DSP algorithms to distinguish COPD from confusable respiratory conditions:

1.  **Expiratory Prolongation (I:E Ratio):** > 1:3 is a hallmark of obstructive disease, distinguishing it from restrictive diseases (like pulmonary fibrosis) or healthy lungs.
2.  **Global High-Frequency Attenuation:** Severe drop in spectral power > 200 Hz distinguishes COPD from Pneumonia (which *increases* high-frequency transmission).
3.  **Polyphonic Low-to-Mid Pitch Wheezes:** Distinguishes COPD from Asthma (which typically features higher-pitched, more widespread wheezing).
4.  **Early Inspiratory Coarse Crackles:** Distinguishes COPD from Pulmonary Fibrosis or Heart Failure (which feature *late* inspiratory *fine* crackles).

### Differential Diagnosis Table (Audio Domain)

| Feature | COPD | Asthma | Pneumonia | Healthy |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Amplitude** | Severely decreased (-10 to -20 dB) | Normal to slightly decreased | Increased over consolidation | Baseline (0 dB reference) |
| **I:E Ratio** | 1:3 to 1:5 (Prolonged) | 1:2 to 1:4 (Prolonged during attack) | 1:1.5 to 1:2 (Normal) | 1:1.5 to 1:2 |
| **Spectral Profile** | Steep roll-off > 200 Hz | Normal base, high-freq wheeze peaks | Enhanced high-freq (up to 2000 Hz) | Gentle roll-off > 400 Hz |
| **Wheeze Type** | Polyphonic, low/mid-pitch (100-800 Hz) | Polyphonic, high-pitch (400-1500 Hz) | Usually absent | Absent |
| **Crackle Type** | Coarse, early inspiratory (100-500 Hz) | Usually absent | Fine, late inspiratory (300-1000 Hz) | Absent |

---

## 5. Synthesis Considerations

To computationally simulate COPD lung sounds, the DSP team should utilize a hybrid approach: subtractive synthesis for the base breath sounds, additive synthesis for wheezes, and stochastic impulse generation for crackles.

**Recommended Computational Approach:**
1.  **Base Breath Sounds:** Generate pink noise. Apply an ADSR envelope mapped to the respiratory cycle. Pass the enveloped noise through a dynamic low-pass filter.
2.  **Wheezes:** Use a bank of 2 to 4 sine wave oscillators. Apply slight frequency modulation (FM) and amplitude modulation (AM) to simulate the organic, chaotic nature of airway fluttering.
3.  **Crackles:** Generate short bursts of white noise or impulses, passed through a bandpass filter, triggered stochastically during the early inspiratory phase.

**Concrete Parameter Ranges (Target Values):**

*   **Respiratory Cycle Timing:**
    *   Inspiration Duration: 1200 ms (± 200 ms jitter).
    *   Expiration Duration: 4000 ms (± 500 ms jitter).
    *   Pause (Post-expiratory): 200–500 ms.
*   **Base Sound Filtering:**
    *   Low-pass filter cutoff: 150–250 Hz.
    *   Filter slope: 24 dB/octave (simulating the heavy acoustic insulation of hyperinflated lungs).
*   **Wheeze Synthesis:**
    *   Oscillator Frequencies: Randomly select 3 frequencies between 150 Hz and 600 Hz.
    *   FM (Vibrato): 5–15 Hz rate, with a depth of 2–5% of the base frequency.
    *   Envelope: Attack 300 ms, Sustain 1500–2500 ms, Release 400 ms. Triggered 500 ms *after* the start of expiration.
*   **Crackle Synthesis:**
    *   Burst duration: 10–15 ms.
    *   Bandpass filter: 150 Hz to 400 Hz.
    *   Density: 3 to 8 crackles clustered in the first 400 ms of the inspiratory phase.

**Common Pitfalls & Avoidance:**
*   **Pitfall:** Making the wheezes sound like a synthesizer (too pure).
    *   *Fix:* Pure sine waves sound artificial. Add a small amount of narrowband noise centered at the sine frequency, and ensure the FM/AM parameters have slight stochastic jitter (Perlin noise works well for modulating the FM rate).
*   **Pitfall:** Making the overall signal too loud.
    *   *Fix:* COPD is defined by *quiet* lungs. The DSP engineer must resist the urge to normalize the audio file to 0 dBFS. The base breath sound should peak around -24 to -18 dBFS, allowing the wheezes to peak around -12 dBFS.
*   **Pitfall:** Static timing.
    *   *Fix:* Human breathing is not a perfect metronome. Introduce a ±10% randomized variance to the I:E durations on every cycle to prevent the simulation from sounding like a mechanical loop.
