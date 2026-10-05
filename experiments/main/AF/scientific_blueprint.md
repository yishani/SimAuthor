# Scientific Blueprint: Atrial Fibrillation (PPG Domain)

## 1. Condition Overview

**Definition and Pathophysiology**
Atrial Fibrillation (AFib) is a supraventricular tachyarrhythmia characterized by uncoordinated atrial activation and consequently ineffective atrial contraction. In the absence of a functional "atrial kick," ventricular filling is solely dependent on passive flow. The atrioventricular (AV) node is bombarded with rapid, irregular electrical impulses, acting as a chaotic filter. This results in an "irregularly irregular" ventricular response, which translates hemodynamically into highly variable stroke volumes and erratic pulse intervals.

**Clinical Significance**
AFib is a major risk factor for ischemic stroke, heart failure, and overall cardiovascular morbidity. Because PPG is the primary modality for wearable heart rate monitors (smartwatches, fitness bands), accurate computational modeling of AFib in the PPG domain is critical for developing and validating automated detection algorithms.

**Subtypes and Severity Gradations**
*   **Paroxysmal:** Episodes terminating spontaneously or with intervention within 7 days.
*   **Persistent:** Continuous AFib lasting >7 days.
*   **Permanent:** AFib where rhythm control interventions are no longer pursued.
*   *Rate-based classifications (Crucial for DSP):* 
    *   AFib with Rapid Ventricular Response (RVR): Mean Heart Rate (HR) > 100 bpm.
    *   Controlled AFib: Mean HR 60–100 bpm.
    *   Slow AFib: Mean HR < 60 bpm.

**Prevalence and Demographics**
AFib affects 1–2% of the general population, with prevalence increasing sharply with age (up to 8–10% in individuals >80 years old). It is more common in males and individuals with hypertension, obesity, or structural heart disease.

---

## 2. Signal Manifestation (PPG Domain)

In the PPG domain, AFib is primarily a disorder of **timing (intervals)** and **amplitude (stroke volume)**, rather than a fundamental change in the underlying single-pulse morphology.

*   **Pulse Interval and Heart-Rate Behavior:**
    *   The peak-to-peak (PP) intervals are "irregularly irregular." There is no predictable pattern, respiratory sinus arrhythmia (RSA) is absent, and compensatory pauses do not exist.
    *   **PP Interval Range:** Highly variable, typically ranging from 400 ms to 1200 ms within a single 30-second window.
    *   **Distribution:** The histogram of PP intervals is typically broad, skewed, and often bimodal or multimodal due to dual AV nodal pathway physiology.

*   **Amplitude and Baseline Behavior:**
    *   **Pulse Amplitude Modulation (PAM):** Highly variable beat-to-beat. Because ventricular filling time varies wildly, stroke volume varies proportionally. 
    *   *Rule of thumb:* A short preceding PP interval results in a low-amplitude pulse; a long preceding PP interval results in a high-amplitude pulse.
    *   **Baseline:** The DC baseline is generally unaffected by AFib itself, though standard respiratory wander (0.15–0.4 Hz) remains present.

*   **Pulse Morphology and Waveform Shape:**
    *   The fundamental shape (systolic peak, diastolic decay) is preserved but dynamically distorted by truncation.
    *   **Systolic Rise:** Transit time from foot to peak remains relatively constant (typically 100–150 ms). The slope (first derivative) scales linearly with the pulse amplitude.
    *   **Diastolic Decay:** Frequently truncated. Before the exponential decay can reach the baseline, the next premature beat initiates a new systolic rise.

*   **Pulse Width:**
    *   Highly variable, dictated entirely by the premature arrival of the subsequent beat. Widths can range from 350 ms (in RVR) to >1000 ms (during long pauses).

*   **Dicrotic Notch Characteristics:**
    *   The presence of the dicrotic notch depends heavily on the PP interval. 
    *   In long intervals (>800 ms), the notch is visible and normal.
    *   In short intervals (<600 ms), the notch is frequently obscured, smoothed out, or completely swallowed by the systolic rise of the next beat.

*   **Acquisition Variability:**
    *   PPG is highly susceptible to motion artifacts. Because AFib relies on interval variability for detection, motion artifacts (which create false peaks) are the primary confounder in real-world acquisition.

---

## 3. Severity and Variability

**Condition Severity (Rate Dependence)**
The morphological appearance of AFib in PPG is heavily dependent on the mean ventricular rate:
*   **AFib with RVR (HR > 110 bpm):** Pulses are severely truncated. The AC amplitude is generally lower due to reduced filling time. The waveform may look like a continuous, jagged sine wave with varying peak heights. Dicrotic notches are almost entirely absent.
*   **Controlled AFib (HR 70 bpm):** Pulses have distinct systolic and diastolic phases. Amplitude variability is highly pronounced. Notches are visible on longer beats.

**Sources of Natural Variability**
*   **Subject-to-Subject:** Vascular compliance affects the sharpness of the systolic peak and the prominence of the dicrotic notch. Older patients (typical AFib demographic) have stiffer arteries, leading to rounded peaks and diminished notches regardless of rhythm.
*   **Beat-to-Beat:** The hallmark of AFib. Stroke volume and pulse transit time fluctuate beat-to-beat based on the preceding RR interval.
*   **Acquisition-related:** Ambient light leakage, sensor pressure, and peripheral perfusion (e.g., cold hands reducing the AC/DC ratio) drastically alter signal-to-noise ratio (SNR).

---

## 4. Key Discriminators

To distinguish AFib from confusable conditions in the PPG domain, algorithms rely on statistical measures of interval and amplitude variability.

1.  **Shannon Entropy (ShE) of PP Intervals:** Measures the complexity/randomness of the interval distribution. AFib has high entropy (typically > 0.7 normalized), whereas normal rhythms have low entropy.
2.  **Root Mean Square of Successive Differences (RMSSD):** Measures short-term interval variability. In AFib, RMSSD is exceptionally high (typically > 70 ms), whereas in Normal Sinus Rhythm it is lower (20–50 ms).
3.  **Pulse Amplitude Variability (PAV):** The coefficient of variation of the AC amplitude. AFib exhibits high PAV (> 15% est.) due to varying stroke volumes, unlike Sinus Arrhythmia where amplitude changes are gradual and respiratory-linked.
4.  **Absence of Compensatory Pauses:** Ectopic beats (PACs/PVCs) feature a short interval followed by a distinct, predictable long interval (compensatory pause). AFib lacks this predictable return to a baseline rate.

### Differential Diagnosis Table (PPG Domain)

| Feature | Atrial Fibrillation (AFib) | Normal Sinus Rhythm (NSR) | Sinus Arrhythmia (RSA) | Ectopic Beats (PAC/PVC) |
| :--- | :--- | :--- | :--- | :--- |
| **PP Interval Rhythm** | Irregularly irregular; chaotic | Regular | Cyclical (respiratory linked) | Mostly regular, isolated disruptions |
| **PP Interval Entropy** | High (> 0.7 normalized) | Low (< 0.3) | Medium (0.3 - 0.5) | Low to Medium |
| **Amplitude Variability** | High, chaotic, beat-to-beat | Low | Moderate, cyclical | High only on the ectopic beat |
| **Compensatory Pause** | Absent | Absent | Absent | Present (long interval after short) |
| **Poincaré Plot ($PP_n$ vs $PP_{{n+1}}$)** | Fan-shaped or dispersed cloud | Tight cluster | Elongated cigar shape along identity line | Cross or star shape (distinct clusters) |

---

## 5. Synthesis Considerations

For a DSP team building a computational PPG simulator, AFib cannot be modeled by simply adding white noise to a regular heart rate. The physiological coupling between timing and hemodynamics must be respected.

### Recommended Computational Approach

1.  **Interval Generation (The Timing Model):**
    *   Do *not* use Gaussian noise. Use a Markov Chain or a randomized sampling from a bimodal/multimodal Gamma or Log-Normal distribution to generate the sequence of PP intervals ($PP_n$).
    *   Ensure the generated sequence has a normalized Shannon Entropy > 0.7 and an RMSSD between 70–150 ms.
2.  **Amplitude Coupling (The Hemodynamic Model):**
    *   Calculate the amplitude of the current beat ($A_n$) as a function of the preceding interval ($PP_{{n-1}}$). 
    *   *Formula concept:* $A_n = A_{{max}} \times (1 - e^{{-k \cdot PP_{{n-1}}}})$. This simulates the exponential nature of ventricular filling. A longer preceding interval yields a larger amplitude.
3.  **Morphology Generation (The Waveform Model):**
    *   Use a standard PPG template (e.g., a sum of two Log-Normal functions or Gaussians representing the systolic and diastolic waves).
    *   Scale the systolic peak by $A_n$.
    *   Scale the systolic rise time slightly inversely to $A_n$ (larger beats have slightly faster rise times).
4.  **Waveform Concatenation and Truncation:**
    *   Place the templates at the times dictated by the $PP_n$ sequence.
    *   Where $PP_n$ is shorter than the template length, simply sum the overlapping tails or truncate the diastolic decay of beat $n$ when the systolic rise of beat $n+1$ begins. This naturally simulates the swallowed dicrotic notches of RVR.
5.  **Noise and Baseline Addition:**
    *   Add a low-frequency respiratory baseline wander (0.15–0.3 Hz).
    *   Add pink noise (1/f) to simulate sensor noise.

### Concrete Parameter Targets for Synthesis

| Parameter | Target Range for AFib Simulation |
| :--- | :--- |
| **Mean Heart Rate** | 60 – 150 bpm (Configurable for RVR vs Controlled) |
| **PP Interval Range** | 350 ms – 1300 ms |
| **RMSSD** | 70 ms – 180 ms |
| **Normalized Shannon Entropy** | 0.70 – 0.95 |
| **Amplitude Coeff. of Variation** | 15% – 35% (est.) |
| **Systolic Rise Time** | 100 ms – 160 ms (Keep relatively constant) |

### Common Pitfalls and How to Avoid Them

*   **Pitfall 1: Uncoupled Amplitude and Intervals.** If you generate chaotic PP intervals but keep the pulse amplitude constant, the resulting PPG will look highly artificial to any machine learning model or clinician. *Solution:* Strictly enforce the $A_n \propto PP_{{n-1}}$ rule.
*   **Pitfall 2: Uniform Random Intervals.** Using a uniform distribution for PP intervals creates a signal that lacks physiological realism. The AV node has refractory periods. *Solution:* Ensure no PP interval is shorter than ~300 ms (absolute refractory limit), and use skewed distributions (Log-Normal).
*   **Pitfall 3: Stretching the Waveform.** Do not time-stretch the entire single-beat PPG template to fit a long PP interval. The systolic phase duration is relatively fixed by cardiac mechanics. *Solution:* Only extend the flat diastolic tail for long intervals; keep the systolic width constant.
