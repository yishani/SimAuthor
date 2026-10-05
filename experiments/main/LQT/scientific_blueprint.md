# Scientific Blueprint: Long QT Syndrome (ECG)

## 1. Condition Overview

**Definition and Pathophysiology**
Long QT Syndrome (LQTS) is an arrhythmogenic disorder characterized by a prolongation of the QT interval on the electrocardiogram (ECG). Pathophysiologically, it is caused by mutations in genes encoding cardiac ion channels (primarily potassium and sodium channels), leading to delayed ventricular repolarization. This delay creates an electrophysiological substrate vulnerable to early afterdepolarizations (EADs), which can trigger a lethal polymorphic ventricular tachycardia known as Torsades de Pointes (TdP).

**Subtypes**
The three most common congenital subtypes account for ~90% of genotyped cases and exhibit distinct ECG morphologies:
*   **LQT1:** Impaired slow delayed rectifier potassium current ($I_{{Ks}}$).
*   **LQT2:** Impaired rapid delayed rectifier potassium current ($I_{{Kr}}$).
*   **LQT3:** Enhanced late inward sodium current ($I_{{Na}}$).

**Prevalence and Demographics**
Congenital LQTS affects approximately 1 in 2,000 to 1 in 2,500 individuals. It is often diagnosed in children, adolescents, or young adults who are otherwise structurally healthy. Acquired LQTS (drug-induced or electrolyte-driven) is significantly more common in general clinical populations.

---

## 2. Signal Manifestation

The following quantitative parameters describe the manifestation of LQTS in **Lead I** and **Lead II**.

### Rhythm and Cycle Organization
*   **Underlying Rhythm:** Typically Normal Sinus Rhythm (NSR).
*   **RR Interval:** 600 ms to 1200 ms (50–100 BPM). *Note: LQT1 patients frequently exhibit mild sinus bradycardia (RR > 1000 ms).*
*   **Cycle Dynamics:** LQTS patients often exhibit impaired QT adaptation to heart rate changes. When the RR interval shortens (tachycardia), the QT interval fails to shorten appropriately.

### Intervals
*   **PR Interval:** Normal (120–200 ms).
*   **QRS Duration:** Normal (80–100 ms).
*   **QT Interval:** Absolute QT is prolonged, typically > 450 ms.
*   **QTc (Corrected QT):** The defining metric. Calculated via Bazett’s formula ($QTc = QT / \sqrt{{RR_{{sec}}}}$).
    *   Normal: < 440 ms (men), < 450 ms (women).
    *   **LQTS Target:** 460 ms to 600 ms.

### Morphology (P, QRS, T)
*   **P-wave:** Normal (smooth, monophasic, positive in Leads I and II).
*   **QRS Complex:** Normal morphology (narrow, sharp).
*   **T-wave:** Morphology is highly subtype-dependent and serves as a primary synthesis target:
    *   *LQT1:* Broad-based, symmetrical, and unusually tall T-wave. The T-wave occupies the entire ST segment.
    *   *LQT2:* Low amplitude, notched (bifid) T-wave. The notch may appear on the ascending limb, peak, or descending limb.
    *   *LQT3:* Late-onset T-wave. The ST segment is isoelectric and abnormally long, followed by a narrow, peaked T-wave.

### Amplitude and Polarity
*   **Lead II (Primary Diagnostic Lead):**
    *   R-wave: 1.0 to 2.5 mV.
    *   T-wave (LQT1): 0.4 to 0.7 mV (tall).
    *   T-wave (LQT2): 0.1 to 0.25 mV (low amplitude, notched).
    *   T-wave (LQT3): 0.3 to 0.5 mV (peaked).
*   **Lead I:**
    *   Generally mirrors Lead II but with approximately 20–30% lower amplitudes due to the typical cardiac axis (+60 degrees). Notching in LQT2 is often visible in Lead I but is most prominent in Lead II.

### Beat-to-Beat Variability
*   **T-Wave Alternans (TWA):** A critical marker of electrical instability in severe LQTS. Manifests as a beat-to-beat alternation in the amplitude, morphology, or polarity of the T-wave.
    *   *Microvolt TWA:* 10–50 µV difference between consecutive T-waves (requires spectral analysis).
    *   *Macroscopic TWA:* > 100 µV (up to 0.5 mV) difference, visible to the naked eye.

---

## 3. Severity and Variability

### Condition Severity
*   **Mild/Borderline:** QTc 450–470 ms. T-wave morphology may appear near-normal.
*   **Moderate:** QTc 480–500 ms. Subtype-specific T-wave morphologies become obvious.
*   **Severe:** QTc > 500 ms. High risk of TdP. Macroscopic T-wave alternans may be present.

### Sources of Natural Variability
*   **Subject-to-Subject:** Genotype dictates the fundamental T-wave shape (broad vs. notched vs. late-onset).
*   **Beat-to-Beat:** QT interval fluctuates slightly with respiration (Respiratory Sinus Arrhythmia), but in LQTS, the repolarization lability is exaggerated.
*   **Acquisition-Related:** Baseline wander (typically < 0.5 Hz) can severely obscure the end of the T-wave, making the exact measurement of the QT interval difficult.

---

## 4. Key Discriminators

### Top Discriminative Features
1.  **QTc Prolongation:** The absolute primary discriminator. A QTc > 480 ms in the absence of bundle branch block is highly specific for LQTS.
2.  **T-Wave Morphology:** The presence of a distinct notch (LQT2) or a prolonged isoelectric ST segment (LQT3) distinguishes LQTS from normal variants.
3.  **Macroscopic T-Wave Alternans:** Rarely seen in normal physiology; highly specific to severe repolarization disorders.

### Differential Diagnosis Table

| Feature | Long QT Syndrome | Normal Sinus Rhythm | Hypokalemia | Short QT Syndrome |
| :--- | :--- | :--- | :--- | :--- |
| **QTc Interval** | **> 460 ms** (often > 500 ms) | 360 – 440 ms | Normal (appears long due to U-wave) | < 340 ms |
| **T-Wave Shape** | Broad, notched, or late-onset | Asymmetric, smooth | Flattened or inverted | Tall, narrow, peaked |
| **ST Segment** | Prolonged (esp. LQT3) | Normal | ST depression | Extremely short / absent |
| **U-Wave** | Usually absent (or fused in LQT2) | Small (< 25% of T-wave) | **Prominent** (fuses with T-wave) | Absent |
| **T-Wave Alternans**| Present in severe cases | Absent | Absent | Absent |

---

## 5. Synthesis Considerations

### Recommended Computational Approach
To synthesize LQTS in the ECG domain, DSP engineers should utilize a dynamical model based on a sum of Gaussian functions (e.g., the McSharry model). In this framework, the ECG cycle is modeled as a trajectory in a 3D state space, where P, Q, R, S, and T waves are generated by Gaussian attractors placed at specific angles ($\theta$) and with specific widths ($b$) and amplitudes ($a$).

To model LQTS, the parameters of the **T-wave Gaussian(s)** must be manipulated.

### Concrete Parameter Ranges (McSharry-style Model)

Assuming a normalized cardiac cycle from $-\pi$ to $+\pi$ (where R-peak is at $\theta = 0$):

*   **Normal T-wave (Reference):**
    *   Center ($\theta_T$): ~ 1.2 rad
    *   Width ($b_T$): ~ 0.3 rad
    *   Amplitude ($a_T$): ~ 0.4 mV (Lead II)

*   **LQT1 Synthesis (Broad T-wave):**
    *   Increase width ($b_T$): **0.5 to 0.6 rad**.
    *   Shift center slightly later ($\theta_T$): **1.3 rad**.
    *   Amplitude ($a_T$): **0.5 to 0.6 mV**.

*   **LQT2 Synthesis (Notched T-wave):**
    *   *Requires TWO Gaussians for the T-wave ($T_1$ and $T_2$).*
    *   $T_1$ Center ($\theta_{{T1}}$): **1.1 rad**, Amplitude: **0.2 mV**, Width: **0.15 rad**.
    *   $T_2$ Center ($\theta_{{T2}}$): **1.4 rad**, Amplitude: **0.15 mV**, Width: **0.15 rad**.
    *   This creates the characteristic "bifid" peak.

*   **LQT3 Synthesis (Late-onset T-wave):**
    *   Shift center significantly later ($\theta_T$): **1.6 to 1.8 rad**.
    *   Keep width narrow ($b_T$): **0.2 rad**.
    *   This naturally creates the long isoelectric ST segment.

*   **T-Wave Alternans (TWA) Implementation:**
    *   Apply an amplitude modulation factor to the T-wave Gaussian: $a_T(i) = a_{{T\_base}} + (-1)^i \cdot \Delta a$, where $i$ is the beat index and $\Delta a$ is **0.05 to 0.15 mV**.

### Common Pitfalls and How to Avoid Them
1.  **Static QT Intervals during HR Variation:**
    *   *Pitfall:* Generating a fixed QT interval of 500 ms while the simulator varies the RR interval from 600 ms to 1000 ms.
    *   *Solution:* Implement a dynamic QT-RR relationship. Use Bazett's formula in reverse: $QT_{{target}} = QTc_{{target}} \times \sqrt{{RR_{{sec}}}}$. Ensure the $QTc_{{target}}$ remains constant (e.g., 500 ms) while the absolute QT scales with the RR interval.
2.  **Widening the QRS Complex:**
    *   *Pitfall:* Accidentally stretching the entire cardiac cycle, resulting in a QRS > 120 ms.
    *   *Solution:* Isolate the time-stretching strictly to the ST segment and T-wave. The P-wave and QRS complex parameters must remain fixed to normal values.
3.  **Unrealistic T-wave Ends (Sharp cutoffs):**
    *   *Pitfall:* The prolonged T-wave extends beyond the end of the simulated cardiac cycle, causing a discontinuity before the next P-wave.
    *   *Solution:* Ensure the Gaussian tails decay to zero smoothly. If simulating severe bradycardia with extreme LQTS, verify that the T-wave completes at least 100 ms before the subsequent P-wave begins.
