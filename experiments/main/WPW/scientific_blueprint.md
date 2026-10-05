# Scientific Blueprint: WPW Syndrome (ECG)

## 1. Condition Overview

**Definition & Pathophysiology**
Wolff-Parkinson-White (WPW) Syndrome is a congenital cardiac pre-excitation syndrome characterized by the presence of an abnormal accessory electrical conduction pathway (the Bundle of Kent) between the atria and the ventricles. This pathway bypasses the atrioventricular (AV) node, which normally delays electrical conduction. Consequently, ventricular depolarization begins earlier than normal (pre-excitation) and occurs via slow cell-to-cell myocardial conduction before the normal His-Purkinje system takes over. 

**Clinical Significance**
While the baseline ECG is abnormal, the primary clinical risk is the propensity for re-entrant tachyarrhythmias, specifically Atrioventricular Reentrant Tachycardia (AVRT), and a risk of rapid ventricular rates during Atrial Fibrillation (AFib), which can degenerate into Ventricular Fibrillation. 

**Subtypes**
*   **Manifest WPW:** Shows the classic baseline ECG pattern (short PR, delta wave).
*   **Concealed WPW:** The accessory pathway only conducts retrogradely; baseline ECG is normal (Not the target of this blueprint).
*   **Intermittent WPW:** The pre-excitation pattern appears and disappears on a beat-to-beat or day-to-day basis.

**Demographics**
Prevalence is approximately 0.1% to 0.3% in the general population. It is more commonly diagnosed in young, otherwise healthy individuals (males > females).

---

## 2. Signal Manifestation

The following parameters describe the baseline sinus rhythm of a patient with manifest WPW Syndrome, specifically focusing on **Lead I** and **Lead II**.

### Rhythm and Cycle Organization
*   **Underlying Rhythm:** Normal Sinus Rhythm (NSR).
*   **Heart Rate:** 60–100 bpm (1.0–1.66 Hz).
*   **RR Interval:** 600–1000 ms. Regular, subject to normal respiratory sinus arrhythmia (RSA).

### Morphological Features
*   **P-wave:** Normal morphology. 
    *   *Duration:* 80–110 ms.
    *   *Amplitude:* +0.1 to +0.2 mV in Leads I and II.
*   **Delta Wave (The Hallmark):** A slurred, low-frequency initial upstroke of the QRS complex.
    *   *Duration:* 30–50 ms.
    *   *Amplitude:* +0.1 to +0.4 mV.
    *   *Slope ($dv/dt$):* Significantly lower than a normal QRS upstroke. The frequency content of the delta wave is concentrated in the 10–20 Hz band, compared to the 40+ Hz band of a normal R-wave.
*   **QRS Complex:** Widened due to the fusion of the pre-excited myocardium (delta wave) and the normally activated myocardium.
    *   *Total Duration:* > 120 ms (Target: 120–160 ms).
    *   *Amplitude:* Often increased. R-wave peaks can reach +1.5 to +2.5 mV in Lead II.
*   **ST Segment and T-wave (Secondary Repolarization Changes):** 
    *   Because depolarization is abnormal, repolarization is also abnormal. 
    *   *Rule of Discordance:* The ST segment and T-wave vector are typically directed *opposite* to the major delta/QRS vector. If the delta/QRS is strongly positive (as in Leads I and II), the ST segment will be slightly depressed (-0.05 to -0.1 mV) and the T-wave will be inverted or biphasic.

### Interval Specifications
*   **PR Interval:** **< 120 ms** (Target: 80–110 ms). Measured from the onset of the P-wave to the onset of the delta wave.
*   **QRS Interval:** **> 120 ms** (Target: 120–160 ms).
*   **QT/QTc Interval:** Usually normal (360–440 ms), though the widened QRS can slightly prolong the absolute QT interval.

### Lead Relationships (Lead I vs. Lead II)
Assuming a typical left-sided or right-sided free wall accessory pathway:
*   **Lead I:** Positive P-wave, positive delta wave, tall R-wave, inverted/flat T-wave.
*   **Lead II:** Positive P-wave, positive delta wave (often more pronounced than Lead I), very tall R-wave (axis is typically +60 degrees, aligning with Lead II), inverted T-wave.

---

## 3. Severity and Variability

### Severity / Degree of Pre-excitation
The "severity" of the WPW ECG pattern depends on the competition between the AV node and the accessory pathway.
*   **High Pre-excitation (e.g., high vagal tone slowing the AV node):** The accessory pathway dominates. The PR interval becomes extremely short (e.g., 80 ms), the delta wave is highly prominent, and the QRS is very wide (e.g., 160 ms).
*   **Low Pre-excitation (e.g., high sympathetic tone accelerating the AV node):** The AV node conducts faster, minimizing the fusion beat. The PR interval approaches normal (110–120 ms), the delta wave is subtle, and the QRS is only mildly widened (110–120 ms).

### Sources of Natural Variability
*   **Beat-to-Beat Variability:** In *Intermittent WPW*, the DSP can model sudden toggling. Beat $N$ may have a PR of 90 ms and QRS of 140 ms, while Beat $N+1$ has a PR of 140 ms and QRS of 90 ms (normal conduction).
*   **Heart Rate Dependence:** At higher heart rates (e.g., exercise), sympathetic tone increases AV node conduction, often causing the delta wave to shrink or disappear entirely.

---

## 4. Key Discriminators

To ensure the simulated signal is not confused with other pathologies, the DSP must strictly adhere to the triad of WPW: **Short PR + Delta Wave + Wide QRS**.

### Top Discriminative Features
1.  **PR Interval:** Must be strictly < 120 ms. This distinguishes WPW from Bundle Branch Blocks (which have wide QRS but normal PR).
2.  **Initial QRS Slope:** The first 30-50 ms of the QRS must have a low $dv/dt$ (slurring), followed by a sharp high $dv/dt$ peak. This distinguishes WPW from Premature Ventricular Contractions (PVCs), which are uniformly wide and slurred throughout.
3.  **P-to-QRS Relationship:** Every QRS must be preceded by a P-wave. This distinguishes WPW from ventricular ectopic rhythms.

### Differential Diagnosis Table

| Feature | WPW Syndrome | Left Bundle Branch Block (LBBB) | Premature Ventricular Contraction (PVC) | Normal Sinus Rhythm |
| :--- | :--- | :--- | :--- | :--- |
| **PR Interval** | **< 120 ms** | 120–200 ms (Normal) | N/A (No preceding P-wave) | 120–200 ms |
| **QRS Duration** | **> 120 ms** | > 120 ms | > 120 ms | 80–110 ms |
| **QRS Onset** | **Slurred (Delta wave)** | Sharp/Rapid | Slurred/Variable | Sharp/Rapid |
| **P-wave Presence** | **Yes (1:1 ratio)** | Yes (1:1 ratio) | No (Dissociated/Absent) | Yes (1:1 ratio) |
| **ST-T Discordance** | **Yes** | Yes | Yes | No (Concordant) |

---

## 5. Synthesis Considerations

### Recommended Computational Approach
A piecewise dynamical model (such as a modified McSharry model using a sum of Gaussian functions) is highly recommended. Standard ECG simulators use 5 Gaussians (P, Q, R, S, T). To simulate WPW, **introduce a 6th Gaussian** specifically for the delta wave, positioned between the P and R waves.

### Concrete Parameter Ranges for DSP

| Parameter | Target Value / Range | DSP Implementation Notes |
| :--- | :--- | :--- |
| **Heart Rate ($f_{{HR}}$)** | 1.2 Hz (72 bpm) | Add 0.05 Hz low-frequency noise for RSA. |
| **P-wave Center ($t_P$)** | -0.15 s (relative to R-peak) | Standard Gaussian, width ($\sigma$) $\approx$ 0.02 s. |
| **Delta Wave Center ($t_\Delta$)** | -0.06 s (relative to R-peak) | **Crucial:** Width ($\sigma$) $\approx$ 0.025 s. Amplitude: +0.2 to +0.4 mV. |
| **R-wave Center ($t_R$)** | 0.00 s | Sharp Gaussian, width ($\sigma$) $\approx$ 0.01 s. Amplitude: +1.5 mV. |
| **T-wave Center ($t_T$)** | +0.25 s (relative to R-peak) | Width ($\sigma$) $\approx$ 0.04 s. **Amplitude:** -0.2 to -0.4 mV (Inverted). |
| **Baseline Wander** | 0.1–0.5 Hz | Apply standard respiratory baseline wander. |

*Note: To achieve the "fusion" look, the trailing edge of the Delta Gaussian must overlap heavily with the leading edge of the R-wave Gaussian.*

### Common Pitfalls and How to Avoid Them

1.  **Pitfall: Sharp Delta Wave.** 
    *   *Issue:* If the delta wave is modeled with high-frequency components, it will look like a premature Q-wave or a notched R-wave, not pre-excitation.
    *   *Solution:* Ensure the delta wave Gaussian has a wider standard deviation ($\sigma$) than the R-wave, acting as a low-pass filtered ramp-up to the main R-peak.
2.  **Pitfall: Normal T-waves.** 
    *   *Issue:* Adding a delta wave to a completely normal ECG template results in concordant, upright T-waves. This is physiologically inaccurate and will be flagged as synthetic by cardiologists.
    *   *Solution:* Hardcode ST-T discordance. If the simulated QRS integral is highly positive, force the ST segment to depress by ~0.1 mV and invert the T-wave.
3.  **Pitfall: PR Interval Too Long.**
    *   *Issue:* If the delta wave is simply added to the *start* of a normal QRS without moving the P-wave closer, the PR interval will remain normal (e.g., 160 ms). 
    *   *Solution:* The distance from the start of the P-wave to the start of the Delta wave must be strictly constrained between 80 ms and 110 ms. Shift the P-wave closer to the QRS complex in the time domain.
