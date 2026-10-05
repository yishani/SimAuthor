# Scientific Blueprint: {condition_name}

## Instructions
Produce a comprehensive Scientific Blueprint for **{condition_name}**
({signal_modality} signal).  This document will be the sole clinical
reference for a DSP team building a computational simulator.

## Modality Specification

{modality_specification}

## Required Sections

### 1. Condition Overview
- Definition, pathophysiology, and clinical significance.
- Key subtypes or severity gradations (if applicable).
- Prevalence and typical patient demographics.

### 2. Signal Manifestation
Describe in quantitative detail how {condition_name} manifests in the
**{signal_modality}** domain.  Follow the Modality Specification above
for which properties to characterise and at what level of detail.

### 3. Severity and Variability
- How the signal characteristics change with condition severity or
  disease stage.
- Sources of natural variability (subject-to-subject, beat-to-beat,
  recording-condition, acquisition-related, or other).

### 4. Key Discriminators
- The 3–5 most important features that distinguish this condition from
  similar conditions in the same modality.
- For each: typical range, how it differs from confounds.
- Include a differential diagnosis table comparing against 2–4 similar
  conditions.

### 5. Synthesis Considerations
- Recommended computational approach for reproducing this condition in
  the {signal_modality} domain.
- Concrete parameter ranges the DSP engineer should target.
- Common pitfalls and how to avoid them.

## Output Format
Write in clear, technical markdown.  Use tables for comparisons and
reference values.  Be SPECIFIC with numbers.  If you are uncertain
about a value, give your best estimate and mark it with (est.).
This document should be 800–1500 words — comprehensive but focused.
