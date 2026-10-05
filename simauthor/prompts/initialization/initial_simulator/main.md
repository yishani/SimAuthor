# Initial Simulator

Construct a self-contained Python simulator for:

- **Clinical condition:** {condition_name}
- **Signal modality:** {signal_modality}

Simulate how the target condition manifests specifically in the requested signal modality. Use your biomedical and computational knowledge to identify the principal physiological or signal-generating mechanisms and implement them as an executable simulator.

The simulator should provide a simple but scientifically meaningful starting point for later refinement. It should:

- organize the synthesis into interpretable components;
- expose important physiological and simulation parameters explicitly;
- generate exactly **{num_samples}** diverse samples;
- represent plausible variation in severity, physiology, subjects, and acquisition where appropriate;
- avoid near-identical outputs that differ only through trivial noise;
- remain modular, robust, and easy to revise;
- use an explicit random seed for reproducibility.

## Modality Specification

{modality_specification}

## Output

Return only a complete executable Python script in one fenced `python` code block. Do not include any text outside the code block.

The script must run without manual modification as:

```bash
python script_000.py