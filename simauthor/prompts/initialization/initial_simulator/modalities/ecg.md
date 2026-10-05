### Output Format
- File format: NumPy `.npy` files
- Array shape: `(n_samples, 2)` — two leads
- Lead ordering: lead I at column 0, lead II at column 1
- Sampling rate: {output_sr} Hz (synthesize internally at {synth_sr} Hz)
- Duration per sample: {duration} seconds
- Number of segments: generate exactly {num_samples} diverse segments
- Units: millivolts (mV)
- Normalization: do not clip; preserve physiological amplitude range

### Output Directory and Naming
- Save samples to: `{output_dir}/`
- Create the output directory if it does not exist
- Filename: include the condition abbreviation and sample index
