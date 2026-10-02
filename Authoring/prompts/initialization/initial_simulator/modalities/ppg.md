### Output Format
- File format: NumPy `.npy` files
- Array shape: `(n_samples,)` — single channel
- Sampling rate: {output_sr} Hz (synthesize internally at {synth_sr} Hz)
- Duration per sample: {duration} seconds
- Number of segments: generate exactly {num_samples} diverse segments
- Units: arbitrary; peak-normalize so maximum absolute value is approximately 1.0

### Output Directory and Naming
- Save samples to: `{output_dir}/`
- Create the output directory if it does not exist
- Filename: include the condition abbreviation and sample index
