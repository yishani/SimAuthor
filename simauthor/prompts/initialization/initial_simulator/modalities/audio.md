### Output Format
- File format: WAV
- Channels: mono
- Synthesize internally at {synth_sr} Hz, then resample to {output_sr} Hz before saving
- Duration per sample: {duration} seconds
- Number of samples: generate exactly {num_samples} diverse samples
- Normalization: peak-normalize to prevent clipping
- Use `scipy.io.wavfile` or `soundfile` to write output files

### Output Directory and Naming
- Save samples to: `{output_dir}/`
- Create the output directory if it does not exist
- Filename: include the condition abbreviation and sample index
