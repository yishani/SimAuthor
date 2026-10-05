# Revision audit

This directory contains the material for the paper's code-revision audit.

- `audit_prompt.md`: semantic labelling instructions.
- `lineage_audit.csv`: one row per revision on the selected lineage.
- `structural_revision_summary.csv`: paper-level structural versus
  parameter-edit summary.
- `lineage_audit.md`: human-readable audit report.

Intermediate extraction records, model responses, and large per-step unified
diffs are omitted. The final audit can be checked against the candidate programs
and parent links retained under `experiments/main/`.
