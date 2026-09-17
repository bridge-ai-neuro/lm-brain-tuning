# Data preparation

The pipeline starts from preprocessed fMRI responses. It does not reproduce
scanner-level NIfTI preprocessing.

## Sources

- Harry Potter: Wehbe et al. (2014), public dataset DOI
  [10.5061/dryad.gt413](https://doi.org/10.5061/dryad.gt413). The experiments
  use the corresponding preprocessed stimulus timing, ROI, and
  subject response arrays.
- Subset Moth: the reading subset of the public
  [Narratives reading/listening fMRI dataset](https://gin.g-node.org/denizenslab/narratives_reading_listening_fmri).

Review and follow each provider's terms before downloading or sharing data.

## Source descriptor

`prepare-data` consumes a JSON or YAML descriptor. All relative paths are
resolved relative to the descriptor itself. Required fields are:

- `dataset`: `hp` or `moth`;
- `subjects`: the paper subject order;
- `runs`: an integer NPY array with fold IDs 0 through 3;
- `responses`: subject-to-NPY mapping, each shaped `(samples, voxels)`;
- `language_masks`: subject-to-boolean-NPY mapping;
- `noise_ceilings`: subject-to-NPY mapping;
- either `contexts`, a JSONL file with `text` and five `chunks` per row, or a
  `stimulus` object containing `words`, `word_times`, and `fmri_times` arrays.

`prepare-data` consumes responses, run mappings, language masks, noise ceilings,
and stimulus arrays exported from the preprocessing environment. The resulting
canonical bundle can be copied to another machine and verified solely from
`manifest.json` and its checksums.

Legacy NumPy object arrays are rejected by default. If the provider's word
array uses object dtype, set `stimulus.trusted_object_arrays` to `true` only
after verifying the source. The generated canonical bundle never uses object
arrays or pickle.

Harry Potter contexts contain the latest 20 words in groups of four. Subset
Moth uses the latest 100 words while retaining at most five TR-aligned groups.
The exact subject lists, four-fold mapping, and five-group representation are
validated during import. Training and evaluation additionally verify that the
bundle's context parameters and subjects match the resolved paper config.

## Canonical bundle

The generated directory contains:

```text
manifest.json
contexts.jsonl
runs.npy
responses/sub-*.npy
language_masks/sub-*.npy
noise_ceilings/sub-*.npy
```

`manifest.json` records SHA-256 checksums for every generated input. The loader
checks sample counts, voxel dimensions, context groups, subject order, and fold
IDs before any training output is created.
