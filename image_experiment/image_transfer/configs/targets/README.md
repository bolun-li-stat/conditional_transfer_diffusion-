# Target-set files

Target classes and auxiliary candidates are selected by researchers before
running a study. The code validates and hashes that decision; it never chooses
classes from observed results.

Every runnable target-set file declares an ID, version, review/freeze flags,
and target entries with a name, synset, supercategory, selection rationale,
and fixed close/medium/far candidate lists. A `main` study requires a reviewed,
frozen file with at least four targets across multiple supercategories unless
an explicitly declared single-target exception is used.

`main_target_set.template.yaml` is intentionally incomplete and cannot pass the
main-study validation gate until a researcher fills and reviews it.
