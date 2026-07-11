# Target-set guide

Target-set files are versioned researcher decisions. Each target records a name, synset, supercategory, selection rationale, and fixed close/medium/far auxiliary candidates. Canonical file content is hashed into grids, checkpoints, results, and pairing keys; changing a target or candidate list creates a different target-set identity.

Stage rules are deliberate:

- `smoke` is synthetic/offline plumbing only.
- `pilot` supports an engineering run and does not imply a reviewed scientific target set.
- `case_study` may use one target, but summaries must retain single-target scope.
- `main` normally requires an external reviewed and frozen file with at least four targets across more than one supercategory. The explicit one-target exception remains case-study scope and is not the default main design.

`targets/french_bulldog_pilot.yaml` is an engineering input and is currently `reviewed: false` and `frozen: false`. `targets/main_target_set.template.yaml` is empty, unreviewed, and unfrozen. Neither file currently authorizes a main study. This is intentional; code must not invent or approve targets.

To prepare a main set:

1. Copy the template to a versioned target-set filename.
2. Add at least four predeclared targets across multiple supercategories.
3. For every target, document its selection rationale and audit all close/medium/far candidate synsets without looking at transfer outcomes.
4. Have the designated researcher record the reviewer and review date, then set `reviewed: true` and `frozen: true`.
5. Point a reviewed main config copy at that file and preserve the file with the generated grid and resolved configuration.

The resolver rejects missing metadata, duplicate targets, incomplete auxiliary groups, too few main targets without the explicit single-target exception, and single-supercategory multi-target main sets. Grid generation also requires a matching passed release-pilot status. An explicit readiness override is recorded but never bypasses target-set validation.

Do not revise targets, auxiliary candidates, group labels, or exclusions after inspecting generated samples or metrics. If a scientific reason requires a change, increment the target-set version, regenerate hashes and grids, and analyze it as a distinct design.
