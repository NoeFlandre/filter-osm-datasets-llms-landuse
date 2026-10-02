# Glossary

This page defines the project terms. Each term has one meaning in all the pages.

| Term | Meaning |
|---|---|
| Bucket | The private Hugging Face Bucket `NoeFlandre/landuse-filter-work`. It holds chunks, parts, manifests, index checkpoints and gates. |
| Chunk | A group of unique sentences that one job generates. A chunk id is the sha256 of the fingerprint and the sorted text hashes. |
| Controller | The `luf g5k run` process on the controller machine. It is the only writer of assignments. |
| Dedup | The method that generates each unique text one time and copies the label to each identical sentence. |
| Decision | The value in the `decision` column of `labels/`: `yes`, `no`, `failed`, `skipped_unsplit` or `pending`. |
| DSpark | The draft model that speeds up the generation of the main model. |
| Fingerprint | A hash of a configuration. `config_fingerprint` covers the settings that change the output. `serving_fingerprint` covers all the settings. |
| Frontend | The site machine of Grid'5000 where you run `oarsub` and `oarstat`. |
| Gate | The pre-registered non-inferiority test. A configuration or a GPU type must pass the gate before production use. |
| Generation | The raw model output for one unique text, with its token counts and provenance. |
| Label | The decision for one sentence position. |
| Namespace | A name that keeps the results of a candidate apart: `<fp>` (production), `<fp>-gpu-<key>` (admission), `<fp>-w<N>` (tuning). |
| OAR | The job scheduler of Grid'5000. |
| Part | A file of results that a job uploads to the bucket. Its name is the sha256 of its bytes. |
| Partial file | A published file that has some answers and some `pending` sentences. |
| Plan | The list of chunks for a dataset. The planner job writes it. |
| Revision | A pinned version of an input dataset. |
| Window | The value of `max_running_requests`. It is the number of concurrent requests on one GPU. |
| Walltime | The maximum run time of a job. |
