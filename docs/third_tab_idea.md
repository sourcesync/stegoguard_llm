# Explorer: third tab idea

George's description, verbatim, from the project conversation:

> So lets think about a third tab and it allows you to first select a text from the original corpus (maybe it allows you to choose the domain and the iteration), then it allows you to choose the "message" and the technique to hide it (these options are of course correlated and depends on the structural viability), then it shows result of embedding the message and then that feeds into the detector result (depending on the detectors that were run). I'm imagine this is a top-down flow chart kind of reactive user interface experience.

Free choice of message is out of scope for now; the technique and message choices are limited to what the adaptive-replay experiment actually generated.

## Data available

`scripts/export_explorer_bundle.py` (as of this commit) adds a `cat` section to `outputs/explorer/explorer_bundle.json`, keyed by each of the 48 adaptive-replay source answers:

- `det` / `feat`: the source answer's own detector result and evaluator features.
- `cands`: all 578 encoded candidates for that answer (rank and synonym), with the payload, bit length, whether it decoded correctly, token count, encoding attempts, which replay rounds used it, and the per-candidate detector result and features. Synonym candidates also carry the word-level substitutions (slot id, bit, original word, substituted word, whether the text actually changed).
- `fails`: the 60 attempts that failed to encode, with the reason (`no_text_change`, `length_mismatch`, `tokenizer_roundtrip_failure`).

Counts and the 10-model voting rule have been checked against the frozen records, including that all 1,152 replay rounds match their catalogue entries.

## Status

Data layer done. The page for this tab has not been built.
