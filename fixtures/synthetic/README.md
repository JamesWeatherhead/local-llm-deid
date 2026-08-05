# Synthetic fixture

Everything in this directory is **synthetic**. The three notes are short,
discharge-style documents written for this repository, and every identifier in
them is invented. They contain no real patient data and are not derived from the
study corpus. Their only purpose is to let the pipeline and scorer run
end-to-end offline: no GPU, no model weights, no PHI.

## Contents

- `notes/TEST001.txt`, `TEST002.txt`, `TEST003.txt`: the synthetic notes.
- `gold/TEST00N.json`: gold annotations in the same PubAnnotation shape as the
  study gold (reference `text`, typed character `denotations`, `identifier_type`
  attributes).
- `name_gazetteer.txt`: the invented names, passed to the offline test stub
  so it can match names (a pattern stub cannot recognise arbitrary names; a
  real language model needs no such list).
- `build_gold.py`: regenerates the gold files from the notes.

## Regenerating the gold

The gold files are checked in, but you can rebuild them from the notes and the
annotation list at the top of `build_gold.py`:

```bash
python3 fixtures/synthetic/build_gold.py
```

It annotates every occurrence of each listed literal, so if you edit a note,
keep each identifier intact on a single line.

## What the notes are designed to exercise

The fixture is deliberately uneven so a run produces a non-trivial score:

- **TEST001** and **TEST003** each name a facility and a city
  (`02_GEOGRAPHIC_SUBDIVISION`) in addition to the usual names, dates, ages,
  phone, email, and MRN.
- **TEST002** contains only identifiers the offline test stub can catch, so it
  comes out completely clean under the stub, a worked example of a
  zero-residual note.

Across the three notes the gold has 26 PHI spans / 335 PHI characters.

## Expected result under the offline test stub

The offline `StubExtractor` targets names (from the gazetteer), dates, ages over
89, phone numbers, email addresses, and MRNs, but **not** geographic
subdivisions. Running the stub pipeline and scoring it against this gold gives a
fixed, hand-checkable result:

| Metric (cumulative pass 2, all notes) | Value |
| --- | --- |
| Character recall | 0.755 (253 / 335) |
| Character precision | 1.000 (the stub only emits grounded literals) |
| Residual PHI characters | 82 (every one geographic) |
| Zero-residual notes | 1 of 3 (TEST002) |

The 82 leaked characters are exactly the facility and city names the stub does
not target, which is why per-type recall is 1.0 for every category it *does*
target and 0.0 for `02_GEOGRAPHIC_SUBDIVISION`. These values are asserted in
`tests/test_end_to_end.py`.

To reproduce:

```bash
PYTHONPATH=src python3 -m deid.run_model --model-id offline-stub \
    --notes-dir fixtures/synthetic/notes --out-dir /tmp/selftest \
    --protocol-dir protocol --offline-stub --name-file fixtures/synthetic/name_gazetteer.txt
PYTHONPATH=src python3 -m deid.metrics \
    --pred-dir /tmp/selftest --gold-dir fixtures/synthetic/gold --out-dir /tmp/selftest_metrics
```
