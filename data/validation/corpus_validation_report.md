# Europeana Corpus Validation Report

## Purpose

This report validates the current working corpus before it is fixed
for comparative single-agent and multi-agent RAG evaluation.

This audit does not alter the corpus. It reports corpus composition,
text coverage, possible duplicates, and a reproducible manual-review
sample.

## Corpus selection

- Raw exploratory records: 300
- Records with English descriptive metadata: 265
- Candidate records before short-story removal: 230
- Records retained in the current working corpus: 220
- Selection rule used for the retained corpus: English descriptive metadata, rights information, and at least 40 words in the main narrative field.

## Main story text

- Minimum story length: 40 words
- 25th percentile: 75.75 words
- Median: 122.5 words
- 75th percentile: 191.0 words
- Maximum: 934 words

## Materials text

- Records with any materials text: 200
- Records with at least 5 materials words: 179
- Records with at least 20 materials words: 114
- Median materials length, where present: 23.0 words

The materials field requires manual review because some records may
contain detailed descriptions of diaries, letters, photographs, medals
or albums, while others may contain only short inventory labels.

## Integrity checks

- Missing title: 0
- Missing story text: 0
- Missing rights field: 0
- Missing Europeana URL: 0
- Possible high-similarity duplicate pairs: 1

## Manual validation sample

A fixed random sample of 12 records was created across short,
medium and long story-text bands. It should be reviewed for:

1. Whether the record can support evidence-based questions.
2. Whether the English text is sufficiently coherent.
3. Whether the materials description contributes distinct information.
4. Whether there are any obvious factual, translation or metadata red flags.

The sample is saved as:

`data/validation/manual_review_sample.csv`

## Decision status

The corpus should only be treated as final after:

1. Reviewing the 12-record sample.
2. Inspecting any possible duplicate pairs.
3. Deciding whether the materials field supports a standalone specialist agent.
