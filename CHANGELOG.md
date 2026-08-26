# Changelog

## 1.0.1 - 2026-08-25

Reliability and reproducibility release:

* makes the gold set, rather than produced output, define the scoring denominator;
* rejects stale/extra documents and malformed or mismatched prediction artifacts;
* fixes conditional validation denominators and separates format usability from finish reason;
* fails closed on reused model-output directories and malformed server responses;
* strips persisted model text by default, uses private output permissions, and requires explicit acknowledgement for raw-content retention or non-loopback endpoints;
* reproduces the manuscript's three-execution bootstrap preparation for recall and completeness;
* aligns Gemma-4 E2B/E4B checkpoint metadata and the resolving E2B GGUF repository with the manuscript; and
* pins GitHub Actions dependencies and expands the regression suite.

No study inference or analysis results were rerun or changed in this release.

## 1.0.0 - 2026-08-05

Initial public companion-code release.
