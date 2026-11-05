"""Local open-weight LLM clinical de-identification: pipeline and scoring.

Modules:
    pipeline   -- deterministic resolver, coordinate mapping, and redaction
    inference  -- llama-server client and an offline deterministic test stub
    run_model  -- drive one model over a note set (two sequential passes)
    metrics    -- re-derive character/span/note/reliability metrics from gold
    agreement  -- compare two independent annotation sets, type agnostically
    presidio_runner -- optional two-pass offset-based comparator
"""
