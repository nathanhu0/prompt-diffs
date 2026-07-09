"""Subliminal data-generation building blocks shared across methods.

Deliberately tiny: the ONLY shared pieces are the pure token helper
(`_common.truncate_ids_to_numbers`) and the drop-only Cloud filter
(`cloud_filter.accept`). Each generation method is a self-contained driver that
runs its own generate -> capture -> truncate -> write loop and writes rows via
`core.subliminal.data.write_rows`. There is intentionally no shared generation
harness / orchestrator.
"""
