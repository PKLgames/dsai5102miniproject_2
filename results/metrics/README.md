# Current numerical metrics

analysis.json contains the complete current analysis: descriptive variance and stability, sparse selection, held-out classification, cluster comparison and cross-window ranks. environment.json records the versions actually used. validation.json records the completed numerical checks. figure_confusion.json contains the held-out confusion counts used in the report.

The same JSON files remain in the results root for compatibility with the supplied current scripts. The metrics copies are byte-identical at delivery. The current scripts read the root files; after rerunning them, refresh these copies before sharing results. Old E1_summary.json through E6_summary.json from the earlier analysis are intentionally not retained because their methods and reported values differ.
