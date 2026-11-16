# Gates

Source: the owner's audit dated 2026-11-14, recorded in `gates.yaml`. The software reads that file; it does not
check licences. All current runs use SYNTHETIC data only.

| Item | Status | Why | Who acts | What unlocks it |
|---|---|---|---|---|
| data:fnspid | blocked | Conflicting licence statements; upstream publisher terms unread | owner | Written permission from the authors plus readable publisher terms |
| data:alpaca_benzinga | blocked | Terms unread; a credential would be needed | owner | Owner reads and records the terms; credential kept outside the repository |
| data:sec_edgar | not adopted | Filings are not news; it would change the question | owner and supervisor | A new research question and pre-registration |
| data:gdelt | rejected | No ticker mapping or headline text | owner | None planned |
| data:user_import | allowed with a licence note | Importing does not clear a licence | owner | A licence note and source name on every import |
| scorer:finbert_prosus | exploratory only | Licence permits non-commercial research; training dates unverified | owner | A documented training cutoff before the test period |
| scorer:finbert_tone | blocked | Licence unresolved | owner | A clear licence statement |
| scorer:lm_lexicon | exploratory only | Version not frozen before 2020 | owner | A dated pre-test-period version, supplied locally |
| scorer:demo_lexicon, scorer:mock_model | demo only | Synthetic demo tools | nobody | Not applicable |
| study:confirmatory | blocked | No licensed real data, no verified-clear scorer | owner and supervisor | Permitted data licence, verified-clear scorer, frozen pre-registration, frozen samples |
| storage:real_data_and_weights | blocked | Nothing real may be stored here | owner | Owner decision after licences are resolved |

Print the live table with `.venv/bin/python -m se.cli gates`. Confirmatory mode is refused in code with every reason listed.
