# AI Use Declaration (to be reviewed and confirmed by the team before submission)

## Tool
* **Claude Code** (Anthropic), used as an agentic coding and writing assistant in a cloud development container, November 2026.

## What the AI tool did
| Task | AI contribution | Human responsibility required |
|---|---|---|
| Requirements extraction from the template, AgriGuard report and the 24B / 26B review reports | drafted the requirement matrix | team to confirm interpretation |
| Dataset search and acquisition | located and downloaded DataCo (GitHub mirror) and SupplyGraph; recorded hashes, commits, licences | team to confirm licence terms with faculty |
| Synthetic data generator, feature pipeline, all experiments, simulator, MILP, decision engine | wrote the code and executed it | team to read, understand and be able to defend every line in the viva |
| Flask backend, MySQL schema, React frontend, AI assistant | wrote the code; ran tests and builds | team to review and run locally |
| Tests (pytest) | wrote and executed; results in `documentation/testing/TEST_REPORT.md` | team to re-run |
| IEEE paper, documentation, slides, viva Q&A | drafted text, tables (generated from result files) and figures | team to rewrite in their own words where required and verify each claim |
| References | proposed and verified bibliographic details via web search (DOIs where found) | team to open and read each cited source |

## What the AI tool did NOT do
* It did not fabricate data presented as real, invent results, or edit numbers by hand: all tables are produced by
  `ml/experiments/paper_tables.py` from saved outputs and checked by `ml/experiments/verify_results.py`.
* It did not execute the Gemini path of the assistant (no API key/network); this is stated as implemented-not-tested.

## Compliance note on the 20 % AI-content rule
The course template limits AI-generated content (text or code) to < 20 % as measured by Turnitin. **Most of the code and text in
this repository was drafted by an AI tool, so the submission as-is would not satisfy that rule.** The team must (a) confirm with the
faculty how the rule applies to source code and AI-assisted analysis, and (b) substantially rewrite the report text and review the
code themselves before submission. No attempt has been made to evade plagiarism or AI-detection tools, and no Turnitin score is
guaranteed.
