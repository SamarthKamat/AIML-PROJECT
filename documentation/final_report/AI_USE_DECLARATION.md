# AI Use Declaration

## Declaration used in the report (Section V)

> **AI assistance disclosure.** Generative AI (Claude Code, Anthropic; 12–13 Nov 2026) was used extensively as a development and writing
> assistant: under the authors' direction it generated most of the code, executed the experiments and drafted the text, and all
> repository commits were produced in AI-assisted sessions. The authors defined the problem, scope, datasets and evaluation
> requirements, reviewed the report and directed its revisions after faculty review. All authors take responsibility for the submitted
> work. The Gemini API is separately a functional component of the deployed assistant and was not used to write this report.

## Types of AI use

| Type | Tool | Role |
|---|---|---|
| Development assistant | Claude Code (Anthropic) | Generated most of the source code (data generator, experiments, simulator, MILP, statistics, Flask/React app) and executed the experiments. |
| Drafting and language | Claude Code (Anthropic) | Drafted the report, slides and viva notes, and revised them after each faculty or peer review. |
| Debugging and checking | Claude Code (Anthropic) | Debugged code, wrote the 73 verification checks and the 87-item report-number audit, and checked references against publisher and index records. |
| Functional component | Google Gemini API | Runtime only: phrases the in-app assistant's answers from tool outputs. Not used to write code or text. Not counted as authorship. |

## Evidence

- **Git history** (`git log`): 12 commits between 12 and 13 Nov 2026. Seven are attributed to "Supply Chain Black Box Team" and five to
  "Claude". All 12 were created in Claude Code sessions, and each carries a `Co-Authored-By: Claude` trailer. No commit was made outside
  an AI-assisted session.
- **Session records**: the experiments, figures, tables and report text were generated and revised in those sessions.
- **Human contributions on record**:
  - the original project brief: problem, scope, required experiments, honesty rules and deliverables;
  - team details;
  - two faculty/peer review documents and the revision requests based on them;
  - deployment configuration (Gemini key set by the user on Render).

## Why no percentage is stated

The available evidence does not support a claim that AI assistance was below 20% of the project work. On the git record it was the
majority of code and text production. The declaration therefore describes the actual nature and extent of the assistance and states no
percentage.

To genuinely reduce AI dependence before submission, the authors could:

1. Re-run `bash run_all.sh` themselves and check the outputs against `experiments/REPORT_NUMBER_AUDIT.md`.
2. Rewrite sections of the report in their own words from their own understanding.
3. Commit that work under their own git identities, so it appears in the history.

If they do, update this declaration to describe exactly that work. Do not change it otherwise.
