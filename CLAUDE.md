# Project context for Claude Code

Max's individual **Assignment I** for *Business Process Prediction, Simulation, and Optimization* (TUM, chair I17, Henryk Mustroph / Prof. Rinderle-Ma). **Master's course.** Submission: **19 Nov 2026, 23:59, Moodle.** Counts 25 % of the grade.

Deliverables: report PDF in the TUM LaTeX template (two-column, max 5 pages excl. references and appendix, English) + this public repo (link in the report).

Local, not versioned: `data/BPI_Challenge_2017.xes.gz` (event log) and `docs/` (assignment PDF, lecture slides, LaTeX template).

## Rules that constrain how Claude helps

- GenAI: the professor relaxed the writing restriction from the assignment PDF, so Claude may help with report text. Max writes the report text himself. Keep the GenAI usage declaration in the report.
- Footnote every Python package/tool with a link; cite scientific literature properly.
- Reproducibility: fix random seeds for data splits, document all ML parameters.
- If a commercial tool computes quality metrics, a Python reimplementation is required.

## Tasks (assignment PDF, Oct 16 2026 + lecture slides)

0. **Abstract:** task, log, findings, main results in 4-5 sentences.
1. **Intro:** domain, origin and purpose of the log, existing BPIC'17 reports (state of the art + limitations). Source: https://ais.win.tue.nl/bpi/2017/challenge.html
2. **3.1 Technical setup:** libraries, visualization, modeling tools (footnotes).
3. **3.2 Simple statistics** - DONE (`src/stats.py`, `results/basic_stats.tex`). Required table plus at least 2 extra statistics; also typical vs long vs short cases.
4. **3.3 Process model:**
   - At least 2 discovery algorithms, models in the appendix.
   - All PM4Py quality metrics, plus at least 2 self-implemented simplicity metrics.
   - **Compare token-based replay vs alignments** (fitness and precision, including runtimes). Find a concrete trace where token replay is misled (silent transitions / a deviation causing follow-up missing/remaining tokens) while the alignment explains it with one cheap move.
   - Final model replays ~80 % of cases (report % of fitting traces AND the fitness value; justify the interpretation).
   - Trade-off discussion along slide 32 (Process Mining II) and slide 36: academic vs consulting/automation perspective. **Grading focuses on why metrics were prioritized.** Professor: precision > generalization (discrete, ordered behaviour space), but still report generalization (unseen decision combinations in simulation). Simplicity: discuss quantitatively and qualitatively.
   - Keep an **iteration log**: one row per model version (change made -> fitness / precision / generalization / simplicity). This table is the "step by step" design story the graders want.
   - Model multiple offers per application as a loop around the O_ block (O_Create Offer repeats in ~27 % of cases).
   - Final BPMN 2.0 remodeled block-structured (SAP Signavio modeler or bpmn.io), exported as BPMN XML, **re-imported into PM4Py and re-evaluated** so the metrics belong to exactly that model. Task names must match log activity labels. Put the BPMN in the report as a PDF figure.
   - **Order matters:** do the Signavio remodel BEFORE decision mining. If the remodel changes the structure, the XORs (decision points) change too.
   - Alignments are the compute bottleneck: develop on a sample or the top variants, run the full log only for final numbers; use timeouts/multiprocessing.
   - PM4Py may need local patches (slide 37); document any patch in the repo.
5. **Decision mining (mandatory for Master's):** alignment-based (de Leoni & van der Aalst 2013), a decision block at every XOR of the final model, rules visualized on the BPMN (text annotations at the gateways).
   - Motivation (professor): XORs introduce uncertainty; a simulation needs a branching rule at each one.
   - PM4Py's built-in decision mining uses token replay, so implement the alignment-based extraction ourselves: align each trace, find the branch taken after each decision place (next visible sync/model move), label = branch.
   - Features: case attributes (LoanGoal, ApplicationType, RequestedAmount) + event/offer attributes known before the decision (CreditScore, OfferedAmount, MonthlyCost, NumberOfTerms, ...); optionally long-tail features from slide 48 (last k resources, elapsed/waiting times, counts). Fixed seed, held-out split, shallow trees.
   - **Slide 52 (mandatory):** compare decision trees vs neural networks + XAI (Park et al. 2022). Plan: per decision point, a shallow DT (rules), an MLP + SHAP, and a surrogate DT on the MLP predictions. Report accuracy, surrogate fidelity, and agreement between SHAP features and tree splits; argue explainability vs accuracy.
6. **Tooling rules:** list every commercial tool used and include screenshots of results (also for Signavio).
7. **3.4 Advanced:** Option 1 (SAP Signavio Process Mining Suite) only if Max has Process Intelligence access (the academic version is excluded). Otherwise Option 2 - candidate: drift analysis (distributional vs concept drift), motivated by the arrival jump around June 2016.

## Findings so far (from `results/extra_stats.json`)

- 31,509 cases, 1,202,267 events, 15,930 variants, 26 activities, 149 resources; case length 38.16 ± 16.72 events; duration 21.90 ± 13.17 days.
- Categorical event attributes: 7 (Action, org:resource, concept:name, EventOrigin, lifecycle:transition, Accepted, Selected) + 2 ID attributes (EventID, OfferID), counted separately.
- Variants needed for 80 % of cases: raw 9,629, lifecycle=complete 610, **A_+O_ complete 65**. This is the main argument for event abstraction (`figures/variant_coverage.pdf`). Keep the trade-off in mind: W_ events carry resources and processing times, which simulation needs.
- Outcomes: A_Pending 17,228, A_Cancelled 10,431, A_Denied 3,752, none 98 (97 of these were active in the last 30 days, i.e. incomplete; filter them and say so).
- Cancelled cases spike at ~31 days, which suggests an automatic cancellation timer. Check the reports; if confirmed, model it as a timer event, and decision mining should rediscover it.
- Weekly arrivals jump around June 2016 (~500 to ~700/week); all cases start in 2016, the log ends 1 Feb 2017.
- Slowest decile: ~1.9 offers per case vs 1.2 in the fastest decile; mostly cancelled.

## Page budget (5 pages)

Abstract + intro ~0.6, setup ~0.3, statistics ~0.6, process model ~2, decision mining ~0.6, advanced ~0.8. Everything else goes to the appendix (discovered models, extra figures). Draft intro and setup while the analysis runs, not at the end.

## Plan

| Dates | Work |
|---|---|
| Nov 4 | DONE: repo, loading, statistics, figures |
| Nov 5-6 | Read reports, draft intro, long/short case analysis |
| Nov 7-9 | Discovery (IM, IMf, Heuristics), all metrics with TBR + alignments + runtimes, own simplicity metrics (e.g. size, CFC/CNC) |
| Nov 10-11 | Iterate final model, Signavio remodel, re-import, verify |
| Nov 12-13 | Decision mining (DT vs MLP+SHAP) |
| Nov 14-15 | Advanced analysis |
| Nov 16-17 | Writing, trade-off section, declaration, README |
| Nov 18-19 | Buffer |

## Conventions

- Scripts in `src/`, one per report section; tables in `results/`, figures in `figures/` (PDF for the report).
- Report: LaTeX in `report/` (TUM template, `latexmk` via `report/latexmkrc`), built by LaTeX Workshop into `report/build/`; it reads `../figures/` and `../results/` directly.
- Python env: `.venv` is a symlink to `~/.venvs/bpic17` (venv refuses paths containing `:`).
- `python src/load.py` caches the log as `data/bpic17.parquet`; other scripts call `load_log()`.
- Figures: one-column width 3.4 in, serif font, validated categorical palette `#2a78d6 #eb6834 #1baf7a #eda100` (fixed order).
- Max prefers complete, runnable scripts and concise prose; explain concepts with concrete numbers first.
