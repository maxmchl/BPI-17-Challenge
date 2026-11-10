# Project context for Claude Code

Max's individual **Assignment I** for *Business Process Prediction, Simulation, and Optimization* (TUM, chair I17, Henryk Mustroph / Prof. Rinderle-Ma). **Master's course.** Submission: **19 Nov 2026, 23:59, Moodle.** Counts 25 % of the grade.

Deliverables: report PDF in the TUM LaTeX template (two-column, max 5 pages excl. references and appendix, English) + this public repo (link in the report).

Local, not versioned: `data/BPI_Challenge_2017.xes.gz` (event log) and `docs/` (assignment PDF, lecture slides, LaTeX template).

## Rules that constrain how Claude helps

- GenAI: the professor relaxed the writing restriction from the assignment PDF, so Claude may help with report text. Max writes the report text himself. Keep the GenAI usage declaration in the report.
- Footnote every Python package/tool with a link; cite scientific literature properly.
- Reproducibility: fix random seeds for data splits, document all ML parameters.
- If a commercial tool computes quality metrics, a Python reimplementation is required.

## Professor's answers (class, Nov 6 2026) - these override earlier assumptions

- **Fitness target:** ~80 % from conformance checking is the baseline. Dropping towards 75 % is acceptable if argued with generalization. **No official precision threshold:** choose one and argue for it.
- **Lifecycle:** remove the lifecycle state for the modelling task (it is more or less irrelevant there); for simulation it is very important. The professor himself kept only the `complete` timestamps of W_ items. Caveat from our data: most W_ items never complete (see Findings), so `complete`-only nearly removes W_Call after offers. Justify whichever W_ representation we use.
- **Bank vs customer:** only W_ events are executed by the bank; A_ and O_ are states the bank does not directly control. Determining which events are executed by the bank is "important and cool" for this assignment. Show it with swimlanes (bank vs customer) in Signavio at the end; PM4Py cannot model swimlanes.
- **Iterations in the data, then rediscover.** No manual editing of the Petri net. The final drawn model may come from Signavio, but the model the quality metrics are computed on must be in the code.
- **Full log vs train/test:** computing fitness on the whole log is fine for representativeness, but the process was recorded over a long time and may have changed (governance etc.). Discuss this; a temporal split is a good way to show it.
- **Alignments:** use multiprocessing (multiple cores) for speed. The professor expects the computation not to take very long in the end.
- **Incomplete variants:** some variants never complete. Argue whether to include them and why (real-life log with errors and human involvement). We exclude the 98 cases without an outcome.
- **Cases TBR cannot handle:** the final model may fit ~80 %; some cases cannot be used with token replay but can with alignments. It is acceptable to argue for removing such cases for simplicity.
- **Decision mining:** do it on the final Petri net, after the metrics work is finished. Decision types: XOR (real options a or b), self-loops (can repeat indefinitely), loop decisions (repeat a subset). Counting loops as XOR or as a separate type is a definition choice we must argue. Expect no good rules for 80-90 % of decisions: find 2-3 decisions with really good rules and explain why the others do not work. For decisions without good rules, report the branching probabilities ("nice touch", also simulation input).
- **Advanced analysis:** a designated SAP Signavio account (with process mining / decision mining) arrives in the week of Nov 9, so plan for Option 1.

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
   - Keep an **iteration log** (`results/iterations.csv`): one row per model version (data change + rediscovery -> fitness / precision / generalization / simplicity). This table is the "step by step" design story the graders want. Planned steps: base IMf 0.10 on log A -> merge vs keep the two O_Sent labels -> log B with W_ work items -> optional: collapse direct repeats / filter non-alignable variants.
   - Multiple offers per application need a loop around the whole offer block (27 % of cases). IMf keeps that loop up to noise 0.10 and filters it from 0.15 on (see Findings).
   - Temporal generalization check: discover on cases started Jan-Jun 2016, evaluate on Jul-Dec.
   - Final BPMN 2.0 remodeled block-structured (SAP Signavio modeler or bpmn.io), exported as BPMN XML, **re-imported into PM4Py and re-evaluated** so the metrics belong to exactly that model. Task names must match log activity labels. Put the BPMN in the report as a PDF figure.
   - **Order matters:** do the Signavio remodel BEFORE decision mining. If the remodel changes the structure, the XORs (decision points) change too.
   - Alignments are the compute bottleneck: develop on a sample or the top variants, run the full log only for final numbers; use timeouts/multiprocessing.
   - PM4Py may need local patches (slide 37); document any patch in the repo.
5. **Decision mining (mandatory for Master's):** alignment-based (de Leoni & van der Aalst 2013) on the final Petri net, rules visualized on the BPMN (text annotations at the gateways). Per the professor: classify decision points (XOR / self-loop / loop decision), find 2-3 with good rules, give branching probabilities for the rest and explain why no rule works. Candidates: channel online vs bank (ApplicationType = Limit raise -> bank); outcome after validation (Pending / Denied / Incomplete); return vs cancel after A_Complete is driven by the 30-day timer, so expect no attribute rule.
   - Motivation (professor): XORs introduce uncertainty; a simulation needs a branching rule at each one.
   - PM4Py's built-in decision mining uses token replay, so implement the alignment-based extraction ourselves: align each trace, find the branch taken after each decision place (next visible sync/model move), label = branch.
   - Features: case attributes (LoanGoal, ApplicationType, RequestedAmount) + event/offer attributes known before the decision (CreditScore, OfferedAmount, MonthlyCost, NumberOfTerms, ...); optionally long-tail features from slide 48 (last k resources, elapsed/waiting times, counts). Fixed seed, held-out split, shallow trees.
   - Leakage: CreditScore, Selected and Accepted are recorded at O_Create Offer but reflect later outcomes (van der Ham 2017); use only features known before the decision.
   - **Slide 52 (mandatory):** compare decision trees vs neural networks + XAI (Park et al. 2022), only on the 2-3 good decision points. Plan: per decision point, a shallow DT (rules), an MLP + SHAP, and a surrogate DT on the MLP predictions. Report accuracy, surrogate fidelity, and agreement between SHAP features and tree splits; argue explainability vs accuracy.
6. **Tooling rules:** list every commercial tool used and include screenshots of results (also for Signavio).
7. **3.4 Advanced: Option 1** (SAP Signavio Process Mining Suite, designated account from the professor in the week of Nov 9): repeat 3.3 there, conformance checking and decision mining if possible, usability/visualization comparison with PM4Py, when to build yourself vs use a commercial suite, assessment of the simulation feature. Screenshots required. Drift (June 2016 jump) stays relevant for the temporal generalization check.

## Findings so far (from `results/extra_stats.json`)

- 31,509 cases, 1,202,267 events, 15,930 variants, 26 activities, 149 resources; case length 38.16 ± 16.72 events; duration 21.90 ± 13.17 days.
- Categorical event attributes: 7 (Action, org:resource, concept:name, EventOrigin, lifecycle:transition, Accepted, Selected) + 2 ID attributes (EventID, OfferID), counted separately.
- Variants needed for 80 % of cases: raw 9,629, lifecycle=complete 610, **A_+O_ complete 65**. This is the main argument for event abstraction (`figures/variant_coverage.pdf`). Keep the trade-off in mind: W_ events carry resources and processing times, which simulation needs.
- Outcomes: A_Pending 17,228, A_Cancelled 10,431, A_Denied 3,752, none 98 (97 of these were active in the last 30 days, i.e. incomplete; filter them and say so).
- Cancelled cases spike at ~31 days, which suggests an automatic cancellation timer. Check the reports; if confirmed, model it as a timer event, and decision mining should rediscover it.
- Weekly arrivals jump around June 2016 (~500 to ~700/week); all cases start in 2016, the log ends 1 Feb 2017.
- Slowest decile: ~1.9 offers per case vs 1.2 in the fastest decile; mostly cancelled.
- Literature notes on six BPIC'17 reports: Claude Project doc `claude/literature_notes_bpic17_reports.md` (copy into `docs/` locally).

## Findings from discovery (Nov 5, log A = A_+O_ complete, 31,411 cases)

- IM: 100 % fitting, precision 0.462, 73 % silent transitions, ext. cyclomatic 77,728 (flower-like). HM: valid WF-net but 0 % fitting, gets stuck after O_Created on the most frequent variant; its precision 0.99 is misleading.
- IMf sweep: 0.0 -> 100 % / P 0.462; 0.05 -> 93.2 % / 0.560; **0.10 -> 88.7 % / 0.592 (base model)**; 0.15 -> 62.3 % / 0.659; 0.20 -> 62.3 % / 0.625 (dominated by 0.15); 0.30 -> 40.0 % / 0.696.
- The cliff 0.10 -> 0.15 is the offer-block loop: multi-offer cases fitting 5,457/8,511 at 0.10, 0/8,511 at 0.15.
- Generalization gap TBR vs alignments (from 0.15 on): two silent self-loops (skip_5 on O_Created, skip_11 on O_Sent mail+online) are never fired by TBR; 0 alignment-fitting cases use them.
- TBR pitfall: PM4Py's default `CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS=False` silently skips events whose activity is not in the model. No IMf model contains "O_Sent (online only)", so at 0.10 TBR marks 193 cases as fitting that alignments do not (variant #66, 36 cases). Rerun TBR with the parameter set to True and report both.
- Knock-on example (IMf 0.2, variant #62, 38 cases): one O_Sent substitution -> alignment fitness 0.895 (one log + one model move) vs TBR 0.789 (3 missing, 4 remaining tokens).
- W_ lifecycle: most work items never reach `complete`: W_Call after offers 341 of 31,264 cases, W_Handle leads 3,473/20,339, W_Call incomplete files 2,659/14,958, W_Validate application 10,890/21,825, W_Complete application 19,059/31,411. They end in ate_abort or withdraw instead.
- Log B timing check (web chat, Nov 10, IMf 0.10, 2 cores): A (A_+O_): 2,588 variants, net 30P/43T, 88.7 % fitting (align), alignments 73 s. **B1 (A_+O_ + W_ `start`)**: 3,997 variants, net 53P/76T (51 silent), 82.0 % fitting, alignments 742 s. B2 (A_+O_ + W_ terminal event): 3,130 variants, net 29P/50T, 73.5 % fitting, alignments 132 s. B1 matches "work actually started by a bank employee" and meets the target; B2 includes automatic withdraws (e.g. 16,740 W_Handle leads) and falls below 75 %.
- Runtime: alignment-based ETC precision aligns every unique prefix, not every variant (A 13,776, B1 26,510, B2 23,880 prefixes), so it is the most expensive metric (IM: 4,866 s). It honours the `multiprocessing`/`cores` keys. Its values match TBR precision within 0.002 in the whole sweep, so use TBR precision (with the not-in-model parameter) during iterations and alignment precision only for the final model and the comparison table.

## Page budget (5 pages)

Abstract + intro ~0.6, setup ~0.3, statistics ~0.6, process model ~2, decision mining ~0.6, advanced ~0.8. Everything else goes to the appendix (discovered models, extra figures). Draft intro and setup while the analysis runs, not at the end.

## Plan

| Dates | Work |
|---|---|
| Nov 4-5 | DONE: repo, statistics, literature, discovery (IM, IMf, HM), IMf sweep |
| Nov 10-11 | Log B (W_ work items), data iterations + iteration log, TBR rerun with parameter, temporal split |
| Nov 11-12 | Fix the final model, Signavio remodel with bank/customer swimlanes |
| Nov 12-13 | Decision mining: classify decisions, 2-3 with rules (DT + MLP/SHAP), probabilities for the rest |
| from Nov ~11 | Option 1 in Signavio, in parallel, once the account arrives |
| Nov 14-17 | Writing, trade-off section, declaration, README |
| Nov 18-19 | Buffer; submit by Nov 19, 23:59 |

## Conventions

- Scripts in `src/`, one per report section; tables in `results/`, figures in `figures/` (PDF for the report).
- Report: LaTeX in `report/` (TUM template, `latexmk` via `report/latexmkrc`), built by LaTeX Workshop into `report/build/`; it reads `../figures/` and `../results/` directly.
- Python env: `.venv` is a symlink to `~/.venvs/bpic17` (venv refuses paths containing `:`).
- `python src/load.py` caches the log as `data/bpic17.parquet`; other scripts call `load_log()`.
- Figures: one-column width 3.4 in, serif font, validated categorical palette `#2a78d6 #eb6834 #1baf7a #eda100` (fixed order).
- Max prefers complete, runnable scripts and concise prose; explain concepts with concrete numbers first.
