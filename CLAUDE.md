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

Report structure (report/main.tex): 1 Introduction; 2 Approach and Results with 2.1 Technical Setup and Preliminaries, 2.2 Basic Analysis, 2.3 Process Model Creation and Validation (incl. decision mining), 2.4 Advanced Analysis.

0. **Abstract:** task, log, findings, main results in 4-5 sentences.
1. **Intro:** domain, origin and purpose of the log, existing BPIC'17 reports (state of the art + limitations). Source: https://ais.win.tue.nl/bpi/2017/challenge.html
2. **2.1 Technical setup:** libraries, visualization, modeling tools (footnotes).
3. **2.2 Basic analysis** - DONE (`src/stats.py`, `results/basic_stats.tex`; W_ statistics `src/w_stats.py`, `results/w_stats.tex`). Required table plus at least 2 extra statistics; also typical vs long vs short cases.
4. **2.3 Process model:**
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
5. **Decision mining (mandatory for Master's, in 2.3):** alignment-based (de Leoni & van der Aalst 2013) on the final Petri net, rules visualized on the BPMN (text annotations at the gateways). Per the professor: classify decision points (XOR / self-loop / loop decision), find 2-3 with good rules, give branching probabilities for the rest and explain why no rule works. Candidates: channel online vs bank (ApplicationType = Limit raise -> bank); outcome after validation (Pending / Denied / Incomplete); return vs cancel after A_Complete is driven by the 30-day timer, so expect no attribute rule.
   - Motivation (professor): XORs introduce uncertainty; a simulation needs a branching rule at each one.
   - PM4Py's built-in decision mining uses token replay, so implement the alignment-based extraction ourselves: align each trace, find the branch taken after each decision place (next visible sync/model move), label = branch.
   - Features: case attributes (LoanGoal, ApplicationType, RequestedAmount) + event/offer attributes known before the decision (CreditScore, OfferedAmount, MonthlyCost, NumberOfTerms, ...); optionally long-tail features from slide 48 (last k resources, elapsed/waiting times, counts). Fixed seed, held-out split, shallow trees.
   - Leakage: CreditScore, Selected and Accepted are recorded at O_Create Offer but reflect later outcomes (van der Ham 2017); use only features known before the decision.
   - **Slide 52 (mandatory):** compare decision trees vs neural networks + XAI (Park et al. 2022), only on the 2-3 good decision points. Plan: per decision point, a shallow DT (rules), an MLP + SHAP, and a surrogate DT on the MLP predictions. Report accuracy, surrogate fidelity, and agreement between SHAP features and tree splits; argue explainability vs accuracy.
6. **Tooling rules:** list every commercial tool used and include screenshots of results (also for Signavio).
7. **2.4 Advanced: Option 1** (SAP Signavio Process Mining Suite, designated account from the professor in the week of Nov 9): repeat 2.3 there, conformance checking and decision mining if possible, usability/visualization comparison with PM4Py, when to build yourself vs use a commercial suite, assessment of the simulation feature. Screenshots required. Drift (June 2016 jump) stays relevant for the temporal generalization check.

## Findings so far (from `results/extra_stats.json`)

- 31,509 cases, 1,202,267 events, 15,930 variants, 26 activities, 149 resources; case length 38.16 ± 16.72 events; duration 21.90 ± 13.17 days.
- Categorical event attributes: 7 (Action, org:resource, concept:name, EventOrigin, lifecycle:transition, Accepted, Selected) + 2 ID attributes (EventID, OfferID), counted separately.
- Variants needed for 80 % of cases: raw 9,629, lifecycle=complete 610, **A_+O_ complete 65**. This is the main argument for event abstraction (`figures/variant_coverage.pdf`). Keep the trade-off in mind: W_ events carry resources and processing times, which simulation needs.
- Outcomes: A_Pending 17,228, A_Cancelled 10,431, A_Denied 3,752, none 98 (97 of these were active in the last 30 days, i.e. incomplete; filter them and say so).
- Cancelled cases spike at ~31 days, which suggests an automatic cancellation timer. Check the reports; if confirmed, model it as a timer event, and decision mining should rediscover it.
- Weekly arrivals jump around June 2016 (~500 to ~700/week); all cases start in 2016, the log ends 1 Feb 2017.
- Slowest decile: ~1.9 offers per case vs 1.2 in the fastest decile; mostly cancelled.
- Literature notes on eight BPIC'17 reports (six professional + the two winners): `docs/literature_notes_bpic17_reports.md` (local, not versioned).
- **Rodrigues et al. 2017 (academic winner):** hand-built Petri net in 19 versions (closest precedent to our iteration log), fitness 97.28 % / precision 98.72 % on 7 clustered activities (events < 2 min apart by the same user merged), so not comparable with our activity-level model; no generalization/simplicity, not reproducible from code.
- **Povalyaeva et al. 2017 (student winner):** no structural drift (ProM concept-drift plug-in), only a volume split in June 2016; feature candidates for the outcome decision: first withdrawal amount, number of terms, monthly cost, requested amount (> 40,000), time to first offer (all in our base feature set except time to first offer, covered by days since first offer).

## Findings from discovery (Nov 5, log A = A_+O_ complete, 31,411 cases)

- IM: 100 % fitting, precision 0.462, 73 % silent transitions, ext. cyclomatic 77,728 (flower-like). HM: valid WF-net but 0 % fitting, gets stuck after O_Created on the most frequent variant; its precision 0.99 is misleading.
- IMf sweep: 0.0 -> 100 % / P 0.462; 0.05 -> 93.2 % / 0.560; **0.10 -> 88.7 % / 0.592 (base model)**; 0.15 -> 62.3 % / 0.659; 0.20 -> 62.3 % / 0.625 (dominated by 0.15); 0.30 -> 40.0 % / 0.696.
- The cliff 0.10 -> 0.15 is the offer-block loop: multi-offer cases fitting 5,457/8,511 at 0.10, 0/8,511 at 0.15.
- Generalization gap TBR vs alignments (from 0.15 on): two silent self-loops (skip_5 on O_Created, skip_11 on O_Sent mail+online) are never fired by TBR; 0 alignment-fitting cases use them.
- TBR pitfall: PM4Py's default `CONSIDER_ACTIVITIES_NOT_IN_MODEL_IN_FITNESS=False` silently skips events whose activity is not in the model. No IMf model contains "O_Sent (online only)", so at 0.10 TBR marks 193 cases as fitting that alignments do not (variant #66, 36 cases).
- **PM4Py bug (patch footnote, slide 37):** with the parameter set to True, PM4Py 2.7.23.11 keeps the unknown activities in one dict for the whole log (token_replay.py l. 1388) and only checks that it is non-empty (l. 941), so every trace replayed after the first one with an unknown activity is unfit (IMf 0.10: 3,143 fitting cases wrongly unfit). `metrics.tbr_fitness` applies the intended rule instead (fit iff TBR fits and all activities are model labels); with it IMf 0.10 gives 88.74 % = alignments. Log fitness, precision and generalization do not depend on the switch. `src/tbr_rerun.py`, `results/tbr_disagreement_imf010.json`.
- Knock-on example (IMf 0.2, variant #62, 38 cases): one O_Sent substitution -> alignment fitness 0.895 (one log + one model move) vs TBR 0.789 (3 missing, 4 remaining tokens).
- W_ lifecycle: most work items never reach `complete`: W_Call after offers 341 of 31,264 cases, W_Handle leads 3,473/20,339, W_Call incomplete files 2,659/14,958, W_Validate application 10,890/21,825, W_Complete application 19,059/31,411. They end in ate_abort or withdraw instead.
- Log B timing check (web chat, Nov 10, IMf 0.10, 2 cores): A (A_+O_): 2,588 variants, net 30P/43T, 88.7 % fitting (align), alignments 73 s. **B1 (A_+O_ + W_ `start`)**: 3,997 variants, net 53P/76T (51 silent), 82.0 % fitting, alignments 742 s. B2 (A_+O_ + W_ terminal event): 3,130 variants, net 29P/50T, 73.5 % fitting, alignments 132 s. B1 matches "work actually started by a bank employee" and meets the target; B2 includes automatic withdraws (e.g. 16,740 W_Handle leads) and falls below 75 %.
- Runtime: alignment-based ETC precision aligns every unique prefix, not every variant (A 13,776, B1 26,510, B2 23,880 prefixes), so it is the most expensive metric (IM: 4,866 s). It honours the `multiprocessing`/`cores` keys. Its values match TBR precision within 0.002 in the whole sweep, so use TBR precision (with the not-in-model parameter) during iterations and alignment precision only for the final model and the comparison table.

## Decisions and results (Nov 11, web chat + local runs)

- **Final model: It6_c75** = IM on the most frequent variants covering 75 % of cases (92 variants) of log **B1c**, evaluated on all 31,411 cases: 89.4 % fitting cases (alignments), fitness 0.993, precision 0.700 (TBR) / 0.701 (alignments), generalization 0.991, size 163, CNC 1.233, silent share 0.55, ext. cyclomatic 59. `results/final_metrics.json`, `results/models/final.{pnml,bpmn}`.
- **Two-stage choice** (`src/final_choice.py`, `results/final_choice.json`): (1) log version by purpose: simulation base model + bank-vs-customer question need the W_ work items at lifecycle start (one per work item started by an employee; every W_ start event has share User_1 = 0). (2) Selection rule only within that version, because ETC precision is not comparable across logs with different activity sets: max precision with >= 80 % fitting (alignments), ties < 0.005 -> lower CNC, lower silent share, more fitting cases.
- **Log B1c** = B1 with merged O_Sent and without O_Created (always directly after O_Create Offer), O_Accepted (= A_Pending per case) and O_Refused (one per open offer on denial; 32 denied cases have none). `results/b1c_redundancy.json`. **No precision claim for B1c vs B1:** at 75 % coverage B1 has 0.710, B1c 0.700 (same 89.4 % fitting). B1c is justified only by removing redundant events (size 163 vs 182).
- **It1 stays as the A_/O_ backbone:** merging the O_Sent labels alone raised fitting from 88.7 % to 93.0 % with an identical net (precision 0.589). Tie-break bug fixed on the way (identical nets were decided by list order).
- **Variant filter beats IMf noise:** discover with IM on the top variants, evaluate on the full log. Best per log version (not comparable across versions): A It4_c70 86.4 % / 0.706; B1 It5_c75 89.4 % / 0.710; B1c It6_c75 89.4 % / 0.700; B1c without rare W_ (It7) 89.6 % / 0.631. IMf rows on B1: 0.10 -> 85.9 % / 0.432. Log C (all activities at lifecycle complete, professor's choice): 5,196 variants, 60 % coverage 82.1 % / 0.594, 70 % 91.3 % / 0.455 -> reason for W_ start. `figures/coverage_tradeoff.pdf`.
- **Precision on the discovery log is always lower than on the full log** (B1c 75 %: 0.511 vs 0.700): ETC penalizes allowed continuations the log never shows. Never confuse discovery filtering with evaluation filtering.
- **Temporal split (final config):** the halves discover different trees. The H1 model (Jan-Jun) is looser: 92.6 % in sample, fits H2 94.3 % at precision ~0.50. The H2 model (Jul-Dec) is tighter: 90.3 % in sample, fits H1 87.6 % at precision ~0.61. Fitting cases by start month 84.5 % (May) to 92.5 % (Sep), `figures/fitting_by_month.pdf`. Reading: mild drift; the later behaviour is more regular, so a model of it is stricter and replays the first half worse. Limitation: each half is filtered separately at 75 % coverage, so part of the difference comes from the variant filter, not from the process. Contrast: Povalyaeva et al. 2017 (student winner) found no structural drift with the ProM concept-drift plug-in, only a volume split in June 2016. (With It1 the trees were identical, 92.1-94.1 % per month.)
- **Conformance on the final model:** fitting by number of offers 1 -> 95.8 %, 2 -> 76.9 %, 3 -> 61.2 %, 4+ -> 50.4 %; TBR default vs not-in-model rule differ on 173 cases (rare W_ not in the model); TBR fires every transition. Knock-on example for the report stays variant #62 on IMf 0.2 (alignment 0.895 vs TBR 0.789).
- **Decision points** (`results/decision_points.csv`, IDs fixed per place in `results/decision_point_ids.json`): 14 (10 XOR, 2 loop decisions, 2 self-loops); kinds: 8 business decisions (D2 multiple offers 26.5 % repeat, D4 documents incomplete / W_Call incomplete files 59.2 %, D5 incomplete loop 39.4 %, D6 call after offers 81.8 %, D7 offer returned 61.3 %, D9 outcome Pending 54.9 / Cancelled 33.2 / Denied 11.9 %, D10 channel A_Submitted 64.8 %, D12 W_Handle leads 17.8 %), 2 optional-activity skips (D1 O_Sent, D8 W_Complete application), 3 bookkeeping (O_Cancelled per open offer, cancellation events fixed by the outcome), 1 repetition (D3 O_Sent). For all points without a mined rule, the branching probabilities in that CSV are the result (simulation input).
- **Decision mining** (`src/decision_mining.py`, `results/decision_mining.csv`, `results/decision_rules.json`, `figures/shap_D*.pdf`): alignment-based, one row per visit of the decision place, features = state after the last event before the decision; GroupShuffleSplit by case 70/30, seed 42; DT depth 3 balanced, MLP (32, 16) + SHAP (permutation, 500 test rows), surrogate DT depth 3. Base features (balanced accuracy DT / MLP, baseline 0.5 resp. 0.33):
  - D10 channel 0.76 / 0.77: ApplicationType = Limit raise -> bank 100 % (1,023 test rows); New credit with RequestedAmount <= 1,300 -> bank 99.6 %; created after 20:47 -> online 98.7 %.
  - D6 call after offers 0.99 / 0.98: the first offer of a case always gets W_Call after offers (100 %); a later offer while work items already run gets no new call (99.6 %). The call is one work item per offer round, not per offer.
  - D9 outcome 0.77 / 0.70: nothing after the last O_Sent and no O_Returned -> A_Cancelled 98.9 % (30-day timer); returned + incomplete -> A_Pending 83-87 %; Denied is hard (no incomplete round, New credit -> Denied only 39 %).
  - Leakage run D9 (+ CreditScore, Selected, Accepted): balanced accuracy 0.91, CreditScore top SHAP feature (0.28, next 0.11) -> confirms the leakage (van der Ham 2017).
  - Surrogate fidelity 0.97-0.99; SHAP top-k vs DT split features agree 0.67 (D10), 1.0 (D6), 0.8 (D9).
  - With vs without resource/time features: D10 with them learns only `last resource = User_1 -> online` (accuracy 1.0, a tautology: the system creates online applications); D6 and D9 unchanged (only small last_hour leaves). Business rules for the BPMN come from the version without resource/time.
- **BPMN guards** (`src/bpmn_annotations.py`, `results/bpmn_annotations.json`): lanes per activity and one annotation per gateway (rules for D6/D9/D10, probabilities for the rest), located by the activities before the place. For the Signavio remodel; if it changes the structure, rerun decision_points + decision_mining on the re-imported model.
- **Signavio remodel re-imported** (`src/signavio_reimport.py`, `results/models/final_bpmn.bpmn` -> `final_signavio.pnml`, `results/final_signavio_metrics.json`): lanes, annotations and associations stripped in a copy (PM4Py reads annotations as nodes). Same 19 visible labels as final.pnml, all log activities, and the **same language** (reachability graphs with silent moves as epsilon, no distinguishing trace). Identical: 89.4 % fitting cases (TBR and alignments), precision 0.700 TBR / 0.701 alignments. Differ only through 8 extra silent transitions from the gateway conversion (39P/50T vs 31P/42T): size 195 vs 163, CNC 1.191 vs 1.233, silent share 0.62 vs 0.55, ext. cardoso 49 vs 41, ext. cyclomatic 69 vs 59, fitness TBR 0.9962 vs 0.9952, alignments 0.99271 vs 0.99275, generalization 0.992 vs 0.991. Report the re-imported numbers as the metrics of the final BPMN.
- **Signavio-generated BPMN** (Process Intelligence, Variant Explorer -> Generate BPMN on the top 92 variants; `src/signavio_discovery_eval.py`, `results/models/signavio_generated.bpmn`, `results/signavio_discovery_metrics.json`): 19 tasks = the It6_c75 activities, 22 XOR, 1 AND, 2 OR gateways (both OR joins), loop markers on O_Create Offer, O_Sent, O_Cancelled; offer loop and validation loop present. PM4Py's direct conversion is **not sound** (OR joins become AND joins, loop markers ignored: every case deadlocks before A_Concept, 15 of 19 tasks dead). **Faithful translation** in a copy (field `translation`): OR1 -> XOR join (inputs mutually exclusive), OR2 -> AND join with a silent skip on the direct online path, loop markers -> XOR loop; sound (37 markings, Woflan agrees). Metrics on B1c: 45.7 % fitting cases, fitness 0.970 (alignments), precision 0.856 TBR / 0.857 alignments, generalization 0.983, size 220, CNC 1.268, silent share 0.68, arc degree 0.651, ext. cyclomatic 62. Low fit is the generated structure, not the translation: W_Complete application is reachable only via AND1 (bank / leads), so online cases without W_Handle leads (top variants 1, 4, 6) have a log move and variants 2, 9 a W_Handle leads model move (5 of the top 10, 8,115 cases). Point for 2.4: Signavio's generated BPMN is precise but underfits, and uses OR joins / loop markers that need a manual translation for PM4Py.
- **Top 10 variants vs final BPMN** (`src/top10_variant_conformance.py`, `results/top10_variant_conformance.csv`): the 10 most frequent B1c variants (41.6 % of cases, all with one offer) fit final_bpmn.bpmn perfectly (alignment fitness 1.0, no deviating moves).
- **Trade-off figure** `figures/tradeoff_panels.pdf`: fitting vs precision / generalization / CNC for every iteration row, final model ringed.
- **CSV vs XES** (`results/csv_xes_check.json`): the CSV in data/ is identical to the 4TU XES (all events, values, timestamps); the pipeline stays on the XES. Report: "XES from 4TU, DOI 10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b".
- **Swimlanes** (`results/activity_roles.csv`): Customer (A_Create Application, A_Submitted, O_Returned), Bank employees (all W_ + employee-recorded A_/O_ states, share User_1 = 0), System (A_Concept 54 %, A_Cancelled 76 %, O_Cancelled 48 % User_1; 30-day timer).
- **W_ statistics** (`results/w_stats.*`): W_ = 63.9 % of all events; 4.07 +- 2.11 work items started per case; e.g. W_Call after offers ends in ate_abort 89.8 %, W_Handle leads in withdraw 82.2 %.
- **Exploration W_ complete** (`src/explore_w_complete.py`, `results/explore_w_complete{.csv,_best.json}`, `figures/explore_w_complete_pareto.pdf`; not in the iteration log): 360 configs (logs C, C without duplicates, without rare W_, only W_ in >= 50 % of cases, clustered events, each +/- collapsed repeats; IM coverage 40-80 %, IMf, ILP, HM). Precision 0.8 with >= 80 % fitting is **not reachable**: the Pareto front drops from 0.899 at 72.7 % to 0.740 at 76.7 %. Best >= 80 %: C-W50 + collapsed repeats, IM c70, 87.7 % fitting, precision 0.729 (alignments), but it keeps only W_Complete application and W_Assess potential fraud and lacks W_Call after offers, W_Call incomplete files, W_Handle leads and W_Validate application, so it cannot serve the simulation / bank-vs-customer question. HM is never sound; ILP is precise only below 75 % fitting.

## Page budget (5 pages)

Abstract + intro ~0.6, setup ~0.3, statistics ~0.6, process model ~2, decision mining ~0.6, advanced ~0.8. Everything else goes to the appendix (discovered models, extra figures). Draft intro and setup while the analysis runs, not at the end.

## Plan

| Dates | Work |
|---|---|
| Nov 4-5 | DONE: repo, statistics, literature, discovery (IM, IMf, HM), IMf sweep |
| Nov 10-11 | DONE: log B1/B1c, iteration log, TBR rerun (PM4Py bug), temporal split, final model It6_c75, decision point inventory |
| Nov 11-12 | Signavio remodel with three swimlanes (Customer / Bank employees / System), re-import + re-evaluate |
| Nov 14 | DONE: decision mining D10/D6/D9 (DT + MLP/SHAP + surrogate, leakage and resource/time runs), BPMN guard texts, trade-off panels, CSV check |
| from Nov ~11 | Option 1 in Signavio, in parallel, once the account arrives |
| Nov 14-17 | Writing, trade-off section, declaration, README |
| Nov 18-19 | Buffer; submit by Nov 19, 23:59 |

## Conventions

- Scripts in `src/`, one per report section; tables in `results/`, figures in `figures/` (PDF for the report).
- Report: LaTeX in `report/` (TUM template, `latexmk` via `report/latexmkrc`), built by LaTeX Workshop into `report/build/`; it reads `../figures/` and `../results/` directly.
- Python env: `.venv` is a symlink to `~/.venvs/bpic17` (venv refuses paths containing `:`).
- `python src/load.py` caches the log as `data/bpic17.parquet`; other scripts call `load_log()`.
- Reproduce order: load -> stats -> figures_stats -> w_stats -> preprocess (redundancy check) -> discovery -> imf_sweep -> tbr_rerun -> conformance_examples -> iterations -> final_choice -> temporal -> final_model -> conformance_examples final -> decision_points -> decision_mining -> bpmn_annotations -> (Signavio remodel) signavio_reimport -> top10_variant_conformance -> signavio_discovery_eval; tradeoff_panels after final_choice; export_signavio after final_choice (Signavio CSVs in data/, not versioned); csv_xes_check anytime. Long runs write their JSON incrementally and skip finished rows.
- Figures: one-column width 3.4 in, serif font, validated categorical palette `#2a78d6 #eb6834 #1baf7a #eda100` (fixed order).
- Max prefers complete, runnable scripts and concise prose; explain concepts with concrete numbers first.
