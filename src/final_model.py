"""Section 2.3 - final model: full metric set, PNML, BPMN export, activity roles (swimlanes).

The configuration (log version, O_Sent handling, variant coverage of the discovery log, IM/IMf
noise threshold) comes from results/final_choice.json (final_choice.py). The model is
rediscovered, not edited; all metrics are computed on the full log of that version.

Outputs
  results/models/final.pnml, figures/appendix/final.pdf            Petri net
  results/models/final.bpmn, figures/appendix/final_bpmn.pdf        BPMN 2.0 (process tree of the same
                                                                    discovery run), for the Signavio import
  results/final_metrics.json                                        all metrics incl. alignment precision
  results/activity_roles.csv                                        who executes / triggers each activity
Run after iterations.py:  python src/final_model.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT, RES as RESOURCE
from preprocess import final_log, discover, discovery_log
from metrics import evaluate, timed

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"
MODEL_DIR = RES / "models"
FIG_DIR = ROOT / "figures" / "appendix"

# Role of each activity and suggested swimlane (three lanes: Customer, Bank employees, System).
# Professor (Nov 6): only W_ events are executed by the bank; A_ and O_ events are state changes
# of the application / offer. Evidence: share of events recorded by User_1, the system account
# (every W_ start event is recorded by an employee, share 0). "triggered_by" is our
# interpretation of who causes the state change (BPIC'17 description + share_user_1).
ROLES = {
    "A_Create Application": ("state", "customer", "Customer",
                             "application created on the website by the customer (65 % User_1), "
                             "otherwise entered by an employee on the customer's behalf"),
    "A_Submitted": ("state", "customer", "Customer",
                    "only in online applications; customer submits the form (100 % User_1)"),
    "A_Concept": ("state", "system", "System",
                  "application becomes a concept after the lead stage; mostly automatic (54 % User_1)"),
    "A_Accepted": ("state", "bank employee", "Bank employees",
                   "first assessment passed, the bank will make an offer (0 % User_1)"),
    "A_Complete": ("state", "bank employee", "Bank employees",
                   "application completed and offers sent; the bank now waits for the customer (0 % User_1)"),
    "A_Validating": ("state", "bank employee", "Bank employees",
                     "returned documents are being validated (0 % User_1)"),
    "A_Incomplete": ("state", "bank employee", "Bank employees",
                     "validation found missing documents; the customer has to send them (0 % User_1)"),
    "A_Pending": ("state", "bank employee", "Bank employees",
                  "outcome: offer accepted after validation, loan to be paid out (0 % User_1)"),
    "A_Denied": ("state", "bank employee", "Bank employees", "outcome: application rejected (0 % User_1)"),
    "A_Cancelled": ("state", "system (timer) / customer", "System",
                    "outcome: mostly automatic after ~30 days without customer reply (76 % User_1), "
                    "otherwise cancelled by an employee at the customer's request"),
    "O_Create Offer": ("state", "bank employee", "Bank employees", "employee creates an offer (0 % User_1)"),
    "O_Created": ("state", "system", "System", "offer object created, always directly after O_Create Offer"),
    "O_Sent (mail and online)": ("state", "bank employee", "Bank employees", "offer sent by mail and online"),
    "O_Sent (online only)": ("state", "bank employee", "Bank employees", "offer sent online only"),
    "O_Sent": ("state", "bank employee", "Bank employees",
               "offer sent to the customer (both channels merged, 0 % User_1)"),
    "O_Returned": ("state", "customer", "Customer",
                   "customer returns the signed offer; recorded by an employee on receipt (0 % User_1)"),
    "O_Accepted": ("state", "bank employee", "Bank employees", "offer accepted, same count as A_Pending per case"),
    "O_Refused": ("state", "bank employee", "Bank employees", "one per open offer when the application is denied"),
    "O_Cancelled": ("state", "system (timer) / bank employee", "System",
                    "one per open offer when the application is cancelled or an offer is replaced; "
                    "mixed (48 % User_1)"),
}
W_DESC = {
    "W_Handle leads": "follow up an incomplete (lead) application",
    "W_Complete application": "complete the application data and create offers",
    "W_Call after offers": "call the customer after offers were sent",
    "W_Validate application": "check the returned documents",
    "W_Call incomplete files": "call the customer about missing documents",
    "W_Assess potential fraud": "fraud check",
    "W_Personal Loan collection": "collection task (rare)",
    "W_Shortened completion": "shortened completion (rare)",
}


def activity_roles(net) -> pd.DataFrame:
    log, _ = final_log(keep_attrs=True)
    in_model = {t.label for t in net.transitions if t.label is not None}
    g = log.groupby(ACT)
    tab = pd.DataFrame({"events": g.size(), "cases": g[CASE].nunique(),
                        "share_user_1": g[RESOURCE].apply(lambda s: (s == "User_1").mean()).round(3),
                        "resources": g[RESOURCE].nunique()})
    rows = []
    for act, r in tab.iterrows():
        if act.startswith("W_"):
            role = ("work item", "bank employee", "Bank employees",
                    f"work item started by an employee (W_ start, 0 % User_1): {W_DESC.get(act, '')}")
        else:
            role = ROLES[act]
        rows.append({"activity": act, "in_model": act in in_model, "type": role[0],
                     "executed_or_triggered_by": role[1], "lane": role[2], "justification": role[3],
                     **r.to_dict()})
    return pd.DataFrame(rows).sort_values(["lane", "type", "activity"])


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log, cfg = final_log()
    print(f"final configuration {cfg['id']}: log {cfg['log']}, merge_osent={cfg['merge_osent']}, "
          f"collapse={cfg['collapse_repeats']}, {cfg.get('algorithm', 'IMf')} noise {cfg['noise_threshold']}, "
          f"coverage {cfg.get('coverage')}", flush=True)

    (net, im, fm), t_disc = timed(discover, log, cfg)
    pm4py.write_pnml(net, im, fm, str(MODEL_DIR / "final.pnml"))
    pm4py.save_vis_petri_net(net, im, fm, str(FIG_DIR / "final.pdf"), rankdir="LR")

    # BPMN from the process tree of the same discovery run (block-structured, gateways = tree operators)
    tree = discover(log, cfg, tree=True)
    bpmn = pm4py.convert_to_bpmn(tree)
    pm4py.write_bpmn(bpmn, str(MODEL_DIR / "final.bpmn"))
    pm4py.save_vis_bpmn(bpmn, str(FIG_DIR / "final_bpmn.pdf"), rankdir="LR")
    tnet, _, _ = pm4py.convert_to_petri_net(tree)
    same = (sorted(t.label or "" for t in tnet.transitions) == sorted(t.label or "" for t in net.transitions)
            and len(tnet.places) == len(net.places) and len(tnet.arcs) == len(net.arcs))
    print(f"process tree -> Petri net identical in size and labels to the discovered net: {same}", flush=True)

    roles = activity_roles(net)
    roles.to_csv(RES / "activity_roles.csv", index=False)
    print(roles[["activity", "in_model", "executed_or_triggered_by", "lane", "share_user_1"]].to_string(index=False))

    out_path = RES / "final_metrics.json"
    meta = {k: cfg.get(k) for k in ["id", "log", "merge_osent", "collapse_repeats", "drop_rare_w",
                                    "algorithm", "noise_threshold", "coverage", "data_change", "cases",
                                    "events", "variants", "activities", "variants_discovery"]}

    def save(r):
        out_path.write_text(json.dumps({**meta, **r, "t_discovery": t_disc,
                                        "tree_equals_net": same}, indent=2, default=str))

    r, _, _ = evaluate(log, net, im, fm, save=save)
    dlog = discovery_log(log, cfg)
    r["precision_tbr_discovery_log"] = (r["precision_tbr"] if dlog is log else
                                        pm4py.precision_token_based_replay(dlog, net, im, fm))
    save(r)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, indent=1))


if __name__ == "__main__":
    main()
