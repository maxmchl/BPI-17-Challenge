"""Section 3.3 - final model: full metric set, PNML, BPMN export, activity roles (swimlanes).

The configuration (log version, O_Sent handling, collapse, IMf noise threshold) comes from
results/final_choice.json (iterations.py). The model is rediscovered, not edited.

Outputs
  results/models/final.pnml, figures/appendix/final.pdf            Petri net
  results/models/final.bpmn, figures/appendix/final_bpmn.pdf        BPMN 2.0 (process tree of the same
                                                                    IMf run), for the Signavio import
  results/final_metrics.json                                        all metrics incl. alignment precision
  results/activity_roles.csv                                        who executes / triggers each activity
Run after iterations.py:  python src/final_model.py
"""
import json
import warnings

import pandas as pd
import pm4py

from load import ROOT, CASE, ACT, RES as RESOURCE
from preprocess import final_log
from metrics import evaluate, timed

warnings.filterwarnings("ignore", category=UserWarning)
RES = ROOT / "results"
MODEL_DIR = RES / "models"
FIG_DIR = ROOT / "figures" / "appendix"

# Role of each activity. Professor (Nov 6): only W_ events are executed by the bank; A_ and O_
# events are state changes of the application / offer that the bank does not control directly.
# "triggered_by" is our interpretation of who causes the state change (BPIC'17 description and
# the share of events recorded by User_1, the system account), "lane" the suggested swimlane.
ROLES = {
    "A_Create Application": ("state", "customer", "Customer",
                             "application created online by the customer or entered by an employee on their behalf"),
    "A_Submitted": ("state", "customer", "Customer",
                    "only in online applications; customer submits the form (always recorded by User_1)"),
    "A_Concept": ("state", "system", "Bank",
                  "application becomes a concept after the lead is handled; partly automatic (User_1)"),
    "A_Accepted": ("state", "bank", "Bank",
                   "first assessment passed, the bank will make an offer (recorded by employees only)"),
    "A_Complete": ("state", "bank", "Bank",
                   "application completed and offers sent; the bank now waits for the customer"),
    "A_Validating": ("state", "bank", "Bank", "returned documents are being validated by the bank"),
    "A_Incomplete": ("state", "bank", "Bank",
                     "validation found missing documents; the customer has to send them (next move: customer)"),
    "A_Pending": ("state", "bank", "Bank", "outcome: offer accepted after validation, loan to be paid out"),
    "A_Denied": ("state", "bank", "Bank", "outcome: application rejected by the bank"),
    "A_Cancelled": ("state", "customer / timer", "Customer",
                    "outcome: customer withdrew or no reply in ~30 days (mostly User_1, i.e. automatic)"),
    "O_Create Offer": ("state", "bank", "Bank", "employee creates an offer"),
    "O_Created": ("state", "system", "Bank", "offer object created, immediately follows O_Create Offer"),
    "O_Sent (mail and online)": ("state", "bank", "Bank", "offer sent to the customer by mail and online"),
    "O_Sent (online only)": ("state", "bank", "Bank", "offer sent to the customer online only"),
    "O_Sent": ("state", "bank", "Bank", "offer sent to the customer (both channels merged)"),
    "O_Returned": ("state", "customer", "Customer", "customer returns the signed offer documents"),
    "O_Accepted": ("state", "bank", "Bank", "offer accepted after validation (together with A_Pending)"),
    "O_Refused": ("state", "bank", "Bank", "offer refused (together with A_Denied)"),
    "O_Cancelled": ("state", "customer / timer", "Customer",
                    "offer cancelled: new offer replaces it, customer withdrew, or timeout"),
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
            role = ("work item", "bank employee", "Bank",
                    f"W_ work item started by a bank resource: {W_DESC.get(act, '')}")
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
    noise = cfg["noise_threshold"]
    print(f"final configuration {cfg['id']}: log {cfg['log']}, merge_osent={cfg['merge_osent']}, "
          f"collapse={cfg['collapse_repeats']}, noise {noise}", flush=True)

    (net, im, fm), t_disc = timed(pm4py.discover_petri_net_inductive, log, noise_threshold=noise)
    pm4py.write_pnml(net, im, fm, str(MODEL_DIR / "final.pnml"))
    pm4py.save_vis_petri_net(net, im, fm, str(FIG_DIR / "final.pdf"), rankdir="LR")

    # BPMN from the process tree of the same IMf run (block-structured, gateways = tree operators)
    tree = pm4py.discover_process_tree_inductive(log, noise_threshold=noise)
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
    meta = {k: cfg[k] for k in ["id", "log", "merge_osent", "collapse_repeats", "noise_threshold",
                                "data_change", "cases", "events", "variants", "activities"]}

    def save(r):
        out_path.write_text(json.dumps({**meta, **r, "t_discovery": t_disc,
                                        "tree_equals_net": same}, indent=2, default=str))

    r, _, _ = evaluate(log, net, im, fm, save=save)
    save(r)
    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}, indent=1))


if __name__ == "__main__":
    main()
