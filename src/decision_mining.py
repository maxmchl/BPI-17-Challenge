"""Section 2.3 (decision mining) - alignment-based decision mining on the final net It6_c75.

Method (de Leoni & van der Aalst 2013): align every case with the final Petri net; every sync or
model move fires a transition. A firing of transition t whose preset contains a decision place p
is one visit of p with label = branch t. The features of a visit describe the case prefix that
the alignment has consumed before the firing (only information known before the decision).
Decision points (IDs from results/decision_point_ids.json, see decision_points.py):
  D10  channel: A_Submitted (online) vs silent skip (application entered by the bank)
  D6   call after offers: W_Call after offers vs silent skip, once per sent offer
  D9   outcome: A_Pending / A_Denied / silent branch to A_Cancelled
All other decision points: branching probabilities in results/decision_points.csv only.

Features (state after the last event of the prefix; NaN if not yet known):
  case       ApplicationType, LoanGoal, RequestedAmount
  offer      of the last offer created so far: OfferedAmount, MonthlyCost, NumberOfTerms,
             FirstWithdrawalAmount, OfferedAmount / RequestedAmount; original label of the last
             O_Sent ("O_Sent (online only)" vs "O_Sent (mail and online)", merged in log B1c)
  counts     offers, O_Returned, A_Incomplete and W_ work items started so far
  elapsed    days from case start / first offer / last O_Sent to the last event of the prefix
  create     weekday and hour of A_Create Application (local time Europe/Amsterdam)
  leakage    CreditScore, Selected, Accepted of the last offer: recorded at O_Create Offer but
             reflect the later outcome (van der Ham 2017). Excluded except in one D9 sensitivity run.
  resource   resource, weekday and hour of the last event of the prefix ("who and when")
Feature sets (FEATURE_SETS): base = case + offer + counts + elapsed + create;
  leakage = base + leakage (D9 only); with_resource_time = base + resource;
  without_resource_time = base without create (no resource or time-of-day feature at all).
  O_Returned so far is our addition to the handoff list (known before the outcome).

Split: GroupShuffleSplit by case, 70/30, random_state=42 (all visits of a case in one part).
Metrics on the test part: accuracy, balanced accuracy, macro F1; baseline = majority class of
the training part.
Models (sklearn, random_state=42 everywhere):
  dt         DecisionTreeClassifier(max_depth=3, class_weight="balanced"); native NaN handling;
             categorical features one-hot (OneHotEncoder, min_frequency=50, rest = infrequent)
  mlp        MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True,
             validation_fraction=0.1, max_iter=300, other parameters default: relu, adam,
             alpha=1e-4, learning_rate_init=1e-3); numeric features: median imputation + missing
             indicator + StandardScaler; categorical: one-hot as above
  surrogate  DecisionTreeClassifier(max_depth=3) trained on the MLP predictions of the training
             part (no class weights: it should imitate the MLP); fidelity = agreement with the
             MLP on the test part
SHAP (Park et al. 2022 compare DT vs NN + XAI): PermutationExplainer on the MLP's predict_proba in
  the transformed feature space, 500 random test rows, background = 100 random training rows,
  max_evals = 2 * (2 * features + 1). Global importance = mean |SHAP| per original feature
  (one-hot columns and missing indicators summed back to their feature per row; multi-class:
  mean over classes). Agreement: share of the DT split features among the top-k SHAP features
  (k = number of distinct DT split features).

Business rules for the BPMN annotations come from without_resource_time (handoff item 7).

Outputs: results/decision_mining.csv (one row per decision point, feature set and model),
         results/decision_rules.json (DT and surrogate rules, leaf rules with test-set support),
         results/decision_mining_shap.json, figures/shap_D10.pdf, figures/shap_D6.pdf,
         figures/shap_D9.pdf (base features), figures/shap_D9_leakage.pdf
Cache:   data/decision_mining_moves.pkl (alignment moves per variant, keyed by final.pnml hash)
Run after decision_points.py:  python src/decision_mining.py
"""
import hashlib
import json
import pickle
import warnings

import numpy as np
import pandas as pd
import pm4py
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from load import load_log, ROOT, CASE, ACT, TS, RES as RESOURCE
from preprocess import final_log
from metrics import align
from conformance_examples import variant_log
from figures_stats import C, W, INK2, plt

warnings.filterwarnings("ignore")
RES = ROOT / "results"
FIG = ROOT / "figures"
NET = RES / "models" / "final.pnml"
CACHE = ROOT / "data" / "decision_mining_moves.pkl"
SEED = 42
TZ = "Europe/Amsterdam"

DECISIONS = {  # decision point -> name and label of the silent branch
    "D10": ("channel (online vs bank)", "bank (no A_Submitted)"),
    "D6": ("call after offers", "no call"),
    "D9": ("outcome", "A_Cancelled"),
}

CASE_F = (["case:ApplicationType", "case:LoanGoal"], ["case:RequestedAmount"])
OFFER_F = (["last_sent_label"], ["OfferedAmount", "MonthlyCost", "NumberOfTerms", "FirstWithdrawalAmount",
                                 "offered_to_requested"])
COUNT_F = ([], ["n_offers", "n_returned", "n_incomplete", "n_work_items"])
ELAPSED_F = ([], ["days_since_start", "days_since_first_offer", "days_since_last_sent"])
CREATE_F = (["create_weekday"], ["create_hour"])
LEAK_F = (["Selected", "Accepted"], ["CreditScore"])
RT_F = (["last_resource", "last_weekday"], ["last_hour"])


def fset(*groups):
    return [c for g in groups for c in g[0]], [n for g in groups for n in g[1]]


FEATURE_SETS = {
    "base": fset(CASE_F, OFFER_F, COUNT_F, ELAPSED_F, CREATE_F),
    "leakage": fset(CASE_F, OFFER_F, COUNT_F, ELAPSED_F, CREATE_F, LEAK_F),
    "with_resource_time": fset(CASE_F, OFFER_F, COUNT_F, ELAPSED_F, CREATE_F, RT_F),
    "without_resource_time": fset(CASE_F, OFFER_F, COUNT_F, ELAPSED_F),
}
# with/without resource and time: a tree with "who and when" features tends to learn only that
# (e.g. User_1, hour <= 16.5); the business rules for the BPMN come from without_resource_time.
RUNS = ([(d, "base") for d in DECISIONS] + [("D9", "leakage")]
        + [(d, fs) for d in DECISIONS for fs in ("with_resource_time", "without_resource_time")])


# ---------------------------------------------------------------- alignments -> decision visits
def moves_per_variant(df):
    """For every variant: list of (log events consumed before the move, fired transition name)."""
    log, _, variants = variant_log(df)
    key = hashlib.md5(NET.read_bytes() + repr(variants).encode()).hexdigest()
    if CACHE.exists():
        cached = pickle.loads(CACHE.read_bytes())
        if cached["key"] == key:
            return cached["moves"]
    net, im, fm = pm4py.read_pnml(str(NET))
    al = align(log, net, im, fm)
    moves = {}
    for var, trace, res in zip(variants, log, al):
        assert tuple(e[ACT] for e in trace) == var, "event order of the EventLog differs from the DataFrame"
        if res is None:
            continue
        k, out = 0, []
        for (log_name, model_name), _ in res["alignment"]:
            if model_name != ">>":
                out.append((k, model_name))
            if log_name != ">>":
                k += 1
        assert k == len(var)
        moves[var] = out
    CACHE.write_bytes(pickle.dumps({"key": key, "moves": moves}))
    return moves


def event_state(ev: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Per event: the case state after this event (features of a decision taken right after it)."""
    ev = ev.copy()
    g = ev.groupby(CASE, sort=False)
    act = ev[ACT]
    offer = act == "O_Create Offer"
    sent = act == "O_Sent"
    for name, m in [("n_offers", offer), ("n_returned", act == "O_Returned"),
                    ("n_incomplete", act == "A_Incomplete"), ("n_work_items", act.str.startswith("W_"))]:
        ev[name] = m.astype(int).groupby(ev[CASE], sort=False).cumsum()
    for c in ["OfferedAmount", "MonthlyCost", "NumberOfTerms", "FirstWithdrawalAmount", "CreditScore",
              "Selected", "Accepted"]:
        ev[c] = ev[c].where(offer).groupby(ev[CASE], sort=False).ffill()
    req = ev["case:RequestedAmount"]
    ev["offered_to_requested"] = ev["OfferedAmount"] / req.where(req > 0)
    orig = raw.set_index("EventID")[ACT]
    label = pd.Series(orig.reindex(ev["EventID"]).to_numpy(), index=ev.index).where(sent)
    ev["last_sent_label"] = label.groupby(ev[CASE], sort=False).ffill().fillna("no O_Sent yet")
    ts = ev[TS]
    start = g[TS].transform("min")
    first_offer = ts.where(offer).groupby(ev[CASE], sort=False).transform("first").where(ev["n_offers"] > 0)
    last_sent = ts.where(sent).groupby(ev[CASE], sort=False).ffill()
    day = pd.Timedelta(days=1)
    ev["days_since_start"] = (ts - start) / day
    ev["days_since_first_offer"] = (ts - first_offer) / day
    ev["days_since_last_sent"] = (ts - last_sent) / day
    create = ts.where(act == "A_Create Application").groupby(ev[CASE], sort=False).transform("first")
    local = create.dt.tz_convert(TZ)
    ev["create_weekday"], ev["create_hour"] = local.dt.day_name(), local.dt.hour + local.dt.minute / 60
    now = ts.dt.tz_convert(TZ)
    ev["last_weekday"], ev["last_hour"] = now.dt.day_name(), now.dt.hour + now.dt.minute / 60
    ev["last_resource"] = ev[RESOURCE]
    return ev


def visits(ev: pd.DataFrame, moves: dict, place_branches: dict) -> pd.DataFrame:
    """One row per visit of a decision place: case, decision point, branch label, row of the last
    event before the decision."""
    traces = ev.groupby(CASE, sort=False)[ACT].agg(tuple)
    offset = ev.groupby(CASE, sort=False).cumcount()
    first_row = pd.Series(ev.index[offset == 0], index=ev.loc[offset == 0, CASE].to_numpy())
    rows = []
    for case, var in traces.items():
        for k, t in moves.get(var, []):
            for dp, label in place_branches.get(t, []):
                rows.append((case, dp, label, first_row[case] + k - 1 if k > 0 else -1))
    return pd.DataFrame(rows, columns=[CASE, "decision_point", "label", "event_row"])


# ---------------------------------------------------------------------------------- models
def preprocess(cat, num, for_mlp: bool):
    num_pipe = (Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]) if for_mlp else "passthrough")
    cat_pipe = Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="missing")),
                         ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=50,
                                                  sparse_output=False))])
    return ColumnTransformer([("num", num_pipe, num), ("cat", cat_pipe, cat)], verbose_feature_names_out=True)


def origin(names, cat, num):
    """Transformed column -> original feature (longest matching feature name)."""
    feats = sorted(cat + num, key=len, reverse=True)
    out = []
    for n in names:
        body = n.split("__", 1)[1].replace("missingindicator_", "")
        out.append(next(f for f in feats if body == f or body.startswith(f + "_")))
    return out


def readable(name: str, cat: list) -> tuple[str, bool]:
    body = name.split("__", 1)[1]
    for f in sorted(cat, key=len, reverse=True):
        if body.startswith(f + "_"):
            return f"{f} = {body[len(f) + 1:]}", True
    return body, False


def leaf_rules(tree, names, cat, X_test, y_test, has_nan) -> list[dict]:
    """Root-to-leaf rules with the class distribution of the test rows that reach the leaf."""
    t = tree.tree_
    leaf = tree.apply(X_test)
    out = []

    def walk(node, conds):
        if t.children_left[node] == -1:
            ys = pd.Series(y_test[leaf == node])
            dist = ys.value_counts(normalize=True).round(3).to_dict()
            out.append({"if": conds or ["(always)"], "then": str(tree.classes_[np.argmax(t.value[node][0])]),
                        "test_rows": int(len(ys)), "test_distribution": dist})
            return
        name, onehot = readable(names[t.feature[node]], cat)
        thr = t.threshold[node]
        miss_left = bool(t.missing_go_to_left[node])
        if onehot:
            left, right = name.replace(" = ", " != "), name
        else:
            left, right = f"{name} <= {thr:.4g}", f"{name} > {thr:.4g}"
        if not onehot and has_nan[t.feature[node]]:  # only for features with missing values
            if miss_left:
                left += " (or missing)"
            else:
                right += " (or missing)"
        walk(t.children_left[node], conds + [left])
        walk(t.children_right[node], conds + [right])

    walk(0, [])
    return out


def scores(y, p) -> dict:
    return {"accuracy": accuracy_score(y, p), "balanced_accuracy": balanced_accuracy_score(y, p),
            "macro_f1": f1_score(y, p, average="macro")}


def split_features(tree, orig) -> list[str]:
    used = tree.tree_.feature[tree.tree_.feature >= 0]
    return list(dict.fromkeys(orig[i] for i in used))


def shap_importance(mlp, Xt_train, Xt_test, orig) -> pd.Series:
    rng = np.random.default_rng(SEED)
    bg = Xt_train[rng.choice(len(Xt_train), min(100, len(Xt_train)), replace=False)]
    ex_rows = Xt_test[rng.choice(len(Xt_test), min(500, len(Xt_test)), replace=False)]
    f = Xt_train.shape[1]
    explainer = shap.PermutationExplainer(mlp.predict_proba, shap.maskers.Independent(bg), seed=SEED)
    sv = explainer(ex_rows, max_evals=2 * (2 * f + 1), silent=True).values  # (rows, features, classes)
    if sv.ndim == 2:
        sv = sv[:, :, None]
    grouped = pd.DataFrame(np.transpose(sv, (0, 2, 1)).reshape(-1, f), columns=orig).T.groupby(level=0).sum().T
    return grouped.abs().mean().sort_values(ascending=False)


def run(dp, fs, data):
    cat, num = FEATURE_SETS[fs]
    X, y, groups = data[cat + num], data["label"].to_numpy(), data[CASE].to_numpy()
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=SEED).split(X, y, groups))
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    majority = pd.Series(ytr).mode()[0]
    base = {"decision_point": dp, "decision": DECISIONS[dp][0], "feature_set": fs,
            "visits": len(X), "cases": int(pd.Series(groups).nunique()), "train_rows": len(tr), "test_rows": len(te),
            "classes": "; ".join(f"{c} {v:.3f}" for c, v in pd.Series(y).value_counts(normalize=True).items())}
    rows = [{**base, "model": "majority baseline", **scores(yte, np.full(len(yte), majority))}]

    pre_dt = preprocess(cat, num, for_mlp=False).fit(Xtr)
    names_dt = list(pre_dt.get_feature_names_out())
    orig_dt = origin(names_dt, cat, num)
    Ttr, Tte = pre_dt.transform(Xtr).astype(float), pre_dt.transform(Xte).astype(float)
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=SEED).fit(Ttr, ytr)
    dt_feats = split_features(dt, orig_dt)
    rows.append({**base, "model": "decision tree (depth 3, balanced)", **scores(yte, dt.predict(Tte)),
                 "split_features": "; ".join(dt_feats)})

    pre_mlp = preprocess(cat, num, for_mlp=True).fit(Xtr)
    orig_mlp = origin(list(pre_mlp.get_feature_names_out()), cat, num)
    Mtr, Mte = pre_mlp.transform(Xtr).astype(float), pre_mlp.transform(Xte).astype(float)
    mlp = MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, validation_fraction=0.1, max_iter=300,
                        random_state=SEED).fit(Mtr, ytr)
    p_tr, p_te = mlp.predict(Mtr), mlp.predict(Mte)
    imp = shap_importance(mlp, Mtr, Mte, orig_mlp)
    k = max(1, len(dt_feats))
    top = list(imp.index[:k])
    rows.append({**base, "model": "MLP (32, 16)", **scores(yte, p_te), "mlp_iterations": mlp.n_iter_,
                 "shap_top": "; ".join(f"{f} {v:.3f}" for f, v in imp.head(5).items()),
                 "shap_topk_in_dt_splits": len(set(top) & set(dt_feats)) / k})

    sur = DecisionTreeClassifier(max_depth=3, random_state=SEED).fit(Ttr, p_tr)
    sur_feats = split_features(sur, orig_dt)
    rows.append({**base, "model": "surrogate tree of the MLP (depth 3)", **scores(yte, sur.predict(Tte)),
                 "fidelity": accuracy_score(p_te, sur.predict(Tte)), "split_features": "; ".join(sur_feats),
                 "shap_topk_in_dt_splits": len(set(imp.index[:max(1, len(sur_feats))]) & set(sur_feats))
                 / max(1, len(sur_feats))})

    has_nan = np.isnan(Ttr).any(axis=0)
    rules = {"decision tree": {"text": export_text(dt, feature_names=names_dt, show_weights=False),
                               "leaves": leaf_rules(dt, names_dt, cat, Tte, yte, has_nan)},
             "surrogate": {"text": export_text(sur, feature_names=names_dt),
                           "leaves": leaf_rules(sur, names_dt, cat, Tte, yte, has_nan)}}
    return rows, rules, imp


def plot_shap(imp: pd.Series, title: str, path):
    top = imp.head(10)[::-1]
    fig, ax = plt.subplots(figsize=(W, 0.25 * len(top) + 0.6))
    ax.barh(range(len(top)), top.to_numpy(), color=C[0], height=0.6)
    ax.set_yticks(range(len(top)), [n.replace("case:", "") for n in top.index], fontsize=7)
    ax.set_xlabel("mean |SHAP value| (MLP)")
    ax.set_title(title, fontsize=8, color=INK2, loc="left")
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main():
    raw = load_log()
    ev, cfg = final_log(keep_attrs=True, df=raw)
    print(f"final configuration {cfg['id']}: {ev[CASE].nunique():,} cases", flush=True)
    moves = moves_per_variant(ev[[CASE, ACT, TS]])
    dpt = pd.read_csv(RES / "decision_points.csv")
    place_branches = {}
    for r in dpt[dpt["decision_point"].isin(DECISIONS)].itertuples():
        label = r.branch_label if r.branch_label != "tau" else DECISIONS[r.decision_point][1]
        place_branches.setdefault(r.branch_transition, []).append((r.decision_point, label))
    state = event_state(ev, raw)
    vis = visits(ev, moves, place_branches)
    print(vis.groupby("decision_point")["label"].value_counts().to_string(), flush=True)
    assert (vis["event_row"] >= 0).all()
    feats = sorted({c for cat, num in FEATURE_SETS.values() for c in cat + num})
    data = vis.join(state[feats], on="event_row")

    rows, rules, shap_out = [], {}, {}
    for dp, fs in RUNS:
        r, rl, imp = run(dp, fs, data[data["decision_point"] == dp].reset_index(drop=True))
        rows += r
        rules[f"{dp} {fs}"] = rl
        shap_out[f"{dp} {fs}"] = imp.round(4).to_dict()
        if fs in ("base", "leakage"):
            name = f"shap_{dp}" + ("_leakage" if fs == "leakage" else "")
            plot_shap(imp, f"{dp} {DECISIONS[dp][0]}" + (" (with leakage features)" if fs == "leakage" else ""),
                      FIG / f"{name}.pdf")
        for x in r:
            print(f"{dp} {fs:22s} {x['model']:38s} acc {x['accuracy']:.3f} bal {x['balanced_accuracy']:.3f} "
                  f"F1 {x['macro_f1']:.3f}" + (f" fidelity {x['fidelity']:.3f}" if "fidelity" in x else ""),
                  flush=True)
        print(rl["decision tree"]["text"], flush=True)
        pd.DataFrame(rows).to_csv(RES / "decision_mining.csv", index=False)
        (RES / "decision_rules.json").write_text(json.dumps(rules, indent=2))
        (RES / "decision_mining_shap.json").write_text(json.dumps(shap_out, indent=2))


if __name__ == "__main__":
    main()
