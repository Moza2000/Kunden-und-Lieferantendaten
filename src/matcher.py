from __future__ import annotations

import argparse
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import pandas as pd
import yaml
from rapidfuzz import fuzz


@dataclass
class MatchConfig:
    raw: dict

    @property
    def threshold(self) -> float:
        return float(self.raw["matching"]["threshold"])


def load_config(path: str) -> MatchConfig:
    with open(path, "r", encoding="utf-8") as f:
        return MatchConfig(yaml.safe_load(f))


def _norm_text(value: str, use_unicode_norm: bool = True) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().upper()
    text = re.sub(r"\s+", " ", text)
    if use_unicode_norm:
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return text


def _norm_alnum(value: str, use_unicode_norm: bool = True) -> str:
    text = _norm_text(value, use_unicode_norm)
    return re.sub(r"[^A-Z0-9]", "", text)


def _norm_digits(value: str) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\D", "", str(value))


def _to_bool_flag(value: str) -> bool:
    val = _norm_text(value, False)
    return val in {"X", "1", "TRUE", "JA", "Y"}


def read_sources(cfg: MatchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    inp = cfg.raw["input"]
    customers = pd.read_csv(inp["customers_csv"], dtype=str, sep=inp.get("sep", ";"), encoding=inp.get("encoding", "utf-8"))
    vendors = pd.read_csv(inp["vendors_csv"], dtype=str, sep=inp.get("sep", ";"), encoding=inp.get("encoding", "utf-8"))
    return customers.fillna(""), vendors.fillna("")


def validate_columns(df: pd.DataFrame, mapping: Dict[str, str], label: str, optional: List[str] = None) -> None:
    optional = optional or []
    required = set(v for k, v in mapping.items() if k not in optional and v)
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Fehlende Spalten in {label}: {missing}")


def prepare_customers(df: pd.DataFrame, cfg: MatchConfig) -> pd.DataFrame:
    m = cfg.raw["columns"]["customers"]
    validate_columns(df, m, "customers")
    unicode_norm = bool(cfg.raw["matching"].get("unicode_normalization", True))

    out = pd.DataFrame()
    out["customer_id"] = df[m["customer_id"]].astype(str)
    out["vendor_id_link"] = df[m["vendor_id_link"]].astype(str)
    out["name"] = df[m["name"]].map(lambda x: _norm_text(x, unicode_norm))
    out["street"] = df[m["street"]].map(lambda x: _norm_text(x, unicode_norm))
    out["house_no"] = df[m["house_no"]].map(lambda x: _norm_text(x, unicode_norm))
    out["address"] = (out["street"] + " " + out["house_no"]).str.strip()
    out["city"] = df[m["city"]].map(lambda x: _norm_text(x, unicode_norm))
    out["country"] = df[m["country"]].map(lambda x: _norm_text(x, False))
    out["vat_id"] = df[m["vat_id"]].map(lambda x: _norm_alnum(x, unicode_norm))
    out["duns"] = df[m["duns"]].map(_norm_digits)
    out["deletion_flag"] = df[m["deletion_flag"]].map(_to_bool_flag)
    out["segment"] = df[m["segment"]].map(lambda x: _norm_text(x, False))
    out["b2c_flag"] = df[m["b2c_flag"]].map(_to_bool_flag)
    out["bank"] = df[m.get("bank", "")].map(lambda x: _norm_alnum(x, unicode_norm)) if m.get("bank") else ""

    # duns plausibilisierung (nur 9-stellig behalten, sonst leer)
    out["duns"] = out["duns"].map(lambda x: x if len(x) == 9 else "")
    return out


def prepare_vendors(df: pd.DataFrame, cfg: MatchConfig) -> pd.DataFrame:
    m = cfg.raw["columns"]["vendors"]
    validate_columns(df, m, "vendors", optional=["segment", "bank"])
    unicode_norm = bool(cfg.raw["matching"].get("unicode_normalization", True))

    out = pd.DataFrame()
    out["vendor_id"] = df[m["vendor_id"]].astype(str)
    out["customer_id_link"] = df[m["customer_id_link"]].astype(str)
    out["name"] = df[m["name"]].map(lambda x: _norm_text(x, unicode_norm))
    out["street"] = df[m["street"]].map(lambda x: _norm_text(x, unicode_norm))
    out["house_no"] = df[m["house_no"]].map(lambda x: _norm_text(x, unicode_norm))
    out["address"] = (out["street"] + " " + out["house_no"]).str.strip()
    out["city"] = df[m["city"]].map(lambda x: _norm_text(x, unicode_norm))
    out["country"] = df[m["country"]].map(lambda x: _norm_text(x, False))
    out["vat_id"] = df[m["vat_id"]].map(lambda x: _norm_alnum(x, unicode_norm))
    out["duns"] = df[m["duns"]].map(_norm_digits)
    out["deletion_flag"] = df[m["deletion_flag"]].map(_to_bool_flag)
    # Segment ist optional - wenn nicht vorhanden, alle auf "ALL" setzen
    if m.get("segment") and m["segment"] in df.columns:
        out["segment"] = df[m["segment"]].map(lambda x: _norm_text(x, False))
    else:
        out["segment"] = "ALL"
    out["credit_bureau_no"] = df[m["credit_bureau_no"]].map(lambda x: _norm_alnum(x, unicode_norm))
    out["bank"] = df[m.get("bank", "")].map(lambda x: _norm_alnum(x, unicode_norm)) if m.get("bank") and m["bank"] in df.columns else ""

    out["duns"] = out["duns"].map(lambda x: x if len(x) == 9 else "")
    out["supplier_cluster_id"] = out["credit_bureau_no"].where(out["credit_bureau_no"] != "", "NO_CREFO")
    return out


def apply_filters(customers: pd.DataFrame, vendors: pd.DataFrame, cfg: MatchConfig) -> Tuple[pd.DataFrame, pd.DataFrame]:
    f = cfg.raw["filters"]
    customers_f = customers[customers["segment"].isin([s.upper() for s in f["customer_allowed_segments"]])].copy()
    if bool(f.get("exclude_b2c", True)):
        customers_f = customers_f[~customers_f["b2c_flag"]]
    # Vendor segment filter nur anwenden wenn Liste nicht leer
    vendor_segments = f.get("vendor_allowed_segments", [])
    if vendor_segments:
        vendors_f = vendors[vendors["segment"].isin([s.upper() for s in vendor_segments])].copy()
    else:
        vendors_f = vendors.copy()
    return customers_f.reset_index(drop=True), vendors_f.reset_index(drop=True)


def _field_score(a: str, b: str, mode: str) -> float | None:
    if a == "" and b == "":
        return None
    if mode == "exact":
        return 1.0 if a == b and a != "" else 0.0
    if a == "" or b == "":
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def build_candidates(customers: pd.DataFrame, vendors: pd.DataFrame) -> pd.DataFrame:
    pairs: List[pd.DataFrame] = []
    for country, c_group in customers.groupby("country"):
        v_group = vendors[vendors["country"] == country]
        if c_group.empty or v_group.empty:
            continue
        pairs.append(c_group.assign(_k=1).merge(v_group.assign(_k=1), on="_k", suffixes=("_c", "_v")).drop(columns=["_k"]))
    return pd.concat(pairs, ignore_index=True) if pairs else pd.DataFrame()


def score_pairs(candidates: pd.DataFrame, cfg: MatchConfig) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    weights = cfg.raw["scoring_weights"]
    missing_policy = cfg.raw["matching"].get("missing_policy", "one_side_missing_zero")
    link_mode = cfg.raw["matching"].get("link_field_mode", "weighted")

    rows = []
    for _, r in candidates.iterrows():
        metrics = {
            "name_score": _field_score(r["name_c"], r["name_v"], "fuzzy"),
            "address_score": _field_score(r["address_c"], r["address_v"], "fuzzy"),
            "city_score": _field_score(r["city_c"], r["city_v"], "fuzzy"),
            "country_score": _field_score(r["country_c"], r["country_v"], "exact"),
            "vat_score": _field_score(r["vat_id_c"], r["vat_id_v"], "exact"),
            "duns_score": _field_score(r["duns_c"], r["duns_v"], "exact"),
            "deletion_score": 1.0 if r["deletion_flag_c"] == r["deletion_flag_v"] else 0.0,
            "bank_score": _field_score(r["bank_c"], r["bank_v"], "exact"),
        }

        linked = (
            (str(r["vendor_id_link"]).strip() != "" and str(r["vendor_id_link"]).strip() == str(r["vendor_id"]).strip())
            or (str(r["customer_id_link"]).strip() != "" and str(r["customer_id_link"]).strip() == str(r["customer_id"]).strip())
        )
        metrics["link_score"] = 1.0 if linked else 0.0

        weighted_sum = 0.0
        weight_sum = 0.0
        field_to_weight = {
            "name_score": weights.get("name", 1.0),
            "address_score": weights.get("address", 1.0),
            "city_score": weights.get("city", 1.0),
            "country_score": weights.get("country", 1.0),
            "vat_score": weights.get("vat_id", 1.0),
            "duns_score": weights.get("duns", 1.0),
            "deletion_score": weights.get("deletion_flag", 0.5),
            "bank_score": weights.get("bank", 1.0),
            "link_score": weights.get("link_fields", 2.0),
        }

        for k, w in field_to_weight.items():
            s = metrics[k]
            if s is None:
                continue
            if missing_policy == "skip" and k in {"vat_score", "duns_score", "bank_score"}:
                source_pair = {
                    "vat_score": (r["vat_id_c"], r["vat_id_v"]),
                    "duns_score": (r["duns_c"], r["duns_v"]),
                    "bank_score": (r["bank_c"], r["bank_v"]),
                }[k]
                if "" in source_pair:
                    continue
            weighted_sum += w * float(s)
            weight_sum += w

        score = weighted_sum / weight_sum if weight_sum > 0 else 0.0
        if link_mode == "override" and linked:
            score = max(score, 1.0)

        row = r.to_dict()
        row.update(metrics)
        row["quote"] = round(score, 4)
        row["linked_via_lifnr_kunnr"] = linked
        rows.append(row)

    scored = pd.DataFrame(rows)
    return scored[scored["quote"] >= cfg.threshold].copy()


def classify_matches(scored: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scored.empty:
        cols = list(scored.columns) + ["match_type", "component_id", "top_match_customer", "top_match_vendor"]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    df = scored.copy()
    df["customer_degree"] = df.groupby("customer_id")["vendor_id"].transform("nunique")
    df["vendor_degree"] = df.groupby("vendor_id")["customer_id"].transform("nunique")

    df["top_match_customer"] = df.groupby("customer_id")["quote"].transform("max") == df["quote"]
    df["top_match_vendor"] = df.groupby("vendor_id")["quote"].transform("max") == df["quote"]

    g = nx.Graph()
    for _, r in df.iterrows():
        cnode = f"C::{r['customer_id']}"
        vnode = f"V::{r['vendor_id']}"
        g.add_edge(cnode, vnode)

    component_map = {}
    for idx, comp in enumerate(nx.connected_components(g), start=1):
        for node in comp:
            component_map[node] = idx

    def _component_id(row: pd.Series) -> int:
        return component_map.get(f"C::{row['customer_id']}")

    df["component_id"] = df.apply(_component_id, axis=1)

    component_stats = {}
    for cid, sub in df.groupby("component_id"):
        c_count = sub["customer_id"].nunique()
        v_count = sub["vendor_id"].nunique()
        component_stats[cid] = (c_count, v_count)

    def _match_type(row: pd.Series) -> str:
        c_count, v_count = component_stats[row["component_id"]]
        if c_count == 1 and v_count == 1:
            return "1_1 Match"
        if c_count >= 2 and v_count >= 2:
            return "N_M Match"
        return "1_N Match"

    df["match_type"] = df.apply(_match_type, axis=1)

    return (
        df[df["match_type"] == "1_1 Match"].copy(),
        df[df["match_type"] == "1_N Match"].copy(),
        df[df["match_type"] == "N_M Match"].copy(),
    )


def export_country_excels(scored: pd.DataFrame, cfg: MatchConfig) -> None:
    out_dir = Path(cfg.raw["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if scored.empty:
        return

    for country, df_country in scored.groupby("country_c"):
        m11, m1n, mnm = classify_matches(df_country)
        file_path = out_dir / f"matches_{country}.xlsx"
        with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
            m11.to_excel(writer, sheet_name="1_1 Match", index=False)
            m1n.to_excel(writer, sheet_name="1_N Match", index=False)
            mnm.to_excel(writer, sheet_name="N_M Match", index=False)


def run_pipeline(config_path: str) -> pd.DataFrame:
    cfg = load_config(config_path)
    customers_raw, vendors_raw = read_sources(cfg)
    customers = prepare_customers(customers_raw, cfg)
    vendors = prepare_vendors(vendors_raw, cfg)
    customers_f, vendors_f = apply_filters(customers, vendors, cfg)

    out_dir = Path(cfg.raw["output"]["directory"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg.raw["output"].get("write_clean_csv", True)):
        customers_f.to_csv(out_dir / "clean_customers.csv", index=False)
        vendors_f.to_csv(out_dir / "clean_vendors.csv", index=False)

    candidates = build_candidates(customers_f, vendors_f)
    scored = score_pairs(candidates, cfg)
    export_country_excels(scored, cfg)
    scored.to_csv(out_dir / "all_matches_scored.csv", index=False)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description="Kunden/Lieferanten Matching")
    parser.add_argument("--config", default="config.example.yaml", help="Pfad zur YAML-Konfiguration")
    args = parser.parse_args()

    scored = run_pipeline(args.config)
    print(f"Fertig. Matches über Threshold: {len(scored)}")


if __name__ == "__main__":
    main()
