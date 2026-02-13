import argparse
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

MATCH_THRESHOLD = 70.0
WEIGHTS = {
    "name_fuzzy": 0.80,
    "street_fuzzy": 0.20,
    "city_fuzzy": 0.10,
    "duns_exact": 0.90,
    "vat_exact": 0.95,
    "cross_id_match": 1.00,
    "crefo_exact": 0.80,
}
TOP_NAME_CANDIDATES = 25

UMLAUT_MAP = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"})


def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().translate(UMLAUT_MAP)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[\.,\-/]", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("strasse", "str")
    text = text.replace("straße", "str")
    text = text.replace("str.", "str")
    return text.strip()


def normalize_id(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", normalize_text(value))


def fuzzy_score(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return float(fuzz.token_sort_ratio(left, right))


def build_address(street: object, house_number: object) -> str:
    return normalize_text(f"{street or ''} {house_number or ''}")


def calculate_match_score(customer: pd.Series, supplier: pd.Series, has_customer_crefo: bool) -> tuple[float, dict[str, float]]:
    details: dict[str, float] = {}

    details["name_fuzzy"] = fuzzy_score(customer["name_norm"], supplier["name_norm"])
    details["street_fuzzy"] = fuzzy_score(customer["address_norm"], supplier["address_norm"])
    details["city_fuzzy"] = fuzzy_score(customer["city_norm"], supplier["city_norm"])

    details["duns_exact"] = 100.0 if customer["duns_norm"] and customer["duns_norm"] == supplier["duns_norm"] else 0.0
    details["vat_exact"] = 100.0 if customer["vat_norm"] and customer["vat_norm"] == supplier["vat_norm"] else 0.0

    cross = (
        (customer["debitor_norm"] and customer["debitor_norm"] == supplier["debitor_norm"])
        or (customer["kreditor_norm"] and customer["kreditor_norm"] == supplier["kreditor_norm"])
        or (customer["debitor_norm"] and customer["debitor_norm"] == supplier["kreditor_norm"])
        or (customer["kreditor_norm"] and customer["kreditor_norm"] == supplier["debitor_norm"])
    )
    details["cross_id_match"] = 100.0 if cross else 0.0

    if has_customer_crefo:
        details["crefo_exact"] = 100.0 if customer["crefo_norm"] and customer["crefo_norm"] == supplier["crefo_norm"] else 0.0
    else:
        details["crefo_exact"] = 0.0

    score = sum((details[key] * weight) for key, weight in WEIGHTS.items()) / sum(WEIGHTS.values())
    return score, details


def prepare_dataframes(customers: pd.DataFrame, suppliers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    customers = customers.copy()
    suppliers = suppliers.copy()

    customer_crefo_column = "Kreditauskunftsnummer" if "Kreditauskunftsnummer" in customers.columns else None

    customers["debitor_norm"] = customers["Debitor"].map(normalize_id)
    customers["kreditor_norm"] = customers["Kreditor"].map(normalize_id)
    customers["duns_norm"] = customers["DUNS Nummer"].map(normalize_id)
    customers["vat_norm"] = customers["Umsatzsteuer-Id.Nr"].map(normalize_id)
    customers["name_norm"] = customers["Name1"].map(normalize_text)
    customers["city_norm"] = customers["Ort"].map(normalize_text)
    customers["country_norm"] = customers["Land"].map(normalize_text)
    customers["address_norm"] = customers.apply(lambda row: build_address(row.get("Straße", ""), row.get("Hausnummer", "")), axis=1)
    customers["crefo_norm"] = customers[customer_crefo_column].map(normalize_id) if customer_crefo_column else ""

    suppliers["kreditor_norm"] = suppliers["Kreditor"].map(normalize_id)
    suppliers["debitor_norm"] = suppliers["Debitor"].map(normalize_id)
    suppliers["duns_norm"] = suppliers["DUNS Nummer"].map(normalize_id)
    suppliers["vat_norm"] = suppliers["Umsatzsteuer-Id.Nr"].map(normalize_id)
    suppliers["crefo_norm"] = suppliers["Kreditauskunftsnummer"].map(normalize_id)
    suppliers["name_norm"] = suppliers["Name"].map(normalize_text)
    suppliers["city_norm"] = suppliers["Ort"].map(normalize_text)
    suppliers["country_norm"] = suppliers["Land"].map(normalize_text)
    suppliers["address_norm"] = suppliers.apply(lambda row: build_address(row.get("Straße", ""), row.get("Hausnummer", "")), axis=1)

    return customers, suppliers, bool(customer_crefo_column)


def build_supplier_indexes(suppliers: pd.DataFrame) -> dict[str, dict[str, set[int]]]:
    indexes = {
        "duns_norm": defaultdict(set),
        "vat_norm": defaultdict(set),
        "debitor_norm": defaultdict(set),
        "kreditor_norm": defaultdict(set),
        "country_norm": defaultdict(set),
    }
    for idx, row in suppliers.iterrows():
        for field in ("duns_norm", "vat_norm", "debitor_norm", "kreditor_norm", "country_norm"):
            if row[field]:
                indexes[field][row[field]].add(idx)
    return indexes


def find_candidate_suppliers(customer: pd.Series, suppliers: pd.DataFrame, indexes: dict[str, dict[str, set[int]]]) -> set[int]:
    candidates: set[int] = set()
    for field in ("duns_norm", "vat_norm", "debitor_norm", "kreditor_norm"):
        value = customer[field]
        if value:
            candidates.update(indexes[field].get(value, set()))

    country_candidates = indexes["country_norm"].get(customer["country_norm"], set()) if customer["country_norm"] else set(suppliers.index)
    if country_candidates:
        supplier_names = {idx: suppliers.at[idx, "name_norm"] for idx in country_candidates}
        name_query = customer["name_norm"]
        if name_query:
            top = process.extract(name_query, supplier_names, scorer=fuzz.token_sort_ratio, limit=TOP_NAME_CANDIDATES)
            candidates.update([candidate_id for _, _, candidate_id in top])
        else:
            candidates.update(list(country_candidates)[:TOP_NAME_CANDIDATES])

    return candidates or set(country_candidates)


def match_entities(customers: pd.DataFrame, suppliers: pd.DataFrame, threshold: float) -> pd.DataFrame:
    indexes = build_supplier_indexes(suppliers)
    has_customer_crefo = "crefo_norm" in customers.columns and customers["crefo_norm"].astype(bool).any()
    matches: list[dict[str, object]] = []

    for c_idx, customer in customers.iterrows():
        candidates = find_candidate_suppliers(customer, suppliers, indexes)
        for s_idx in candidates:
            supplier = suppliers.loc[s_idx]
            score, details = calculate_match_score(customer, supplier, has_customer_crefo)
            if score >= threshold:
                matches.append(
                    {
                        "customer_index": c_idx,
                        "supplier_index": s_idx,
                        "customer_debitor": customer["Debitor"],
                        "supplier_kreditor": supplier["Kreditor"],
                        "customer_name": customer["Name1"],
                        "supplier_name": supplier["Name"],
                        "score": round(score, 2),
                        **{f"detail_{k}": round(v, 2) for k, v in details.items()},
                    }
                )

    if not matches:
        return pd.DataFrame(columns=["customer_index", "supplier_index", "customer_debitor", "supplier_kreditor", "score"])

    return pd.DataFrame(matches).sort_values("score", ascending=False)


def build_output_tables(customers: pd.DataFrame, suppliers: pd.DataFrame, matches: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    customer_match_counts = matches.groupby("customer_index")["supplier_index"].nunique() if not matches.empty else pd.Series(dtype=int)
    supplier_match_counts = matches.groupby("supplier_index")["customer_index"].nunique() if not matches.empty else pd.Series(dtype=int)

    unmatched_customers = customers.loc[~customers.index.isin(customer_match_counts.index)].copy()
    unmatched_customers["entity_type"] = "customer"
    unmatched_customers["entity_id"] = unmatched_customers["Debitor"]

    unmatched_suppliers = suppliers.loc[~suppliers.index.isin(supplier_match_counts.index)].copy()
    unmatched_suppliers["entity_type"] = "supplier"
    unmatched_suppliers["entity_id"] = unmatched_suppliers["Kreditor"]

    unique_entities = pd.concat(
        [
            unmatched_customers[["entity_type", "entity_id", "Name1", "Land", "LöSp"]].rename(columns={"Name1": "name", "LöSp": "lock_flag"}),
            unmatched_suppliers[["entity_type", "entity_id", "Name", "Land", "LöVm"]].rename(columns={"Name": "name", "LöVm": "lock_flag"}),
        ],
        ignore_index=True,
    )

    multi_customer_ids = set(customer_match_counts[customer_match_counts > 1].index.tolist())
    multi_supplier_ids = set(supplier_match_counts[supplier_match_counts > 1].index.tolist())
    multi_matches = matches[
        matches["customer_index"].isin(multi_customer_ids) | matches["supplier_index"].isin(multi_supplier_ids)
    ].copy()
    if not multi_matches.empty:
        multi_matches["customer_match_count"] = multi_matches["customer_index"].map(customer_match_counts)
        multi_matches["supplier_match_count"] = multi_matches["supplier_index"].map(supplier_match_counts)

    one_to_one = matches[
        matches["customer_index"].isin(customer_match_counts[customer_match_counts == 1].index)
        & matches["supplier_index"].isin(supplier_match_counts[supplier_match_counts == 1].index)
    ].copy()

    return unique_entities, multi_matches, one_to_one


def write_outputs(unique_entities: pd.DataFrame, multi_matches: pd.DataFrame, one_to_one: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    unique_entities.to_excel(output_dir / "unique_entities.xlsx", index=False)
    multi_matches.to_excel(output_dir / "multi_matches.xlsx", index=False)
    one_to_one.to_excel(output_dir / "one_to_one_matches.xlsx", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Match customers and suppliers with weighted fuzzy logic")
    parser.add_argument("--customers", required=True, help="Path to customer CSV (KNA1 export)")
    parser.add_argument("--suppliers", required=True, help="Path to supplier CSV (LFA1 export)")
    parser.add_argument("--output-dir", default="output", help="Directory for Excel outputs")
    parser.add_argument("--threshold", type=float, default=MATCH_THRESHOLD, help="Match threshold (0-100)")
    parser.add_argument("--sep", default=",", help="CSV separator")
    args = parser.parse_args()

    customers = pd.read_csv(args.customers, sep=args.sep, dtype=str).fillna("")
    suppliers = pd.read_csv(args.suppliers, sep=args.sep, dtype=str).fillna("")

    customers, suppliers, _ = prepare_dataframes(customers, suppliers)
    matches = match_entities(customers, suppliers, args.threshold)
    unique_entities, multi_matches, one_to_one = build_output_tables(customers, suppliers, matches)
    write_outputs(unique_entities, multi_matches, one_to_one, Path(args.output_dir))


if __name__ == "__main__":
    main()
