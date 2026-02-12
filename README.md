# Kunden- und Lieferantendaten Matching (Jupyter + CLI)

Dieses Projekt vergleicht Kunden- und Lieferantenstammdaten, berechnet eine konfigurierbare Match-Quote und erzeugt pro Land eine Excel-Datei mit drei Match-Sheets:

- `1_1 Match`
- `1_N Match`
- `N_M Match`

Die **Kreditauskunftsnummer (Crefo)** wird **nicht** in die Quote einbezogen, aber als `supplier_cluster_id` zur Gruppierung angezeigt.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Projektstruktur

- `src/matcher.py` – Pipeline + CLI
- `config.example.yaml` – konfigurierbare Feldzuordnung, Filter, Gewichte, threshold
- `data/customers.csv` – Beispiel-Kundendaten
- `data/vendors.csv` – Beispiel-Lieferantendaten
- `notebook/Kunden_Lieferanten_Matching.ipynb` – Jupyter-Notebook
- `output/` – erzeugte Ergebnisdateien

## Nutzung (CLI)

```bash
python src/matcher.py --config config.example.yaml
```

Ergebnisse:

- `output/all_matches_scored.csv`
- `output/clean_customers.csv`
- `output/clean_vendors.csv`
- `output/matches_<LAND>.xlsx`

## Jupyter Notebook

Notebook starten:

```bash
jupyter lab
```

Dann `notebook/Kunden_Lieferanten_Matching.ipynb` öffnen und die Zellen ausführen.

## Fachregeln (umgesetzt)

- Segmentfilter Kunde: `HC01`, `ZODR`, `ZKZG`, `ZCPL`
- Segmentfilter Lieferant: `SPES`
- B2C-Kunden werden ausgeschlossen
- Matching-Felder: Name, Adresse, Ort, Land, USt-ID, DUNS, optional Bank, Linkfelder, Löschvermerk
- Crefo nur informativ (kein Scoring)
- Threshold frei konfigurierbar (z. B. `0.70`)
- Länderspezifischer Excel-Export mit 3 Sheets
