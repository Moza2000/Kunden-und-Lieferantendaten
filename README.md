# Kunden-/Lieferanten-Matching

Dieses Skript matched KNA1- und LFA1-CSV-Dateien über eine gewichtete Score-Logik mit Fuzzy-Matching.

## Ausführung

```bash
python match_entities.py --customers kundentabelle.csv --suppliers lieferanten.csv --sep ';' --output-dir output
```

## Output

- `unique_entities.xlsx` (Kunden/Lieferanten ohne Match)
- `multi_matches.xlsx` (1:n oder n:1 Matches)
- `one_to_one_matches.xlsx` (eindeutige 1:1 Matches)
