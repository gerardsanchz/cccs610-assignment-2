# CCCS 610, Assignment 2: code, data, results and figures

Supporting material for Sebastián Vielmas's report "Assignment 2: Roads, Images, Text and Compromise" (McGill University, School of Continuing Studies, CCCS 610 Digital Thinking for Data Analysis, sessions 5 to 8). The report cites files here by path.

| Folder | Holds |
|---|---|
| `code/` | One Python script per session (`m05_roads.py`, `m06_leaves.py`, `m07_words.py`, `m08_*.py`). Each states how to run it in its docstring. |
| `results/` | The JSON files behind every number and table in the report. |
| `figures/` | The figures, the contact sheet of the photo steps, and the two animated GIFs. |
| `data/` | The inputs, below. |
| `warm-up-nlp-applications.md` | The author's warm-up note cited in session 7, item 4. |

## Data sources and licences

| File | Source | Licence |
|---|---|---|
| `results/m05-real-source.json` | Wikidata query service: population (P1082, 2021) and coordinates (P625) of the most populous Québec municipalities | CC0 1.0 |
| `data/leaves-original.jpg`, `data/leaves-1500.jpg` | "Autumn leaves on the ground in Quebec City.jpg", Wilfredor, Wikimedia Commons, taken 6 October 2018 (the second file is the same photo reduced to 1500 by 1000 pixels) | CC0 1.0 |
| `data/maria-chapdelaine-13525.txt` | Louis Hémon, *Maria Chapdelaine*, Project Gutenberg ebook 13525, with the Project Gutenberg header removed | Public domain in the USA, per the Project Gutenberg page |
| `data/m08-fuel-consumption-ratings-2026.csv` | Natural Resources Canada, 2026 Fuel Consumption Ratings (2026-10-06) | Open Government Licence – Canada |
| `data/m08-quebec-houses.csv.gz` | Rôle d'évaluation foncière du Québec, municipality 23027 (Ville de Québec), roll 2026, open data. Detached houses used as dwellings, one dwelling each, with their civic address, lot and floor area, year built and assessed value. It holds no owner names. | CC BY 4.0, attribution to the open-data publisher: https://donneesouvertes.affmunqc.net/role/RL23027_2026.xml |

Contains information licensed under the Open Government Licence – Canada.

The code and the report text are by Sebastián Vielmas; no licence is granted beyond what a reader needs to check the work.
