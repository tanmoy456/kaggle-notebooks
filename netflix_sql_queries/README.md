# Netflix Data Exploration with SQL

SQL practice using the Netflix Movies & TV Shows dataset — imported into a local MySQL server and explored via MySQL Workbench.

---

## Dataset

| Detail | Info |
|--------|------|
| **Name** | Netflix Movies and TV Shows |
| **Source** | [Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows) |
| **License** | CC0 — Public Domain |
| **Records** | 8,807 titles (Movies & TV Shows) |

The dataset contains listings of all movies and TV shows available on Netflix, along with details such as cast, directors, ratings, release year, duration, country, genre categories, and descriptions.

### Columns

| Column | Description |
|--------|-------------|
| `show_id` | Unique ID for every title (`s1`, `s2`, ...) |
| `type` | `Movie` or `TV Show` |
| `title` | Title of the show/movie |
| `director` | Director(s) — can be empty |
| `cast` | Actors involved — can be empty |
| `country` | Country where the title was produced — can be empty |
| `date_added` | Date it was added on Netflix (e.g. `September 25, 2021`) |
| `release_year` | Original release year |
| `rating` | Content rating (`PG-13`, `TV-MA`, `TV-14`, etc.) |
| `duration` | Runtime in minutes (movies) or number of seasons (TV shows) |
| `listed_in` | Genre categories |
| `description` | Short synopsis |

---

## The Encoding Problem

The raw CSV downloaded from Kaggle is encoded in **UTF-8** and contains **7,174 non-ASCII characters** — mostly accented characters in international cast/director names and smart punctuation in descriptions.

### Examples of non-ASCII characters in the data

| Character | Unicode | Count | Where it appears |
|-----------|---------|-------|------------------|
| `é` | U+00E9 | 973 | Actor/director names (e.g. `André`, `José`) |
| `á` | U+00E1 | 767 | Names (e.g. `García`, `Fialová`) |
| `'` | U+2019 | 525 | Smart quotes in descriptions (e.g. `India's`) |
| `–` | U+2013 | 608 | En dashes in descriptions |
| `—` | U+2014 | 284 | Em dashes in descriptions |
| `ü` | U+00FC | 384 | Names (e.g. `Düzyatan`, `Müller`) |
| `ñ` | U+00F1 | 201 | Spanish names (e.g. `Señor`, `Ibáñez`) |

### What goes wrong in MySQL Workbench

When tried to import the raw CSV using the **Table Data Import Wizard**, Workbench throws this error:

```
Unhandled exception: 'ascii' codec can't decode byte 0xe2
in position 1941: ordinal not in range(128)
```

**Why it happens:** The wizard has a known bug — it internally reads the file through Python's ASCII codec, regardless of the encoding dropdown showing `utf-8`. ASCII only handles bytes 0–127. The first multi-byte UTF-8 character it encounters (`'` at byte position 1941, in the Kota Factory description: `"...to train India's finest collegiate minds..."`) causes the crash.

**This is a Workbench bug, not a problem with the CSV file.** The file is valid UTF-8.

### How to fix it — clean the CSV

Before importing, convert the file to pure ASCII by transliterating non-ASCII characters to their closest equivalents.

**Option A — Quick one-liner (macOS/Linux terminal):**

```bash
iconv -f UTF-8 -t ASCII//TRANSLIT netflix_titles.csv > netflix_titles_clean.csv
```

**Option B — Python script (more control):**

```python
import unicodedata

with open("netflix_titles.csv", "r", encoding="utf-8") as f:
    text = f.read()

# Replace smart punctuation with ASCII equivalents
replacements = {
    "\u2018": "'",     # left single quote  -> apostrophe
    "\u2019": "'",     # right single quote  -> apostrophe
    "\u201C": '"',     # left double quote   -> straight quote
    "\u201D": '"',     # right double quote  -> straight quote
    "\u2013": "-",     # en dash             -> hyphen
    "\u2014": "--",    # em dash             -> double hyphen
    "\u2026": "...",   # ellipsis            -> three dots
}

for old, new in replacements.items():
    text = text.replace(old, new)

# Transliterate accented characters (é -> e, ñ -> n, etc.)
clean = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")

with open("netflix_titles_clean.csv", "w") as f:
    f.write(clean)

print("Done — clean file written.")
```

**What the cleaning does:**

- `Niewöhner` → `Niewohner`
- `Hasanović` → `Hasanovic`
- `India's` → `India's`
- `a tempting prize awaits — with deadly high stakes` → `a tempting prize awaits -- with deadly high stakes`

All 8,807 rows are preserved. Only the encoding of special characters changes.

The cleaned CSV file (`netflix_titles_clean.csv`) is included in the `datasets/` folder of this repo.

---

## How to Import into MySQL Workbench (No Schema Required)

The Table Data Import Wizard auto-creates the table — no need to write `CREATE TABLE` manually.

### Prerequisites

- MySQL Server installed and running
- MySQL Workbench installed and connected to your local server
- The cleaned CSV file (`netflix_titles_clean.csv`)

### Steps

1. **Open Workbench** and connect to local MySQL server.

2. **Create a database** (if don't have one). In a query tab, run:
   ```sql
   CREATE DATABASE netflix;
   USE netflix;
   ```

3. **Open the Import Wizard:**
   - In the left Navigator panel, expand the database under SCHEMAS.
   - Right-click on **Tables** → **Table Data Import Wizard**.

4. **Select the file:**
   - Browse to `netflix_titles_clean.csv`.
   - Click **Next**.

5. **Choose target:**
   - Select **Create new table**.
   - Pick the database (e.g. `netflix`) and give the table a name (e.g. `titles`).
   - Click **Next**.

6. **Review columns:**
   - The wizard auto-detects column names from the CSV header.
   - It guesses data types (`VARCHAR`, `INT`, `TEXT`).
   - Can rename columns or change types here if needed.
   - Click **Next**.

7. **Run the import:**
   - Click **Next** to start importing.
   - Wait for it to finish — it shows a progress bar and row count.
   - Click **Finish**.

8. **Refresh and verify:**
   - Right-click in the SCHEMAS panel → **Refresh All**.
   - New table should appear under your database.
   - Run a quick check:
     ```sql
     SELECT COUNT(*) FROM titles;
     -- Should return 8807

     SELECT * FROM titles LIMIT 5;
     ```

---
