# Ranked Pairs Voting

This document describes how the department uses ranked-choice voting and how to run the script that tallies the votes. Hiring decisions are the most common use (who to invite for on-campus interviews, who to make an offer to), but the same method works for any departmental vote with more than two options.

## Running a vote

### Step 1: Create the Qualtrics survey

Set up a Qualtrics survey with a single question listing all options. Use the "Form field" question type, with one text field per option where voters enter a positive integer (1 = top choice, 2 = second choice, etc.).

![Qualtrics survey setup example](qualtrics-setup.png)

Voters can give different options the same number (a tie in their personal ranking), skip options they don't want to rank, and use a threshold option (e.g. "Do not hire", "None of the above") to indicate that anything ranked below it is unacceptable.

### Step 2: Configure security settings

To ensure each faculty member can vote only once while keeping votes anonymous (even from you):

1. **Enable Invitation-Only Access:**
   - Go to **Survey** → **Survey Options** (icon on the left) → **Security**
   - Set **Survey Access** to **"Invitation only"**
   - Enable **"Prevent multiple submissions"**

2. **Enable Anonymization:**
   - In the same Security tab, enable **"Anonymize responses"**
   - This disconnects responses from contact records

References:
- [Qualtrics: Security Survey Options](https://www.qualtrics.com/support/survey-platform/survey-module/survey-options/survey-protection/)
- [Qualtrics: Anonymize Responses](https://www.qualtrics.com/support/survey-platform/sp-administration/data-privacy-tab/anonymous-responses-admin/)

### Step 3: Collect votes

1. Go to **Distributions** → **Emails** → **Compose Email**
2. Click **Select Contacts** in the recipient field
3. Create a new contact list (or select existing):
   - Click **New Contact List**
   - Name it (e.g., "2025 Instructor Hiring Vote: Zoom interviews")
   - Enter emails manually or import a CSV
4. Compose the invitation email
5. Click **Send**

Each voter receives a unique, one-time-use link. Reminders can be sent to non-respondents without exposing votes.

### Step 4: Export from Qualtrics

1. Go to **Data & Analysis** in the Qualtrics survey
2. Click **Export & Import** → **Export Data**
3. Choose **CSV** format
4. Select **"Export values"** (not "Export labels")
5. Download and save the file

### Step 5: Run the app

Double-click `RankedPairsVoting.exe` to launch the GUI: select the exported ballot CSV, click "Run Ranked Pairs", and results are saved to the same folder as the CSV. A sample ballot file `sample-ballots.csv` is included for testing.

Or, from the command line:

```
python ranked_pairs_voting.py your-ballots.csv --verbose
```

Either method produces a results spreadsheet (`survey-results YYYY-MM-DD.xlsx`) and a final ranking graph (`resolved-ranking.pdf`).

## Method

The script uses a variant of Ranked Pairs (Tideman 1987). The output is a partial order on the candidates: every pair that the votes resolve cleanly is ordered, and any pair that the votes leave genuinely tied is left unordered, to be settled by the chair. When the votes resolve all pairs, the partial order is a linear ranking and there is nothing for the chair to do.

### Step 1: Build the head-to-head table

For each unordered pair of candidates `{A, B}`, count two numbers from the ballots:

- `n(A, B)` = the number of ballots that rank `A` strictly above `B`
- `n(B, A)` = the number of ballots that rank `B` strictly above `A`

Ballots that rank `A` and `B` equally, or that omit one of them, contribute to neither count.

If `n(A, B) > n(B, A)`, record an arrow `A → B` with margin `n(A, B) − n(B, A)`. If the two counts are exactly equal, record nothing for that pair.

### Step 2: Sort the arrows by margin

List every recorded arrow from largest margin to smallest. Group together any arrows that have exactly the same margin; these will be processed simultaneously.

### Step 3: Lock in arrows, group by group

Start with a blank diagram showing only the candidate names and no arrows. Then, working from the largest-margin group of arrows down to the smallest:

1. Tentatively draw every arrow in the current group on top of the diagram.
2. For each arrow `A → B` in the current group, check whether you can now follow arrows from `B` back to `A` (using any combination of arrows already locked in plus the tentative arrows from this group). If you can, the arrow `A → B` is part of a cycle — erase it.
3. The arrows from the current group that survive Step 3.2 are now locked in permanently.
4. Move on to the next group.

### Step 4: Read off the partial order

The locked-in diagram defines a partial ordering of the candidates: `A` is ranked above `B` if and only if there is a (directed) path from `A` to `B`. Some pairs may have no path between them in either direction; those are the unresolved pairs.

If for every two candidates there is a path one way or the other, the partial order is total and you can read off a linear ranking directly: the candidate with no incoming arrows is first, then remove that candidate and repeat.

Otherwise, the chair must choose any total ordering that is consistent with the diagram: the chair essentially adds arrows to the diagram so that there are no cycles and a total order is obtained. The script's spreadsheet groups tied candidates together with a `~` prefix, and `resolved-ranking.pdf` shows the diagram so the chair can see which orderings are acceptable. (This tie-breaking step is due to George Gilbert.)

### Why it works

The procedure guarantees a Condorcet winner when one exists. If some candidate `C` would beat every other candidate head-to-head, then for each other `X` the recorded arrow is `C → X` and there is no arrow into `C` anywhere. The arrow `C → X` can never be erased, because erasure requires a path from `X` back to `C`, which would have to enter `C` along an arrow that doesn't exist. So `C` ends up at the top of the diagram with a path to every other candidate.

It also avoids vote-splitting: similar candidates do not hurt each other, because every pair is judged on its own head-to-head margin rather than on first-place counts.

The cost of refusing to make arbitrary choices on equal-margin conflicts is that the procedure can return a partial order rather than a linear one.

## Output

### The Excel file

**Results sheet:**

| Candidate | Ranked Pairs Rank | Borda Score |
|-----------|-------------------|-------------|
| Smith     | 1                 | +24         |
| Jones     | 2                 | +12         |
| ...       | ...               | ...         |

The Ranked Pairs rank is the final ordering (1 = winner). The Borda score is the sum of pairwise margins across all matchups (wins minus losses); higher is better. Borda is what the department used historically and serves as a sanity check, but isn't as useful. Ranked Pairs and Borda usually agree; when they disagree, Ranked Pairs is the official result.

**Matchups sheet:** a matrix of head-to-head margins. Entry (A, B) shows how many more voters preferred A over B than B over A. Positive means A is preferred.

**Notes sheet:** warnings about ties requiring chair intervention, or ballots with missing votes.

### The graph

`resolved-ranking.pdf` shows the final ranking after applying Ranked Pairs as a clean top-to-bottom ordering. This is the official result. If there are ties, the title says "CHAIR REVIEW NEEDED".

## Ties

When the procedure leaves some pairs unresolved (because of equal head-to-head support or a rock-paper-scissors situation like A > B > C > A), the script flags it: the spreadsheet shows approximate rankings with a `~` prefix and `resolved-ranking.pdf` is titled "CHAIR REVIEW NEEDED". The chair then chooses any total ordering consistent with the diagram, as described in Step 4 of the Method. The algorithm has already resolved everything it can; the remaining ambiguity is a genuine tie in voter preferences.

## Edge cases

**Missing votes.** If a voter leaves some candidates unranked, those candidates are treated as "no preference" for that voter by default — they don't count as wins or losses against other candidates. Use `--missing-votes worst` to treat unranked candidates as the worst choice instead.

**Threshold options.** For hiring votes, include a "Do not hire" option; for other votes, something like "None of the above", as appropriate.

## Command-line options

- `--verbose`, `-v`: show progress information
- `--output FOLDER`: save results to a different folder
- `--missing-votes skip|worst|error`: how to handle missing votes

The Python script requires Python 3.9+, `pandas`, `networkx`, `openpyxl`, and `graphviz` (plus a system Graphviz install for PDF graph output). The `.exe` bundles everything.

## Building the executable

To rebuild the standalone `.exe`: ensure Python and pip are installed, run `build_exe_flet.bat`, and the executable will be created in the `dist` folder.

---

*For more on Ranked Pairs: [Wikipedia](https://en.wikipedia.org/wiki/Ranked_pairs). For the mathematical details, see `AlgorithmProposalGTG1.pdf`.*
