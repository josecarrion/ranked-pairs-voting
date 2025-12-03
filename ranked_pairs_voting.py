#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Ranked Pairs Voting System for Faculty Hiring

Processes Qualtrics survey ballots and ranks candidates using:
1. Tideman Ranked Pairs method (Condorcet-consistent)
2. Modified Borda count (pairwise margin sums)

Outputs:
- Excel file with rankings and matchup matrix
- PDF graphs showing victory relationships

When ties cannot be automatically resolved, the script flags this
for manual intervention by the committee chair.

Usage:
    python ranked_pairs_voting.py ballots.csv
    python ranked_pairs_voting.py ballots.csv --output results.xlsx

Author: José Carrión (j.carrion@tcu.edu)
Original: 2022-11-19
Refactored: 2025
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import networkx as nx
import pandas as pd

# Optional: socialchoice library for additional methods
try:
    from socialchoice import PairwiseBallotBox
    SOCIALCHOICE_AVAILABLE = True
except ImportError:
    SOCIALCHOICE_AVAILABLE = False


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class VotingResults:
    """Container for all voting results."""
    ranked_pairs_ranking: list[str]
    borda_scores: dict[str, float]
    matchup_matrix: pd.DataFrame
    ranked_pairs_graph: nx.DiGraph
    victory_graph: nx.DiGraph
    unused_edges: list[tuple[str, str, float]]
    has_ties: bool
    tie_description: str


# =============================================================================
# Ballot Loading and Parsing
# =============================================================================

def load_qualtrics_csv(filepath: Path) -> pd.DataFrame:
    """
    Load a Qualtrics CSV export file.

    Args:
        filepath: Path to the CSV file

    Returns:
        Raw DataFrame from the CSV

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is empty or malformed
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Ballot file not found: {filepath}")

    df = pd.read_csv(filepath, sep=",", dtype=str)

    if df.empty:
        raise ValueError(f"Ballot file is empty: {filepath}")

    if len(df) < 3:
        raise ValueError(
            f"Ballot file has fewer than 3 rows. Expected Qualtrics format with "
            f"header rows. Got {len(df)} rows."
        )

    return df


def find_question_columns(df: pd.DataFrame) -> list[str]:
    """
    Find the columns containing ballot questions (Q#_# format).

    Qualtrics exports have metadata columns followed by question columns.
    Question columns are named like Q6_1, Q6_2, etc.

    Args:
        df: Raw Qualtrics DataFrame

    Returns:
        List of column names containing ballot data
    """
    question_pattern = re.compile(r'^Q\d+_\d+$')
    question_cols = [col for col in df.columns if question_pattern.match(col)]

    if not question_cols:
        # Fallback: assume last N columns after standard Qualtrics metadata
        # Standard metadata columns: StartDate through UserLanguage (17 columns)
        metadata_cols = [
            'StartDate', 'EndDate', 'Status', 'IPAddress', 'Progress',
            'Duration (in seconds)', 'Finished', 'RecordedDate', 'ResponseId',
            'RecipientLastName', 'RecipientFirstName', 'RecipientEmail',
            'ExternalReference', 'LocationLatitude', 'LocationLongitude',
            'DistributionChannel', 'UserLanguage'
        ]
        # Find where metadata ends
        for i, col in enumerate(df.columns):
            if col not in metadata_cols and not col.startswith('{'):
                question_cols = list(df.columns[i:])
                break

    if not question_cols:
        raise ValueError("Could not identify question columns in the CSV")

    return question_cols


def extract_candidate_names(df: pd.DataFrame, question_cols: list[str]) -> list[str]:
    """
    Extract candidate names from the Qualtrics description row.

    The second row (index 0 after header) contains descriptions like:
    "Please write one positive integer... - Doran, Charles"

    We extract the part after " - ".

    Args:
        df: Raw Qualtrics DataFrame
        question_cols: Columns containing ballot questions

    Returns:
        List of candidate names in column order
    """
    description_row = df.iloc[0]
    candidates = []

    for col in question_cols:
        desc = str(description_row[col])
        # Extract text after " - " (the candidate name)
        match = re.search(r' - (.+)$', desc)
        if match:
            candidates.append(match.group(1).strip())
        else:
            # Fallback: use column name
            candidates.append(col)

    return candidates


def parse_ballots(
    df: pd.DataFrame,
    question_cols: list[str],
    missing_vote_strategy: str = "skip"
) -> tuple[list[list[Optional[int]]], list[str]]:
    """
    Parse ballot data from the DataFrame.

    Args:
        df: Raw Qualtrics DataFrame
        question_cols: Columns containing ballot questions
        missing_vote_strategy: How to handle missing votes
            - "skip": Treat as no preference (won't count in pairwise comparisons)
            - "worst": Assign worst possible rank
            - "error": Raise an error

    Returns:
        Tuple of (ballot_matrix, problematic_response_ids)
        ballot_matrix: List of ballots, each ballot is list of rankings (or None)
        problematic_response_ids: List of response IDs with issues
    """
    # Skip the first two rows (description and JSON metadata)
    ballot_df = df.iloc[2:][question_cols].copy()

    ballots = []
    problematic_ids = []

    for idx, row in ballot_df.iterrows():
        ballot = []
        has_problem = False

        for col in question_cols:
            value = row[col]

            if pd.isna(value) or value == '' or value is None:
                if missing_vote_strategy == "error":
                    response_id = df.iloc[idx].get('ResponseId', f'row {idx}')
                    raise ValueError(
                        f"Missing vote in ballot {response_id} for {col}"
                    )
                elif missing_vote_strategy == "worst":
                    ballot.append(999)  # Will be normalized later
                else:  # skip
                    ballot.append(None)
                has_problem = True
            else:
                try:
                    rank = int(float(value))
                    if rank < 1:
                        raise ValueError(f"Rank must be positive, got {rank}")
                    ballot.append(rank)
                except (ValueError, TypeError) as e:
                    if missing_vote_strategy == "error":
                        response_id = df.iloc[idx].get('ResponseId', f'row {idx}')
                        raise ValueError(
                            f"Invalid vote '{value}' in ballot {response_id}: {e}"
                        )
                    ballot.append(None)
                    has_problem = True

        ballots.append(ballot)
        if has_problem:
            response_id = df.iloc[idx + 2].get('ResponseId', f'row {idx}')
            problematic_ids.append(response_id)

    return ballots, problematic_ids


# =============================================================================
# Pairwise Comparison Building
# =============================================================================

def build_pairwise_comparisons(
    ballots: list[list[Optional[int]]],
    candidates: list[str]
) -> list[tuple[str, str, str]]:
    """
    Convert ranked ballots to pairwise win/loss/tie comparisons.

    For each pair of candidates, each ballot contributes:
    - "win" if the first candidate is ranked higher (smaller number)
    - "loss" if the first candidate is ranked lower
    - "tie" if they have equal rank
    - Nothing if either has no ranking

    Args:
        ballots: List of ballots (each is list of integer rankings or None)
        candidates: List of candidate names

    Returns:
        List of (candidate_a, candidate_b, outcome) tuples
    """
    comparisons = []
    n_candidates = len(candidates)

    for ballot in ballots:
        for i in range(n_candidates - 1):
            for j in range(i + 1, n_candidates):
                rank_i = ballot[i]
                rank_j = ballot[j]

                # Skip if either candidate wasn't ranked
                if rank_i is None or rank_j is None:
                    continue

                if rank_i < rank_j:
                    comparisons.append((candidates[i], candidates[j], "win"))
                elif rank_i > rank_j:
                    comparisons.append((candidates[i], candidates[j], "loss"))
                else:
                    comparisons.append((candidates[i], candidates[j], "tie"))

    return comparisons


def build_matchup_matrix(
    comparisons: list[tuple[str, str, str]],
    candidates: list[str]
) -> pd.DataFrame:
    """
    Build the pairwise matchup matrix.

    Entry (i, j) is the margin of victory of candidate i over candidate j
    (positive means i beats j more often than j beats i).

    Args:
        comparisons: List of pairwise comparisons
        candidates: List of candidate names

    Returns:
        DataFrame with matchup margins
    """
    # Initialize counts
    wins = {c: {d: 0 for d in candidates} for c in candidates}

    for c1, c2, outcome in comparisons:
        if outcome == "win":
            wins[c1][c2] += 1
        elif outcome == "loss":
            wins[c2][c1] += 1
        # ties don't affect the count

    # Build margin matrix
    sorted_candidates = sorted(candidates)
    matrix_data = []

    for c1 in sorted_candidates:
        row = []
        for c2 in sorted_candidates:
            if c1 == c2:
                row.append(None)
            else:
                margin = wins[c1][c2] - wins[c2][c1]
                row.append(margin)
        matrix_data.append(row)

    return pd.DataFrame(
        matrix_data,
        index=sorted_candidates,
        columns=sorted_candidates
    )


def compute_borda_scores(matchup_matrix: pd.DataFrame) -> dict[str, float]:
    """
    Compute modified Borda scores from the matchup matrix.

    The Borda score for a candidate is the sum of their margins
    against all other candidates.

    Args:
        matchup_matrix: Pairwise margin matrix

    Returns:
        Dictionary of candidate -> Borda score
    """
    scores = {}
    for candidate in matchup_matrix.index:
        row = matchup_matrix.loc[candidate]
        # Sum all non-null values (use pd.notna to handle NaN properly)
        total = 0
        for v in row:
            if pd.notna(v):
                total += v
        scores[candidate] = total
    return scores


# =============================================================================
# Ranked Pairs Algorithm
# =============================================================================

def build_victory_graph(matchup_matrix: pd.DataFrame) -> nx.DiGraph:
    """
    Build a directed graph of victories.

    An edge from A to B means A beats B (positive margin).
    Edge weight is the margin of victory.

    Args:
        matchup_matrix: Pairwise margin matrix

    Returns:
        Directed graph with victory edges
    """
    g = nx.DiGraph()
    g.add_nodes_from(matchup_matrix.index)

    for c1 in matchup_matrix.index:
        for c2 in matchup_matrix.columns:
            if c1 != c2:
                margin = matchup_matrix.loc[c1, c2]
                if margin is not None and margin > 0:
                    g.add_edge(c1, c2, margin=margin)

    return g


def would_create_cycle(graph: nx.DiGraph, u: str, v: str) -> bool:
    """
    Check if adding edge (u, v) would create a cycle.

    A cycle would be created if there's already a path from v to u.

    Args:
        graph: Current graph
        u: Source node
        v: Target node

    Returns:
        True if adding (u, v) would create a cycle
    """
    return nx.has_path(graph, v, u)


def run_ranked_pairs(matchup_matrix: pd.DataFrame) -> tuple[nx.DiGraph, list]:
    """
    Run the Tideman Ranked Pairs algorithm.

    1. Sort all edges by margin of victory (descending)
    2. Add each edge if it doesn't create a cycle
    3. Result is a DAG that can be topologically sorted

    For ties in margin, we don't add edges that would create cycles
    when combined with other edges of the same margin.

    Args:
        matchup_matrix: Pairwise margin matrix

    Returns:
        Tuple of (ranked_pairs_graph, unused_edges)
    """
    # Build list of all edges with positive margins
    edges = []
    for c1 in matchup_matrix.index:
        for c2 in matchup_matrix.columns:
            if c1 != c2:
                margin = matchup_matrix.loc[c1, c2]
                if margin is not None and margin > 0:
                    edges.append((c1, c2, margin))

    # Sort by margin (descending)
    edges.sort(key=lambda x: -x[2])

    # Group edges by margin
    edges_by_margin = []
    if edges:
        current_margin = edges[0][2]
        current_group = []
        for e in edges:
            if e[2] == current_margin:
                current_group.append(e)
            else:
                edges_by_margin.append(current_group)
                current_margin = e[2]
                current_group = [e]
        edges_by_margin.append(current_group)

    # Build the ranked pairs graph
    rp_graph = nx.DiGraph()
    rp_graph.add_nodes_from(matchup_matrix.index)
    unused_edges = []

    for group in edges_by_margin:
        # For edges with same margin, we need to check which ones
        # would form a cycle if all were added
        # First, create a temporary graph with all edges in this group
        temp_graph = rp_graph.copy()
        for u, v, margin in group:
            temp_graph.add_edge(u, v, margin=margin)

        # Find which edges are part of cycles
        for u, v, margin in group:
            # Check if this edge is in a cycle in the temp graph
            if is_edge_in_cycle(temp_graph, u, v):
                unused_edges.append((u, v, margin))
            else:
                # Safe to add
                rp_graph.add_edge(u, v, margin=margin)

    return rp_graph, unused_edges


def is_edge_in_cycle(graph: nx.DiGraph, u: str, v: str) -> bool:
    """
    Check if edge (u, v) is part of any cycle in the graph.

    An edge is in a cycle if there's a path from v back to u.

    Args:
        graph: The graph to check
        u: Source node
        v: Target node

    Returns:
        True if the edge is part of a cycle
    """
    if not graph.has_edge(u, v):
        return False
    return nx.has_path(graph, v, u)


def simplify_graph(graph: nx.DiGraph, ranking: list[str]) -> nx.DiGraph:
    """
    Remove redundant edges from the graph for cleaner visualization.

    An edge is redundant if there's another path between the same nodes.

    Args:
        graph: The ranked pairs graph
        ranking: The topological ordering

    Returns:
        Simplified graph with redundant edges removed
    """
    simplified = graph.copy()

    for i in range(len(ranking) - 1):
        for j in range(i + 1, len(ranking)):
            u, v = ranking[i], ranking[j]
            if simplified.has_edge(u, v):
                # Check if there's another path
                temp = simplified.copy()
                temp.remove_edge(u, v)
                if nx.has_path(temp, u, v):
                    simplified.remove_edge(u, v)

    return simplified


def get_ranking_from_graph(graph: nx.DiGraph) -> tuple[list[str], bool, str]:
    """
    Extract a ranking from the ranked pairs graph.

    If the graph is a proper DAG with a unique topological sort,
    that gives the ranking. Otherwise, there are ties that need
    manual resolution.

    Args:
        graph: The ranked pairs graph

    Returns:
        Tuple of (ranking, has_ties, tie_description)
    """
    if not nx.is_directed_acyclic_graph(graph):
        # This shouldn't happen if ranked pairs ran correctly
        return (
            list(graph.nodes()),
            True,
            "Graph contains cycles - algorithm error"
        )

    # Check if there's a unique topological ordering
    # This is true iff the graph is a tournament (fully connected DAG)
    ranking = list(nx.topological_sort(graph))

    # Check for ties by seeing if the graph is "semiconnected"
    # A semiconnected graph has a unique topological sort
    if not nx.is_semiconnected(graph):
        # Find the ambiguous positions
        ambiguous = find_ambiguous_positions(graph, ranking)
        tie_desc = (
            f"Ties detected - chair intervention needed. "
            f"Ambiguous positions: {ambiguous}"
        )
        return ranking, True, tie_desc

    return ranking, False, ""


def find_ambiguous_positions(graph: nx.DiGraph, ranking: list[str]) -> list[str]:
    """
    Find positions in the ranking that are ambiguous.

    Args:
        graph: The ranked pairs graph
        ranking: Current topological ordering

    Returns:
        List of candidate names with ambiguous positions
    """
    ambiguous = []

    for i in range(len(ranking) - 1):
        # Check if there's an edge between adjacent candidates
        if not graph.has_edge(ranking[i], ranking[i + 1]):
            # No direct edge - these candidates could be swapped
            ambiguous.extend([ranking[i], ranking[i + 1]])

    # Remove duplicates while preserving order
    seen = set()
    result = []
    for c in ambiguous:
        if c not in seen:
            seen.add(c)
            result.append(c)

    return result


# =============================================================================
# Output Generation
# =============================================================================

def create_results_excel(
    results: VotingResults,
    candidates: list[str],
    output_path: Path,
    problematic_ballots: list[str]
) -> None:
    """
    Create Excel file with voting results.

    Sheets:
    - Results: Final ranking with both methods
    - Matchups: Pairwise margin matrix
    - Unused edges: Edges not included due to cycles
    - Notes: Any warnings or issues

    Args:
        results: VotingResults object
        candidates: List of candidate names
        output_path: Where to save the Excel file
        problematic_ballots: List of response IDs with issues
    """
    with pd.ExcelWriter(output_path) as writer:
        # Results sheet
        results_data = []
        for i, candidate in enumerate(results.ranked_pairs_ranking):
            results_data.append({
                'Candidate': candidate,
                'Ranked Pairs Rank': i + 1 if not results.has_ties else '?',
                'Borda Score': results.borda_scores.get(candidate, 0)
            })

        # Sort by Borda score as backup when ties exist
        if results.has_ties:
            results_data.sort(key=lambda x: -x['Borda Score'])
            for i, row in enumerate(results_data):
                row['Ranked Pairs Rank'] = f"~{i + 1}"

        results_df = pd.DataFrame(results_data)
        results_df.to_excel(writer, sheet_name='Results', index=False)

        # Matchups sheet
        results.matchup_matrix.to_excel(writer, sheet_name='Matchups')

        # Unused edges sheet (if any)
        if results.unused_edges:
            unused_df = pd.DataFrame(
                results.unused_edges,
                columns=['Winner', 'Loser', 'Margin']
            )
            unused_df.to_excel(writer, sheet_name='Unused Edges', index=False)

        # Notes sheet
        notes = []
        if results.has_ties:
            notes.append({
                'Type': 'WARNING',
                'Message': results.tie_description
            })
            notes.append({
                'Type': 'INFO',
                'Message': 'See ranked-pairs-graph.pdf to manually resolve ties'
            })

        if problematic_ballots:
            notes.append({
                'Type': 'WARNING',
                'Message': f'Ballots with missing/invalid votes: {", ".join(problematic_ballots)}'
            })

        if notes:
            notes_df = pd.DataFrame(notes)
            notes_df.to_excel(writer, sheet_name='Notes', index=False)


def create_graph_pdf(
    graph: nx.DiGraph,
    output_path: Path,
    title: str = ""
) -> bool:
    """
    Create a PDF visualization of the graph using pygraphviz.

    Args:
        graph: NetworkX directed graph
        output_path: Where to save the PDF
        title: Optional title for the graph

    Returns:
        True if successful, False otherwise
    """
    try:
        agraph = nx.nx_agraph.to_agraph(graph)
    except ImportError:
        return False

    # Styling
    agraph.node_attr['shape'] = 'box'
    agraph.node_attr['fontname'] = 'DejaVu Sans'
    agraph.node_attr['style'] = 'filled'
    agraph.node_attr['fillcolor'] = '#E8E8E8'
    agraph.edge_attr['arrowsize'] = '0.5'
    agraph.graph_attr['splines'] = 'ortho'
    agraph.graph_attr['fontname'] = 'DejaVu Sans'
    agraph.graph_attr['fontsize'] = '24'
    agraph.graph_attr['rankdir'] = 'TB'

    if title:
        agraph.graph_attr['label'] = title

    # Group nodes at same level (cycles = ties)
    try:
        for cycle in sorted(nx.simple_cycles(graph), key=len, reverse=True):
            agraph.add_subgraph(cycle, rank='same')
    except nx.NetworkXError:
        pass  # No cycles

    agraph.draw(str(output_path), prog='dot')
    return True


def _get_bundled_graphviz_path() -> Optional[Path]:
    """
    Get the path to bundled Graphviz executables when running as a packaged app.

    When packaged with PyInstaller/flet pack, bundled files are in sys._MEIPASS.
    Returns None if not running as packaged app or if graphviz not bundled.
    """
    import os

    # Check if running as packaged executable (PyInstaller)
    if hasattr(sys, '_MEIPASS'):
        bundled_path = Path(sys._MEIPASS) / 'graphviz_bin'
        if bundled_path.exists():
            return bundled_path

    # Also check relative to the script (for development/testing)
    script_dir = Path(__file__).parent
    local_graphviz = script_dir / 'graphviz_bin'
    if local_graphviz.exists():
        return local_graphviz

    return None


def create_graph_graphviz(
    graph: nx.DiGraph,
    output_path: Path,
    title: str = ""
) -> bool:
    """
    Create a PDF visualization of the graph using the graphviz package.

    This is an alternative to pygraphviz that's easier to install.
    Requires Graphviz to be installed on the system or bundled with the app.

    Args:
        graph: NetworkX directed graph
        output_path: Where to save the PDF
        title: Optional title for the graph

    Returns:
        True if successful, False otherwise
    """
    try:
        import graphviz
    except ImportError:
        return False

    if len(graph.nodes()) == 0:
        return False

    # Check for bundled Graphviz and add to PATH if found
    bundled_path = _get_bundled_graphviz_path()
    if bundled_path:
        import os
        os.environ['PATH'] = str(bundled_path) + os.pathsep + os.environ.get('PATH', '')
        # Also set font path so Graphviz finds bundled DejaVu Sans font
        os.environ['GDFONTPATH'] = str(bundled_path)

    # Create a new directed graph
    dot = graphviz.Digraph(comment=title)

    # Graph attributes - use DejaVu Sans to match pygraphviz output
    dot.attr(rankdir='TB', splines='ortho', fontname='DejaVu Sans', fontsize='24')
    dot.attr(pad='0.75', margin='0.5,0.75')  # margin = horizontal,vertical

    if title:
        # Use HTML-like label with line breaks for spacing below title
        dot.attr(label=f'<<BR/><B>{title}</B><BR/><BR/><BR/>>', labelloc='t')

    # Node attributes - no border (color matches fill)
    dot.attr('node', shape='box', style='filled', fillcolor='#E8E8E8',
             color='#E8E8E8', fontname='DejaVu Sans', fontsize='11')

    # Edge attributes
    dot.attr('edge', arrowsize='0.5')

    # Add nodes
    for node in graph.nodes():
        dot.node(str(node))

    # Add edges
    for u, v in graph.edges():
        dot.edge(str(u), str(v))

    # Group nodes at same level (for ties/cycles)
    try:
        for cycle in sorted(nx.simple_cycles(graph), key=len, reverse=True):
            with dot.subgraph() as s:
                s.attr(rank='same')
                for node in cycle:
                    s.node(str(node))
    except nx.NetworkXError:
        pass  # No cycles

    # Render
    output_path = Path(output_path)
    # graphviz adds the extension automatically, so we need to handle this
    output_stem = output_path.parent / output_path.stem

    try:
        dot.render(str(output_stem), format='pdf', cleanup=True)
        return True
    except graphviz.ExecutableNotFound:
        print("Warning: Graphviz executable not found. Please install Graphviz.")
        return False
    except Exception as e:
        print(f"Warning: Graph rendering failed: {e}")
        return False


def create_graph(
    graph: nx.DiGraph,
    output_path: Path,
    title: str = ""
) -> tuple[bool, Path]:
    """
    Create a graph visualization, trying pygraphviz first, then graphviz package.

    Args:
        graph: NetworkX directed graph
        output_path: Where to save the graph
        title: Optional title for the graph

    Returns:
        Tuple of (success, actual_output_path)
    """
    # Try pygraphviz first (best quality, PDF output)
    if create_graph_pdf(graph, output_path, title):
        return True, output_path

    # Fall back to graphviz package (also good quality, PDF output)
    if create_graph_graphviz(graph, output_path, title):
        return True, output_path

    return False, output_path


# =============================================================================
# Main Entry Point
# =============================================================================

def process_ballots(
    input_path: Path,
    output_dir: Optional[Path] = None,
    missing_vote_strategy: str = "skip",
    verbose: bool = False
) -> VotingResults:
    """
    Process a Qualtrics ballot file and compute rankings.

    Args:
        input_path: Path to the Qualtrics CSV export
        output_dir: Directory for output files (default: same as input)
        missing_vote_strategy: How to handle missing votes
        verbose: Print progress information

    Returns:
        VotingResults object with all computed data
    """
    if output_dir is None:
        output_dir = input_path.parent

    if verbose:
        print(f"Loading ballots from {input_path}")

    # Load and parse
    df = load_qualtrics_csv(input_path)
    question_cols = find_question_columns(df)
    candidates = extract_candidate_names(df, question_cols)
    ballots, problematic = parse_ballots(df, question_cols, missing_vote_strategy)

    if verbose:
        print(f"Found {len(candidates)} candidates: {candidates}")
        print(f"Found {len(ballots)} ballots")
        if problematic:
            print(f"Warning: {len(problematic)} ballots had issues")

    # Build pairwise comparisons
    comparisons = build_pairwise_comparisons(ballots, candidates)
    matchup_matrix = build_matchup_matrix(comparisons, candidates)
    borda_scores = compute_borda_scores(matchup_matrix)

    if verbose:
        print("Computed matchup matrix and Borda scores")

    # Run ranked pairs
    rp_graph, unused_edges = run_ranked_pairs(matchup_matrix)
    ranking, has_ties, tie_desc = get_ranking_from_graph(rp_graph)

    if verbose:
        print(f"Ranked pairs complete. Ties: {has_ties}")
        if has_ties:
            print(f"  {tie_desc}")

    # Build victory graph for visualization
    victory_graph = build_victory_graph(matchup_matrix)
    simplified_victory = simplify_graph(victory_graph, ranking)

    # Simplify ranked pairs graph for cleaner visualization
    # (removes transitive edges: if A > B > C, don't need A > C edge)
    simplified_rp_graph = simplify_graph(rp_graph, ranking)

    # Create results object
    results = VotingResults(
        ranked_pairs_ranking=ranking,
        borda_scores=borda_scores,
        matchup_matrix=matchup_matrix,
        ranked_pairs_graph=rp_graph,
        victory_graph=simplified_victory,
        unused_edges=unused_edges,
        has_ties=has_ties,
        tie_description=tie_desc
    )

    # Generate outputs
    date_str = time.strftime("%Y-%m-%d")
    excel_path = output_dir / f"survey-results {date_str}.xlsx"
    rp_graph_path = output_dir / "ranked-pairs-graph.pdf"
    victory_graph_path = output_dir / "victory-graph.pdf"

    create_results_excel(results, candidates, excel_path, problematic)

    if verbose:
        print(f"Saved results to {excel_path}")

    # Generate graphs (use simplified versions for cleaner output)
    title = "RANKED PAIRS - CHAIR REVIEW NEEDED" if has_ties else "Ranked Pairs Result"
    create_graph_pdf(simplified_rp_graph, rp_graph_path, title)
    create_graph_pdf(simplified_victory, victory_graph_path, "Victory Graph")

    if verbose:
        print(f"Saved graphs to {rp_graph_path} and {victory_graph_path}")

    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Process Qualtrics voting ballots using Ranked Pairs method",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python ranked_pairs_voting.py ballots.csv
    python ranked_pairs_voting.py ballots.csv --output ./results/
    python ranked_pairs_voting.py ballots.csv --missing-votes worst --verbose
        """
    )

    parser.add_argument(
        "input",
        type=Path,
        help="Path to the Qualtrics CSV export file"
    )

    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Output directory (default: same as input file)"
    )

    parser.add_argument(
        "--missing-votes",
        choices=["skip", "worst", "error"],
        default="skip",
        help="How to handle missing votes: skip (ignore), worst (assign worst rank), error (fail)"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print progress information"
    )

    args = parser.parse_args()

    try:
        results = process_ballots(
            input_path=args.input,
            output_dir=args.output,
            missing_vote_strategy=args.missing_votes,
            verbose=args.verbose
        )

        print("\n" + "=" * 60)
        print("VOTING RESULTS")
        print("=" * 60)

        if results.has_ties:
            print("\n⚠️  TIES DETECTED - CHAIR INTERVENTION NEEDED")
            print(f"   {results.tie_description}")
            print("\n   See 'ranked-pairs-graph.pdf' to resolve ties manually.\n")

        print("\nRanking (by Ranked Pairs, with Borda scores):\n")
        for i, candidate in enumerate(results.ranked_pairs_ranking):
            rank_str = f"{i + 1}." if not results.has_ties else f"~{i + 1}."
            borda = results.borda_scores.get(candidate, 0)
            print(f"  {rank_str:4} {candidate:40} (Borda: {borda:+.0f})")

        print("\n" + "=" * 60)

        if results.has_ties:
            sys.exit(1)  # Non-zero exit for scripts to detect

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
