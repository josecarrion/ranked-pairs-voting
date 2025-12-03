#!/usr/bin/env python3
"""
Ranked Pairs Voting - Desktop Application

A graphical interface for processing Qualtrics voting ballots
using the Ranked Pairs (Tideman) method.

Usage:
    python voting_app.py

Or double-click the file on Windows.

Requirements:
    pip install flet pandas networkx pygraphviz
"""

import multiprocessing
import os
import subprocess
import sys
import time
from pathlib import Path

import flet as ft

# Import the voting logic from our refactored script
from ranked_pairs_voting import (
    load_qualtrics_csv,
    find_question_columns,
    extract_candidate_names,
    parse_ballots,
    build_pairwise_comparisons,
    build_matchup_matrix,
    compute_borda_scores,
    run_ranked_pairs,
    get_ranking_from_graph,
    build_victory_graph,
    simplify_graph,
    create_results_excel,
    create_graph,
    VotingResults,
)


def main(page: ft.Page):
    """Main application entry point."""

    # Page configuration
    page.title = "Ranked Pairs Voting"
    page.window.width = 800
    page.window.height = 700
    page.padding = 30
    page.theme_mode = ft.ThemeMode.LIGHT

    # State
    selected_file = None
    output_dir = None

    # --- UI Components ---

    # Title
    title = ft.Text(
        "Ranked Pairs Voting",
        size=32,
        weight=ft.FontWeight.BOLD,
    )

    subtitle = ft.Text(
        "Process Qualtrics ballots using the Tideman method",
        size=16,
        color=ft.Colors.GREY_700,
    )

    # File selection
    file_path_text = ft.Text(
        "No file selected",
        size=14,
        color=ft.Colors.GREY_600,
        width=500,
    )

    def on_file_picked(e: ft.FilePickerResultEvent):
        nonlocal selected_file, output_dir
        if e.files and len(e.files) > 0:
            selected_file = e.files[0].path
            output_dir = str(Path(selected_file).parent)
            file_path_text.value = selected_file
            file_path_text.color = ft.Colors.BLACK
            run_button.disabled = False
        else:
            selected_file = None
            file_path_text.value = "No file selected"
            file_path_text.color = ft.Colors.GREY_600
            run_button.disabled = True
        page.update()

    file_picker = ft.FilePicker(on_result=on_file_picked)
    page.overlay.append(file_picker)

    pick_file_button = ft.ElevatedButton(
        "Select Ballot CSV",
        icon=ft.Icons.FOLDER_OPEN,
        on_click=lambda _: file_picker.pick_files(
            allowed_extensions=["csv"],
            dialog_title="Select Qualtrics Export CSV",
        ),
    )

    # Options
    missing_votes_dropdown = ft.Dropdown(
        label="Handle missing votes",
        value="skip",
        width=320,
        options=[
            ft.dropdown.Option("skip", "Skip (treat as no preference)"),
            ft.dropdown.Option("worst", "Treat as worst rank"),
        ],
    )

    # Progress/status area
    status_text = ft.Text(
        "",
        size=14,
        color=ft.Colors.GREY_700,
    )

    progress_bar = ft.ProgressBar(
        visible=False,
        width=400,
    )

    # Results area
    results_container = ft.Container(
        visible=False,
        padding=20,
        border_radius=10,
        bgcolor=ft.Colors.GREY_100,
        content=ft.Column([]),
    )

    def process_ballots():
        """Run the voting algorithm and update the UI."""
        nonlocal selected_file, output_dir

        if not selected_file:
            return

        # Show progress
        progress_bar.visible = True
        progress_bar.value = None  # Indeterminate
        status_text.value = "Loading ballots..."
        status_text.color = ft.Colors.BLUE_700
        run_button.disabled = True
        results_container.visible = False
        page.update()

        try:
            # Load and parse
            status_text.value = "Parsing ballot data..."
            page.update()

            input_path = Path(selected_file)
            df = load_qualtrics_csv(input_path)
            question_cols = find_question_columns(df)
            candidates = extract_candidate_names(df, question_cols)

            missing_strategy = missing_votes_dropdown.value
            ballots, problematic = parse_ballots(df, question_cols, missing_strategy)

            # Compute results
            status_text.value = f"Processing {len(ballots)} ballots for {len(candidates)} candidates..."
            page.update()

            comparisons = build_pairwise_comparisons(ballots, candidates)
            matchup_matrix = build_matchup_matrix(comparisons, candidates)
            borda_scores = compute_borda_scores(matchup_matrix)

            rp_graph, unused_edges = run_ranked_pairs(matchup_matrix)
            ranking, has_ties, tie_desc = get_ranking_from_graph(rp_graph)

            victory_graph = build_victory_graph(matchup_matrix)
            simplified_victory = simplify_graph(victory_graph, ranking)

            # Simplify the ranked pairs graph for cleaner visualization
            # (removes transitive edges: if A > B > C, don't need A > C edge)
            simplified_rp_graph = simplify_graph(rp_graph, ranking)

            results = VotingResults(
                ranked_pairs_ranking=ranking,
                borda_scores=borda_scores,
                matchup_matrix=matchup_matrix,
                ranked_pairs_graph=rp_graph,
                victory_graph=simplified_victory,
                unused_edges=unused_edges,
                has_ties=has_ties,
                tie_description=tie_desc,
            )

            # Save outputs
            status_text.value = "Saving results..."
            page.update()

            date_str = time.strftime("%Y-%m-%d")
            excel_path = Path(output_dir) / f"survey-results {date_str}.xlsx"
            rp_graph_path = Path(output_dir) / "ranked-pairs-graph.pdf"
            victory_graph_path = Path(output_dir) / "victory-graph.pdf"

            create_results_excel(results, candidates, excel_path, problematic)

            # Try to create graphs (pygraphviz -> matplotlib fallback)
            graphs_created = False
            graph_files = []
            try:
                title_text = "CHAIR REVIEW NEEDED" if has_ties else "Ranked Pairs Result"
                rp_success, rp_actual_path = create_graph(simplified_rp_graph, rp_graph_path, title_text)
                victory_success, victory_actual_path = create_graph(simplified_victory, victory_graph_path, "Victory Graph")

                if rp_success:
                    graph_files.append(rp_actual_path.name)
                if victory_success:
                    graph_files.append(victory_actual_path.name)

                graphs_created = rp_success or victory_success
            except Exception as graph_error:
                print(f"Graph generation failed: {graph_error}")

            # Show results
            progress_bar.visible = False

            # Build results display
            results_rows = []

            # Header
            if has_ties:
                results_rows.append(
                    ft.Container(
                        padding=10,
                        border_radius=5,
                        bgcolor=ft.Colors.AMBER_100,
                        content=ft.Row([
                            ft.Icon(ft.Icons.WARNING, color=ft.Colors.AMBER_700),
                            ft.Text(
                                "Ties detected - chair intervention needed",
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.AMBER_900,
                            ),
                        ]),
                    )
                )
                results_rows.append(ft.Text(tie_desc, size=12, color=ft.Colors.GREY_700))
            else:
                results_rows.append(
                    ft.Container(
                        padding=10,
                        border_radius=5,
                        bgcolor=ft.Colors.GREEN_100,
                        content=ft.Row([
                            ft.Icon(ft.Icons.CHECK_CIRCLE, color=ft.Colors.GREEN_700),
                            ft.Text(
                                "Complete ranking determined",
                                weight=ft.FontWeight.BOLD,
                                color=ft.Colors.GREEN_900,
                            ),
                        ]),
                    )
                )

            results_rows.append(ft.Divider())
            results_rows.append(ft.Text("Ranking:", weight=ft.FontWeight.BOLD, size=16))

            # Ranking table
            for i, candidate in enumerate(ranking):
                rank_str = f"{i + 1}." if not has_ties else f"~{i + 1}."
                borda = borda_scores.get(candidate, 0)
                results_rows.append(
                    ft.Row([
                        ft.Text(rank_str, width=40, weight=ft.FontWeight.BOLD),
                        ft.Text(candidate, width=350),
                        ft.Text(f"Borda: {borda:+.0f}", color=ft.Colors.GREY_600),
                    ])
                )

            results_rows.append(ft.Divider())

            # Output files
            results_rows.append(ft.Text("Output files:", weight=ft.FontWeight.BOLD))
            results_rows.append(ft.Text(f"  {excel_path.name}", size=12))
            if graphs_created:
                for gf in graph_files:
                    results_rows.append(ft.Text(f"  {gf}", size=12))
            else:
                results_rows.append(
                    ft.Text(
                        "  (Graphs not generated - install matplotlib: pip install matplotlib)",
                        size=12,
                        color=ft.Colors.ORANGE_700,
                    )
                )

            if problematic:
                results_rows.append(ft.Divider())
                results_rows.append(
                    ft.Text(
                        f"Note: {len(problematic)} ballot(s) had missing/invalid votes",
                        color=ft.Colors.ORANGE_700,
                        size=12,
                    )
                )

            # Open folder button
            def open_output_folder(_):
                if sys.platform == "win32":
                    os.startfile(output_dir)
                elif sys.platform == "darwin":
                    subprocess.run(["open", output_dir])
                else:
                    subprocess.run(["xdg-open", output_dir])

            results_rows.append(ft.Container(height=10))
            results_rows.append(
                ft.ElevatedButton(
                    "Open Output Folder",
                    icon=ft.Icons.FOLDER,
                    on_click=open_output_folder,
                )
            )

            results_container.content = ft.Column(results_rows, spacing=5)
            results_container.visible = True

            status_text.value = "Done!"
            status_text.color = ft.Colors.GREEN_700

        except Exception as e:
            progress_bar.visible = False
            status_text.value = f"Error: {str(e)}"
            status_text.color = ft.Colors.RED_700
            results_container.visible = False

        finally:
            run_button.disabled = False
            page.update()

    run_button = ft.ElevatedButton(
        "Run Ranked Pairs",
        icon=ft.Icons.PLAY_ARROW,
        disabled=True,
        style=ft.ButtonStyle(
            bgcolor={
                ft.ControlState.DEFAULT: ft.Colors.BLUE_700,
                ft.ControlState.DISABLED: ft.Colors.GREY_400,
            },
            color={
                ft.ControlState.DEFAULT: ft.Colors.WHITE,
                ft.ControlState.DISABLED: ft.Colors.GREY_600,
            },
        ),
        on_click=lambda _: process_ballots(),
    )

    # Help text
    help_text = ft.Container(
        padding=15,
        border_radius=10,
        bgcolor=ft.Colors.BLUE_50,
        content=ft.Column([
            ft.Text("How to use:", weight=ft.FontWeight.BOLD, color=ft.Colors.BLUE_900),
            ft.Text(
                "1. Export your Qualtrics survey results as CSV\n"
                "2. Click 'Select Ballot CSV' and choose the file\n"
                "3. Click 'Run Ranked Pairs' to process\n"
                "4. Results will be saved to the same folder as your CSV",
                size=13,
                color=ft.Colors.BLUE_800,
            ),
        ], spacing=5),
    )

    # --- Layout ---

    page.add(
        ft.Column([
            title,
            subtitle,
            ft.Container(height=20),

            # File selection row
            ft.Row([
                pick_file_button,
                file_path_text,
            ], spacing=20),

            ft.Container(height=10),

            # Options row
            ft.Row([
                missing_votes_dropdown,
            ]),

            ft.Container(height=20),

            # Run button and status
            ft.Row([
                run_button,
                status_text,
            ], spacing=20),

            progress_bar,

            ft.Container(height=20),

            # Results
            results_container,

            ft.Container(height=20),

            # Help
            help_text,

        ], spacing=5, scroll=ft.ScrollMode.AUTO, expand=True)
    )


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for PyInstaller on Windows
    ft.app(target=main)
