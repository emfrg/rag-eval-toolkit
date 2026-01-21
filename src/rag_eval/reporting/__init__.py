"""Reporting module - console output and JSON export."""

from rag_eval.reporting.console import print_comparison_table, print_results_summary
from rag_eval.reporting.json_export import export_results

__all__ = [
    "print_comparison_table",
    "print_results_summary",
    "export_results",
]
