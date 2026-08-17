"""Run the complete weekly Project Ted planning workflow."""

from project_ted.delivery import send_weekly_report
from project_ted.fpl import fetch_planning_context
from project_ted.news import FootballNewsSearch
from project_ted.report import render_weekly_report
from project_ted.weekly import run_weekly_planning


def run_weekly_job() -> str:
    """Run both agents and email their weekly planning report."""

    context = fetch_planning_context()
    news = FootballNewsSearch()

    weekly_run = run_weekly_planning(context, news)
    report = render_weekly_report(weekly_run, context)

    return send_weekly_report(report, weekly_run)


def main() -> None:
    """Run the scheduled job and print its delivery identifier."""

    email_id = run_weekly_job()
    print(f"Project Ted report delivered: {email_id}")
