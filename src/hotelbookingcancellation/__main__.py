"""Support file to run the project from the CLI.

HotelBookingCancellation file for ensuring the package is executable
as `hotelbookingcancellation` and
`python -m hotelbookingcancellation`
"""
import importlib
from pathlib import Path
from typing import List

from click import Command, Group
from kedro.framework.cli.utils import KedroCliError, load_entry_points
from kedro.framework.project import configure_project


def _find_run_command(package_name: str) -> Command:
    try:
        project_cli = importlib.import_module(f"{package_name}.cli")
        # fail gracefully if cli.py does not exist
    except ModuleNotFoundError as exc:
        if f"{package_name}.cli" not in str(exc):
            raise
        plugins = load_entry_points("project")
        run = _find_run_command_in_plugins(plugins) if plugins else None
        if run:
            # use run command from installed plugin if it exists
            return run
        # use run command from `kedro.framework.cli.project`
        from kedro.framework.cli.project import run

        return run
    # fail badly if cli.py exists, but has no `cli` in it
    if not hasattr(project_cli, "cli"):
        raise KedroCliError(f"Cannot load commands from {package_name}.cli")
    return project_cli.run


def _find_run_command_in_plugins(plugins: List[Group]) -> Command:
    for group in plugins:
        if "run" in group.commands:
            return group.commands["run"]


def main(*args: str, **kwargs: str):
    """Entry point for running the project from the CLI."""
    package_name = Path(__file__).parent.name
    configure_project(package_name)
    run = _find_run_command(package_name)
    run(*args, **kwargs)


if __name__ == "__main__":
    main()
