import click

from . import add as add_cmd
from . import compare as compare_cmd
from . import create as create_cmd
from . import grid as grid_cmd
from . import plot as plot_cmd
from . import run as run_cmd
from . import update as update_cmd


@click.group()
def main() -> None:
    """Unified experiment management tool for GPUSSSP."""


@main.command()
@click.argument("xp_name")
def create(xp_name: str) -> None:
    """Create a new experiment branch."""

    create_cmd.handle(xp_name)


@main.command()
@click.argument("run_targets")
@click.argument("params")
@click.argument("metrics", required=False, default="")
def add(run_targets: str, params: str, metrics: str) -> None:
    """Add an instrumentation commit to the current experiment branch."""

    add_cmd.handle(run_targets, params, metrics)


@main.command()
def run() -> None:
    """Run all experiments in the current branch."""

    run_cmd.handle()


@main.command()
@click.argument("xp_name", required=False)
def update(xp_name: str | None) -> None:
    """Rebase experiment branch on main and drop result commits."""

    update_cmd.handle(xp_name)


@main.command()
@click.argument("xp_name", required=False)
@click.option("--device", help="Filter by device hash.")
@click.option("--variant", help="Filter variants by substring match.")
@click.option(
    "--metrics",
    default="time",
    help="Comma-separated metrics to compare (default: time).",
)
def compare(
    xp_name: str | None,
    device: str | None,
    variant: str | None,
    metrics: str,
) -> None:
    """Compare experiment results and print summary tables."""

    metrics_list = [m.strip() for m in metrics.split(",") if m.strip()]
    compare_cmd.handle(xp_name, device=device, variant=variant, metrics=metrics_list)


@main.command()
@click.argument("xp_name", required=False)
@click.option("--device", help="Filter by device hash.")
@click.option("--variant", help="Filter variants by substring match.")
def plot(
    xp_name: str | None,
    device: str | None,
    variant: str | None,
) -> None:
    """Generate plots for experiment results."""

    plot_cmd.handle(xp_name, device=device, variant=variant)


@main.command()
@click.argument("xp_name", required=False)
@click.option("--device", help="Filter by device hash.")
@click.option("--variant", help="Filter variants by substring match.")
@click.option("--primary-param", help="First parameter for rows.")
@click.option("--secondary-param", help="Second parameter for cells.")
@click.option(
    "--show",
    type=click.Choice(["winners", "top3", "all"]),
    default="winners",
    help="Which combos to show in Grid 2: winners, top3, or all.",
)
def grid(
    xp_name: str | None,
    device: str | None,
    variant: str | None,
    primary_param: str,
    secondary_param: str,
    show: str,
) -> None:
    """Show grids: best secondary-param per primary-param/rank, and speedup for combos."""

    grid_cmd.handle(
        xp_name,
        device=device,
        variant=variant,
        param1=primary_param,
        param2=secondary_param,
        show=show,
    )


@main.command(name="_rebase", hidden=True)
@click.argument("xp_name")
@click.argument("todo_file")
def rebase_helper(xp_name: str, todo_file: str) -> None:
    """Hidden helper for interactive rebase to drop result commits."""

    update_cmd.rebase_helper(xp_name, todo_file)
