"""Where the generated figures go.

The paper lives in its own repository (`iscc-overleaf`), a sibling of this one inside the iscc
workspace, so a regenerated figure shows up as a diff in the paper repo instead of a copy step
somebody has to remember. Every `validate_*.py` writes through `figure_path`.

Resolution order:

1. ``$ISCC_PAPER_DIR`` — an explicit paper checkout (CI, or a layout that is not the workspace).
2. ``../iscc-overleaf`` — the workspace sibling. The normal case.
3. ``./manuscript`` — the legacy in-repo location, so a bare clone of the public repository still
   runs end to end without the paper repo next to it.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def paper_dir():
    """The root of the paper checkout (the directory holding paper.tex)."""
    env = os.environ.get("ISCC_PAPER_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    sibling = os.path.join(os.path.dirname(REPO), "iscc-overleaf")
    if os.path.isdir(sibling):
        return sibling
    return os.path.join(REPO, "manuscript")


def figures_dir():
    """The figure directory of the paper checkout, created if it does not exist yet."""
    d = os.path.join(paper_dir(), "figures")
    os.makedirs(d, exist_ok=True)
    return d


def figure_path(name):
    """Absolute path to write the figure `name` to."""
    return os.path.join(figures_dir(), name)
