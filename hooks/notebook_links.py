"""Make cross-notebook links work on the built site.

Notebooks link to each other as ``[Title](other.ipynb)`` — the form that works when you open them
in Jupyter, and the form the docs conventions ask for. mkdocs-jupyter does not rewrite those hrefs
and mkdocs serves directory URLs, so on the published site every one of them 404s. (Found
2026-08-27: 31 dead links across the tutorials, none of which `mkdocs build --strict` reports,
because it does not look inside converted notebook HTML.)

This rewrites ``href="other.ipynb"`` to ``href="../other/"`` at build time and leaves the notebook
source alone, so both the site and Jupyter navigation work.

A page's own notebook is copied next to its ``index.html`` for download; that self-link is left
untouched.
"""
import os
import re

_LINK = re.compile(r'href="([A-Za-z0-9_./-]+)\.ipynb"')


def on_post_page(output, page, config, **kwargs):
    src = getattr(getattr(page, "file", None), "src_path", "") or ""
    if not src.startswith("tutorials/"):
        return output
    own = os.path.splitext(os.path.basename(src))[0]

    def repl(m):
        target = os.path.basename(m.group(1))
        if target == own:                      # the downloadable copy of this page's own notebook
            return m.group(0)
        return f'href="../{target}/"'

    return _LINK.sub(repl, output)
