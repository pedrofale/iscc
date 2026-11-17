"""Plots and images of a GenotypeTumor (Muller plots, tissue maps, phylogenies, H&E).

A mixin of :class:`~iscc.tumor.models.count.GenotypeTumor`; it holds no state of its own."""

import numpy as np
from ...constants import normal_names


class PlottingMixin:
    def plot_muller(self, ax=None, by_drivers=False, **kwargs):
        """Render a Muller plot of the clonal dynamics (clone frequencies over steps, nested by ancestry).

        ``by_drivers=True`` colours by distinct DRIVER-mutation combinations (Noble's demon
        convention) instead of by genotype, collapsing passenger-only diversity; combine with
        ``min_freq`` for a size threshold. Additional keywords (``ax``, ``colormap``,
        ``normalize``) select the target axes and control the styling."""
        from .. import viz
        driver_map = self._driver_signatures() if by_drivers else None
        return viz.plot_muller(self.traces, self.genotypes_parents, ax=ax,
                               driver_map=driver_map, **kwargs)

    def _driver_signatures(self):
        """Map every cancer genotype id -> a hashable signature of its mutated DRIVER genes (the
        oncogene/TSG loci, ``driver_types != 0``, carrying an SNV). Covers ancestors too (needed to
        build the driver-clone ancestry). Genotypes sharing a signature are one 'driver clone'."""
        dts = getattr(self.selection, "driver_types", None)
        if not dts:
            return {}
        driver_idx = np.flatnonzero(np.concatenate(dts) != 0)
        if driver_idx.size == 0:
            return {}
        out = {}
        for gid, rep in self.genotypes.items():
            if gid in normal_names or not hasattr(rep, "get_snvs"):
                continue
            vafs = rep.get_snvs()
            out[str(gid)] = tuple(int(i) for i in driver_idx[vafs[driver_idx] > 0])
        return out

    def _functional_signatures(self):
        """Map every cancer genotype id -> a hashable signature of its mutated FUNCTIONAL genes: ALL the
        selection axes (oncogene/TSG + dispersal + immune/treatment resistance + breach/stromal/met
        survival) carrying an SNV. Unlike ``_driver_signatures`` (onc/TSG only), sharing a signature
        means sharing the mutations selection acts on, so a sweep on ANY axis (breach, met_survival,
        treatment_resistance, ...) becomes one functional clone -> one Muller band -> visible selection."""
        s = self.selection
        parts = [s.get_oncogenes(), s.get_tsgs(), s.get_dispersal_genes(), s.get_immune_resistant(),
                 s.get_treatment_resistant(), s.get_breach(), s.get_stromal_survival(),
                 s.get_met_survival(), s.get_drug_tolerance()]
        parts = [np.asarray(p, dtype=int) for p in parts if len(p)]
        if not parts:
            return {}
        func_idx = np.unique(np.concatenate(parts))
        out = {}
        for gid, rep in self.genotypes.items():
            if gid in normal_names or not hasattr(rep, "get_snvs"):
                continue
            vafs = rep.get_snvs()
            out[str(gid)] = tuple(int(i) for i in func_idx[vafs[func_idx] > 0])
        return out

    def _stage_colors(self):
        """{gid -> rgba} colouring each CANCER genotype by its STAGE-DOMINANT driver — the highest arc
        stage it has activated (from its heritable traits), on the 7-stage scheme of ``_stage_of``:
        1 proliferation (division rate raised by an onc/TSG mutation), 2 breach (duct escape),
        3 stromal survival, 4 met survival, 5 drug tolerance, 6 chemo resistance; 0 = no driver. Normal
        cells keep their type colours. A categorical view: a few colours tracking the selection
        cascade.

        Returns the map plus the sorted set of stages actually present, for the legend."""
        from .. import viz
        from ...constants import normal_cmap_rgba
        out, present = {}, set()
        for gid, rep in self.genotypes.items():
            if getattr(rep, "type", None) != "cancer":
                out[str(gid)] = normal_cmap_rgba.get(rep.type, (0.8, 0.8, 0.8, 1.0))
                continue
            s = self._stage_of(rep)
            out[str(gid)] = viz.STAGE_PALETTE[s]
            present.add(s)
        return out, sorted(present)

    def _stage_of(self, rep):
        """Stage-dominant driver of one CANCER genotype, on the SAME 7-stage scheme as
        ``iscc.tumor.arc.stage_of``: 0 none, 1 proliferation (division raised above baseline), 2 breach
        (duct escape), 3 stromal survival, 4 met survival, 5 drug tolerance, 6 chemo resistance — the
        highest arc stage its heritable traits have activated. The return value INDEXES
        ``viz.STAGE_PALETTE`` / ``viz.STAGE_NAMES``. Robust to missing evo keys."""
        ep = rep.evolutionary_parameters
        div = ep.get("division_rate", 0.0)
        base = getattr(rep, "baseline_rates", {}).get("division_rate", div)
        # Drug escape as the ENGINE defines it: protection is max(treatment_resistance, drug_tolerance)
        # (see _apply_treatment), so a pure persister takes zero drug and must not read as sensitive;
        # tolerance is its own stage, and claims the cell only when it STRICTLY exceeds resistance —
        # the same tie-break _apply_treatment uses to decide who pays the persister cost. See
        # arc.stage_of, which this mirrors.
        tr = ep.get("treatment_resistance", 0)
        dt = ep.get("drug_tolerance", 0.0)
        if max(tr, dt) > 0:
            return 5 if dt > tr else 6
        if ep.get("met_survival", 0) > 0:
            return 4
        if ep.get("stromal_survival", 0) > 0:
            return 3
        if ep.get("breach", 0) > 0:
            return 2
        if div > base + 1e-9:
            return 1
        return 0

    def plot_tissue(self, ax=None, color="state", cmap=None, figsize=(7, 7), min_freq=0.02):
        """Deme-resolution tissue map built directly from the per-deme genotype COUNTS.

        Unlike ``plot_grid`` (which reads the possibly-subsampled ``cell_data`` and so goes holey at
        cm-scale when ``max_cells`` thins the tissue), this colours EVERY occupied deme from the full
        counts, so the spatial overview is always complete. Primary grid only.

        Parameters
        ----------
        color : {"state", "cancer_frac", "clone", "stage"}
            ``"state"`` (default): categorical — empty / stroma / duct(normal) / DCIS (cancer in a
            duct) / IDC (cancer in stroma). ``"cancer_frac"``: the per-deme cancer fraction as a
            heatmap. ``"clone"``: colour each cancer deme by its dominant clone's DRIVER-clone colour
            (the same driver-collapsed palette as ``plot_muller(by_drivers=True)`` and
            ``plot_phylogeny``), on a pale-stroma background. ``"stage"``: colour each cancer deme by
            the most advanced trait of its dominant clone (none, proliferation, breach, stromal
            survival, metastatic survival, drug tolerance, chemotherapy resistance) on a pale-stroma
            background.
        min_freq : float, optional
            For ``color="clone"``, the size threshold (fraction of the cancer population) below which
            driver clones are merged into their nearest surviving ancestor before colouring — matched
            to the Muller's ``min_freq`` so the two share a palette. Default 0.02.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        G = self.grid_size
        gid_is_can = {g: self._is_cancer(g) for g in self.genotypes}
        if color == "cancer_frac":
            img = np.full(G * G, np.nan)
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                tot = sum(d.values())
                if tot:
                    img[di] = sum(c for g, c in d.items() if gid_is_can[g]) / tot
            im = ax.imshow(img.reshape(G, G), cmap=cmap or "magma", vmin=0, vmax=1, interpolation="nearest")
            plt.colorbar(im, ax=ax, fraction=0.046, label="cancer fraction")
        elif color == "clone":
            from .. import viz
            clone_colors = viz.functional_clone_colors(
                self.traces, self.genotypes_parents, driver_map=self._driver_signatures(), min_freq=min_freq)
            default = (0.6, 0.6, 0.6, 1.0)
            rgba = np.tile(np.array([0.95, 0.91, 0.91, 1.0]), (G * G, 1))   # pale-stroma background
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                best_g, best_c = None, 0
                for g, c in d.items():
                    if gid_is_can[g] and c > best_c:
                        best_g, best_c = g, c
                if best_g is not None:
                    rgba[di] = clone_colors.get(str(best_g), default)
            ax.imshow(rgba.reshape(G, G, 4), interpolation="nearest")
        elif color == "stage":
            from .. import viz
            from matplotlib.colors import to_hex
            from matplotlib.patches import Patch
            img = np.full(G * G, -1, dtype=int)             # -1 = empty/stroma background
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                best_g, best_c = None, 0
                for g, c in d.items():
                    if gid_is_can[g] and c > best_c:
                        best_g, best_c = g, c
                if best_g is not None:
                    img[di] = self._stage_of(self.genotypes[best_g])
            img = img.reshape(G, G)
            ax.imshow(np.zeros((G, G)), cmap=ListedColormap(["#f3e7e7"]))    # pale stroma
            cm = ListedColormap(list(viz.STAGE_PALETTE))
            ax.imshow(np.ma.masked_where(img < 0, img), cmap=cm, vmin=0,
                      vmax=len(viz.STAGE_PALETTE) - 1, interpolation="nearest")
            names = ["none", "proliferation", "breach", "stromal", "met survival", "drug tolerance",
                     "chemo resist"]
            present = sorted({int(v) for v in img.ravel() if v >= 0})
            if present:
                ax.legend(handles=[Patch(color=to_hex(viz.STAGE_PALETTE[i]), label=names[i]) for i in present],
                          loc="upper right", fontsize=8, framealpha=0.9)
        else:
            # 0 empty, 1 stroma, 2 duct(normal), 3 DCIS (cancer in duct), 4 IDC (cancer in stroma)
            img = np.zeros(G * G, dtype=int)
            for di in range(min(len(self.demes), G * G)):
                d = self.demes[di]
                ncan = sum(c for g, c in d.items() if gid_is_can[g])
                induct = self.gland_id is not None and self.gland_id[di] >= 0
                img[di] = (3 if induct else 4) if ncan > 0 else (2 if (induct and d) else (1 if d else 0))
            cm = cmap or ListedColormap(["white", "#f3d3d3", "#2e8b57", "#4b0082", "#d62728"])
            ax.imshow(img.reshape(G, G), cmap=cm, vmin=0, vmax=4, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        return ax

    def plot_phylogeny(self, ax=None, min_freq=0.02, figsize=(7, 7), linewidth=1.4):
        """Radial CLADE phylogeny drawn inline with matplotlib (Cassiopeia-style).

        Rather than subsampling cells, the whole cancer population is COLLAPSED to driver clones — the
        same functional collapse the by_drivers Muller uses, size-merged at ``min_freq`` — and the tree
        is the clade genealogy: each surviving clade is one node, placed radially (founder at the centre,
        depth = clade ancestry), coloured by its driver-clone colour (matching ``plot_muller(by_drivers)``
        and ``plot_tissue("clone")``) and sized by its cell count.

        Renders straight to ``ax`` (no image file); returns the matplotlib ``Axes``. Primary grid only.
        """
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from .. import viz
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        driver_map = self._driver_signatures()
        # the SAME display basis the by_drivers Muller draws on: full population collapsed to driver
        # clones and size-merged at min_freq (no cell subsampling).
        basis_counts, basis_parents, _gmap, basis_cols = viz._display_basis(
            self.traces, self.genotypes_parents, driver_map=driver_map, min_freq=min_freq)
        present = [str(c) for c in basis_cols]
        if not present:
            raise ValueError("no cancer clones to build a phylogeny")
        clone_colors = viz.functional_clone_colors(
            self.traces, self.genotypes_parents, driver_map=driver_map, min_freq=min_freq)
        default = (0.6, 0.6, 0.6, 1.0)
        # clade genealogy: each clade -> its parent clade (from the collapsed basis)
        pmap = ({str(c): str(basis_parents.iloc[0][c]) for c in basis_parents.columns}
                if basis_parents.shape[1] else {})
        presset = set(present)
        children, roots = defaultdict(list), []
        for c in present:
            p = pmap.get(c)
            (children[p].append(c) if (p in presset and p != c) else roots.append(c))
        root = roots[0] if len(roots) == 1 else "__root__"
        if len(roots) != 1:
            for r in roots:
                children[root].append(r)
        nodes = set(present) | {root}
        peak = {c: float(basis_counts[c].max()) if c in basis_counts else 0.0 for c in present}
        pmax = max(peak.values()) or 1.0
        # radial layout: DFS leaf order -> even angles; internal angle = mean of children; radius = depth
        depth = {root: 0}
        order, post, stack = [], [], [(root, False)]
        while stack:
            n, done = stack.pop()
            if done:
                post.append(n); continue
            ch = children.get(n, [])
            if not ch:
                order.append(n); post.append(n); continue
            stack.append((n, True))
            for c in reversed(ch):
                depth[c] = depth[n] + 1
                stack.append((c, False))
        nleaf = max(len(order), 1)
        angle = {lf: 2 * np.pi * (i + 0.5) / nleaf for i, lf in enumerate(order)}
        for n in post:
            if n not in angle:
                angle[n] = float(np.mean([angle[c] for c in children[n]]))
        maxd = max(depth.values()) or 1
        col = {n: tuple(np.asarray(clone_colors.get(n, default), float).tolist()) for n in nodes}

        def r_of(n):                                # founder at the centre, clades on rings by depth
            return depth[n] / maxd
        # arc at each clade's radius spanning its children, radial spokes out to each child clade
        for n in nodes:
            ch = children.get(n, [])
            if not ch:
                continue
            rn = r_of(n)
            cangs = [angle[c] for c in ch]
            a0, a1 = min(cangs), max(cangs)
            arc = np.linspace(a0, a1, max(2, int(np.ceil((a1 - a0) / 0.02)) + 1))
            ax.plot(rn * np.cos(arc), rn * np.sin(arc), color=col[n], lw=linewidth, solid_capstyle="round")
            for c in ch:
                ac, rc = angle[c], r_of(c)
                ax.plot([rn * np.cos(ac), rc * np.cos(ac)], [rn * np.sin(ac), rc * np.sin(ac)],
                        color=col[c], lw=linewidth, solid_capstyle="round")
        # one marker per clade, sized by its cell count
        for c in present:
            s = 40 + 360 * np.sqrt(peak.get(c, 0) / pmax)
            ax.scatter([r_of(c) * np.cos(angle[c])], [r_of(c) * np.sin(angle[c])], s=s,
                       color=col[c], edgecolors="black", linewidth=0.4, zorder=3)
        ax.set_aspect("equal"); ax.axis("off")
        ax.set_xlim(-1.12, 1.12); ax.set_ylim(-1.12, 1.12)
        return ax

    def he_image(self, px=6, darkness=0.8, sigma_frac=0.5):
        """Dense H&E-like morphology image built from the per-deme cell COUNTS.

        The tissue *image* a histology/Visium slide sits on. Unlike rasterising the (possibly
        subsampled) ``cell_data`` — which is a THIN section and so renders faint and holey at
        cm-scale — this paints EVERY occupied deme from the full counts, so denser tissue (packed
        DCIS ducts, invasive masses) reads dark and looser stroma reads pale, giving a complete,
        high-contrast image. Cancer nuclei tint toward hematoxylin purple; stroma stays eosin pink.

        The image is in the raw deme-grid frame (``px`` pixels per deme). To overlay a placed Visium
        slide, shift the spot coordinates back into this frame by the placement translation,
        ``spot_coords.mean(0) - section_cell_crd.mean(0)``.

        Parameters
        ----------
        px : int, default 6
            Pixels per deme.
        darkness : float, default 0.8
            Maximum staining of the densest tissue.
        sigma_frac : float, default 0.5
            Gaussian smoothing sigma as a fraction of ``px``.

        Returns
        -------
        image : numpy.ndarray, shape (G*px, G*px, 3), float32
            RGB H&E morphology image in ``[0, 1]``.
        px : float
            Pixels per deme (the scale factor mapping deme coords onto the image).
        """
        from scipy.ndimage import gaussian_filter
        G = self.grid_size
        gid_is_can = {g: self._is_cancer(g) for g in self.genotypes}
        tot = np.zeros(G * G); can = np.zeros(G * G)
        for di in range(min(len(self.demes), G * G)):
            d = self.demes[di]
            if not d:
                continue
            tot[di] = sum(d.values())
            can[di] = sum(c for g, c in d.items() if gid_is_can[g])
        tot = tot.reshape(G, G); can = can.reshape(G, G)
        frac = np.divide(can, tot, out=np.zeros_like(can), where=tot > 0)
        ref = np.percentile(tot[tot > 0], 98) if np.any(tot > 0) else 1.0
        dens = np.clip(tot / (ref or 1.0), 0.0, 1.0)
        dens = gaussian_filter(np.kron(dens, np.ones((px, px))), sigma=px * sigma_frac)
        fr = gaussian_filter(np.kron(frac, np.ones((px, px))), sigma=px * sigma_frac)
        d = darkness * dens
        # eosin-pink stroma darkening toward hematoxylin purple where cancer nuclei are dense
        image = np.stack([1 - d * (0.35 + 0.10 * fr),
                          1 - d * (0.62 + 0.33 * fr),
                          1 - d * (0.33 + 0.05 * fr)], axis=-1)
        return np.clip(image, 0.0, 1.0).astype(np.float32), float(px)

    def plot_grid(self, color=None, ax=None, **kwargs):
        """Plot the spatial deme grid, colouring each deme by a per-cell attribute.

        Rebuilds the per-cell data if stale, then draws the grid coloured by ``color``
        (default the dominant clone / cell type). When the metastatic compartment is enabled
        and no explicit ``ax`` is passed, both the primary and met grids are drawn side by
        side with a shared clone colormap.

        Parameters
        ----------
        color : str or list of str, optional
            Attribute key(s) to colour by; defaults to the dominant clone / cell type.
        ax : matplotlib.axes.Axes, optional
            Axes to draw into; forces the single-panel primary view. A new figure is created
            when ``None``.

        Returns
        -------
        matplotlib.axes.Axes
            The axes drawn into.
        """
        from .. import viz
        if (self.cell_data is None or getattr(self, "_cell_data_stale", False)
                or self.cell_data["cell_type"].shape[0] != self.get_tumor_size()):
            self.make_cell_data()
        # With the metastatic deposit enabled, `plot_grid()` shows BOTH grids (primary + met) side by
        # side with a shared clone colormap, so the same clone is the same colour across compartments.
        # (Pass an explicit `ax` to force the single-panel primary view instead.)
        if self._met_enabled and ax is None and "cell_compartment" in self.cell_data:
            return self.plot_grid_compartments(color=color, **kwargs)
        return viz.plot_grid(self.cell_data, self.grid_size, self.traces,
                             self.genotypes_parents, color=color, ax=ax, **kwargs)

    def plot_grid_compartments(self, color=None, axes=None, by_drivers=False, by_stage=False, **kwargs):
        """Two spatial grids (primary + met) side by side, shared clone colormap. ``by_drivers``
        colours by functional clone (matching the driver-collapsed 2-band Muller); ``by_stage`` colours
        by the stage-dominant driver; pass ``min_freq`` to also merge below-threshold clones."""
        from .. import viz
        if (self.cell_data is None or getattr(self, "_cell_data_stale", False)
                or self.cell_data["cell_type"].shape[0] != self.get_tumor_size()):
            self.make_cell_data()
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        driver_map = self._functional_signatures() if by_drivers else None
        return viz.plot_grid_compartments(self.cell_data, self.grid_size, self.met_grid_size,
                                          self.traces, self.genotypes_parents, color=color,
                                          axes=axes, driver_map=driver_map, **kwargs)

    def plot_muller_compartments(self, axes=None, by_drivers=False, by_stage=False, star_seeder=True,
                                 **kwargs):
        """Two-band Muller (primary over met) sharing one colormap with the grids. ``by_drivers``
        collapses genotypes to FUNCTIONAL clones and ``min_freq`` (a fraction) merges below-threshold
        clones, so selective sweeps (DCIS→IDC breach, the met founder, post-chemo resistant escape) show
        as bands instead of a passenger rainbow. ``by_stage`` instead colours by the STAGE-DOMINANT
        driver. ``star_seeder`` stars the first met-seeding clone's band in BOTH panels at the
        seeding moment (so a minor founder is findable)."""
        from .. import viz
        driver_map = self._functional_signatures() if by_drivers else None
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        if star_seeder and "star_genotype" not in kwargs:
            seeders = [e for e in self.events if e.get("event") == "seeding"]
            if seeders:
                kwargs["star_genotype"] = str(seeders[0]["genotype"])
                for i, tr in enumerate(self.traces):
                    if any(str(g) not in normal_names for g in tr.get("met_counts", {})):
                        kwargs["star_gen"] = i
                        break
        return viz.plot_muller_compartments(self.traces, self.genotypes_parents, axes=axes,
                                            driver_map=driver_map, **kwargs)

    def plot_muller_founders(self, ax=None, **kwargs):
        """A SINGLE primary Muller highlighting the clone(s) that seeded the metastasis, so you can
        see which — usually minor — primary population founds the met. Founders are read from the
        seeding events; a primary clone is highlighted iff it shares a functional signature with one."""
        from .. import viz
        founders = {e["genotype"] for e in self.events if e.get("event") == "seeding"}
        return viz.plot_muller_founders(self.traces, self.genotypes_parents, founders,
                                        self._functional_signatures(), ax=ax, **kwargs)

    def plot_clone_tree(self, ax=None, by_stage=False, by_drivers=False, **kwargs):
        """Ground-truth clone tree: the TRUE genotypes_parents genealogy collapsed to the
        display clones, each node at its true mutational depth, sized by peak population and coloured by
        ``by_stage`` (stage-dominant driver) or ``by_drivers`` (functional clone). Pass ``min_freq`` to
        set the collapse threshold (default 0.02)."""
        from .. import viz
        if by_stage:
            colors, present = self._stage_colors()
            kwargs.setdefault("clone_colors", colors)
            kwargs.setdefault("color_legend", viz.stage_legend(present))
        driver_map = self._functional_signatures() if by_drivers else None
        return viz.plot_clone_tree(self.traces, self.genotypes_parents, driver_map=driver_map,
                                   ax=ax, **kwargs)
