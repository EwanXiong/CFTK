"""
report_generator.py — collect all PNG results and embed into HTML report.
All PNGs are base64-encoded so the output is a single self-contained file.
Groups and sample membership are read from cftk_init.json (no hardcoding).
"""

import base64
import datetime
import glob
import json
import os
import re


def _b64(png_path, max_width=2400, jpeg_quality=95):
    """
    Convert image to JPEG (quality=88) and base64-encode.
    JPEG at quality 88 is visually near-lossless but ~70% smaller than PNG.
    Downscales to max_width if image is wider.
    Falls back to raw PNG if PIL unavailable.
    """
    try:
        from PIL import Image
        import io
        img = Image.open(png_path)
        # convert to RGB (JPEG does not support alpha)
        if img.mode in ("RGBA", "P", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            alpha = img.split()[-1] if img.mode in ("RGBA", "LA") else None
            bg.paste(img, mask=alpha)
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")
        # downscale if wider than max_width
        if img.width > max_width:
            ratio = max_width / img.width
            img   = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=jpeg_quality, optimize=True)
        data = base64.b64encode(buf.getvalue()).decode()
        return "data:image/jpeg;base64," + data
    except Exception:
        with open(png_path, "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _img_tag(png_path, alt="", style="width:100%;", zoomable=True):
    if not png_path or not os.path.exists(png_path) or os.path.isdir(png_path):
        return f'<div class="fig-placeholder"><span class="hint">{alt} (not found)</span></div>'
    src = _b64(png_path)
    img = f'<img src="{src}" alt="{alt}" style="{style}"'
    if zoomable:
        img += f' onclick="openLightbox(this.src)" title="Click to zoom"'
    img += '>'
    return img


def _b64_svg(svg_path):
    """Embed SVG as base64 data URI."""
    if not svg_path or not os.path.exists(svg_path):
        return ""
    with open(svg_path, "rb") as f:
        return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode()


def _img_tag_svg(svg_path, alt=""):
    """Render SVG as inline <img> with zoom support."""
    if not svg_path or not os.path.exists(svg_path):
        return f'<div class="fig-placeholder"><span class="hint">{alt} (not found)</span></div>'
    src = _b64_svg(svg_path)
    return (f'<img src="{src}" alt="{alt}" '
            f'style="width:100%;max-height:320px;object-fit:contain;padding:8px;" '
            f'onclick="openLightbox(this.src)" title="Click to zoom">')


def _multiqc_png_gallery(multiqc_dir, uid_prefix):
    """
    Build a dropdown gallery from PNG files exported by MultiQC.
    Scans multiqc_plots/png/ under multiqc_dir.
    Converts raw filenames to readable labels:
      cutadapt_filtered_reads_plot-cnt → Cutadapt Filtered Reads (Count)
    """
    png_dir = os.path.join(multiqc_dir, "multiqc_plots", "png")
    PH = '<div class="fig-placeholder" style="height:80px;"><span class="hint">{}</span></div>'
    if not os.path.exists(png_dir):
        return PH.format("MultiQC PNG plots not found — re-run process with --export flag")

    pngs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    if not pngs:
        return PH.format("No PNG plots found in multiqc_plots/png/")

    def _label(fname):
        """Convert filename to readable label."""
        stem = os.path.splitext(os.path.basename(fname))[0]
        # suffix mappings
        stem = stem.replace("-cnt", " (Count)")
        stem = stem.replace("-pct", " (Percent)")
        stem = stem.replace("_plot_3_", " — ")
        stem = stem.replace("_plot_", " — ")
        stem = stem.replace("_", " ")
        return stem.title()

    images = [(_label(p), p) for p in pngs]
    return _dropdown_gallery("Figure:", images,
                             uid=re.sub(r'[^a-zA-Z0-9_]', '_', uid_prefix))




def _mbias_from_groups(rd, groups):
    """
    Build M-bias SVG gallery per group.
    File format: {sample_name}_OT.svg / {sample_name}_OB.svg
    Each dropdown item shows OT and OB side by side in one row.
    """
    meth_dirs = [
        os.path.join(rd, "1_process", "4_methylation"),
        os.path.join(rd, "4_methylation"),
    ]
    meth_dir = next((d for d in meth_dirs if os.path.exists(d)), None)
    if not meth_dir:
        return "<p><em>No methylation results found.</em></p>"

    # scan for *_OT.svg, derive OB path
    ot_files = sorted(glob.glob(os.path.join(meth_dir, "*_OT.svg")))
    if not ot_files:
        return "<p><em>No M-bias SVG files found in 4_methylation/.</em></p>"

    # map sample_name → (OT_path, OB_path)
    sample_svgs = {}
    for ot in ot_files:
        stem = os.path.basename(ot).replace("_OT.svg", "")
        ob   = ot.replace("_OT.svg", "_OB.svg")
        sample_svgs[stem] = (ot, ob if os.path.exists(ob) else None)

    html = ""
    for grp, members in groups.items():
        grp_items = []
        for sname in members:
            match = sample_svgs.get(sname)
            if match:
                grp_items.append((sname, match))
        if not grp_items:
            continue

        onchange_js = (
            "var idx=this.selectedIndex;"
            "var panels=this.parentNode.parentNode.querySelectorAll('.dd-panel');"
            "for(var i=0;i<panels.length;i++){"
            "panels[i].style.display=(i===idx)?'block':'none';}"
        )
        opts = "".join(f'<option>{sname}</option>' for sname, _ in grp_items)

        panels = ""
        for i, (sname, (ot, ob)) in enumerate(grp_items):
            ot_img = _img_tag_svg(ot, f"{sname} OT")
            ob_img = _img_tag_svg(ob, f"{sname} OB") if ob else ""
            # OT and OB side by side in one row
            row = (
                f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;">'
                f'  <div class="fig-card">'
                f'    <div class="fig-img">{ot_img}</div>'
                f'    <div class="fig-body"><div class="fig-label">OT strand</div></div>'
                f'  </div>'
                f'  <div class="fig-card">'
                f'    <div class="fig-img">{ob_img if ob_img else _img_tag(None, "OB")}</div>'
                f'    <div class="fig-body"><div class="fig-label">OB strand</div></div>'
                f'  </div>'
                f'</div>'
            )
            panels += (
                f'<div class="dd-panel" style="display:{("block" if i == 0 else "none")};">'
                f'{row}</div>'
            )

        html += (
            f'<div style="margin-bottom:4px;font-size:12px;font-weight:500;'
            f'color:var(--ink3);font-family:var(--mono)">{grp}</div>'
            f'<div style="margin-bottom:24px;">'
            f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
            f'<label style="font-size:12px;font-weight:500;color:var(--ink3);'
            f'font-family:var(--mono);">Sample:</label>'
            f'<select onchange="{onchange_js}" '
            f'style="font-size:12px;padding:4px 10px;border:1px solid var(--rule);'
            f'border-radius:4px;background:white;color:var(--ink2);cursor:pointer;">'
            f'{opts}</select></div>'
            f'{panels}</div>'
        )
    return html or "<p><em>No M-bias results found for configured samples.</em></p>"


def _multiqc_named_galleries(multiqc_dir, gallery_groups, uid_prefix):
    """
    Build multiple independent dropdown galleries from MultiQC PNG exports.
    gallery_groups: list of (gallery_title, [filename_stem_prefix, ...])
      e.g. [("Filtered Reads", ["cutadapt_filtered_reads"]),
             ("Trimmed Sequences", ["cutadapt_trimmed_sequences"])]
    Files are matched by prefix; all PNGs in multiqc_plots/png/ not matched
    by any group fall into the last group as fallback.
    """
    png_dir = os.path.join(multiqc_dir, "multiqc_plots", "png")
    PH = ('<div class="fig-placeholder" style="height:80px;">'
          '<span class="hint">MultiQC PNG not found — re-run with --export</span></div>')
    if not os.path.exists(png_dir):
        return PH

    all_pngs = sorted(glob.glob(os.path.join(png_dir, "*.png")))
    if not all_pngs:
        return PH

    def _label(fname):
        stem = os.path.splitext(os.path.basename(fname))[0]
        stem = stem.replace("-cnt", " (Count)")
        stem = stem.replace("-pct", " (Percent)")
        stem = stem.replace("_plot_3_", " — ")
        stem = stem.replace("_plot_", " — ")
        stem = stem.replace("_", " ")
        return stem.strip().title()

    # group pngs by prefix
    grouped = {title: [] for title, _ in gallery_groups}
    matched = set()
    for title, prefixes in gallery_groups:
        for png in all_pngs:
            bname = os.path.basename(png)
            if any(bname.startswith(p) for p in prefixes):
                grouped[title].append((_label(png), png))
                matched.add(png)

    html = ""
    for title, prefixes in gallery_groups:
        images = grouped[title]
        if not images:
            continue
        html += f'<div style="margin-bottom:20px;">'
        html += (
            f'<div style="font-size:12px;font-weight:500;color:var(--ink3);'
            f'font-family:var(--mono);margin-bottom:8px;">{title}</div>'
        )
        html += _dropdown_gallery("Figure:", images,
                                  uid=re.sub(r'[^a-zA-Z0-9_]', '_',
                                             f"{uid_prefix}_{title}"))
        html += "</div>"
    return html or PH


def _sec_sample_overview(groups):
    """Render sample/group overview table with per-group row background color."""
    COLORS = [
        ("rgba(29,107,85,.10)",  "#1a6b55"),   # teal
        ("rgba(192,90,44,.10)",  "#c05a2c"),   # coral
        ("rgba(90,74,138,.10)",  "#5a4a8a"),   # purple
        ("rgba(176,122,24,.10)", "#b07a18"),   # amber
        ("rgba(30,95,160,.10)",  "#1e5fa0"),   # blue
    ]
    rows = ""
    for gi, (grp, members) in enumerate(groups.items()):
        row_bg, fg = COLORS[gi % len(COLORS)]
        for i, sname in enumerate(members):
            grp_cell = ""
            if i == 0:
                grp_cell = (
                    f'<td rowspan="{len(members)}" '
                    f'style="vertical-align:middle;border-top:1px solid var(--rule);'
                    f'background:{row_bg};">'
                    f'<span class="grp-badge" '
                    f'style="background:{row_bg};color:{fg};'
                    f'border:1px solid {fg}4d;">{grp}</span></td>'
                )
            rows += (
                f'<tr>'
                f'<td style="border-top:1px solid var(--rule);background:{row_bg};'
                f'color:var(--ink2);">{i+1}</td>'
                f'<td style="border-top:1px solid var(--rule);background:{row_bg};'
                f'font-family:var(--mono);color:var(--ink);">{sname}</td>'
                f'{grp_cell}'
                f'</tr>'
            )

    n_total   = sum(len(v) for v in groups.values())
    scrollable = "scrollable" if n_total > 5 else ""
    return f"""
    <section class="section" id="part_overview">
      <div class="section-header">
        <span class="section-num">OVERVIEW</span>
        <h2 class="section-title">Sample Overview</h2>
        <span class="section-tag">{n_total} samples · {len(groups)} groups</span>
      </div>
      <div class="sample-table-wrap {scrollable}">
        <table class="sample-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Sample</th>
              <th>Group</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
      </div>
    </section>"""


def _find(results_dir, *parts, pattern="*.png"):
    path = os.path.join(results_dir, *parts, pattern)
    files = sorted(glob.glob(path))
    return files[0] if files else None


def _find_all(results_dir, *parts, pattern="*.png"):
    path = os.path.join(results_dir, *parts, pattern)
    return sorted(glob.glob(path))


def _uid(prefix=""):
    import random, string
    # sanitize prefix: replace non-alphanumeric with underscore
    safe = re.sub(r'[^a-zA-Z0-9]', '_', prefix)
    return safe + ''.join(random.choices(string.ascii_lowercase, k=8))


def _dropdown_gallery(title, images, uid=None):
    """
    Dropdown gallery. Uses onchange attribute with fully self-contained JS
    expression — no external functions, no addEventListener, no DOMContentLoaded.
    The onchange JS string iterates panel ids and sets display directly.
    """
    if not images:
        return f"<p><em>No {title} results found.</em></p>"
    uid = uid or _uid("dd_")
    uid = re.sub(r'[^a-zA-Z0-9_]', '_', uid)
    n   = len(images)

    # Build the onchange JS as a plain string (no f-string inside).
    # It reads select.value and shows/hides panels by id.
    # Written without any curly braces so it's safe inside f-strings.
    js_parts = []
    js_parts.append("var v=this.value;")
    for i in range(n):
        js_parts.append(
            f"document.getElementById('{uid}_p{i}').style.display="
            f"(v==='{i}')?'block':'none';"
        )
    onchange_js = "".join(js_parts)

    # option tags
    opts = "".join(
        f'<option value="{i}">{label}</option>'
        for i, (label, _) in enumerate(images)
    )

    # panel divs — Python controls initial display state
    panels = ""
    for i, (label, path) in enumerate(images):
        disp = "block" if i == 0 else "none"
        panels += (
            f'<div id="{uid}_p{i}" style="display:{disp};">'
            f'<div class="fig-card">'
            f'<div class="fig-img">{_img_tag(path, label)}</div>'
            f'<div class="fig-body"><div class="fig-label">{label}</div></div>'
            f'</div></div>'
        )

    return (
        f'<div style="margin-bottom:20px;">'
        f'<div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">'
        f'<label style="font-size:12px;font-weight:500;color:var(--ink3);'
        f'font-family:var(--mono);">{title}</label>'
        f'<select onchange="{onchange_js}"'
        f' style="font-size:12px;padding:4px 10px;border:1px solid var(--rule);'
        f'border-radius:4px;background:white;color:var(--ink2);cursor:pointer;">'
        f'{opts}</select></div>'
        f'{panels}'
        f'</div>'
    )


def _read_perf_table(tsv_path):
    """Render modality_performance.tsv as clean HTML table."""
    if not tsv_path or not os.path.exists(tsv_path):
        return "<p><em>No modality performance data.</em></p>"
    try:
        import pandas as pd
        import ast, re as _re
        df = pd.read_csv(tsv_path, sep="\t", index_col=0)

        # extract best_roc_auc_mean and best_classifier name only
        rows = []
        for idx, row in df.iterrows():
            best_auc  = row.get("best_roc_auc_mean", "—")
            best_clf_raw = row.get("best_classifier, idx", "—")
            # parse classifier name from tuple string: ('RandomForestClassifier', ...)
            m = _re.search(r"'([^']+)'", str(best_clf_raw))
            clf_name = m.group(1) if m else str(best_clf_raw)
            # shorten class names
            clf_name = clf_name.replace("Classifier","").replace("Regression","Reg")

            # get per-clf mean AUCs from the array column
            mean_col = [c for c in df.columns if "mean" in c and "best" not in c]
            if mean_col:
                raw = row[mean_col[0]]
                try:
                    vals = ast.literal_eval(str(raw))
                    clf_aucs = " / ".join(f"{v:.3f}" for v in vals)
                except Exception:
                    clf_aucs = str(raw)
            else:
                clf_aucs = "—"

            rows.append({
                "Modality":          idx,
                "Clf AUCs (RFC/LR/SVC)": clf_aucs,
                "Best Classifier":   clf_name,
                "Best AUC":          f"{best_auc:.4f}" if isinstance(best_auc, float) else best_auc,
            })

        tdf = pd.DataFrame(rows).set_index("Modality")
        return tdf.to_html(classes="result-table", border=0, escape=False)
    except Exception as e:
        return f"<p><em>Could not load table: {e}</em></p>"


# ── Section builders ──────────────────────────────────────────────────────────

def _sec_power(rd):
    img = _img_tag(_find(rd, "0_power", pattern="power_cumulative.png"), "Power curve")
    return f"""
    <section class="section" id="part0">
      <div class="section-header">
        <span class="section-num">PART 00</span>
        <h2 class="section-title">Power Analysis</h2>
        <span class="section-tag">power</span>
      </div>
      <div class="fig-card">
        <div class="fig-img tall">{img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 0A · Power vs CpG proportion</div>
          <div class="fig-caption">Complementary ECDF of per-CpG detection power
            at different read depths.</div>
        </div>
      </div>
    </section>"""


def _sec_process(rd, groups=None):
    groups = groups or {}

    def _multiqc_html(*subdirs):
        """Find multiqc_report.html, try multiple path patterns."""
        candidates = [
            os.path.join(rd, "1_process", *subdirs, "multiqc", "multiqc_report.html"),
            os.path.join(rd, *subdirs, "multiqc", "multiqc_report.html"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return candidates[0]  # first candidate for placeholder

    def _multiqc_dir(*subdirs):
        """Locate the multiqc output directory."""
        candidates = [
            os.path.join(rd, "1_process", *subdirs, "multiqc"),
            os.path.join(rd, *subdirs, "multiqc"),
        ]
        return next((d for d in candidates if os.path.exists(d)), candidates[0])

    trim_gallery = _multiqc_named_galleries(
        _multiqc_dir("1_trimming"),
        [
            ("Filtered Reads",    ["cutadapt_filtered_reads"]),
            ("Trimmed Sequences", ["cutadapt_trimmed_sequences"]),
        ],
        uid_prefix="trim"
    )
    align_gallery = _multiqc_named_galleries(
        _multiqc_dir("2_alignment"),
        [
            ("Alignment",          ["samtools_alignment"]),
            ("Flagstat & Coverage", ["samtools-flagstat", "samtools-stats"]),
        ],
        uid_prefix="align"
    )
    mbias_block   = _mbias_from_groups(rd, groups)

    return f"""
    <section class="section" id="part1">
      <div class="section-header">
        <span class="section-num">PART 01</span>
        <h2 class="section-title">Data Processing</h2>
        <span class="section-tag">process</span>
      </div>
      <p style="font-size:13px;color:var(--ink2);margin-bottom:24px;">
        Raw sequencing data processed through adapter trimming (Trim Galore),
        bisulfite alignment (bwameth), duplicate removal (sambamba),
        and methylation calling (MethylDackel).
      </p>

      <h3 class="subsec-title" id="part1_1">1.1 Trim Galore QC</h3>
      {trim_gallery}

      <h3 class="subsec-title" id="part1_2">1.2 bwameth Alignment QC</h3>
      {align_gallery}

      <h3 class="subsec-title" id="part1_3">1.3 M-bias</h3>
      {mbias_block}
    </section>"""


def _sec_qc(rd, groups=None):
    groups = groups or {}
    # 2.1 methylation distribution
    meth_img = _img_tag(
        _find(rd, "2_qc", "1_methylation_distribution", pattern="*.png"),
        "Methylation distribution")

    # 2.2 fragment length — all PNGs in subdir, build dropdown
    frag_dir  = os.path.join(rd, "2_qc", "2_fragment_length")
    frag_pngs = sorted(glob.glob(os.path.join(frag_dir, "*.png"))) if os.path.exists(frag_dir) else []

    def _frag_label(fname):
        stem = os.path.splitext(os.path.basename(fname))[0]
        stem = stem.replace("fragment_length", "").strip("_")
        return "All Samples" if not stem else stem.replace("_", " ").title()

    frag_items = sorted(
        [(_frag_label(p), p) for p in frag_pngs],
        key=lambda x: (0 if x[0] == "All Samples" else (2 if "Comparison" in x[0] else 1), x[0])
    )
    frag_block = _dropdown_gallery("Select figure:", frag_items, uid="frag_dd")

    # 2.3 dinucleotide
    dinuc_img = _img_tag(
        _find(rd, "2_qc", "3_dinucleotide_freq", pattern="*.png"),
        "Dinucleotide frequency")

    return f"""
    <section class="section" id="part2">
      <div class="section-header">
        <span class="section-num">PART 02</span>
        <h2 class="section-title">QC Analysis</h2>
        <span class="section-tag">qc</span>
      </div>

      <h3 class="subsec-title" id="part2_1">2.1 Methylation Distribution</h3>
      <div class="fig-card" style="margin-bottom:28px;">
        <div class="fig-img tall">{meth_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 2A · Methylation β-value distribution</div>
          <div class="fig-caption">CpG methylation β-value density across all samples.</div>
        </div>
      </div>

      <h3 class="subsec-title" id="part2_2">2.2 Fragment Length Distribution</h3>
      {frag_block}

      <h3 class="subsec-title" id="part2_3">2.3 Dinucleotide Frequency</h3>
      <div class="fig-card">
        <div class="fig-img tall">{dinuc_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 2C · Dinucleotide frequency</div>
          <div class="fig-caption">AT/GC dinucleotide fractions relative to
            fragment center. 10-bp periodicity reflects nucleosome positioning.</div>
        </div>
      </div>

    </section>"""


def _sec_differential(rd):
    diff_base = os.path.join(rd, "3_differential")
    modalities = []
    if os.path.exists(diff_base):
        modalities = sorted([
            d for d in os.listdir(diff_base)
            if os.path.isdir(os.path.join(diff_base, d)) and d != "dmr"
        ])

    # build per-plot-type dropdown (PCA / Violin / Heatmap) across modalities
    def _plot_dropdown(plot_type, stem, uid_prefix):
        items = []
        for mod in modalities:
            p = _find(rd, "3_differential", mod, pattern=f"{stem}.png")
            if p:
                items.append((mod.upper(), p))
        return _dropdown_gallery(f"Modality:", items, uid=f"{uid_prefix}_{stem}")

    pca_block  = _plot_dropdown("PCA",     "pca",     "diff")
    viol_block = _plot_dropdown("Violin",  "violin",  "diff")
    hmap_block = _plot_dropdown("Heatmap", "heatmap", "diff")

    # DMR volcano
    dmr_img = _img_tag(
        _find(rd, "3_differential", "dmr", pattern="dmr_volcano.png"), "DMR volcano")

    return f"""
    <section class="section" id="part3">
      <div class="section-header">
        <span class="section-num">PART 03</span>
        <h2 class="section-title">Differential Analysis</h2>
        <span class="section-tag">diff</span>
      </div>

      <h3 class="subsec-title" id="part3_1">3.1 PCA</h3>
      {pca_block}

      <h3 class="subsec-title" id="part3_2">3.2 Violin</h3>
      {viol_block}

      <h3 class="subsec-title" id="part3_3">3.3 Heatmap</h3>
      {hmap_block}

      <h3 class="subsec-title" id="part3_4">3.4 DMR Analysis</h3>
      <div class="fig-card">
        <div class="fig-img tall">{dmr_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 3D · DMR Volcano</div>
          <div class="fig-caption">DMR volcano: mean methylation difference vs
            −log₁₀(q-value). Gene labels: top differentially methylated promoters.</div>
        </div>
      </div>
    </section>"""


def _sec_fragmentomics(rd, groups):
    """
    groups: dict {group_name: [sample_name, ...]}  from cftk_init.json samples
    """
    # collect all sample names across groups (flat)
    all_samples = set()
    for members in groups.values():
        all_samples.update(members)

    def _sample_dropdown(analysis_dir, file_pattern, strip_suffix, uid_prefix):
        """Build per-group dropdowns for per-sample figures."""
        all_pngs = sorted(glob.glob(os.path.join(analysis_dir, file_pattern)))
        # map sample_name → path
        name_to_path = {}
        for p in all_pngs:
            stem = os.path.splitext(os.path.basename(p))[0]
            # strip known suffixes (.markdup, then the analysis suffix)
            clean = stem.replace(".markdup", "")
            for sfx in [strip_suffix, strip_suffix.replace("*","")]:
                clean = clean.replace(sfx, "").strip("_").strip(".")
            for s in all_samples:
                if clean == s or stem.startswith(s):
                    name_to_path[s] = p
                    break

        blocks = ""
        for grp, members in groups.items():
            items = [(s, name_to_path[s]) for s in members if s in name_to_path]
            if items:
                blocks += f'<div style="margin-bottom:4px;font-size:12px;font-weight:500;color:var(--ink3);font-family:var(--mono)">{grp}</div>'
                blocks += _dropdown_gallery("Sample:", items, uid=f"{uid_prefix}_{grp}")
        return blocks

    def _group_and_compare(analysis_dir, grp_pat, cmp_pat, groups):
        """Render group-mean images and comparison image."""
        html = ""
        for grp in groups:
            imgs = sorted(glob.glob(os.path.join(analysis_dir, grp_pat.replace("{grp}", grp))))
            if imgs:
                html += f"""
        <div class="fig-card" style="margin-bottom:16px;">
          <div class="fig-img tall">{_img_tag(imgs[0], f"{grp} mean")}</div>
          <div class="fig-body">
            <div class="fig-label">{grp} — Group Mean</div>
          </div>
        </div>"""
        if cmp_pat:
            cmp_imgs = [p for p in sorted(glob.glob(os.path.join(analysis_dir, cmp_pat)))
                        if os.path.isfile(p)]
            if cmp_imgs:
                html += f"""
        <div class="fig-card" style="margin-bottom:16px;">
          <div class="fig-img tall">{_img_tag(cmp_imgs[0], "Comparison")}</div>
          <div class="fig-body">
            <div class="fig-label">Group Comparison</div>
          </div>
        </div>"""
        return html

    # ── DELFI ─────────────────────────────────────────────────────────────────
    delfi_dir     = os.path.join(rd, "4_fragmentomics", "delfi")
    delfi_samples = _sample_dropdown(delfi_dir, "*_delfi_genome.png", "_delfi_genome", "delfi")
    delfi_groups  = _group_and_compare(delfi_dir, "delfi_{grp}.png", "delfi_comparison.png",
                                       list(groups.keys()))

    # ── End-motif ─────────────────────────────────────────────────────────────
    em_dir     = os.path.join(rd, "4_fragmentomics", "end_motif")
    em_samples = _sample_dropdown(em_dir, "*mer_top20.png", "mer_top20", "em")
    em_groups  = _group_and_compare(em_dir, "end_motif_{grp}.png", "", list(groups.keys()))
    # end_motif has no comparison, only group means
    em_groups_html = ""
    for grp in groups:
        imgs = sorted(glob.glob(os.path.join(em_dir, f"end_motif_{grp}.png")))
        if imgs:
            em_groups_html += f"""
        <div class="fig-card" style="margin-bottom:16px;">
          <div class="fig-img tall">{_img_tag(imgs[0], f"{grp} mean")}</div>
          <div class="fig-body">
            <div class="fig-label">{grp} — Group Mean Top 20</div>
          </div>
        </div>"""

    # ── Cleavage ──────────────────────────────────────────────────────────────
    cl_dir = os.path.join(rd, "4_fragmentomics", "cleavage")
    cl_html = ""
    for grp in groups:
        imgs = sorted(glob.glob(os.path.join(cl_dir, f"cleavage_{grp}_samples.png")))
        if imgs:
            cl_html += f"""
        <div class="fig-card" style="margin-bottom:16px;">
          <div class="fig-img tall">{_img_tag(imgs[0], f"{grp} samples")}</div>
          <div class="fig-body">
            <div class="fig-label">{grp} — All Samples</div>
          </div>
        </div>"""
    cmp_imgs = sorted(glob.glob(os.path.join(cl_dir, "cleavage_comparison.png")))
    if cmp_imgs:
        cl_html += f"""
        <div class="fig-card" style="margin-bottom:16px;">
          <div class="fig-img tall">{_img_tag(cmp_imgs[0], "Comparison")}</div>
          <div class="fig-body">
            <div class="fig-label">Group Comparison</div>
          </div>
        </div>"""

    # ── WPS ───────────────────────────────────────────────────────────────────
    wps_dir     = os.path.join(rd, "4_fragmentomics", "wps")
    wps_samples = _sample_dropdown(wps_dir, "*.wps_profile.png", ".wps_profile", "wps")

    return f"""
    <section class="section" id="part4">
      <div class="section-header">
        <span class="section-num">PART 04</span>
        <h2 class="section-title">Fragmentomics</h2>
        <span class="section-tag">frag</span>
      </div>

      <h3 class="subsec-title" id="part4_2">4.1 DELFI</h3>
      {delfi_samples or '<p><em>No per-sample DELFI results.</em></p>'}
      {delfi_groups  or '<p><em>No group DELFI results.</em></p>'}

      <h3 class="subsec-title" id="part4_3">4.2 End Motif</h3>
      {em_samples    or '<p><em>No per-sample end-motif results.</em></p>'}
      {em_groups_html or '<p><em>No group end-motif results.</em></p>'}

      <h3 class="subsec-title" id="part4_4">4.3 Cleavage</h3>
      {cl_html       or '<p><em>No cleavage results.</em></p>'}

      <h3 class="subsec-title" id="part4_5">4.4 WPS</h3>
      {wps_samples   or '<p><em>No per-sample WPS results.</em></p>'}
    </section>"""


def _sec_mesa(rd):
    perf_tsv = os.path.join(rd, "5_mesa", "modality_performance.tsv")
    roc_img  = _img_tag(_find(rd, "5_mesa", pattern="mesa_roc.png"),  "ROC curve")
    hmap_img = _img_tag(_find(rd, "5_mesa", pattern="mesa_heatmap.png"), "Prediction heatmap")
    sp_img   = _img_tag(_find(rd, "5_mesa", pattern="mesa_spearman.png"), "Spearman correlation")
    perf_tbl = _read_perf_table(perf_tsv)

    return f"""
    <section class="section" id="part5">
      <div class="section-header">
        <span class="section-num">PART 05</span>
        <h2 class="section-title">MESA Multimodal Modeling</h2>
        <span class="section-tag">mesa</span>
      </div>

      <h3 class="subsec-title" id="part5_1">5.1 Modality Performance</h3>
      {perf_tbl}

      <h3 class="subsec-title" id="part5_2">5.2 ROC Curve</h3>
      <div class="fig-card" style="margin-bottom:20px;">
        <div class="fig-img tall">{roc_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 5A · ROC curves (LOOCV)</div>
          <div class="fig-caption">ROC curves for each modality and the MESA
            multimodal model. AUC values shown in legend.</div>
        </div>
      </div>

      <h3 class="subsec-title" id="part5_3">5.3 Prediction Heatmap</h3>
      <div class="fig-card" style="margin-bottom:20px;">
        <div class="fig-img tall">{hmap_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 5B · Per-sample LOOCV predictions</div>
          <div class="fig-caption">LOOCV prediction probability heatmap per modality.
            Samples sorted by true label.</div>
        </div>
      </div>

      <h3 class="subsec-title" id="part5_4">5.4 Spearman Correlation</h3>
      <div class="fig-card">
        <div class="fig-img tall">{sp_img}</div>
        <div class="fig-body">
          <div class="fig-label">Figure 5C · Modality Spearman correlation</div>
          <div class="fig-caption">Pairwise Spearman correlation of LOOCV
            prediction scores across modalities.</div>
        </div>
      </div>
    </section>"""


# ── Sidebar builder ───────────────────────────────────────────────────────────

def _build_sidebar(rd, groups):
    diff_base  = os.path.join(rd, "3_differential")
    modalities = sorted([
        d for d in os.listdir(diff_base)
        if os.path.isdir(os.path.join(diff_base, d)) and d != "dmr"
    ]) if os.path.exists(diff_base) else []

    mod_links = "".join(
        f'<a href="#part3_{i+1}" class="nav-link nav-sub">'
        f'<span class="dot"></span>{i+1} · {m.upper()}</a>'
        for i, m in enumerate(["PCA", "Violin", "Heatmap", "DMR"])
    )

    return f"""
    <div class="nav-section">
      <div class="nav-label">Analysis</div>
      <a href="#part_overview" class="nav-link nav-top"><span class="dot"></span>Overview</a>
      <a href="#part0" class="nav-link nav-top"><span class="dot"></span>0 · Power</a>
      <a href="#part1" class="nav-link nav-top"><span class="dot"></span>1 · Processing</a>
      <a href="#part1_1" class="nav-link nav-sub"><span class="dot"></span>1.1 · Trim Galore</a>
      <a href="#part1_2" class="nav-link nav-sub"><span class="dot"></span>1.2 · Alignment</a>
      <a href="#part1_3" class="nav-link nav-sub"><span class="dot"></span>1.3 · M-bias</a>

      <a href="#part2" class="nav-link nav-top"><span class="dot"></span>2 · QC</a>
      <a href="#part2_1" class="nav-link nav-sub"><span class="dot"></span>2.1 · Methylation</a>
      <a href="#part2_2" class="nav-link nav-sub"><span class="dot"></span>2.2 · Fragment Length</a>
      <a href="#part2_3" class="nav-link nav-sub"><span class="dot"></span>2.3 · Dinucleotide</a>

      <a href="#part3" class="nav-link nav-top"><span class="dot"></span>3 · Differential</a>
      <a href="#part3_1" class="nav-link nav-sub"><span class="dot"></span>3.1 · PCA</a>
      <a href="#part3_2" class="nav-link nav-sub"><span class="dot"></span>3.2 · Violin</a>
      <a href="#part3_3" class="nav-link nav-sub"><span class="dot"></span>3.3 · Heatmap</a>
      <a href="#part3_4" class="nav-link nav-sub"><span class="dot"></span>3.4 · DMR</a>

      <a href="#part4" class="nav-link nav-top"><span class="dot"></span>4 · Fragmentomics</a>
      <a href="#part4_2" class="nav-link nav-sub"><span class="dot"></span>4.1 · DELFI</a>
      <a href="#part4_3" class="nav-link nav-sub"><span class="dot"></span>4.2 · End Motif</a>
      <a href="#part4_4" class="nav-link nav-sub"><span class="dot"></span>4.3 · Cleavage</a>
      <a href="#part4_5" class="nav-link nav-sub"><span class="dot"></span>4.4 · WPS</a>

      <a href="#part5" class="nav-link nav-top"><span class="dot"></span>5 · MESA</a>
      <a href="#part5_1" class="nav-link nav-sub"><span class="dot"></span>5.1 · Performance</a>
      <a href="#part5_2" class="nav-link nav-sub"><span class="dot"></span>5.2 · ROC</a>
      <a href="#part5_3" class="nav-link nav-sub"><span class="dot"></span>5.3 · Heatmap</a>
      <a href="#part5_4" class="nav-link nav-sub"><span class="dot"></span>5.4 · Spearman</a>
    </div>"""


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_report(args):
    rd       = getattr(args, "results_dir", "results")
    out_dir  = getattr(args, "output_dir",  "results/report")
    project  = getattr(args, "project_name", "cftk_project")
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    # load group info from config
    cfg_path = getattr(args, "config", None)
    groups   = {}  # {group_name: [sample_name, ...]}
    if cfg_path and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
        for grp, members in cfg.get("samples", {}).items():
            groups[grp] = [s["name"] for s in members]
    else:
        # fallback from args.groups list
        for g in getattr(args, "groups", []):
            groups[g] = []

    os.makedirs(out_dir, exist_ok=True)

    sidebar  = _build_sidebar(rd, groups)
    sections = (
        _sec_sample_overview(groups)
        + _sec_power(rd)
        + _sec_process(rd, groups=groups)
        + _sec_qc(rd, groups=groups)
        + _sec_differential(rd)
        + _sec_fragmentomics(rd, groups)
        + _sec_mesa(rd)
    )

    tmpl_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    with open(tmpl_path) as f:
        template = f.read()

    replacements = {
        "<!-- {SIDEBAR} -->":     sidebar,
        "<!-- {SECTIONS} -->":    sections,
        "<!-- {PROJECT_NAME} -->": project,
        "<!-- {GROUPS} -->":      " · ".join(groups.keys()),
        "<!-- {DATE} -->":        date_str,
        "<!-- {REPORT_DATE} -->": date_str,
    }
    html = template
    for k, v in replacements.items():
        html = html.replace(k, v)

    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    print(f"[report] saved → {out_path}")
    return out_path
