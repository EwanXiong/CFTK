"""
report_generator.py — collect all PNG results and embed into HTML report.
All PNGs are base64-encoded so the output is a single self-contained file.
"""

import base64
import datetime
import glob
import json
import os
import re


def _b64(png_path):
    """Return base64 data-URI string for a PNG file."""
    with open(png_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def _img_tag(png_path, alt="", style="width:100%;"):
    if not png_path or not os.path.exists(png_path):
        return f'<div class="fig-placeholder"><span class="hint">{alt}</span></div>'
    return f'<img src="{_b64(png_path)}" alt="{alt}" style="{style}">'


def _find(results_dir, *parts, pattern="*.png"):
    """Glob for the first matching PNG under results_dir/parts."""
    path = os.path.join(results_dir, *parts, pattern)
    files = sorted(glob.glob(path))
    return files[0] if files else None


def _find_all(results_dir, *parts, pattern="*.png"):
    path = os.path.join(results_dir, *parts, pattern)
    return sorted(glob.glob(path))


def _read_tsv_html(tsv_path, max_rows=20):
    """Convert a TSV file to an HTML table string."""
    if not tsv_path or not os.path.exists(tsv_path):
        return "<p><em>No data available.</em></p>"
    import pandas as pd
    try:
        df = pd.read_csv(tsv_path, sep="\t", index_col=0).head(max_rows)
        return df.to_html(classes="result-table", border=0, float_format="{:.4f}".format)
    except Exception:
        return "<p><em>Could not load table.</em></p>"


# ── Section builders ──────────────────────────────────────────────────────────

def _sec_power(rd):
    img = _img_tag(_find(rd, "0_power", pattern="power_cumulative.png"),
                   "Power cumulative curve")
    return f"""
    <section class="section" id="part0">
      <div class="section-header">
        <span class="section-num">PART 00</span>
        <h2 class="section-title">Power Analysis</h2>
        <span class="section-tag">power</span>
      </div>
      <div class="fig-grid cols-1">
        <div class="fig-card">
          <div class="fig-img tall">{img}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 0A · Power vs CpG proportion</div>
            <div class="fig-caption">Complementary ECDF of per-CpG detection power
              at different read depths. Red dashed line: 80% power threshold.</div>
          </div>
        </div>
      </div>
    </section>"""


def _sec_process(rd):
    return f"""
    <section class="section" id="part1">
      <div class="section-header">
        <span class="section-num">PART 01</span>
        <h2 class="section-title">Data Processing</h2>
        <span class="section-tag">process</span>
      </div>
      <p style="font-size:13px;color:var(--ink2)">
        Raw sequencing data were processed through a 6-step pipeline:
        adapter trimming (Trim Galore), alignment (bwameth),
        duplicate removal (sambamba), methylation calling (MethylDackel),
        nucleosome occupancy (DANPOS3), and WPS calculation.
      </p>
    </section>"""


def _sec_qc(rd):
    meth    = _img_tag(_find(rd, "2_qc", "1_methylation_distribution", pattern="methylation_distribution.png"),
                       "Methylation distribution")
    frag    = _img_tag(_find(rd, "2_qc", "2_fragment_length", pattern="fragment_length.png"),
                       "Fragment length")
    dinuc   = _img_tag(_find(rd, "2_qc", "3_dinucleotide_freq", pattern="dinucleotide_freq.png"),
                       "Dinucleotide frequency")
    return f"""
    <section class="section" id="part2">
      <div class="section-header">
        <span class="section-num">PART 02</span>
        <h2 class="section-title">QC Analysis</h2>
        <span class="section-tag">qc</span>
      </div>
      <div class="fig-grid cols-3">
        <div class="fig-card">
          <div class="fig-img">{meth}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 2A · Methylation distribution</div>
            <div class="fig-caption">CpG methylation β-value density across all samples.</div>
          </div>
        </div>
        <div class="fig-card">
          <div class="fig-img">{frag}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 2B · Fragment length</div>
            <div class="fig-caption">Mean fragment length distribution. Peak marks
              mono-nucleosomal cfDNA.</div>
          </div>
        </div>
        <div class="fig-card">
          <div class="fig-img">{dinuc}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 2C · Dinucleotide frequency</div>
            <div class="fig-caption">AT/GC dinucleotide fractions relative to
              fragment center. 10-bp periodicity reflects nucleosome positioning.</div>
          </div>
        </div>
      </div>
    </section>"""


def _sec_differential(rd):
    modalities = []
    diff_base  = os.path.join(rd, "3_differential")
    if os.path.exists(diff_base):
        modalities = [
            d for d in os.listdir(diff_base)
            if os.path.isdir(os.path.join(diff_base, d)) and d != "dmr"
        ]

    mod_blocks = ""
    for i, mod in enumerate(sorted(modalities)):
        pca  = _img_tag(_find(rd, "3_differential", mod, pattern="pca.png"), f"{mod} PCA")
        viol = _img_tag(_find(rd, "3_differential", mod, pattern="violin.png"), f"{mod} violin")
        hmap = _img_tag(_find(rd, "3_differential", mod, pattern="heatmap.png"), f"{mod} heatmap")
        letter = chr(65 + i * 3)
        mod_blocks += f"""
        <h3 style="font-family:var(--serif);font-size:16px;margin:24px 0 12px;font-weight:400">
          3.{i+1} {mod.upper()}
        </h3>
        <div class="fig-grid cols-3">
          <div class="fig-card">
            <div class="fig-img">{pca}</div>
            <div class="fig-body">
              <div class="fig-label">Figure 3{letter} · PCA</div>
              <div class="fig-caption">PCA of {mod} matrix. Groups separated along PC1.</div>
            </div>
          </div>
          <div class="fig-card">
            <div class="fig-img">{viol}</div>
            <div class="fig-body">
              <div class="fig-label">Figure 3{chr(ord(letter)+1)} · Violin</div>
              <div class="fig-caption">Per-sample mean {mod} distribution.
                Mann-Whitney U p-value shown.</div>
            </div>
          </div>
          <div class="fig-card">
            <div class="fig-img">{hmap}</div>
            <div class="fig-body">
              <div class="fig-label">Figure 3{chr(ord(letter)+2)} · Heatmap</div>
              <div class="fig-caption">Hierarchical clustering of top-variance {mod} features.
                Z-score normalized.</div>
            </div>
          </div>
        </div>"""

    # DMR volcano
    dmr_vol = _img_tag(_find(rd, "3_differential", "dmr", pattern="dmr_volcano.png"),
                       "DMR volcano")
    mod_blocks += f"""
        <h3 style="font-family:var(--serif);font-size:16px;margin:32px 0 12px;font-weight:400">
          3.{len(modalities)+1} DMR Analysis
        </h3>
        <div class="fig-grid cols-1">
          <div class="fig-card">
            <div class="fig-img">{dmr_vol}</div>
            <div class="fig-body">
              <div class="fig-label">Figure DMR · Volcano plot</div>
              <div class="fig-caption">DMR volcano: mean methylation difference vs
                −log₁₀(q-value). Gene labels: top differentially methylated promoters.</div>
            </div>
          </div>
        </div>"""

    return f"""
    <section class="section" id="part3">
      <div class="section-header">
        <span class="section-num">PART 03</span>
        <h2 class="section-title">Differential Analysis</h2>
        <span class="section-tag">diff</span>
      </div>
      {mod_blocks}
    </section>"""


def _sec_fragmentomics(rd):
    # DELFI
    delfi_imgs = _find_all(rd, "4_fragmentomics", "delfi", pattern="*_genome.png")
    delfi_cards = ""
    for i, p in enumerate(delfi_imgs):
        name = os.path.basename(p).replace("_delfi_genome.png", "")
        delfi_cards += f"""
        <div class="fig-card">
          <div class="fig-img">{_img_tag(p, name)}</div>
          <div class="fig-body">
            <div class="fig-label">DELFI · {name}</div>
            <div class="fig-caption">Genome-wide DELFI score (corrected ratio).</div>
          </div>
        </div>"""

    # End-motif
    em_imgs = _find_all(rd, "4_fragmentomics", "end_motif", pattern="*_top20.png")
    em_cards = ""
    for p in em_imgs:
        name = os.path.basename(p).replace("_top20.png", "")
        em_cards += f"""
        <div class="fig-card">
          <div class="fig-img">{_img_tag(p, name)}</div>
          <div class="fig-body">
            <div class="fig-label">End-motif · {name}</div>
            <div class="fig-caption">Top 20 most frequent k-mer end motifs.</div>
          </div>
        </div>"""

    # Cleavage
    cl_img = _img_tag(_find(rd, "4_fragmentomics", "cleavage",
                             pattern="cleavage_profile.png"),
                      "Cleavage profile")

    # WPS
    wps_imgs = _find_all(rd, "4_fragmentomics", "wps", pattern="*_profile.png")
    wps_cards = ""
    for p in wps_imgs:
        name = os.path.basename(p).replace("_profile.png", "")
        wps_cards += f"""
        <div class="fig-card">
          <div class="fig-img">{_img_tag(p, name)}</div>
          <div class="fig-body">
            <div class="fig-label">WPS · {name}</div>
            <div class="fig-caption">Genome-wide mean WPS profile.</div>
          </div>
        </div>"""

    return f"""
    <section class="section" id="part4">
      <div class="section-header">
        <span class="section-num">PART 04</span>
        <h2 class="section-title">Fragmentomics</h2>
        <span class="section-tag">frag</span>
      </div>

      <h3 style="font-family:var(--serif);font-size:16px;margin:16px 0 12px;font-weight:400">
        4.1 DELFI
      </h3>
      <div class="fig-grid cols-2">{delfi_cards or '<p>No DELFI results.</p>'}</div>

      <h3 style="font-family:var(--serif);font-size:16px;margin:24px 0 12px;font-weight:400">
        4.2 End Motif
      </h3>
      <div class="fig-grid cols-2">{em_cards or '<p>No end-motif results.</p>'}</div>

      <h3 style="font-family:var(--serif);font-size:16px;margin:24px 0 12px;font-weight:400">
        4.3 Cleavage Profile
      </h3>
      <div class="fig-grid cols-1">
        <div class="fig-card">
          <div class="fig-img tall">{cl_img}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 4C · Cleavage profile</div>
            <div class="fig-caption">Aggregate cleavage proportion around CTCF motifs.</div>
          </div>
        </div>
      </div>

      <h3 style="font-family:var(--serif);font-size:16px;margin:24px 0 12px;font-weight:400">
        4.4 WPS
      </h3>
      <div class="fig-grid cols-2">{wps_cards or '<p>No WPS results.</p>'}</div>
    </section>"""


def _sec_mesa(rd):
    roc  = _img_tag(_find(rd, "5_mesa", pattern="mesa_roc.png"), "ROC")
    hmap = _img_tag(_find(rd, "5_mesa", pattern="mesa_heatmap.png"), "Prediction heatmap")
    sp   = _img_tag(_find(rd, "5_mesa", pattern="mesa_spearman.png"), "Spearman correlation")
    perf = _read_tsv_html(os.path.join(rd, "5_mesa", "modality_performance.tsv"))

    return f"""
    <section class="section" id="part5">
      <div class="section-header">
        <span class="section-num">PART 05</span>
        <h2 class="section-title">MESA Multimodal Modeling</h2>
        <span class="section-tag">mesa</span>
      </div>

      <h3 style="font-family:var(--serif);font-size:16px;margin:16px 0 12px;font-weight:400">
        5.1 Modality Performance
      </h3>
      {perf}

      <h3 style="font-family:var(--serif);font-size:16px;margin:24px 0 12px;font-weight:400">
        5.2 LOOCV Results
      </h3>
      <div class="fig-grid cols-2">
        <div class="fig-card">
          <div class="fig-img">{roc}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 5A · ROC curves</div>
            <div class="fig-caption">ROC curves for each modality and the MESA
              multimodal model under LOOCV. AUC values shown in legend.</div>
          </div>
        </div>
        <div class="fig-card">
          <div class="fig-img">{hmap}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 5B · Per-sample predictions</div>
            <div class="fig-caption">LOOCV prediction probability heatmap per modality.
              Samples sorted by true label.</div>
          </div>
        </div>
      </div>
      <div class="fig-grid cols-1" style="margin-top:20px">
        <div class="fig-card">
          <div class="fig-img">{sp}</div>
          <div class="fig-body">
            <div class="fig-label">Figure 5C · Modality Spearman correlation</div>
            <div class="fig-caption">Pairwise Spearman correlation of per-sample
              prediction scores. Low correlation validates modality complementarity.</div>
          </div>
        </div>
      </div>
    </section>"""


# ── Main entry ────────────────────────────────────────────────────────────────

def generate_report(args):
    rd         = getattr(args, "results_dir", "results")
    out_dir    = getattr(args, "output_dir", "results/report")
    project    = getattr(args, "project_name", "cftk_project")
    groups     = getattr(args, "groups", ["Control", "Case"])
    date_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    os.makedirs(out_dir, exist_ok=True)

    # load template
    tmpl_path = os.path.join(os.path.dirname(__file__), "report_template.html")
    with open(tmpl_path) as f:
        template = f.read()

    # build section HTML
    sections = (
        _sec_power(rd)
        + _sec_process(rd)
        + _sec_qc(rd)
        + _sec_differential(rd)
        + _sec_fragmentomics(rd)
        + _sec_mesa(rd)
    )

    # fill placeholders
    html = template
    replacements = {
        "<!-- {SECTIONS} -->":      sections,
        "<!-- {PROJECT_NAME} -->":  project,
        "<!-- {GROUPS} -->":        " · ".join(groups),
        "<!-- {DATE} -->":          date_str,
        "<!-- {REPORT_DATE} -->":   date_str,
    }
    for k, v in replacements.items():
        html = html.replace(k, v)

    out_path = os.path.join(out_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)

    print(f"[report] report saved → {out_path}")
    return out_path
