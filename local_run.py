"""Cross-platform local runner for the MatPlotAgent workflow."""
from __future__ import annotations

import argparse, base64, json, os, re, shutil, subprocess, sys, time
from pathlib import Path
from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark_data"
BUNDLED_UI_FONT = (
    ROOT.parent.parent / "RenderingModule" / "Assets" / "Resources" /
    "Fonts" / "Poppins-Bold.ttf"
)

def api_client():
    provider = os.getenv("MATPLOT_PROVIDER", "").lower()
    use_qwen = provider == "qwen" or (not provider and bool(os.getenv("DASHSCOPE_API_KEY")))
    key = os.getenv("DASHSCOPE_API_KEY") if use_qwen else os.getenv("OPENAI_API_KEY")
    if not key:
        name = "DASHSCOPE_API_KEY" if use_qwen else "OPENAI_API_KEY"
        raise SystemExit(f"Missing {name}. Set it before running.")
    default_url = ("https://dashscope.aliyuncs.com/compatible-mode/v1" if use_qwen
                   else "https://api.openai.com/v1")
    return OpenAI(api_key=key, base_url=os.getenv("OPENAI_BASE_URL", default_url))

def selected_model():
    provider = os.getenv("MATPLOT_PROVIDER", "").lower()
    use_qwen = provider == "qwen" or (not provider and bool(os.getenv("DASHSCOPE_API_KEY")))
    # Grid materialization only asks the model to write Python from a text
    # contract.  It does not send an image when --no-visual-refine is used, so
    # a large vision model adds latency without improving the input signal.
    # Keep this separate from MATPLOT_SUMMARY_MODEL, which still receives the
    # finished grid image in S4DAnalysisService/digest.py.
    return os.getenv(
        "MATPLOT_CODE_MODEL",
        os.getenv("MATPLOT_MODEL", "qwen-flash" if use_qwen else "gpt-4.1-mini"),
    )

def complete(messages, max_tokens=5000):
    try:
        result = api_client().chat.completions.create(
            model=selected_model(), messages=messages,
            temperature=0, max_completion_tokens=max_tokens)
    except APIConnectionError as exc:
        raise SystemExit("Cannot connect to OPENAI_BASE_URL. Check the URL, proxy, and network.") from exc
    except APIStatusError as exc:
        detail = getattr(exc, "body", None) or str(exc)
        raise SystemExit(f"OpenAI API request failed ({exc.status_code}): {detail}") from exc
    return result.choices[0].message.content or ""

def code_from(answer):
    blocks = re.findall(r"```(?:python)?\s*([\s\S]*?)```", answer, re.I)
    return "\n".join(blocks).strip() if blocks else answer.strip()

def listing(workspace):
    entries = []
    for path in sorted(workspace.iterdir()):
        if not path.is_file() or path.suffix.lower() in {".png", ".py", ".log"}:
            continue
        entry = f"- {path.name}"
        if path.suffix.lower() in {".csv", ".tsv", ".json", ".txt"}:
            try:
                preview = path.read_text(encoding="utf-8", errors="replace")[:4000]
                entry += f"\n  Exact content preview:\n{preview}"
            except OSError:
                pass
        entries.append(entry)
    return "\n".join(entries)

def generate(query, workspace, output, feedback="", previous_code=""):
    prompt = f"""Write a complete Python script for this scientific visualization request:
{query}

Working-directory files:
{listing(workspace) or '- none'}

Read files by relative path, use a non-interactive backend, do not invent data
when input data exists, and save the final figure as {output}. Open every JSON
or text file explicitly with encoding='utf-8' (the runner may execute on Windows,
whose default encoding is not UTF-8). Return only one
fenced Python code block. Keep figsize at or below 20x20 inches and dpi at or
below 200. Render at 200 dpi. Use a consistent readable type scale: subplot
titles at least 14 pt, axis labels at least 13 pt, tick labels at least 11 pt,
and legends/colorbars at least 11 pt. When Poppins-Bold.ttf is present, register
it with `matplotlib.font_manager.fontManager.addfont` and use its resolved family
name for all chart text. Place annotations inside axes coordinates; do not let artists far
outside an axis combine with bbox_inches='tight' to create an enormous image.
When creating a dynamic subplot grid, use plt.subplots(..., squeeze=False) and
index axes[row, column] directly. squeeze=False already returns a two-dimensional
axes array even for a 1x1 grid; never wrap that result in another np.array."""
    if feedback:
        prompt += f"""

The previous script is included below. Modify it rather than redesigning the
plot from scratch. Treat the review as advice: apply only issues supported by
the request, data, image, and code. Preserve correct column names, statistics,
and chart types. Audit the data axis of every subplot: for a vertical histogram
data is on x (use axvline for data values); for a horizontal histogram data is
on y (use axhline); for boxplots the mapping depends on orientation.

Previous script:
```python
{previous_code}
```

Review and static-analysis findings:
{feedback}
"""
    return code_from(complete([
        {"role": "system", "content": "You are an expert scientific visualization programmer."},
        {"role": "user", "content": prompt}]))

def execute(code, workspace, stem, timeout, expected_output=""):
    script, log_path = workspace / f"{stem}.py", workspace / f"{stem}.log"
    script.write_text(code, encoding="utf-8")
    env = os.environ.copy(); env.setdefault("MPLBACKEND", "Agg")
    started_at = time.time()
    try:
        result = subprocess.run([sys.executable, script.name], cwd=workspace, env=env,
                                capture_output=True, text=True, timeout=timeout)
        log, ok = (result.stdout or "") + (result.stderr or ""), result.returncode == 0
    except subprocess.TimeoutExpired as exc:
        log, ok = f"Execution timed out after {timeout}s.\n{exc}", False
    log_path.write_text(log, encoding="utf-8")
    # Some otherwise valid agent scripts paraphrase the requested output name.
    # Recover a PNG created by this execution and normalize it to the contract
    # filename instead of throwing away a completed plot and spending another
    # model call on a repair that only changes the filename.
    if ok and expected_output:
        expected = workspace / expected_output
        if not valid_png(expected):
            candidates = sorted(
                (
                    path for path in workspace.glob("*.png")
                    if path != expected
                    and path.stat().st_mtime >= started_at - 1.0
                    and valid_png(path)
                ),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if candidates:
                shutil.copy2(candidates[0], expected)
    return ok, log

def valid_png(path):
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (OSError, SyntaxError, Image.DecompressionBombError):
        return False

def render_contract_fallback(workspace, output):
    """Render a validated S4D contract after both agent code attempts fail.

    MatPlotAgent remains the primary generator. This bounded renderer prevents a
    transient code-generation mistake from turning an otherwise valid 3x3 job
    into nine unavailable panels.
    """
    contract_path = workspace / "grid_contract.json"
    data_path = workspace / "grid_data.csv"
    if not contract_path.is_file() or not data_path.is_file():
        return False

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        import numpy as np
        import pandas as pd

        bundled_font = workspace / "Poppins-Bold.ttf"
        font_family = "DejaVu Sans"
        if bundled_font.is_file():
            font_manager.fontManager.addfont(str(bundled_font))
            font_family = font_manager.FontProperties(
                fname=str(bundled_font)).get_name()
        plt.rcParams.update({
            "font.family": font_family,
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
        })

        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        frame = pd.read_csv(data_path, encoding="utf-8")
        frame = frame.dropna(subset=["value"])
        columns = contract["grid"]["columns"]
        rows = contract["grid"]["rows"]
        encoding = contract["encoding"]
        layout = contract["layout"]
        chart_type = contract.get("chartType", "horizontal_heatmap")
        minimum = float(encoding["minimum"])
        maximum = float(encoding["maximum"])
        unit = str(encoding["unit"])
        cmap = plt.get_cmap(encoding.get("colorMap", "viridis"))
        figure, axes = plt.subplots(
            len(rows), len(columns), squeeze=False,
            figsize=(float(layout["figureWidthInches"]), float(layout["figureHeightInches"])),
        )
        last_mappable = None
        cell_order = []
        for row_index, row in enumerate(rows):
            for column_index, column in enumerate(columns):
                cell_id = f"{column['id']}__{row['id']}"
                cell_order.append(cell_id)
                axis = axes[row_index, column_index]
                cell = frame.loc[frame["cell_id"] == cell_id]
                values = cell["value"].to_numpy(dtype=float)
                title = f"{row['label']} x {column['label']}"
                axis.set_title(title, fontsize=14, fontweight="semibold")
                if values.size == 0:
                    axis.text(.5, .5, "No valid values", ha="center", va="center",
                              transform=axis.transAxes)
                    axis.set_axis_off()
                    continue
                if chart_type in {"bar_chart", "histogram"}:
                    edges = np.linspace(minimum, maximum, 13)
                    counts, _ = np.histogram(values, bins=edges)
                    centers = (edges[:-1] + edges[1:]) / 2
                    colors = cmap(np.linspace(.2, .85, len(counts)))
                    axis.bar(centers, counts, width=np.diff(edges) * .88,
                             color=colors, edgecolor="none")
                    axis.set_xlim(minimum, maximum)
                    axis.set_xlabel(unit); axis.set_ylabel("Count")
                elif chart_type == "scatter_plot":
                    last_mappable = axis.scatter(
                        cell["x_index"], cell["y_index"], c=values, cmap=cmap,
                        vmin=minimum, vmax=maximum, s=1.2, linewidths=0,
                        rasterized=True,
                    )
                    axis.set_aspect("equal", adjustable="box")
                    axis.set_xlabel("x_index"); axis.set_ylabel("y_index")
                elif chart_type == "line_chart":
                    profile = cell.groupby("x_index", sort=True)["value"].mean()
                    axis.plot(profile.index.to_numpy(), profile.to_numpy(),
                              color=cmap(.62), linewidth=1.8)
                    axis.set_ylim(minimum, maximum)
                    axis.set_xlabel("x_index"); axis.set_ylabel(f"Mean {unit}")
                elif chart_type == "pie_chart":
                    counts, edges = np.histogram(
                        values, bins=np.linspace(minimum, maximum, 7)
                    )
                    keep = counts > 0
                    labels = [
                        f"[{edges[i]:.2f}, {edges[i + 1]:.2f})"
                        for i in range(6) if keep[i]
                    ]
                    colors = cmap(np.linspace(.05, .95, 6))[keep]
                    wedges, _, _ = axis.pie(
                        counts[keep], colors=colors, startangle=90,
                        autopct=lambda percent: f"{percent:.1f}%" if percent >= 3 else "",
                        pctdistance=.72, textprops={"fontsize": 11},
                    )
                    axis.legend(wedges, labels, loc="center left",
                                bbox_to_anchor=(1.0, .5), fontsize=11, frameon=False)
                    axis.set_aspect("equal")
                elif chart_type == "box_plot":
                    axis.boxplot(values, orientation="vertical", showfliers=True)
                    axis.set_ylim(minimum, maximum); axis.set_xticks([1], ["Distribution"])
                    axis.set_ylabel(unit)
                elif chart_type == "violin_plot":
                    axis.violinplot(values, showmeans=True, showmedians=True,
                                    showextrema=True)
                    axis.set_ylim(minimum, maximum); axis.set_xticks([1], ["Distribution"])
                    axis.set_ylabel(unit)
                else:
                    spatial = contract["spatialGrid"]
                    width = int(spatial["width"])
                    height = int(spatial["height"])
                    pivot = cell.pivot_table(index="y_index", columns="x_index",
                                             values="value", aggfunc="mean")
                    image = pivot.reindex(index=range(height), columns=range(width)).to_numpy(float)
                    georef = spatial.get("georeference")
                    if georef:
                        x_axis = georef["x"]
                        y_axis = georef["y"]
                        extent = [
                            float(x_axis["minimum"]) - float(x_axis["step"]) / 2,
                            float(x_axis["maximum"]) + float(x_axis["step"]) / 2,
                            float(y_axis["minimum"]) - float(y_axis["step"]) / 2,
                            float(y_axis["maximum"]) + float(y_axis["step"]) / 2,
                        ]
                        last_mappable = axis.imshow(
                            image, origin="lower", extent=extent, cmap=cmap,
                            vmin=minimum, vmax=maximum,
                            interpolation="nearest", aspect="auto",
                        )
                        axis.set_xlabel(
                            "Longitude (degrees east) | "
                            f"CRS: {georef['coordinateReference']}"
                        )
                        axis.set_ylabel("Latitude (degrees north)")
                        axis.set_xticks(np.linspace(
                            float(x_axis["minimum"]), float(x_axis["maximum"]),
                            min(width, 6),
                        ))
                        axis.set_yticks(np.linspace(
                            float(y_axis["minimum"]), float(y_axis["maximum"]),
                            min(height, 6),
                        ))
                    else:
                        last_mappable = axis.imshow(
                            image, origin="lower", cmap=cmap, vmin=minimum,
                            vmax=maximum, interpolation="nearest", aspect="auto",
                        )
                        axis.set_xlabel("x_index"); axis.set_ylabel("y_index")
        if layout.get("showColorbar") and last_mappable is not None:
            colorbar = figure.colorbar(
                last_mappable, ax=axes.ravel().tolist(), label=unit,
                fraction=.025, pad=.02)
            colorbar.ax.tick_params(labelsize=11)
            colorbar.set_label(unit, size=13)
        if chart_type == "pie_chart":
            figure.subplots_adjust(right=.72, wspace=.45, hspace=.5)
        else:
            figure.tight_layout()
        figure.savefig(workspace / output, dpi=200)
        plt.close(figure)
        metadata = {
            "cellOrder": cell_order,
            "colorMap": encoding.get("colorMap", "viridis"),
            "minimum": minimum, "maximum": maximum, "unit": unit,
            "figureWidth": layout["figureWidthInches"],
            "figureHeight": layout["figureHeightInches"],
            "spatialWidth": contract["spatialGrid"]["width"],
            "spatialHeight": contract["spatialGrid"]["height"],
            "outputFilename": output,
            "fallbackAfterAgentFailure": True,
        }
        (workspace / "chart_result.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        (workspace / "contract_fallback.log").write_text(
            f"Rendered {chart_type} after MatPlotAgent code execution failed.\n",
            encoding="utf-8",
        )
        return valid_png(workspace / output)
    except Exception as exc:
        (workspace / "contract_fallback.log").write_text(
            f"Fallback failed: {type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        return False

def finish_contract_fallback(workspace, output):
    """Commit a fallback image using the same output contract as the agent."""
    initial = workspace / "initial.png"
    if not render_contract_fallback(workspace, initial.name):
        return False
    final = workspace / output
    if final != initial:
        shutil.copy2(initial, final)
    print(f"Done with validated contract fallback: {final}")
    return True

def static_plot_warnings(code):
    warnings = []
    hist_axes = set(re.findall(r"\b(\w+)\.hist\s*\(", code))
    for axis in hist_axes:
        horizontal = bool(re.search(
            rf"\b{re.escape(axis)}\.hist\s*\([^\n]*orientation\s*=\s*['\"]horizontal['\"]", code))
        wrong = "axvline" if horizontal else "axhline"
        right = "axhline" if horizontal else "axvline"
        if re.search(rf"\b{re.escape(axis)}\.{wrong}\s*\(", code):
            warnings.append(
                f"{axis}.hist has {'horizontal' if horizontal else 'vertical'} orientation, but {axis}.{wrong} "
                f"is also used. Verify whether data-valued reference lines should use {axis}.{right}."
            )
    return warnings

def preflight_code_warnings(code, output):
    """Catch generated scripts that are predictably unsafe before execution."""
    warnings = []
    for match in re.finditer(
        r"open\s*\(\s*(['\"])([^'\"]+\.(?:json|txt))\1(?P<args>[^)]*)\)",
        code,
        re.I,
    ):
        if "encoding" not in match.group("args"):
            warnings.append(
                f"Open {match.group(2)!r} explicitly with encoding='utf-8'. "
                "The script runs on Windows and must not use the GBK default."
            )
    if re.search(
        r"counts\s*,\s*_\s*=\s*pd\.cut\s*\([^\n]+\)\.value_counts\s*\(",
        code,
    ):
        warnings.append(
            "pd.cut(...).value_counts() returns one Series, not a (counts, edges) "
            "tuple. For the required bar distribution use "
            "counts, _ = np.histogram(values, bins=edges)."
        )
    if re.search(
        r"\.bar\s*\([^\n]*\bcolor\s*=\s*(?:colormap|cmap_name|cmap)\b",
        code,
    ):
        warnings.append(
            "A colormap name such as 'viridis' is not a Matplotlib bar color. "
            "Evaluate it first, e.g. bar_colors = plt.get_cmap(colormap)("
            "np.linspace(0.2, 0.85, len(counts))), then pass color=bar_colors."
        )
    redundant_axes_wrapper = re.search(
        r"axes\s*=\s*np\.array\s*\(\s*\[\s*\[\s*axes\s*\]\s*\]\s*\)",
        code,
    )
    if re.search(r"plt\.subplots[\s\S]{0,300}squeeze\s*=\s*False", code) and \
            redundant_axes_wrapper:
        warnings.append(
            "plt.subplots(..., squeeze=False) already returns a 2D axes array for a "
            "1x1 grid. Remove the conditional np.array wrapper around axes; it creates "
            "extra dimensions and makes axes[row, column] a NumPy array instead of an Axes."
        )
    if re.search(r"(?:fig\.)?suptitle\s*\([^\n]*(?:rawText|rawIntent)", code):
        warnings.append(
            "Do not place the full raw user intent in fig.suptitle(). Long prompts "
            "combined with bbox_inches='tight' create panoramic images and make the "
            "actual chart tiny. Use a short task label inside the canonical figure."
        )
    if re.search(r"(?:plt|fig)\.colorbar\s*\(", code) and "showColorbar" not in code:
        warnings.append(
            "Read contract['layout']['showColorbar'] and add a colorbar only when it "
            "is true. S4D uses one shared scale and deliberately assigns the visible "
            "colorbar to a bounded subset of cells."
        )
    if re.search(r"\.iterrows\s*\(", code):
        warnings.append(
            "Do not use DataFrame.iterrows(). The uploaded facet-grid CSV can contain "
            "millions of rows; reshape it with pandas pivot/pivot_table or NumPy indexing."
        )
    if re.search(r"\.itertuples\s*\(", code):
        warnings.append(
            "Do not loop over DataFrame.itertuples(). Reshape the complete columns with "
            "pandas pivot/pivot_table or NumPy indexing."
        )
    # Do not reject every ``some_list.index(...)`` call.  The former broad
    # check also rejected harmless lookups across the three row/column labels
    # and could prevent an otherwise valid MatPlotAgent figure from running.
    # Pixel-wise coordinate loops are caught explicitly below; those are the
    # expensive/unsafe cases this preflight is intended to prevent.
    if re.search(r"\.index\.map\s*\(", code):
        warnings.append(
            "Do not call Index.map() while filling pixels. A pivoted DataFrame is already "
            "the complete image array; reindex once and use .to_numpy()."
        )
    coordinate_loops = re.findall(
        r"for\s+\w+\s*,\s*\w+\s+in\s+enumerate\s*\(\s*"
        r"(?:x_indices|y_indices|full_x_range|full_y_range)\s*\)",
        code,
    )
    if coordinate_loops:
        warnings.append(
            "Do not loop over x/y coordinate sequences to fill an image. Reindex the "
            "pivoted DataFrame once and call .to_numpy()."
        )
    if re.search(
        r"for\s+\w+\s+in\s+(?:x_indices|y_indices|full_x|full_y|"
        r"full_x_range|full_y_range)\s*:",
        code,
    ):
        warnings.append(
            "Do not loop directly over x/y coordinate sequences to fill an image. "
            "Reindex the pivoted DataFrame once and call .to_numpy()."
        )
    if re.search(
        r"\[\s*(?:y_indices|y_idx_arr)\s*,\s*"
        r"(?:x_indices|x_idx_arr)\s*\]\s*=",
        code,
    ):
        warnings.append(
            "Do not scatter-assign a 2D pivot with two differently sized 1D index "
            "arrays. Use pivoted.reindex(index=full_y, columns=full_x).to_numpy(dtype=float)."
        )
    if re.search(
        r"(?:full_x|full_y|full_x_indices|full_y_indices)"
        r"\s*\[\s*(?:x_ticks|y_ticks)\s*\]",
        code,
    ) and not re.search(
        r"np\.asarray\s*\(\s*(?:full_x|full_y|full_x_indices|full_y_indices)\s*\)"
        r"\s*\[\s*(?:x_ticks|y_ticks)\s*\]",
        code,
    ):
        warnings.append(
            "Do not index a Python range/list directly with a NumPy tick array. "
            "Use np.asarray(full_x)[x_ticks] and np.asarray(full_y)[y_ticks]."
        )
    if re.search(r"for\s+\w+\s+in\s+\w*(?:pivot|pivoted)\.(?:index|columns)\s*:", code):
        warnings.append(
            "Do not loop over pivot index/columns to copy values into another grid. "
            "Use pivot.reindex(index=..., columns=...).to_numpy() directly."
        )
    if re.search(r"len\s*\(\s*[^()\n]+\.shape\s*\[\s*\d+\s*\]\s*\)", code):
        warnings.append(
            "Do not call len() on data.shape[n]; shape[n] is already an integer."
        )

    save_targets = re.findall(
        r"(?:savefig\s*\(\s*|output_image\s*=\s*)['\"]([^'\"]+\.png)['\"]",
        code,
        re.I,
    )
    if save_targets and output not in save_targets:
        warnings.append(
            f"The script must create {output!r}, but its literal PNG output target(s) are "
            f"{save_targets!r}. Use the exact requested filename."
        )
    return warnings

def inspect_plot(query, image, code, data_context):
    data = base64.b64encode(image.read_bytes()).decode("ascii")
    return complete([{"role": "user", "content": [
        {"type": "text", "text": f"""Review this scientific plot using the request, exact code, and data context.
For every subplot state: (1) what variable is plotted, (2) whether data values
are on x or y, (3) what the other axis means, and (4) whether axhline/axvline
matches that mapping. Identify the exact offending code expression for every
correctness issue. Do not provide replacement code and do not speculate about
data-loading problems contradicted by the supplied context.

Request:
{query}

Data context:
{data_context}

Current code:
```python
{code}
```

Static warnings:
{chr(10).join(static_plot_warnings(code)) or 'none'}"""},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}
    ]}], 2500)

def choose_better_plot(query, initial, refined, initial_code, refined_code):
    def image_part(path):
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
    review = complete([{"role": "user", "content": [
        {"type": "text", "text": f"""Compare two generated scientific plots against the request. Image 1 is
INITIAL and Image 2 is REFINED. Check data-axis semantics, visible data marks,
scales, labels, layout, and every explicit requirement. Executability alone is
not quality. Penalize axes stretched by reference lines and plots whose actual
marks are compressed or invisible. Return exactly one first line:
CHOICE: INITIAL
or
CHOICE: REFINED
Then give concise reasons.

Request: {query}

INITIAL code:
```python
{initial_code}
```

REFINED code:
```python
{refined_code}
```"""},
        image_part(initial), image_part(refined)
    ]}], 2000)
    first_line = review.strip().splitlines()[0].strip().upper() if review.strip() else ""
    choice = "refined" if first_line == "CHOICE: REFINED" else "initial"
    return choice, review

def benchmark_query(example_id, workspace):
    items = json.loads((BENCHMARK / "benchmark_instructions.json").read_text(encoding="utf-8"))
    item = next((x for x in items if int(x["id"]) == example_id), None)
    if not item: raise SystemExit(f"Unknown benchmark example: {example_id}")
    data_dir = BENCHMARK / "data" / str(example_id)
    if data_dir.exists():
        for path in data_dir.iterdir():
            if path.is_file(): shutil.copy2(path, workspace / path.name)
    return item["simple_instruction"]

def main():
    parser = argparse.ArgumentParser(description="Run MatPlotAgent locally")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--prompt"); source.add_argument("--example", type=int)
    parser.add_argument("--data", action="append", default=[])
    parser.add_argument("--workspace", default="./workspace/local")
    parser.add_argument("--output", default="final.png")
    parser.add_argument("--no-visual-refine", action="store_true")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()
    workspace = Path(args.workspace).resolve(); workspace.mkdir(parents=True, exist_ok=True)
    if BUNDLED_UI_FONT.is_file():
        shutil.copy2(BUNDLED_UI_FONT, workspace / "Poppins-Bold.ttf")
    for index, raw in enumerate(args.data):
        path = Path(raw).resolve()
        if not path.is_file(): raise SystemExit(f"Data file does not exist: {path}")
        # S4D sends one CSV per cell. Give it a stable, semantic name so code
        # generation never has to copy a UUID from the prompt correctly.
        target = workspace / ("grid_data.csv" if index == 0 and path.suffix.lower() == ".csv" else path.name)
        if path != target:
            shutil.copy2(path, target)
    query = args.prompt or benchmark_query(args.example, workspace)
    (workspace / "request.txt").write_text(query, encoding="utf-8")
    try:
        code = generate(query, workspace, "initial.png")
    except SystemExit:
        if finish_contract_fallback(workspace, args.output):
            return
        raise
    for attempt in range(1, 3):
        preflight = preflight_code_warnings(code, "initial.png")
        if not preflight:
            break
        (workspace / f"initial_preflight_{attempt}.txt").write_text(
            "\n".join(f"- {item}" for item in preflight) + "\n",
            encoding="utf-8",
        )
        code = generate(
            query,
            workspace,
            "initial.png",
            "Mandatory pre-execution corrections:\n"
            + "\n".join(f"- {item}" for item in preflight),
            previous_code=code,
        )
    remaining_preflight = preflight_code_warnings(code, "initial.png")
    if remaining_preflight:
        if finish_contract_fallback(workspace, args.output):
            return
        raise SystemExit(
            "Generated code failed mandatory pre-execution checks after two revisions:\n"
            + "\n".join(f"- {item}" for item in remaining_preflight)
        )
    ok, log = execute(code, workspace, "generated_initial", args.timeout, "initial.png")
    initial = workspace / "initial.png"
    if not ok or not valid_png(initial):
        try:
            code = generate(query, workspace, "initial.png", "Execution failed, no valid image was created, or the image dimensions were unsafe:\n" + log[-6000:], previous_code=code)
            ok, log = execute(code, workspace, "generated_repair", args.timeout, "initial.png")
        except SystemExit:
            if finish_contract_fallback(workspace, args.output):
                return
            raise
    if not ok or not valid_png(initial):
        if not render_contract_fallback(workspace, "initial.png"):
            raise SystemExit(f"Generation failed; inspect logs in {workspace}")
    final = workspace / args.output
    if args.no_visual_refine: shutil.copy2(initial, final)
    else:
        data_context = listing(workspace)
        feedback = inspect_plot(query, initial, code, data_context)
        (workspace / "visual_feedback.txt").write_text(feedback, encoding="utf-8")
        candidate = workspace / "refined_candidate.png"
        refined = generate(query, workspace, candidate.name, feedback, previous_code=code)
        ok, log = execute(refined, workspace, "generated_refined", args.timeout, candidate.name)
        if not valid_png(candidate):
            shutil.copy2(initial, final)
            (workspace / "refinement_failed.log").write_text(log, encoding="utf-8")
        else:
            choice, review = choose_better_plot(query, initial, candidate, code, refined)
            (workspace / "selection_review.txt").write_text(review, encoding="utf-8")
            shutil.copy2(candidate if choice == "refined" else initial, final)
            (workspace / "selected_version.txt").write_text(choice + "\n", encoding="utf-8")
    print(f"Done: {final}")

if __name__ == "__main__": main()
