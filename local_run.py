"""Cross-platform local runner for the MatPlotAgent workflow."""
from __future__ import annotations

import argparse, base64, json, os, re, shutil, subprocess, sys
from pathlib import Path
from openai import APIConnectionError, APIStatusError, OpenAI
from PIL import Image

ROOT = Path(__file__).resolve().parent
BENCHMARK = ROOT / "benchmark_data"

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
    return os.getenv("MATPLOT_MODEL", "qwen-vl-max" if use_qwen else "gpt-4.1-mini")

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
when input data exists, and save the final figure as {output}. Return only one
fenced Python code block. Keep figsize at or below 20x20 inches and dpi at or
below 200. Place annotations inside axes coordinates; do not let artists far
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

def execute(code, workspace, stem, timeout):
    script, log_path = workspace / f"{stem}.py", workspace / f"{stem}.log"
    script.write_text(code, encoding="utf-8")
    env = os.environ.copy(); env.setdefault("MPLBACKEND", "Agg")
    try:
        result = subprocess.run([sys.executable, script.name], cwd=workspace, env=env,
                                capture_output=True, text=True, timeout=timeout)
        log, ok = (result.stdout or "") + (result.stderr or ""), result.returncode == 0
    except subprocess.TimeoutExpired as exc:
        log, ok = f"Execution timed out after {timeout}s.\n{exc}", False
    log_path.write_text(log, encoding="utf-8")
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
    if re.search(r"plt\.subplots\s*\([^)]*squeeze\s*=\s*False", code, re.S) and re.search(
        r"axes\s*=\s*np\.array\s*\(\s*\[\s*\[?\s*axes\s*\]?\s*\]\s*\)",
        code,
    ):
        warnings.append(
            "plt.subplots(..., squeeze=False) already returns a 2D axes array for a "
            "1x1 grid. Remove the conditional np.array wrapper around axes; it creates "
            "extra dimensions and makes axes[row, column] a NumPy array instead of an Axes."
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
    for raw in args.data:
        path = Path(raw).resolve()
        if not path.is_file(): raise SystemExit(f"Data file does not exist: {path}")
        shutil.copy2(path, workspace / path.name)
    query = args.prompt or benchmark_query(args.example, workspace)
    (workspace / "request.txt").write_text(query, encoding="utf-8")
    code = generate(query, workspace, "initial.png")
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
        raise SystemExit(
            "Generated code failed mandatory pre-execution checks after two revisions:\n"
            + "\n".join(f"- {item}" for item in remaining_preflight)
        )
    ok, log = execute(code, workspace, "generated_initial", args.timeout)
    initial = workspace / "initial.png"
    if not ok or not valid_png(initial):
        code = generate(query, workspace, "initial.png", "Execution failed, no valid image was created, or the image dimensions were unsafe:\n" + log[-6000:], previous_code=code)
        ok, log = execute(code, workspace, "generated_repair", args.timeout)
    if not ok or not valid_png(initial):
        raise SystemExit(f"Generation failed; inspect logs in {workspace}")
    final = workspace / args.output
    if args.no_visual_refine: shutil.copy2(initial, final)
    else:
        data_context = listing(workspace)
        feedback = inspect_plot(query, initial, code, data_context)
        (workspace / "visual_feedback.txt").write_text(feedback, encoding="utf-8")
        candidate = workspace / "refined_candidate.png"
        refined = generate(query, workspace, candidate.name, feedback, previous_code=code)
        ok, log = execute(refined, workspace, "generated_refined", args.timeout)
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
