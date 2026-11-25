"""Production Optimisation Shiny — phase 1 interface and data/domain workflow.
Reference: Production Optimisation Demo v1.0.20 (PySide6).
"""
from pathlib import Path
import zipfile
import tempfile
import base64
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive
from matplotlib.ticker import ScalarFormatter, MaxNLocator

HERE = Path(__file__).parent


def card(title, *children):
    return ui.div(ui.h3(title, class_="card-title"), *children, class_="section-card")


def disabled(label):
    return ui.tags.button(label, disabled=True, title="Not available in this web preview", class_="btn-locked")


def upload(id, label, accept):
    return ui.input_file(id, label, accept=accept, multiple=False, width="100%")


def table_view(df, limit=200):
    if df is None or df.empty:
        return pd.DataFrame({"Information": ["No data loaded yet"]})
    return df.head(limit)


app_ui = ui.page_fluid(
    ui.include_css(HERE / "www" / "styles.css"),
    ui.div(
        ui.div(ui.h1("Production Optimisation App"),
               ui.p("Preview interface for loading saved domains, emulators, TS-EMO/SHAP results, and PCP decision support."),
               class_="app-heading"),
        ui.span("Status: ready", class_="status-pill"),
        class_="top-header"),
    ui.div(
        ui.div(
            ui.div(ui.span("1", class_="wf-num wf-num-active"), ui.span("Domain"), class_="wf-step wf-active"),
            ui.span("→", class_="wf-arrow"),
            ui.div(ui.span("2", class_="wf-num"), ui.span("Emulator"), class_="wf-step"),
            ui.div(
                ui.div(ui.div(ui.span("4", class_="wf-num"), ui.span("SHAP"), class_="wf-step"), class_="wf-branch wf-branch-top"),
                ui.div(ui.div(ui.span("3", class_="wf-num"), ui.span("TS-EMO"), class_="wf-step"),
                       ui.span("→", class_="wf-arrow"),
                       ui.div(ui.span("5", class_="wf-num"), ui.span("PCPs"), class_="wf-step"), class_="wf-branch wf-branch-bottom"),
                class_="wf-fork"),
            class_="wf-diagram"),
        ui.div(ui.strong("Workflow Guide: "),
               "Complete the core setup (1–2), then either ", ui.strong("interpret"),
               " how inputs influence each objective using SHAP (4), or proceed with ",
               ui.strong("optimization"), " by building the Pareto front and extracting optimal solutions via TS-EMO & PCP (3–5).",
               class_="wf-guide"),
        ui.div(ui.strong("Tab symbols:"), class_="wf-symbol-title"),
        ui.p("⚠️  indicates the standard workflow when running the app from the beginning.", class_="wf-symbol"),
        ui.p("💾  indicates that you can skip earlier steps by loading a saved file to visualize results or continue the analysis.", class_="wf-symbol"),
        class_="workflow-graphic"),
    ui.navset_card_tab(
        ui.nav_panel("Data & Domain",
            ui.div(ui.h2("Step 1: Data & Domain"),
                   ui.p("Load experimental data, review inferred variables, and create or restore a domain table."), class_="step-heading"),
            card("Load experimental data", upload("experimental", "Experimental CSV", [".csv"]),
                 ui.output_text("data_status")),
            card("Dataset preview", ui.output_data_frame("data_preview")),
            card("Variable selection and settings",
                 ui.p("Review the detected columns. Domain creation uses the selected variable roles and observed bounds."),
                 ui.output_data_frame("variable_preview"),
                 ui.p("Click a variable row to edit its name, type, objective, bounds and display precision.", class_="hint"),
                 ui.output_text("variable_status")),
            card("Domain options",
                 ui.div(ui.input_action_button("create_domain", "Create Domain", class_="btn-action"),
                        class_="button-row"),
                 upload("domain_file", "Or load saved domain_table.csv", [".csv"]),
                 ui.output_text("domain_status")),
            card("Domain table", ui.output_data_frame("domain_preview"),
                 ui.download_button("download_domain", "Export domain table", class_="btn-other")),
            value="data"),
        ui.nav_panel("Emulator Training",
            ui.div(ui.h2("Step 2: Emulator Training"), ui.p("Saved-model preview. Training and prediction are disabled in the first Shiny release."), class_="step-heading"),
            card("Emulator", upload("emulator_zip", "Load emulator_model.zip or emulator_model.7z", [".zip", ".7z"]),
                 ui.output_text("emulator_status"),
                 disabled("Train Emulator"), disabled("Load Prediction Form"), disabled("Predict Outputs"),
                 ui.p("Model ZIP is inspected but not executed. Prediction remains disabled in this release.", class_="hint")),
            card("Saved emulator parity plot", ui.output_ui("emulator_plot_view")),
            value="emulator"),
        ui.nav_panel("TS-EMO Analysis",
            ui.div(ui.h2("Step 3: TS-EMO Analysis"), ui.p("Inspect saved optimisation data; running new TS-EMO studies is disabled."), class_="step-heading"),
            card("Load saved results", upload("hv_file", "HV_coverage_data.csv", [".csv"]),
                 upload("pareto_file", "pareto_data_with_metadata.csv or pareto_data_with_metadata_per_NB.csv", [".csv"]),
                 disabled("Run TS-EMO")),
            card("HV coverage", ui.layout_columns(
                ui.div(ui.h4("Hypervolume coverage plot"), ui.output_ui("hv_plot_view")),
                ui.div(ui.h4("HV coverage data"), ui.output_data_frame("hv_preview")),
                col_widths=(6,6))),
            card("Pareto solutions", ui.output_data_frame("pareto_preview")),
            ui.p("The per_NB file contains the same solutions with DNB yield, CO₂e and cost normalised to NB production.", class_="hint"),
            value="tsemo"),
        ui.nav_panel("SHAP Analysis",
            ui.div(ui.h2("Step 4: SHAP Analysis"), ui.p("Inspect saved SHAP explanations. Attribution generation is disabled."), class_="step-heading"),
            ui.layout_columns(
                card("SHAP computation (preview unavailable)",
                     ui.input_numeric("shap_samples", "SHAP samples", 10, min=1, max=10000),
                     ui.input_numeric("shap_batch", "Batch size", 5, min=1, max=100),
                     disabled("Run SHAP"), disabled("Stop SHAP")),
                card("Load previously generated SHAP data",
                     upload("shap_file", "SHAP master Excel workbook", [".xlsx"]),
                     ui.input_action_button("visualize_shap", "Visualize SHAP File", class_="btn-action")),
                col_widths=(5,7)),
            card("Customize plot order",
                 ui.p("Reorder objectives and input variables, then click Apply Order."),
                 ui.layout_columns(ui.div(ui.h4("Objectives — drag to reorder"), ui.output_ui("shap_objective_drag")),
                                   ui.div(ui.h4("Inputs — drag to reorder"), ui.output_ui("shap_variable_drag")), col_widths=(6,6)),
                 ui.input_action_button("apply_shap_order", "Apply Order", class_="btn-action"),
                 ui.output_text("shap_status")),
            card("SHAP panel", ui.output_ui("shap_panel_view")),
            value="shap"),
        ui.nav_panel("PCPs",
            ui.div(ui.h2("Step 5: PCPs"), ui.p("Explore Pareto candidates, apply objective bounds and select representative optimal solutions."), class_="step-heading"),
            ui.layout_columns(
                card("Run a new PCP analysis",
                     upload("pcp_file", "Load Pareto CSV", [".csv"]),
                     ui.input_action_button("pcp_use_active", "Use Active Pareto", class_="btn-other"),
                     ui.input_numeric("pcp_n", "Selected points", 5, min=1, max=10000),
                     ui.output_text("pcp_source_status"),
                     ui.input_action_button("pcp_run", "Run PCP Analysis", class_="btn-action")),
                card("Load previously selected optimal solutions",
                     upload("pcp_selected_file", "Load Selected CSV", [".csv"]),
                     ui.input_action_button("pcp_visualize", "Visualize PCP", class_="btn-action"),
                     ui.output_text("pcp_status")), col_widths=(6,6)),
            ui.div(ui.input_action_button("pcp_settings_open", "Objective bounds & display settings", class_="btn-action"),
                   class_="button-row"),
            card("PCP plot", ui.layout_columns(
                ui.input_selectize("pcp_legend", "Legend categoricals", choices=[], multiple=True),
                ui.download_button("pcp_download_plot", "Save Plot (PNG)", class_="btn-other"), col_widths=(8,4)),
                ui.output_ui("pcp_plot_view")),
            card("Optimal Pareto solutions",
                ui.layout_columns(ui.input_numeric("pcp_preview_rows", "Rows", 12, min=1, max=200),
                    ui.input_checkbox("pcp_constrained", "Constrained solutions", False),
                    ui.input_checkbox("pcp_utopia", "Show utopia distance", False),
                    ui.download_button("pcp_download_selected", "Save selected CSV", class_="btn-other"),
                    col_widths=(2,3,3,4)),
                ui.output_ui("pcp_filters_ui"), ui.output_text("pcp_constraints_status"),
                ui.output_ui("pcp_utopia_point"), ui.output_data_frame("pcp_preview"),
                ui.download_button("pcp_download_constrained", "Save constrained CSV", class_="btn-other")),
            value="pcp"),
        id="main_tab"),
    ui.tags.script("""
    function setupShapDrag() {
      document.querySelectorAll('[data-shap-sort]').forEach(list => {
        if (list.dataset.bound === 'yes') return;
        list.dataset.bound = 'yes';
        let moving = null;
        list.querySelectorAll('[draggable="true"]').forEach(item => {
          item.addEventListener('dragstart', e => {moving = item; e.dataTransfer.effectAllowed = 'move'; item.style.opacity='.5';});
          item.addEventListener('dragend', () => {item.style.opacity='1'; moving=null;});
          item.addEventListener('dragover', e => e.preventDefault());
          item.addEventListener('drop', e => {
            e.preventDefault();
            if (!moving || moving===item) return;
            let items=Array.from(list.children);
            list.insertBefore(moving, items.indexOf(moving)<items.indexOf(item) ? item.nextSibling : item);
            if(window.Shiny) Shiny.setInputValue(list.dataset.shapSort,
              Array.from(list.children).map(x=>x.dataset.value), {priority:'event'});
          });
        });
      });
    }
    new MutationObserver(setupShapDrag).observe(document.documentElement,{subtree:true,childList:true});
    document.addEventListener('DOMContentLoaded',setupShapDrag);
    """),
    ui.input_action_button("workflow_help", "?", class_="floating-help", title="Workflow guide"),
    class_="app-shell"
)


def read_csv(info):
    if not info:
        return None
    return pd.read_csv(info[0]["datapath"])


def infer_role(name, df):
    parts = str(name).split("_-_")
    if len(parts) > 1:
        token = parts[1].strip().lower()
        if token in ("maximize", "minimize", "decision"):
            return token.capitalize()
    return "Decision" if pd.api.types.is_numeric_dtype(df[name]) else "Categorical"


def server(input, output, session):
    domain_state = reactive.value(None)
    variable_overrides = reactive.value({})
    variable_message = reactive.value("")
    errors = reactive.value("")
    pcp_active = reactive.value(True)
    pcp_selected_state = reactive.value(None)
    pcp_message = reactive.value("Waiting for PCP analysis.")
    pcp_bound_settings = reactive.value({})
    pcp_decimal_settings = reactive.value({})

    @reactive.calc
    def data():
        return read_csv(input.experimental())

    @reactive.calc
    def domain_loaded():
        return read_csv(input.domain_file())

    @reactive.calc
    def hv():
        return read_csv(input.hv_file())

    @reactive.calc
    def pareto():
        return read_csv(input.pareto_file())

    @reactive.calc
    def pcp():
        return read_csv(input.pcp_file())

    @reactive.calc
    def pcp_selected_upload():
        return read_csv(input.pcp_selected_file())

    @reactive.calc
    def shap():
        info = input.shap_file()
        if not info:
            return None
        return pd.read_excel(info[0]["datapath"])

    @reactive.calc
    def shap_sources():
        info = input.shap_file()
        if not info:
            return []
        path = info[0]["datapath"]
        xls = pd.ExcelFile(path)
        if "__metadata__" in xls.sheet_names:
            meta = pd.read_excel(path, sheet_name="__metadata__")
            if not {"sheet_name", "objective_name"}.issubset(meta.columns):
                return []
            return [(str(row["objective_name"]), frame)
                    for _, row in meta.iterrows() if str(row["sheet_name"]) in xls.sheet_names
                    for frame in [pd.read_excel(path, sheet_name=str(row["sheet_name"]))]
                    if any(str(c).endswith("_SHAP") for c in frame.columns)]
        return [(name, frame) for name in xls.sheet_names if not name.startswith("__")
                for frame in [pd.read_excel(path, sheet_name=name)]
                if any(str(c).endswith("_SHAP") for c in frame.columns)]

    @reactive.calc
    def domain():
        loaded = domain_loaded()
        return loaded if loaded is not None else domain_state.get()

    @reactive.calc
    def variables():
        df = data()
        if df is None:
            return None
        rows = []
        for col in df.columns:
            if str(col).strip().lower() in ("id", "unnamed: 0"):
                continue
            series = df[col]
            numeric = pd.to_numeric(series, errors="coerce")
            role = infer_role(col, df)
            rows.append({"Original Column": col, "Display Name": str(col).split("_-_")[0],
                         "Role": role, "Type": "Continuous" if numeric.notna().any() else "Categorical",
                         "Lower Bound": numeric.min() if numeric.notna().any() else np.nan,
                         "Upper Bound": numeric.max() if numeric.notna().any() else np.nan, "Decimals": 4})
        result = pd.DataFrame(rows)
        for col, edits in variable_overrides.get().items():
            match = result["Original Column"] == col
            for key, val in edits.items():
                result.loc[match, key] = val
        return result

    selected_variable = reactive.value(None)

    @reactive.effect
    @reactive.event(input.variable_preview_selected_rows)
    def _open_variable_editor():
        selection = input.variable_preview_selected_rows()
        df = variables()
        if df is None or not selection or selection[0] >= len(df):
            return
        row = df.iloc[selection[0]]
        original = str(row["Original Column"])
        selected_variable.set(original)
        def bound_string(v):
            return "" if pd.isna(v) else str(v)
        ui.modal_show(ui.modal(
            ui.p("Edit the selected variable. Changes are applied when you click Save changes."),
            ui.input_text("edit_name", "Display name", value=str(row["Display Name"])),
            ui.layout_columns(
                ui.input_select("edit_role", "Variable role / objective", choices=["Decision", "Maximize", "Minimize", "Ignore"], selected=str(row["Role"])),
                ui.input_select("edit_type", "Variable type", choices=["Continuous", "Categorical"], selected=str(row["Type"])),
                col_widths=(6, 6)),
            ui.layout_columns(
                ui.input_text("edit_lower", "Lower bound", value=bound_string(row["Lower Bound"])),
                ui.input_text("edit_upper", "Upper bound", value=bound_string(row["Upper Bound"])),
                col_widths=(6, 6)),
            ui.input_select("edit_decimals", "Display decimal places", choices={str(i): str(i) for i in range(0, 11)}, selected=str(int(row["Decimals"]))),
            title=f"Variable settings — {row['Display Name']}",
            easy_close=True,
            footer=ui.div(ui.modal_button("Cancel"), ui.input_action_button("apply_variable", "Save changes", class_="btn-action"), class_="button-row")
        ))

    @reactive.effect
    @reactive.event(input.apply_variable)
    def _apply_variable():
        selected = selected_variable.get()
        if not selected:
            return
        def bound(value):
            value = str(value or "").strip()
            return float(value) if value else np.nan
        try:
            lower, upper = bound(input.edit_lower()), bound(input.edit_upper())
            if np.isfinite(lower) and np.isfinite(upper) and lower > upper:
                raise ValueError("Lower bound must not exceed upper bound.")
        except ValueError as exc:
            ui.notification_show(f"Invalid bounds: {exc}", type="error")
            return
        updated = dict(variable_overrides.get())
        updated[selected] = {"Display Name": input.edit_name().strip() or selected,
                             "Role": input.edit_role(), "Type": input.edit_type(),
                             "Lower Bound": lower, "Upper Bound": upper,
                             "Decimals": int(input.edit_decimals())}
        variable_overrides.set(updated)
        variable_message.set(f"Saved settings for {updated[selected]['Display Name']}.")
        ui.modal_remove()

    @render.text
    def variable_status():
        return variable_message.get()

    @reactive.effect
    @reactive.event(input.create_domain)
    def _create_domain():
        vars_df = variables()
        if vars_df is None or vars_df.empty:
            errors.set("Load experimental data before creating a domain.")
            return
        domain_state.set(vars_df.copy())
        errors.set("Created a draft domain table from inferred variable roles and observed bounds. Review before use; Summit domain construction is not yet ported.")

    def inspect_emulator_archive():
        info = input.emulator_zip()
        if not info:
            return None
        uploaded = Path(info[0]["datapath"])
        filename = str(info[0].get("name", "")).lower()
        if filename.endswith(".7z"):
            try:
                import py7zr
            except ImportError as exc:
                raise RuntimeError("7z support requires py7zr. Run: python -m pip install py7zr") from exc
            with py7zr.SevenZipFile(uploaded, mode="r") as archive:
                names = [n for n in archive.getnames() if not n.endswith("/")]
            return ("7z", names)
        if filename.endswith(".zip"):
            with zipfile.ZipFile(uploaded) as archive:
                names = [n for n in archive.namelist() if not n.endswith("/")]
            return ("ZIP", names)
        raise ValueError("Please upload a .zip or .7z archive.")

    @render.text
    def emulator_status():
        if not input.emulator_zip():
            return "No emulator archive uploaded."
        try:
            kind, names = inspect_emulator_archive()
            return f"{kind} loaded: {len(names)} files. Prediction remains disabled. Files: " + ", ".join(names[:8]) + (" ..." if len(names) > 8 else "")
        except Exception as exc:
            return f"Could not inspect emulator archive: {exc}"

    @render.ui
    def emulator_plot_view():
        info = input.emulator_zip()
        if not info:
            return ui.p("Upload emulator_model.zip or emulator_model.7z; an embedded parity plot will appear here automatically.")
        try:
            kind, names = inspect_emulator_archive()
            matches = [n for n in names if n.lower().endswith((".png", ".jpg", ".jpeg")) and "parity" in n.lower()]
            if not matches:
                return ui.p("Archive loaded, but no parity plot was found inside it. The original desktop application may store the parity plot separately.")
            name = matches[0]
            if kind == "ZIP":
                with zipfile.ZipFile(info[0]["datapath"]) as archive:
                    raw = archive.read(name)
            else:
                import py7zr
                with tempfile.TemporaryDirectory() as tmp:
                    with py7zr.SevenZipFile(info[0]["datapath"], mode="r") as archive:
                        archive.extract(path=tmp, targets=[name])
                    raw = (Path(tmp) / name).read_bytes()
            mime = "image/png" if name.lower().endswith(".png") else "image/jpeg"
            return ui.tags.img(src=f"data:{mime};base64,{base64.b64encode(raw).decode()}", style="max-width:100%;max-height:680px;object-fit:contain;")
        except Exception as exc:
            return ui.p(f"Could not read plot from emulator archive: {exc}")

    def image_from_figure(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=145, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return ui.tags.img(src="data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(), style="width:100%;height:auto;object-fit:contain;")

    @render.ui
    def hv_plot_view():
        df = hv()
        if df is None or df.empty:
            return ui.p("Load HV_coverage_data.csv to view the plot.")
        xcol = next((c for c in df.columns if "iteration" in c.lower()), df.columns[0])
        ycol = next((c for c in df.columns if "hv_coverage" in c.lower()), None)
        if ycol is None:
            return ui.p("HV coverage column not found.")
        fig, ax = plt.subplots(figsize=(7.4, 4.6))
        ax.plot(pd.to_numeric(df[xcol], errors="coerce"), pd.to_numeric(df[ycol], errors="coerce"), color="#2563eb", linewidth=2)
        ax.set(xlabel=xcol, ylabel=ycol, title="Hypervolume coverage")
        ax.grid(alpha=.2)
        return image_from_figure(fig)

    def shap_drag_list(items, input_id):
        return ui.tags.div(*[ui.tags.div("☰  " + str(name), draggable="true", **{"data-value": str(name)},
                         style="cursor:grab;padding:9px 12px;margin:5px 0;border:1px solid #cbd5e1;border-radius:6px;background:#f8fafc;")
                         for name in items], **{"data-shap-sort": input_id},
                         style="min-height:40px;")

    @render.ui
    def shap_objective_drag():
        return shap_drag_list([name for name, _ in shap_sources()], "shap_drag_objectives")

    @render.ui
    def shap_variable_drag():
        sources = shap_sources()
        if not sources: return ui.p("Load a workbook first.")
        df = sources[0][1]
        names = [str(c)[:-6] for c in df.columns if str(c).endswith("_Value") and str(c)[:-6]+"_SHAP" in df.columns]
        return shap_drag_list(names, "shap_drag_variables")

    shap_render_counter = reactive.value(0)
    shap_message = reactive.value("Load a SHAP workbook to generate the panel.")

    @reactive.effect
    @reactive.event(input.visualize_shap, input.apply_shap_order)
    def _render_shap_request():
        with reactive.isolate():
            next_count = shap_render_counter.get() + 1
        shap_render_counter.set(next_count)
        shap_message.set("SHAP panel refreshed from saved data.")

    @render.text
    def shap_status():
        return shap_message.get()

    @render.ui
    def shap_panel_view():
        sources = shap_sources()
        if not sources:
            return ui.p("SHAP panel will appear after an existing SHAP workbook is loaded.")
        shap_render_counter.get()
        try:
            obj_order = list(input.shap_drag_objectives() or [])
        except Exception:
            obj_order = []
        try:
            var_order = list(input.shap_drag_variables() or [])
        except Exception:
            var_order = []
        if obj_order:
            by_name = dict(sources)
            sources = [(n, by_name[n]) for n in obj_order if n in by_name] + [(n,d) for n,d in sources if n not in obj_order]
        if not sources:
            return ui.p("No objective sheets found.")
        vars_found = [str(c)[:-6] for c in sources[0][1].columns if str(c).endswith("_Value") and str(c)[:-6]+"_SHAP" in sources[0][1].columns]
        vars_found = [v for v in var_order if v in vars_found] + [v for v in vars_found if v not in var_order]
        if not vars_found:
            return ui.p("No paired *_Value / *_SHAP columns found in the workbook.")
        ncols = 2 if len(sources)>1 else 1
        nrows = (len(sources)+ncols-1)//ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(9*ncols, 5.8*nrows), squeeze=False)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("shap_blue_red", ["#1f77b4", "#ff0050"])
        scatter = None
        for k,(name,df) in enumerate(sources):
            ax = axs.flat[k]
            for j,v in enumerate(vars_found):
                if v+"_SHAP" not in df.columns or v+"_Value" not in df.columns:
                    continue
                x = pd.to_numeric(df[v+"_SHAP"], errors="coerce").to_numpy(dtype=float)
                vals = pd.to_numeric(df[v+"_Value"], errors="coerce").to_numpy(dtype=float)
                finite = np.isfinite(vals)
                lo,hi = (np.nanmin(vals[finite]), np.nanmax(vals[finite])) if finite.any() else (0.,1.)
                color = np.clip((vals-lo)/(hi-lo),0,1) if hi>lo else np.full(len(vals),.5)
                rng=np.random.default_rng(42+j+k*13)
                jitter=rng.uniform(-.16,.16,len(x))
                scatter=ax.scatter(x, j+jitter, c=color, cmap=cmap, vmin=0,vmax=1, s=19, alpha=.75, edgecolors="none")
            ax.set_yticks(range(len(vars_found)), vars_found)
            ax.invert_yaxis()
            ax.axvline(0, color="#64748b", lw=1)
            ax.grid(alpha=.13)
            ax.set_xlabel(f"SHAP value – impact on {name}", fontsize=10, labelpad=12)
            ax.tick_params(axis="x", labelsize=9)
            formatter=ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3,4))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.get_offset_text().set_fontsize(9)
            ax.set_title(name)
        for ax in list(axs.flat)[len(sources):]: ax.axis("off")
        if scatter is not None:
            fig.subplots_adjust(left=.17, right=.88, bottom=.10, top=.93, hspace=.58, wspace=.48)
            cbar_ax=fig.add_axes([.92,.20,.015,.60])
            fig.colorbar(scatter,cax=cbar_ax,label="Feature value (low → high)")
        else:
            fig.subplots_adjust(left=.17, right=.91, bottom=.10, top=.93, hspace=.58, wspace=.48)
        return image_from_figure(fig)


    @render.text
    def data_status():
        df = data()
        return "Waiting for CSV file." if df is None else f"Loaded {len(df):,} rows and {len(df.columns)} columns."

    @render.text
    def domain_status():
        d = domain()
        return errors.get() or ("No domain loaded." if d is None else f"Domain table ready: {len(d)} variables.")

    def rounded_table(df):
        if df is None:
            return table_view(df)
        result = table_view(df).copy()
        settings = variables()
        original_decimals = {}
        display_decimals = {}
        if settings is not None:
            for _, r in settings.iterrows():
                original_decimals[str(r["Original Column"])] = int(r["Decimals"])
                display_decimals[str(r["Display Name"])] = int(r["Decimals"])
        for col in result.select_dtypes(include=["number"]).columns:
            name = str(col)
            if name == "Decimals":
                continue
            digits = original_decimals.get(name, display_decimals.get(name))
            if digits is None:
                # Match Pareto columns whose units/metadata suffix differ from the original data.
                base = name.split("_-_")[0]
                base = base.split("_std_METADATA")[0].split("_variance_METADATA")[0].split("_noise_METADATA")[0]
                digits = display_decimals.get(base)
            if digits is None:
                # Per-NB Pareto columns have different units but retain the same objective stem.
                for key, precision in {**original_decimals, **display_decimals}.items():
                    stem = str(key).split("_-_")[0].split("_(")[0].lower()
                    if stem and len(stem) >= 3 and name.lower().startswith(stem):
                        digits = precision
                        break
            if digits is None:
                source = pcp_source()
                if source is not None and col in source.columns:
                    numeric_cols = list(source.select_dtypes(include="number").columns)
                    if col in numeric_cols:
                        try:
                            val = input[f"pcp_dec_{numeric_cols.index(col)}"]()
                            digits = int(val) if val is not None else None
                        except Exception:
                            pass
            digits = 4 if digits is None else digits
            # Shiny DataGrid transmits numeric floats as numbers and will discard
            # trailing zeros. Strings preserve the chosen displayed precision.
            result[col] = result[col].map(lambda v: "" if pd.isna(v) else f"{float(v):.{digits}f}")
        return result

    @render.data_frame
    def data_preview():
        d = data()
        if d is not None:
            d = d.drop(columns=[c for c in d.columns if str(c).lower() in ("id", "unnamed: 0")], errors="ignore")
        return render.DataGrid(rounded_table(d), filters=False)

    @render.data_frame
    def variable_preview():
        return render.DataGrid(rounded_table(variables()).drop(columns=["Original Column"], errors="ignore"), filters=False, selection_mode="row")

    @render.data_frame
    def domain_preview():
        d = domain()
        if d is not None and "Original Column" in d.columns:
            d = d.drop(columns=["Original Column"])
        return render.DataGrid(rounded_table(d), filters=False)

    @render.data_frame
    def hv_preview():
        return render.DataGrid(table_view(hv()), filters=False)

    @render.data_frame
    def pareto_preview():
        return render.DataGrid(rounded_table(pareto()), filters=False)

    @render.data_frame
    def shap_preview():
        return render.DataGrid(table_view(shap()), filters=False)

    # PCP data selection, objective constraints, diversity selection, plotting and exports.
    @reactive.calc
    def pcp_source():
        if pcp_active.get():
            return pareto() if pareto() is not None else pcp()
        return pcp() if pcp() is not None else pareto()

    def is_pcp_helper(col):
        name = str(col).lower()
        return (name in ("id", "row #", "originalrow", "utopia distance")
                or name.startswith(("unnamed", "__", "_"))
                or "metadata" in name or "variance" in name or "_std" in name
                or "_noise" in name or "utopia_dist" in name)

    def pcp_match_column(name, cols):
        name = str(name)
        if name in cols:
            return name
        stem = name.split("_-_")[0].split("_(")[0].lower()
        if not stem or stem == "nan":
            return None
        return next((c for c in cols if str(c).lower().startswith(stem) and len(stem) > 2), None)

    @reactive.calc
    def pcp_objectives():
        df = pcp_source()
        if df is None:
            df = pcp_selected_upload()
        if df is None:
            return []
        cols = [c for c in df.columns if not is_pcp_helper(c)
                and pd.to_numeric(df[c], errors="coerce").notna().any()]
        # Prefer explicit roles from the experimental data/domain, matching renamed
        # per-NB objective units by their stable objective stem.
        chosen = []
        for settings in (variables(), domain()):
            if settings is None or "Role" not in settings.columns:
                continue
            for _, row in settings.iterrows():
                if str(row["Role"]).lower() not in ("maximize", "minimize"):
                    continue
                for name in (row.get("Original Column", ""), row.get("Display Name", "")):
                    match = pcp_match_column(name, cols)
                    if match and match not in chosen:
                        chosen.append(match)
        # For a Pareto CSV, objective columns can be identified independently of
        # experimental data; use the four measured outputs, not metadata.
        keywords = ("yield", "co2", "cost", "emission", "energy", "purity", "selectivity")
        inferred = [c for c in cols if any(word in str(c).lower() for word in keywords)]
        for c in inferred:
            if c not in chosen:
                chosen.append(c)
        return chosen

    def pcp_direction(col):
        for settings in (variables(), domain()):
            if settings is None or "Role" not in settings.columns:
                continue
            for _, row in settings.iterrows():
                if str(row["Role"]).lower() not in ("maximize", "minimize"):
                    continue
                if pcp_match_column(row.get("Original Column", row.get("Display Name", "")), [col]):
                    return str(row["Role"]).capitalize()
        return "Minimize" if any(k in str(col).lower() for k in ("cost", "co2", "emission", "energy")) else "Maximize"

    def pcp_norm(df, objectives):
        result = np.empty((len(df),len(objectives)))
        for j,c in enumerate(objectives):
            x = pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)
            valid = np.isfinite(x)
            lo,hi = (np.nanmin(x[valid]),np.nanmax(x[valid])) if valid.any() else (0.,1.)
            norm = np.clip((x-lo)/(hi-lo),0,1) if hi>lo else np.full(len(x),.5)
            if pcp_direction(c)=="Minimize": norm = 1-norm
            result[:,j] = np.nan_to_num(norm,nan=.5)
        return result

    @reactive.effect
    @reactive.event(input.pcp_use_active)
    def _pcp_use_active():
        pcp_active.set(True)
        pcp_message.set("Using Pareto results from TS-EMO.")

    @reactive.effect
    @reactive.event(input.pcp_file)
    def _pcp_use_file():
        pcp_active.set(False)
        pcp_selected_state.set(None)

    @render.text
    def pcp_source_status():
        df=pcp_source()
        return "Available Pareto points: " + (str(len(df)) if df is not None else "—")

    @reactive.effect
    def _pcp_update_legend():
        df=pcp_source()
        if df is None: return
        categorical=[c for c in df.columns if df[c].dtype==object and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        ui.update_selectize("pcp_legend", choices=categorical, selected=categorical[:2])

    def pcp_digits(col):
        overrides = pcp_decimal_settings.get()
        if col in overrides:
            return int(overrides[col])
        settings = variables()
        if settings is not None:
            for _, r in settings.iterrows():
                if pcp_match_column(r["Original Column"], [col]) or pcp_match_column(r["Display Name"], [col]):
                    return int(r["Decimals"])
        return 4

    def pcp_label(col):
        settings = variables()
        if settings is not None:
            for _, r in settings.iterrows():
                if pcp_match_column(r["Original Column"], [col]) or pcp_match_column(r["Display Name"], [col]):
                    return str(r["Display Name"])
        return str(col).replace("_-_", " ").replace("_", " ")

    @reactive.effect
    @reactive.event(input.pcp_settings_open)
    def _open_pcp_settings():
        df = pcp_source()
        if df is None or df.empty:
            ui.notification_show("Load Pareto data before editing objective settings.", type="warning")
            return
        controls = []
        bounds = pcp_bound_settings.get()
        for j, c in enumerate(pcp_objectives()):
            current = bounds.get(c, {})
            controls.append(ui.div(
                ui.h5(f"{pcp_label(c)} — {pcp_direction(c)}"),
                ui.layout_columns(
                    ui.input_numeric(f"pcp_modal_low_{j}", "Lower bound", current.get("low")),
                    ui.input_numeric(f"pcp_modal_high_{j}", "Upper bound", current.get("high")),
                    ui.input_select(f"pcp_modal_dec_{j}", "Display decimals",
                        choices={str(i): str(i) for i in range(11)}, selected=str(pcp_digits(c))),
                    col_widths=(4, 4, 4)),
                class_="mb-3"))
        ui.modal_show(ui.modal(
            ui.p("Bounds are drawn as reference ticks on each PCP axis. When 'Constrained solutions' is enabled, they also filter the table."),
            *controls,
            footer=ui.div(ui.input_action_button("pcp_settings_save", "Apply settings & refresh plot", class_="btn-action"),
                          ui.input_action_button("pcp_settings_cancel", "Cancel")),
            title="PCP objective bounds & display settings", size="l", easy_close=False))

    @reactive.effect
    @reactive.event(input.pcp_settings_cancel)
    def _cancel_pcp_settings():
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.pcp_settings_save)
    def _save_pcp_settings():
        new_bounds = dict(pcp_bound_settings.get())
        new_decimals = dict(pcp_decimal_settings.get())
        for j, c in enumerate(pcp_objectives()):
            low = input[f"pcp_modal_low_{j}"]()
            high = input[f"pcp_modal_high_{j}"]()
            if low is not None and high is not None and float(low) > float(high):
                ui.notification_show(f"Invalid bounds for {pcp_label(c)}: lower exceeds upper.", type="error")
                return
            new_bounds[c] = {"low": low, "high": high}
            new_decimals[c] = int(input[f"pcp_modal_dec_{j}"]())
        pcp_bound_settings.set(new_bounds)
        pcp_decimal_settings.set(new_decimals)
        ui.modal_remove()
        ui.notification_show("PCP settings applied. Plot and table updated.", type="message")

    def pcp_filtered(df):
        if df is None: return None
        mask=np.ones(len(df),dtype=bool)
        for c in pcp_objectives():
            if c not in df.columns: continue
            entry = pcp_bound_settings.get().get(c, {})
            lo, hi = entry.get("low"), entry.get("high")
            x=pd.to_numeric(df[c],errors="coerce")
            if lo is not None: mask &= (x>=float(lo)).to_numpy()
            if hi is not None: mask &= (x<=float(hi)).to_numpy()
        return df.loc[mask].copy()

    def pcp_select(df,n):
        objectives=pcp_objectives()
        if df is None or df.empty or not objectives: return df
        norm=pcp_norm(df,objectives)
        # Start closest to the ideal (utopia), then greedily cover the remaining
        # objective space with maximally separated solutions.
        dist=np.linalg.norm(1-norm,axis=1)
        selected=[int(np.argmin(dist))]
        mindist=np.linalg.norm(norm-norm[selected[0]],axis=1)
        for _ in range(min(max(1,n),len(df))-1):
            mindist[selected]=-1
            nxt=int(np.argmax(mindist))
            selected.append(nxt)
            mindist=np.minimum(mindist,np.linalg.norm(norm-norm[nxt],axis=1))
            mindist[selected]=-1
        result=df.iloc[selected].copy()
        result.insert(0,"Row #",df.index.to_numpy()[selected]+1)
        result["Utopia distance"]=dist[selected]
        return result

    @reactive.effect
    @reactive.event(input.pcp_run)
    def _run_pcp():
        df=pcp_source()
        if df is None or df.empty:
            ui.notification_show("Load a Pareto CSV or use active Pareto first.",type="error")
            return
        chosen=pcp_select(pcp_filtered(df),int(input.pcp_n() or 5))
        pcp_selected_state.set(chosen)
        pcp_message.set(f"PCP analysis complete: {len(chosen)} representative solutions selected from {len(df)} Pareto points.")

    @reactive.effect
    @reactive.event(input.pcp_visualize)
    def _visualize_pcp():
        df=pcp_selected_upload()
        if df is None:
            ui.notification_show("Load a saved selected-solutions CSV first.",type="error")
            return
        pcp_selected_state.set(df)
        pcp_message.set(f"Visualizing {len(df)} previously selected solutions.")

    @render.text
    def pcp_status():
        return pcp_message.get()

    @reactive.calc
    def pcp_selected():
        return pcp_selected_state.get()

    @render.text
    def pcp_constraints_status():
        df=pcp_selected()
        if df is None: return "Run PCP Analysis or visualize a saved selection."
        valid=pcp_filtered(df)
        return f"{len(valid)} of {len(df)} selected solutions satisfy the current objective bounds."

    @render.ui
    def pcp_filters_ui():
        df=pcp_selected()
        if df is None: return ui.p("Categorical filters appear after selecting solutions.")
        categorical=[c for c in df.columns if df[c].dtype==object and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        return ui.div(*[ui.input_select(f"pcp_cat_{j}",str(c),
                choices={"__all__":"All",**{str(v):str(v) for v in df[c].dropna().unique()}},
                selected="__all__") for j,c in enumerate(categorical)])

    def pcp_table_data():
        df=pcp_selected()
        if df is None: return None
        df=pcp_filtered(df) if input.pcp_constrained() else df.copy()
        categorical=[c for c in df.columns if df[c].dtype==object and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        for j,c in enumerate(categorical):
            try: val=input[f"pcp_cat_{j}"]()
            except Exception: continue
            if val and val!="__all__": df=df[df[c].astype(str)==val]
        if input.pcp_utopia() and "Utopia distance" not in df.columns:
            objectives = [c for c in pcp_objectives() if c in df.columns]
            if objectives:
                source = pcp_source()
                if source is None: source = df
                norm = np.empty((len(df), len(objectives)))
                for j, c in enumerate(objectives):
                    ref = pd.to_numeric(source[c], errors="coerce")
                    vals = pd.to_numeric(df[c], errors="coerce")
                    lo, hi = ref.min(), ref.max()
                    z = np.clip((vals-lo)/(hi-lo), 0, 1) if hi > lo else np.full(len(df), .5)
                    if pcp_direction(c) == "Minimize": z = 1-z
                    norm[:, j] = np.nan_to_num(z, nan=.5)
                df["Utopia distance"] = np.linalg.norm(1-norm, axis=1)
        if not input.pcp_utopia() and "Utopia distance" in df.columns:
            df=df.drop(columns=["Utopia distance"])
        return df.drop(columns=[c for c in df.columns if is_pcp_helper(c)], errors="ignore")

    @render.ui
    def pcp_utopia_point():
        if not input.pcp_utopia():
            return ui.div()
        source = pcp_source()
        if source is None:
            source = pcp_selected()
        if source is None or source.empty:
            return ui.p("No Pareto data available for the utopia point.")
        objectives = [c for c in pcp_objectives() if c in source.columns]
        if not objectives:
            return ui.p("No objectives available for the utopia point.")
        settings = variables()
        rows = []
        for c in objectives:
            values = pd.to_numeric(source[c], errors="coerce").dropna()
            if values.empty:
                continue
            direction = pcp_direction(c)
            ideal = values.min() if direction == "Minimize" else values.max()
            decimals = 4
            if settings is not None:
                for _, r in settings.iterrows():
                    if pcp_match_column(r["Original Column"], [c]):
                        decimals = int(r["Decimals"])
                        break
            rows.append(ui.tags.tr(ui.tags.td(str(c)), ui.tags.td(f"{ideal:.{decimals}f}"),
                                   ui.tags.td(f"[{direction.lower()}]")))
        return ui.div(ui.strong("Utopia point"),
                      ui.tags.table(ui.tags.tbody(*rows), class_="table table-sm"),
                      class_="section-card")

    @render.data_frame
    def pcp_preview():
        df=pcp_table_data()
        view = rounded_table(df).head(int(input.pcp_preview_rows() or 12))
        for col in pcp_objectives():
            if col in view.columns and col in df.columns:
                digits = pcp_digits(col)
                view[col] = pd.to_numeric(df[col].head(len(view)),errors="coerce").map(
                    lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        return render.DataGrid(view,filters=False)

    @render.ui
    def pcp_plot_view():
        df=pcp_selected()
        if df is None or df.empty: return ui.p("Run PCP Analysis or load a selected-solutions CSV to see the plot.")
        fig=pcp_figure(df)
        return image_from_figure(fig)

    def pcp_figure(df):
        objectives=pcp_objectives()
        source=pcp_source()
        if source is None: source=df
        if len([c for c in objectives if c in df.columns]) < 2:
            objectives = [c for c in df.columns if not is_pcp_helper(c) and any(
                k in str(c).lower() for k in ("yield", "co2", "cost", "emission"))]
        # PCP axes are only the optimization objectives, never metadata or decisions.
        cols=[c for c in objectives if c in df.columns]
        if len(cols)<2:
            fig,ax=plt.subplots(figsize=(10,3))
            ax.text(.5,.5,"At least two objective columns are required.",ha="center",va="center")
            ax.axis("off")
            return fig
        fig,ax=plt.subplots(figsize=(max(12,len(cols)*1.8),7.5))
        fig.subplots_adjust(left=.09,right=.88,bottom=.28,top=.89)
        x=np.arange(len(cols))
        values=np.zeros((len(df),len(cols)))
        for j,c in enumerate(cols):
            ref=source[c] if c in source else df[c]
            if pd.api.types.is_numeric_dtype(ref):
                refnum=pd.to_numeric(ref,errors="coerce")
                lo,hi=refnum.min(),refnum.max()
                val=pd.to_numeric(df[c],errors="coerce").to_numpy(dtype=float)
                values[:,j]=np.clip((val-lo)/(hi-lo),0,1) if hi>lo else .5
            else:
                categories=sorted(ref.dropna().astype(str).unique())
                codes={v:i for i,v in enumerate(categories)}
                values[:,j]=[codes.get(str(v),0)/max(1,len(categories)-1) for v in df[c]]
        score_cols = [c for c in objectives if c in df.columns]
        norm = np.zeros((len(df),len(score_cols)))
        for k,c in enumerate(score_cols):
            reference = pd.to_numeric(source[c] if c in source.columns else df[c],errors="coerce")
            vals = pd.to_numeric(df[c],errors="coerce")
            mn,mx = reference.min(),reference.max()
            z = np.clip((vals-mn)/(mx-mn),0,1) if mx>mn else np.full(len(df),.5)
            if pcp_direction(c)=="Minimize": z=1-z
            norm[:,k]=np.nan_to_num(z,nan=.5)
        dist=np.linalg.norm(1-norm,axis=1)
        # Match desktop app: best (closest to utopia) = blue, worst = red.
        cvals = (dist-dist.min())/(np.ptp(dist) or 1)
        color = np.column_stack([cvals, np.zeros_like(cvals), 1-cvals, np.full_like(cvals, .9)])
        for i,row in enumerate(values):
            ax.plot(x,row,color=color[i],alpha=.83,lw=1.6)
        for j,c in enumerate(cols):
            ax.axvline(j,color="#cbd5e1",lw=.8,zorder=0)
        # Draw bounds in raw units, mapped onto each objective's normalized axis.
        for j, c in enumerate(cols):
            entry = pcp_bound_settings.get().get(c, {})
            ref = pd.to_numeric(source[c] if c in source.columns else df[c], errors="coerce")
            mn, mx = ref.min(), ref.max()
            for key, sign, ls in (("low", "≥", "-"), ("high", "≤", "--")):
                bound = entry.get(key)
                if bound is None: continue
                yy = .5 if not np.isfinite(mx-mn) or mx == mn else (float(bound)-mn)/(mx-mn)
                if not np.isfinite(yy): continue
                yy = np.clip(yy, 0, 1)
                ax.plot([j-.055,j+.055],[yy,yy],color="#374151",lw=1.6,ls=ls,zorder=8)
                ax.annotate(f"{sign} {float(bound):.{pcp_digits(c)}f}", (j, yy),
                    xytext=(7 if j<len(cols)-1 else -7,0),textcoords="offset points",
                    ha="left" if j<len(cols)-1 else "right",va="center",fontsize=9,
                    bbox=dict(facecolor="white",edgecolor="#cbd5e1",alpha=.95,boxstyle="round,pad=.2"),zorder=9)
        labels=[f"{pcp_label(c).replace('(', chr(10)+'(')}\n[{pcp_direction(c)}]" for c in cols]
        ax.set_xticks(x, labels, fontsize=10)
        ax.set_yticks([0,1], ["Min", "Max"])
        ax.set_xlim(0,len(cols)-1);ax.set_ylim(-.03,1.03)
        ax.set_ylabel("Normalized Scale")
        ax.set_title(f"Parallel coordinates — {len(df)} selected solutions")
        ax.grid(axis="y",alpha=.18)
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize
        from matplotlib.colors import ListedColormap
        gradient = np.linspace(0,1,256)
        blue_to_red = ListedColormap(np.column_stack([gradient,np.zeros(256),1-gradient]))
        sm=ScalarMappable(norm=Normalize(vmin=0,vmax=1),cmap=blue_to_red)
        cb=fig.colorbar(sm,ax=ax,pad=.04,shrink=.70,label="Proximity to the Utopia Point")
        cb.set_ticks([0,1]); cb.set_ticklabels(["Best", "Worst"])
        return fig

    @render.download_button(filename="pcp_selected_solutions.csv")
    def pcp_download_selected():
        df=pcp_selected()
        yield df.to_csv(index=False) if df is not None else ""

    @render.download_button(filename="pcp_constrained_solutions.csv")
    def pcp_download_constrained():
        df=pcp_table_data()
        yield df.to_csv(index=False) if df is not None else ""

    @render.download_button(filename="pcp_plot.png", media_type="image/png")
    def pcp_download_plot():
        df=pcp_selected()
        if df is None: return
        fig=pcp_figure(df)
        buf=io.BytesIO()
        fig.savefig(buf,format="png",dpi=200,bbox_inches="tight")
        plt.close(fig)
        yield buf.getvalue()

    @render.download_button(filename="domain_table_draft.csv")
    def download_domain():
        df = domain()
        yield (df.to_csv(index=False) if df is not None else "")

    @reactive.effect
    @reactive.event(input.workflow_help)
    async def _help():
        tab = input.main_tab()
        steps = {
            "data": ("Data & Domain", [("Load experimental_data.csv", data() is not None), ("Create or load domain_table.csv", domain() is not None)], "Complete this tab before proceeding to later analyses."),
            "emulator": ("Emulator Training", [("Upload emulator_model.zip or .7z", bool(input.emulator_zip())), ("Predict outputs (not yet available)", False)], "Prediction is intentionally disabled in this Shiny phase."),
            "tsemo": ("TS-EMO Analysis", [("Load HV_coverage_data.csv", hv() is not None), ("Load Pareto CSV (original or per_NB)", pareto() is not None)], "The per_NB file uses normalised output units; it is not a separate optimisation run."),
            "shap": ("SHAP Analysis", [("Load SHAP_beeswarm_data.xlsx", shap() is not None)], "SHAP generation is unavailable."),
            "pcp": ("PCPs", [("Load or reuse Pareto data", pcp() is not None or pareto() is not None), ("Run PCP analysis or visualize saved selections", pcp_selected() is not None)], "Set objective bounds, select representative solutions, inspect parallel coordinates and export your selection."),
        }
        title, actions, note = steps.get(tab, steps["data"])
        content = [ui.h3(title)]
        for label, done in actions:
            content.append(ui.div(("✓  " if done else "○  ") + label, class_="help-done" if done else "help-pending"))
        content.append(ui.p(note, class_="hint"))
        await session.send_custom_message("noop", {}) if False else None
        ui.modal_show(ui.modal(*content, title="Workflow guide", easy_close=True, footer=ui.modal_button("Close")))


app = App(app_ui, server)
