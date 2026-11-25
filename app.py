"""Production Optimisation Shiny — phase 1 interface and data/domain workflow.
Reference: Production Optimisation Demo v1.0.20 (PySide6).
"""
from pathlib import Path
import zipfile
import hashlib
import os
import re
import tempfile
import base64
import io
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from shiny import App, ui, render, reactive
from matplotlib.ticker import ScalarFormatter, MaxNLocator

HERE = Path(__file__).parent


def load_direct_emulator(root):
    import torch
    paths = list(root.rglob("emulator_model.json"))
    if len(paths) != 1:
        raise ValueError("Expected one emulator_model.json")
    path = paths[0]
    config = json.loads(path.read_text())
    params = config["experiment_params"]
    if params["regressor_name"] != "ANNRegressor" or params.get("descriptors_features"):
        raise ValueError("Only ANNRegressor models without descriptor encoding are supported.")
    variables = config["domain"]
    inputs = [v for v in variables if not v["is_objective"]]
    nums = [v for v in inputs if v["type"] == "ContinuousVariable"]
    cats = [v for v in inputs if v["type"] == "CategoricalVariable"]
    outputs = [next(v for v in variables if v["name"] == name and v["is_objective"]) for name in params["output_variable_names"]]
    models = []
    for i, prep in enumerate(params["predictors"]):
        state = torch.load(path.parent / f"{params['model_name']}_predictor_{i}.pt", map_location="cpu", weights_only=True)
        if set(state) != {"input_layer.weight", "input_layer.bias", "output_layer.weight", "output_layer.bias"}:
            raise ValueError("Unsupported ANN architecture")
        w = state["input_layer.weight"]
        if w.shape[1] != len(nums) + sum(len(v["levels"]) for v in cats) or state["output_layer.weight"].shape != (len(outputs), w.shape[0]):
            raise ValueError("Model dimensions do not match saved domain")
        models.append((state, prep))
    if not models: raise ValueError("No predictors found")
    def predict(vals):
        numeric = np.array([float(vals[v["name"]]) for v in nums])
        categorical = []
        for v in cats:
            value = vals[v["name"]]
            if value not in v["levels"]: raise ValueError(f"Unknown category for {v['name']}")
            categorical.extend(float(value == level) for level in v["levels"])
        predictions = []
        with torch.inference_mode():
            for state, prep in models:
                ip = prep["input_preprocessor"]["num"]
                op = prep["output_preprocessor"]
                x = torch.tensor(np.r_[(numeric - ip["mean_"]) / ip["scale_"], categorical], dtype=torch.float32)
                y = torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(x, state["input_layer.weight"], state["input_layer.bias"])), state["output_layer.weight"], state["output_layer.bias"]).numpy()
                y *= np.asarray(op["scale_"])
                y += np.asarray(op["mean_"])
                clip = params.get("clip", True)
                for j, v in enumerate(outputs):
                    if clip is True or isinstance(clip, list) and v["name"] in clip:
                        y[j] = np.clip(y[j], *v["bounds"])
                predictions.append(y)
        return dict(zip(params["output_variable_names"], np.mean(predictions, axis=0)))
    return predict, variables, outputs


def card(title, *children):
    return ui.div(ui.h3(title, class_="card-title"), *children, class_="section-card")


def disabled(label):
    return ui.tags.button(label, disabled=True, title="Not available in this web preview", class_="btn-locked")


def upload(id, label, accept):
    return ui.div(ui.input_file(id, ("⚠️ " if id == "experimental" else "💾 ") + label, accept=accept, multiple=False, width="100%"), ui.output_text(id + "_filename", inline=True), class_="upload-wrap")


def display_name(name):
    return str(name).split("_-_")[0].replace("_", " ").strip()

def clean_variable_view(df):
    df = df.copy()
    if "Display Name" in df: df["Display Name"] = df["Display Name"].map(display_name)
    return df

def display_table(df):
    return df.rename(columns=display_name)

def table_view(df, limit=200):
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.head(0)
    return df.head(limit)


app_ui = ui.page_fluid(
    ui.tags.style('body {font-family: \'Segoe UI\',Arial,sans-serif; background:#f5f8fc; color:#20324d;}\n.app-shell {max-width: 1580px; margin:auto; padding:18px 26px 70px;}\n.top-header {display:flex; justify-content:space-between; align-items:center; gap:20px; background:linear-gradient(120deg,#e8f3ff,#f6faff); padding:22px 26px; border:1px solid #c9def5; border-radius:12px; margin-bottom:14px;}\n.app-heading h1 {font-size:27px; font-weight:750; color:#14346c; margin:0 0 6px;}\n.app-heading p {margin:0; color:#536b89;}\n.status-pill {white-space:nowrap; color:#19613c; font-weight:600;}\n.workflow {display:flex; gap:13px; flex-wrap:wrap; padding:14px 20px; border:1px solid #cddff1; background:white; border-radius:9px; margin-bottom:12px; color:#345c91; font-weight:600;}\n.case-accordion {margin-bottom:14px;}\n.main-tabs {background:white; border-radius:10px;}\n.main-tabs .nav-tabs {border-bottom:2px solid #7ba8d8; background:#eef5fd; padding:8px 8px 0; gap:4px;}\n.main-tabs .nav-link {font-weight:650; color:#335a8e; border:1px solid #c7d9ef; border-bottom:0; background:#e8f1fc;}\n.main-tabs .nav-link.active {color:#14346c; background:white; border-color:#7ba8d8;}\n.main-tabs .tab-content {padding:18px 20px 35px;}\n.step-heading {padding:18px 22px; background:#edf5ff; border:1px solid #d3e5f9; border-radius:10px; margin-bottom:15px;}\n.step-heading h2 {font-size:23px; color:#14346c; font-weight:700; margin:0 0 8px;}\n.step-heading p {margin:0;color:#4e6483;}\n.section-card {background:white; border:1px solid #d5e2ef; border-radius:10px; box-shadow:0 3px 14px #142a4d0b; padding:18px 20px; margin-bottom:15px;}\n.card-title {font-size:17px; color:#1c457c; font-weight:700; margin:0 0 13px;}\n.btn-action {background:#20834c !important; color:white !important; border:0 !important;}\n.btn-other {background:#7550a5 !important; color:white !important; border:0 !important;}\n.btn-locked {border:1px solid #b8c4d4; color:#8591a3; background:#e8ecf2; border-radius:6px; padding:8px 13px; margin:5px 8px 8px 0; cursor:not-allowed;}\n.button-row {display:flex; gap:12px; margin-bottom:16px;}\n.floating-help {position:fixed !important; right:26px; bottom:24px; width:54px; height:54px; border-radius:50% !important; padding:0 !important; display:flex; align-items:center; justify-content:center; font-size:27px; font-weight:800; color:white !important; background:#d62828 !important; border:3px solid white !important; box-shadow:0 4px 15px #0003; z-index:1050;}\n.help-done,.help-pending {padding:11px 14px; border-radius:8px; margin:8px 0;}\n.help-done {background:#dcfce7; color:#166534;}.help-pending {background:#f1f5f9; color:#334155;}\n.hint {color:#596a7f; font-size:14px; margin-top:12px;}\n@media(max-width:700px){.app-shell{padding:10px 10px 70px}.top-header{flex-direction:column;align-items:start}.workflow{gap:7px}.main-tabs .tab-content{padding:12px 8px}.floating-help{right:15px;bottom:15px}}\n\n/* Desktop-inspired workflow map (responsive, no images or JS needed) */\n.workflow-graphic{background:#fafafa;border:1px solid #e7eaf0;border-radius:12px;padding:28px 30px 24px;margin-bottom:16px;color:#34465b;}\n.wf-diagram{display:flex;align-items:center;gap:24px;min-height:165px;flex-wrap:wrap;margin-bottom:14px;}\n.wf-step{display:inline-flex;flex-shrink:0;align-items:center;gap:12px;min-width:156px;min-height:62px;padding:12px 14px;border:2px solid #dadada;background:#f1f1f1;border-radius:10px;box-shadow:0 3px 5px #00000012;font-weight:700;color:#1b2634;}\n.wf-active{background:#337ec4;border-color:#337ec4;color:white;}\n.wf-num{display:inline-flex;justify-content:center;align-items:center;min-width:33px;height:33px;padding:0 6px;background:#bdbdbd;border-radius:7px;color:white;font-weight:750;font-size:19px;}\n.wf-num-active{background:#053c84;}\n.wf-arrow{font-size:29px;font-weight:300;color:#b6b6b6;}\n.wf-fork{position:relative;display:flex;flex-direction:column;gap:20px;margin-left:30px;padding-left:56px;border-left:3px solid #bcbcbc;}\n.wf-branch{position:relative;display:flex;align-items:center;gap:25px;min-height:62px;}\n.wf-branch::before{content:"";position:absolute;left:-56px;top:50%;width:36px;border-top:3px solid #bcbcbc;}\n.wf-fork::before{content:"";position:absolute;top:50%;left:-48px;width:48px;border-top:2px solid #bcbcbc;}\n.wf-guide{border-left:4px solid #2682ef;padding:5px 0 5px 22px;margin:14px 0 18px;line-height:1.55;}\n.wf-symbol-title{font-weight:650;margin:10px 0;}\n.wf-symbol{margin:12px 0;}\n@media(max-width:900px){.workflow-graphic{padding:16px}.wf-diagram{gap:12px}.wf-step{min-width:125px}.wf-fork{margin-left:12px;padding-left:25px}.wf-branch::before{left:-25px;width:16px}.wf-fork::before{left:-40px;width:15px}}\n@media(max-width:650px){.wf-diagram{display:grid;grid-template-columns:1fr;gap:10px}.wf-arrow{display:none}.wf-fork{margin:0 0 0 14px;padding-left:16px;gap:10px}.wf-fork::before{display:none}.wf-branch{flex-wrap:wrap}.wf-branch::before{left:-16px;width:10px}.wf-step{min-height:52px}.wf-branch-bottom>.wf-arrow{display:none}}\n'),
    ui.tags.style("""
    .shiny-input-container .input-group-btn .btn, .shiny-input-container .input-group .btn-file {background:#68737f!important;color:white!important;border-color:#68737f!important;}
    .action-button:not(.floating-help):not(#constraints_open):not(.btn-locked):not(#pcp_settings_cancel):not(.btn-option) {background:#20834c!important;color:white!important;border-color:#20834c!important;}
    .shiny-download-link, .btn-other.shiny-download-link {background:#7550a5!important;color:white!important;border-color:#7550a5!important;}
    .btn-locked {opacity:.65;}
    .action-button.btn-option {background:#1769aa!important;color:white!important;border-color:#1769aa!important;}
    .pcp-size-controls {display:flex;flex-direction:column;align-items:center;gap:8px;width:100%;}
    .pcp-size-controls .shiny-input-container {width:min(100%,340px)!important;}
    .pcp-size-footer {display:flex;justify-content:center;width:100%;}
    #constraints_open {display:inline-flex;align-items:center;justify-content:center;font-family:Arial,sans-serif!important;font-size:26px!important;font-weight:900!important;line-height:1!important;}
    .pcp-tip-columns {text-align:center;align-items:start;}
    .pcp-tip-columns b {text-align:center;}
    .pcp-tip strong {margin-bottom:8px;}

    """),
    ui.tags.style(".floating-help{top:18px !important;bottom:auto !important;right:18px !important;} .top-header{padding-right:85px;} .frozen-controls{opacity:.72;} .preview-progress{height:12px;width:100%;}"),
    ui.tags.style(".modal-content{border:2px solid #93b4d8;border-radius:14px;box-shadow:0 12px 36px #183c6226}.modal-title{font-weight:700;text-align:center;width:100%}.pcp-interactive svg{width:100%;height:auto}.pcp-interactive [id^=pcpline] path{cursor:pointer;pointer-events:stroke}.pcp-interactive [id^=pcpline]:hover path{stroke-width:4!important}"),
    ui.div(
        ui.div(ui.h1("Production Optimisation App"),
               ui.p("Preview interface for loading saved domains, emulators, TS-EMO/SHAP results, and PCP decision support."),
               class_="app-heading"),
        ui.span("●  Preview ready", class_="status-pill"),
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
        ui.p("⚠️ indicates the standard workflow when running the app from the beginning.", class_="wf-symbol"),
        ui.p("💾 indicates that you can skip earlier steps by loading a saved file to visualize results or continue the analysis.", class_="wf-symbol"),
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
            ui.div(ui.h2("Step 2: Emulator Training"), ui.p("Load a trained emulator and predict outputs. Training is disabled."), class_="step-heading"),
            ui.layout_columns(
                card("Training parameters", ui.tags.fieldset(
                    ui.input_numeric("emu_epochs", "Epochs", 1000, min=1, max=50000),
                    ui.input_numeric("emu_folds", "CV folds", 2, min=2, max=20),
                    ui.input_numeric("emu_test", "Test size", .10, min=.05, max=.40, step=.05),
                    disabled("Train Emulator"), disabled("Stop Training"), disabled=True, class_="frozen-controls")),
                card("Load existing emulator", upload("emulator_zip", "Load emulator_model.zip or emulator_model.7z", [".zip", ".7z"]),
                 ui.output_text("emulator_status")), col_widths=(4,8)),
            card("Saved emulator parity plot", ui.output_ui("emulator_plot_view")),
            ui.output_ui("prediction_section"),
            value="emulator"),
        ui.nav_panel("TS-EMO Analysis",
            ui.div(ui.h2("Step 3: TS-EMO Analysis"), ui.p("Inspect saved optimisation data; running new TS-EMO studies is disabled."), class_="step-heading"),
             ui.layout_columns(
                card("TS-EMO controls (preview unavailable)", ui.tags.fieldset(
                     ui.input_select("ts_mode", "Run mode", ["Fresh Start", "Continue Previous Run"]),
                     ui.input_select("ts_design", "Initial design", ["LHS", "SOBOL"]),
                     ui.input_numeric("ts_initial", "Initial steps", 3), ui.input_numeric("ts_steps", "Total steps", 10),
                     ui.input_checkbox("ts_auto", "Auto-stop on HV convergence", False),
                     ui.layout_columns(ui.input_numeric("ts_window", "Window", 5), ui.input_numeric("ts_tol", "Tol.", .5)),
                     ui.input_checkbox("ts_target", "Terminate on HV target", False),
                     ui.layout_columns(ui.input_numeric("ts_percent", "Target (%)", 80), ui.input_numeric("ts_min", "Min steps", 5)),
                     ui.input_checkbox("ts_time", "Terminate on total runtime", False), ui.input_numeric("ts_minutes", "Limit (min)", 4),
                     disabled("Run TS-EMO"), disabled("Stop Run"), disabled=True, class_="frozen-controls")),
                card("Load previous HV results", disabled("Load Continue File"), ui.input_file("ts_files", "💾 Select HV coverage CSV", accept=[".csv"], multiple=False, width="100%"),
                     ui.output_text("ts_files_filename"), ui.output_text("ts_load_status")), col_widths=(6,6)),
             ui.p("Status: Waiting — optimisation disabled in preview."),
             ui.tags.progress(value="0", max="100", class_="preview-progress"), ui.p("0/0     Elapsed: —     ETA: —"),
             ui.output_text("final_hv_summary"),
            card("HV coverage", ui.layout_columns(
                ui.div(ui.h4("Hypervolume coverage plot"), ui.output_ui("hv_plot_view")),
                ui.div(ui.h4("HV coverage data"), ui.output_data_frame("hv_preview")),
                col_widths=(6,6))),
            card("Pareto solutions", disabled("Show Pareto solutions")),
            value="tsemo"),
        ui.nav_panel("SHAP Analysis",
            ui.div(ui.h2("Step 4: SHAP Analysis"), ui.p("Inspect saved SHAP explanations. Attribution generation is disabled."), class_="step-heading"),
            ui.layout_columns(
                card("SHAP computation (preview unavailable)",
                     ui.tags.fieldset(ui.input_numeric("shap_samples", "SHAP samples", 10, min=1, max=10000),
                     ui.input_numeric("shap_batch", "Batch size", 5, min=1, max=100),
                     disabled("Run SHAP"), disabled("Stop SHAP"), disabled=True, class_="frozen-controls")),
                card("Load previously generated SHAP data",
                     upload("shap_file", "SHAP master Excel workbook", [".xlsx"]),
                     ui.p("Saved workbook loads automatically.", class_="hint")),
                col_widths=(5,7)),
            card("Customize plot order",
                 ui.p("Drag objectives and input variables to reorder the plots."),
                 ui.layout_columns(ui.div(ui.h4("Objectives — drag to reorder"), ui.output_ui("shap_objective_drag")),
                                   ui.div(ui.h4("Inputs — drag to reorder"), ui.output_ui("shap_variable_drag")), col_widths=(6,6)),
                 ui.output_text("shap_status")),
             ui.p("Status: Waiting — SHAP computation disabled in preview."),
             ui.tags.progress(value="0", max="100", class_="preview-progress"), ui.p("0/0     Elapsed: —     ETA: —"),
            card("SHAP panel", ui.output_ui("shap_panel_view")),
            value="shap"),
        ui.nav_panel("PCPs",
            ui.div(ui.h2("Step 5: PCPs"), ui.p("Explore Pareto candidates, apply objective bounds and select representative optimal solutions."), class_="step-heading"),
            card("Run a new PCP analysis",
                 upload("pcp_file", "Load Pareto CSV", [".csv"]),
                 ui.input_numeric("pcp_n", "Selected points", 5, min=1, max=10000),
                 ui.output_text("pcp_source_status"),
                 ui.input_action_button("pcp_run", "Run PCP Analysis", class_="btn-action"),
                 ui.output_text("pcp_status")),
            card("PCP plot",
                ui.input_selectize("pcp_legend", "Legend categoricals", choices=[], multiple=True, options={"plugins": ["remove_button"], "hideSelected": True}),
                ui.output_ui("pcp_plot_view"),
                ui.div(ui.input_action_button("utopia_open", "View utopia point", class_="btn-option"),
                       ui.input_action_button("pcp_settings_open", "Objective bounds & display settings", class_="btn-option"),
                       ui.input_action_button("pcp_size_open", "Plot dimensions", class_="btn-option"),
                       style="display:flex;gap:12px;flex-wrap:wrap;margin-bottom:12px;"),
                ui.download_button("pcp_download_plot", "Save Plot (PNG)", class_="btn-other")),
            card("Optimal Pareto solutions",
                ui.div(
                    ui.div(ui.input_checkbox("pcp_constrained", "Constrained solutions", False), ui.input_action_button("constraints_open", "!", title="View failed solutions", style="border-radius:50%;background:#e9edf2;color:#43566e;font-weight:bold;width:28px;height:28px;padding:0;"), style="display:flex;align-items:center;gap:8px;"),
                    class_="pcp-toolbar", style="display:flex;align-items:center;justify-content:flex-start;flex-wrap:wrap;gap:18px;"),
                ui.output_ui("pcp_filters_ui"), ui.output_text("pcp_constraints_status"),
                ui.output_data_frame("pcp_preview"),
                ui.div(ui.download_button("pcp_download_selected", "Save selected CSV", class_="btn-other"), ui.download_button("pcp_download_constrained", "Save constrained CSV", class_="btn-other"), style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px;")),
            value="pcp"),
        id="main_tab"),
    ui.tags.style(".pcp-toolbar .shiny-input-container{width:auto!important;margin:0!important}.pcp-toolbar .form-group,.pcp-toolbar .checkbox{margin:0!important}.pcp-toolbar .form-check{margin:0!important;min-height:0}.pcp-toolbar label{margin-bottom:0}.pcp-toolbar>div{flex:0 0 auto}.uploaded-names{font-size:.85rem;color:#475569;margin-top:4px;overflow-wrap:anywhere}.pcp-tip strong{display:block;text-align:center}.pcp-tip-columns{display:grid;grid-template-columns:1fr 1fr;gap:18px;white-space:pre-wrap}.pcp-tip-columns b{display:block;margin-bottom:5px}"),
    ui.tags.script("""
    function pcpTip(box,line,e){
      const tip=box.querySelector('.pcp-tip'), r=box.getBoundingClientRect();
      const data=JSON.parse(box.dataset.labels)[Number(line.id.slice(7))];
      tip.replaceChildren(); const title=document.createElement('strong'); title.textContent=data.title.replace(/^Solution\\s+/, ''); title.className='pcp-solution-chip'; title.setAttribute('aria-label',data.title); tip.append(title);
      const cols=document.createElement('div'); cols.className='pcp-tip-columns';
      for(const [heading, values] of [['Inputs',data.inputs],['Outputs',data.outputs]]){const col=document.createElement('div'), h=document.createElement('b');h.textContent=heading;col.append(h,document.createTextNode(values.join('\\n')));cols.append(col);}tip.append(cols);
      tip.style.display='block'; tip.style.left=Math.max(0,Math.min(e.clientX-r.left+14,box.clientWidth-tip.offsetWidth))+'px'; tip.style.top=Math.max(20,e.clientY-r.top+12)+'px';
    }
    function leader(box){const tip=box.querySelector('.pcp-tip'), l=box.querySelector('.pcp-leader'); const x=+box.dataset.ax,y=+box.dataset.ay, dx=tip.offsetLeft+tip.offsetWidth/2-x,dy=tip.offsetTop+tip.offsetHeight/2-y; Object.assign(l.style,{display:'block',left:x+'px',top:y+'px',width:Math.hypot(dx,dy)+'px',transform:'rotate('+Math.atan2(dy,dx)+'rad)'});}
    document.addEventListener('pointerover',e=>{const line=e.target.closest('[id^=pcpline]');if(!line)return;const box=line.closest('.pcp-interactive');if(box&&!box.dataset.pinned)pcpTip(box,line,e);});
    document.addEventListener('pointerout',e=>{const line=e.target.closest('[id^=pcpline]');if(!line)return;const box=line.closest('.pcp-interactive');if(box&&!box.dataset.pinned)box.querySelector('.pcp-tip').style.display='none';});
    document.addEventListener('click',e=>{const line=e.target.closest('[id^=pcpline]');if(!line)return;const box=line.closest('.pcp-interactive'),tip=box.querySelector('.pcp-tip');if(box.dataset.pinned===line.id){delete box.dataset.pinned;tip.style.display='none';tip.style.pointerEvents='none';box.querySelector('.pcp-leader').style.display='none';return;}pcpTip(box,line,e);box.dataset.pinned=line.id;const r=box.getBoundingClientRect();box.dataset.ax=e.clientX-r.left;box.dataset.ay=e.clientY-r.top;tip.style.pointerEvents='auto';leader(box);});
    document.addEventListener('pointerdown',e=>{const tip=e.target.closest('.pcp-tip');if(!tip)return;const box=tip.closest('.pcp-interactive');if(!box.dataset.pinned)return;e.preventDefault();const x=e.clientX-tip.offsetLeft,y=e.clientY-tip.offsetTop;tip.setPointerCapture(e.pointerId);tip.onpointermove=ev=>{tip.style.left=Math.max(0,Math.min(box.clientWidth-tip.offsetWidth,ev.clientX-x))+'px';tip.style.top=Math.max(20,ev.clientY-y)+'px';leader(box);};tip.onpointerup=()=>{tip.onpointermove=null;};});
    function uploadColors(){document.querySelectorAll('.shiny-file-input-progress .progress-bar').forEach(b=>{const t=b.textContent.toLowerCase();const done=parseFloat(b.style.width)>=100; b.style.backgroundColor=/error|fail/.test(t)?'#c62828':done?'#198754':'';});}
    new MutationObserver(uploadColors).observe(document.documentElement,{subtree:true,childList:true,characterData:true});
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
    ui.tags.style('\n:root {--ink:#182b46;--muted:#52647b;--border:#dce5ef;--surface:#fff;--bs-body-font-family:Inter, "Segoe UI",system-ui,sans-serif;--bs-body-color:var(--ink);--bs-body-bg:#f1f5f9;}\nbody {font-family:var(--bs-body-font-family);background:#f1f5f9;color:var(--ink);font-size:17px;line-height:1.55;}\n.app-shell {max-width:1560px;padding:24px clamp(12px,3vw,40px) 64px;}\n.top-header {background:#173451;border:0;border-radius:18px;padding:28px 30px;box-shadow:0 8px 24px #17345116;}\n.app-heading h1 {color:white;font-size:clamp(23px,2.4vw,32px);letter-spacing:-.035em;}\n.app-heading p {color:#d5e4f3;max-width:850px;font-size:16px;}\n.status-pill {background:#e4f7ec;color:#16633d;border-radius:99px;padding:7px 14px;font-size:14px;}\n.workflow-graphic {background:white;border:1px solid var(--border);border-radius:16px;padding:22px 26px;}\n.wf-diagram {min-height:140px;gap:18px;justify-content:center;}\n.wf-step {background:#f3f6fa;border:1px solid #dce5ef;box-shadow:none;border-radius:13px;min-height:52px;min-width:135px;font-size:16px;}\n.wf-active {background:#e8f2ff;border-color:#a7c8ef;color:#194c83;}\n.wf-num {border-radius:50%;background:#dce6f1;color:#385775;font-size:17px;min-width:30px;height:30px;}\n.wf-num-active {background:#2368ae;color:white;}\n.wf-guide {background:#f4f8fd;border-left:3px solid #3978bd;border-radius:0 8px 8px 0;padding:12px 16px;font-size:16px;}\n.wf-symbol {font-size:15px;color:var(--muted);margin:5px 0;}\n#main_tab {gap:6px;flex-wrap:wrap;}\n.nav-tabs {border:0!important;background:#eaf0f7!important;padding:8px!important;border-radius:12px;gap:6px;}\n.nav-tabs .nav-link {border:0!important;border-radius:8px!important;background:transparent!important;color:#415875!important;font-weight:650;padding:10px 16px;}\n.nav-tabs .nav-link.active {background:#245d99!important;color:white!important;box-shadow:0 2px 5px #17345120;}\n.nav-tabs .nav-link:not(.active):hover {background:#dce7f4!important;}\n.step-heading {background:transparent;border:0;border-radius:0;padding:14px 4px 16px;margin-bottom:8px;}\n.step-heading h2 {font-size:27px;letter-spacing:-.025em;color:#173451;}\n.step-heading p {font-size:16px;color:var(--muted);}\n.section-card {border:1px solid var(--border);border-radius:15px;box-shadow:0 3px 12px #19334e08;padding:22px;margin-bottom:16px;min-width:0;}\n.card-title {font-size:19px;letter-spacing:-.015em;color:#173451;margin-bottom:16px;padding-bottom:10px;border-bottom:1px solid #e8edf4;}\n.section-card h4 {font-size:16px;font-weight:650;color:#435a76;}\n.btn,.btn-locked {border-radius:9px!important;font-weight:600;font-size:16px;min-height:38px;padding:8px 15px;transition:filter .15s,box-shadow .15s;}\n.btn:not(:disabled):hover {filter:brightness(.94);box-shadow:0 2px 6px #172b4620;}\n.btn:focus-visible {outline:3px solid #8db9ec;outline-offset:2px;}\n.btn-locked,fieldset:disabled .btn {background:#e9edf2!important;color:#68778a!important;border:1px solid #d5dee8!important;box-shadow:none;opacity:1;}\n.floating-help,#constraints_open {border-radius:50%!important;min-height:0;}\n.form-control,.form-select,.selectize-input {border:1px solid #cbd7e4!important;border-radius:8px!important;box-shadow:none!important;min-height:38px;}\n.form-label,.control-label {font-weight:600;color:#43566e;font-size:15px;}\n.upload-wrap {padding:12px 14px;background:#f7f9fc;border:1px dashed #c9d7e6;border-radius:10px;margin-bottom:12px;}\n.upload-wrap .shiny-input-container {margin-bottom:4px;}\n.shiny-file-input-progress {border-radius:99px;overflow:hidden;}\n[id$="_filename"] {font-size:14px;color:#52647b;overflow-wrap:anywhere;}\n[id$="_status"]:not(:empty) {font-size:15px;color:#405875;background:#f1f5fa;border-radius:8px;padding:8px 12px;margin-top:8px;overflow-wrap:anywhere;}\nshiny-data-frame {font-size:15px;--shiny-datagrid-header-bg:#eaf1f9;}\nshiny-data-frame table, .section-card table {font-variant-numeric:tabular-nums;}\nshiny-data-frame th,.section-card th {background:#eaf1f9!important;color:#294866!important;font-weight:650;}\nshiny-data-frame td,.section-card td {border-color:#e6edf5!important;}\n.modal-content {border:1px solid #a9c1db!important;border-radius:18px!important;box-shadow:0 18px 65px #172b4630!important;background:white;}\n.modal-header {padding:22px 24px;border-bottom:1px solid #e7edf4;}\n.modal-title {font-size:22px;color:#173451;letter-spacing:-.02em;}\n.modal-body {padding:24px;}\n.modal-footer {padding:16px 24px;border-top:1px solid #e7edf4;gap:8px;}\n.pcp-tip {box-shadow:0 8px 24px #17345122;border-radius:12px!important;}\n[data-shap-sort] {display:flex;flex-wrap:wrap;gap:8px;align-content:flex-start;}\n[data-shap-sort] > div {margin:0!important;background:#edf4fc!important;border-color:#d0e0f2!important;color:#294866;font-size:15px;}\n[data-shap-sort] > div:hover {background:#dceafa!important;}\n@media(max-width:700px){.app-shell{padding:12px 10px 40px}.top-header{padding:20px 64px 20px 18px}.section-card{padding:15px}.workflow-graphic{padding:16px}.nav-tabs .nav-link{padding:8px 10px;font-size:15px}.wf-diagram{justify-content:start}}\n'),
    ui.tags.style('body {--bs-body-font-size:17px;} .form-control,.form-select,.selectize-input,.selectize-dropdown {font-size:16px!important;} #workflow_help.floating-help {position:fixed!important;top:18px!important;right:18px!important;bottom:auto!important;width:54px!important;height:54px!important;min-height:54px!important;padding:0!important;display:flex;align-items:center;justify-content:center;font-size:27px!important;font-weight:800;color:white!important;background:#d62828!important;border:3px solid white!important;border-radius:50%!important;box-shadow:0 4px 15px #0003!important;z-index:1050;}'),
    ui.tags.script("\n    document.addEventListener('click', async e => {\n      const button=e.target.closest('#pcp_download_plot');\n      if(!button)return;\n      const box=document.querySelector('.pcp-interactive'), tip=box?.querySelector('.pcp-tip'), svg=box?.querySelector('svg');\n      if(!svg||!tip||getComputedStyle(tip).display==='none')return;\n      e.preventDefault();e.stopImmediatePropagation();\n      try {\n        const r=svg.getBoundingClientRect(), t=tip.getBoundingClientRect(), b=box.getBoundingClientRect();\n        const left=Math.min(0,t.left-r.left), top=Math.min(0,t.top-r.top-20), right=Math.max(r.width,t.right-r.left), bottom=Math.max(r.height,t.bottom-r.top);\n        const scale=2.5, canvas=document.createElement('canvas');canvas.width=Math.ceil((right-left)*scale);canvas.height=Math.ceil((bottom-top)*scale);\n        const ctx=canvas.getContext('2d');ctx.scale(scale,scale);ctx.translate(-left,-top);ctx.fillStyle='white';ctx.fillRect(left,top,right-left,bottom-top);\n        const clone=svg.cloneNode(true);clone.setAttribute('xmlns','http://www.w3.org/2000/svg');clone.setAttribute('width',r.width);clone.setAttribute('height',r.height);clone.style.width=r.width+'px';clone.style.height=r.height+'px';\n        const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml;charset=utf-8'}));\n        try {const img=new Image();img.src=url;await img.decode();ctx.drawImage(img,0,0,r.width,r.height);} finally {URL.revokeObjectURL(url);}\n        const x=t.left-r.left,y=t.top-r.top;\n        if(box.dataset.pinned){ctx.strokeStyle='#475569';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(b.left-r.left+Number(box.dataset.ax),b.top-r.top+Number(box.dataset.ay));ctx.lineTo(x+t.width/2,y+t.height/2);ctx.stroke();}\n        ctx.fillStyle='white';ctx.strokeStyle='#93b4d8';ctx.beginPath();if(ctx.roundRect)ctx.roundRect(x,y,t.width,t.height,10);else ctx.rect(x,y,t.width,t.height);ctx.fill();ctx.stroke();\n        const chip=tip.querySelector('.pcp-solution-chip');if(chip){const q=chip.getBoundingClientRect();ctx.fillStyle='#245d99';ctx.beginPath();ctx.ellipse(q.left-r.left+q.width/2,q.top-r.top+q.height/2,q.width/2,q.height/2,0,0,2*Math.PI);ctx.fill();}\n        const walker=document.createTreeWalker(tip,NodeFilter.SHOW_TEXT);let node;\n        while(node=walker.nextNode()){\n          const style=getComputedStyle(node.parentElement);ctx.font=`${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;ctx.fillStyle=style.color;ctx.textBaseline='middle';\n          // Draw each rendered line at its actual browser position, including wrapping.\n          let start=0, prev=null, str='';const range=document.createRange();\n          const flush=()=>{if(prev&&str)ctx.fillText(str,prev.left-r.left,prev.top-r.top+prev.height/2);};\n          for(let i=0;i<node.length;i++){range.setStart(node,i);range.setEnd(node,i+1);const q=range.getBoundingClientRect();if(!q.width){flush();prev=null;str='';continue;}if(prev&&Math.abs(q.top-prev.top)>1){flush();prev=null;str='';}if(!prev)prev=q;str+=node.textContent[i];}flush();\n        }\n        canvas.toBlob(blob=>{if(!blob)return;const u=URL.createObjectURL(blob),a=document.createElement('a');a.href=u;a.download='pcp_plot.png';a.click();setTimeout(()=>URL.revokeObjectURL(u),1000);},'image/png');\n      } catch(err){console.error(err);alert('Could not export the annotated plot. Please retry.');}\n    },true);\n"),
    ui.tags.style("""
    .pcp-tip .pcp-solution-chip {position:absolute;top:-17px;left:50%;transform:translateX(-50%);min-width:34px;height:34px;padding:0 8px;display:flex;align-items:center;justify-content:center;border-radius:99px;background:#245d99;color:white;border:2px solid white;font-size:15px;font-weight:750;box-shadow:0 2px 5px #17345133;}
    .pcp-tip {padding-top:24px!important;}
    #pcp_legend + .selectize-control .item {border-radius:99px!important;background:#e8f2ff!important;color:#245d99!important;padding:4px 9px!important;}
    #main_tab .nav-link, .main-tabs .nav-tabs .nav-link {font-size:19px!important;}
    @media(max-width:700px){#main_tab .nav-link,.main-tabs .nav-tabs .nav-link{font-size:17px!important;}}
    #constraints_open {background:#e9edf2!important;color:#43566e!important;border:1px solid #cbd5e1!important;width:28px!important;height:28px!important;min-height:28px!important;padding:0!important;font-size:20px!important;font-family:var(--bs-body-font-family)!important;}
    #workflow_help.floating-help {font-family:var(--bs-body-font-family)!important;}
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
    pcp_selected_state = reactive.value(None)
    pcp_message = reactive.value("Waiting for PCP analysis.")
    pcp_bound_settings = reactive.value({})
    def register_filename(id):
        @output(id=id + "_filename")
        @render.text
        def filename():
            files = input[id]()
            return ", ".join(f["name"] for f in files) if files else ""
    for upload_id in ("experimental", "domain_file", "emulator_zip", "ts_files", "shap_file", "pcp_file"):
        register_filename(upload_id)
    pcp_padding = reactive.value((.35, .275))
    pcp_size = reactive.value((19, 9))
    pcp_decimal_settings = reactive.value({})

    @reactive.calc
    def data():
        return read_csv(input.experimental())

    @reactive.calc
    def domain_loaded():
        return read_csv(input.domain_file())

    hv_state = reactive.value(None)
    pareto_state = reactive.value(None)
    ts_message = reactive.value("Select an HV coverage CSV file.")

    @reactive.effect
    @reactive.event(input.ts_files)
    def _load_ts_results():
        try:
            files = input.ts_files() or []
            found = {}
            for f in files:
                df = pd.read_csv(f["datapath"])
                name = f["name"].lower()
                kind = "hv" if any("hv_coverage" in str(c).lower() for c in df.columns) else None
                if kind is None: raise ValueError(f"Unrecognized result file: {f['name']}")
                if kind in found: raise ValueError(f"Select only one {kind} file at a time.")
                found[kind] = df
            if not found: raise ValueError("Select at least one CSV file.")
            if "hv" in found: hv_state.set(found["hv"])
            ts_message.set("Loaded: " + ", ".join(found))
        except Exception as exc:
            ts_message.set(str(exc))

    @render.text
    def ts_load_status(): return ts_message.get()

    @reactive.calc
    def hv(): return hv_state.get()

    @reactive.calc
    def pcp():
        return pareto_state.get()

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
            return "Emulator archive loaded successfully."
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

    prediction_model = reactive.value(None)
    prediction_data = reactive.value(None)
    prediction_message = reactive.value("")

    @reactive.effect
    def _load_prediction():
        info = input.emulator_zip()
        raw, cfg = data(), domain()
        prediction_model.set(None)
        prediction_data.set(None)
        if not info:
            prediction_message.set("")
            return
        if raw is None or cfg is None:
            prediction_message.set("Load experimental data and create or load a domain in Tab 1.")
            return
        try:
            digest = hashlib.sha256(Path(info[0]["datapath"]).read_bytes()).hexdigest()
            if digest not in ('5796e439761554954e427ab30722ddb2083485be1a129411904578928041b1f1', 'bf2dddaa296544910e7f2db94301086dca22648eaf2a55cc563c05ff0bb3b197'):
                raise ValueError("Upload an approved emulator archive (ZIP or 7z).")
            clean = pd.DataFrame()
            specs = []
            for _, row in cfg.iterrows():
                role = str(row.get("Role", "")).lower()
                if role == "ignore": continue
                name = str(row.get("Internal Name", row.get("Display Name", "")))
                original = str(row.get("Original Column", ""))
                if original not in raw.columns:
                    original = next((c for c in raw.columns if str(c).split("_-_")[0] == name), name)
                if original not in raw.columns:
                    raise ValueError(f"Cannot match domain variable {name} to experimental data.")
                label = display_name(original)
                categorical = str(row.get("Type", "")).lower() == "categorical"
                objective = role in ("maximize", "minimize")
                if categorical and not objective:
                    clean[name] = raw[original].astype(str)
                    levels = str(row.get("Levels", "")).strip()
                    levels = [x.strip() for x in levels.split(",") if x.strip()] if levels and levels.lower() != "nan" else sorted(clean[name].unique())
                    specs.append((name,label,levels,None,None,0))
                else:
                    clean[name] = pd.to_numeric(raw[original], errors="raise")
                    lo, hi = row.get("Lower Bound", np.nan), row.get("Upper Bound", np.nan)
                    if pd.isna(lo) or pd.isna(hi):
                        nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?", str(row.get("Bounds", "")))
                        if len(nums)==2: lo,hi=map(float,nums)
                    lo = float(clean[name].min() if pd.isna(lo) else lo)
                    hi = float(clean[name].max() if pd.isna(hi) else hi)
                    if not objective: specs.append((name,label,None,lo,hi,int(row.get("Decimals",4))))
            with tempfile.TemporaryDirectory(prefix="shiny_emulator_") as tmp:
                root=Path(tmp)
                kind,names=inspect_emulator_archive()
                for n in names:
                    if not (root/n).resolve().is_relative_to(root.resolve()): raise ValueError("Unsafe archive path")
                if kind=="ZIP":
                    with zipfile.ZipFile(info[0]["datapath"]) as z: z.extractall(root)
                else:
                    import py7zr
                    with py7zr.SevenZipFile(info[0]["datapath"],mode="r") as z: z.extractall(root)
                predict, variables, outputs = load_direct_emulator(root)
                expected = {v["name"] for v in variables if not v["is_objective"]}
                if expected != {v[0] for v in specs} or not {v["name"] for v in outputs}.issubset(clean.columns):
                    raise ValueError("Tab 1 domain names do not match the saved emulator domain.")
                saved = {v["name"]: v for v in variables}
                specs = [(name, label, saved[name].get("levels"), *(saved[name].get("bounds", [None, None])), digits) for name,label,levels,lo,hi,digits in specs]
            prediction_model.set((predict,outputs,clean,specs,None))
            prediction_message.set("Emulator loaded. Edit inputs and click Predict Outputs.")
        except Exception as exc:
            prediction_message.set(f"Could not load emulator: {exc}")

    @render.ui
    def prediction_section():
        if not input.emulator_zip():
            return None
        if prediction_model.get() is None:
            return card("Predict outputs", ui.output_text("prediction_status"))
        return card("Predict outputs", ui.div(
            ui.div(ui.h4("Input conditions"), ui.output_ui("prediction_form"),
                   ui.input_action_button("predict_outputs", "Predict Outputs", class_="btn-action")),
            ui.div(ui.h4("Predicted outputs", style="text-align:center;"),
                   ui.output_text("prediction_status"), ui.output_ui("prediction_results"),
                   style="padding:16px; background:#f8fafc; border:1px solid #dbe3ed; border-radius:10px; align-self:start;"),
            ))

    @render.ui
    def prediction_form():
        model=prediction_model.get()
        if model is None: return ui.p("Load a valid emulator archive first.")
        _,_,clean,specs,_=model
        controls=[]
        for j,(name,label,levels,lo,hi,digits) in enumerate(specs):
            default=clean[name].iloc[0]
            if levels is not None: controls.append(ui.input_select(f"pred_{j}",label,levels,selected=str(default)))
            else: controls.append(ui.input_numeric(f"pred_{j}",label,float(default),step=10**(-min(6,max(0,digits)))))
        return ui.div(*controls, style="display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;")

    @reactive.effect
    @reactive.event(input.predict_outputs)
    def _predict():
        prediction_data.set(None)
        try:
            model=prediction_model.get()
            if model is None: raise ValueError("Load a valid emulator archive first.")
            emul,dom,clean,specs,DataSet=model
            vals={}; mask=np.ones(len(clean),dtype=bool); warnings=[]
            for j,(name,label,levels,lo,hi,digits) in enumerate(specs):
                value=input[f"pred_{j}"]()
                if levels is None:
                    value=float(value)
                    if not np.isfinite(value): raise ValueError(f"Invalid value for {label}")
                    if not lo<=value<=hi: warnings.append(label)
                    mask &= np.isclose(clean[name].astype(float),value,rtol=0,atol=1e-10)
                else: mask &= clean[name].astype(str).eq(str(value)).to_numpy()
                vals[name]=value
            pred=emul(vals)
            rows=[]
            for var in dom:
                name=var["name"]
                actual=clean.loc[mask,name]
                rows.append({"Objective":var["description"],"Original":name,"Predicted":float(pred[name]),"Actual":float(actual.iloc[0]) if len(actual) else np.nan,"Direction":"Maximize" if var["maximize"] else "Minimize"})
            prediction_data.set(pd.DataFrame(rows))
            prediction_message.set("Prediction complete." + (" Outside domain: "+", ".join(warnings) if warnings else ""))
        except Exception as exc: prediction_message.set(f"Prediction failed: {exc}")

    @render.text
    def prediction_status(): return prediction_message.get()

    @render.ui
    def prediction_results():
        df = prediction_data.get()
        if df is None or df.empty:
            return ui.p("Enter input conditions and click Predict Outputs.", style="text-align:center;color:#64748b;")
        def fmt(v, digits):
            return "—" if pd.isna(v) else f"{v:.{digits}f}"
        tiles = []
        for _, row in df.iterrows():
            digits = pcp_digits(row["Objective"])
            tiles.append(ui.div(
                ui.div(display_name(row["Objective"]), style="font-weight:600;color:#334155;overflow-wrap:anywhere;"),
                ui.div(fmt(row["Predicted"], digits), style="font-size:1.65rem;font-weight:700;color:#1d4ed8;font-variant-numeric:tabular-nums;"),
                ui.div("Predicted", style="font-size:.8rem;color:#64748b;"),
                ui.div("Actual: " + fmt(row["Actual"], digits), style="margin-top:8px;font-variant-numeric:tabular-nums;"),
                ui.div(str(row["Direction"]), style="font-size:.8rem;color:#64748b;"),
                style="background:white;border:1px solid #dbe3ed;border-radius:8px;padding:16px;text-align:center;min-width:0;"))
        return ui.div(*tiles, style=f"display:grid;grid-template-columns:repeat({len(tiles)},minmax(160px,1fr));gap:12px;margin:16px auto 0;overflow-x:auto;")

    @render.text
    def final_hv_summary():
        df=hv()
        if df is None or df.empty: return "Final HV coverage: —"
        col=next((c for c in df.columns if "hv_coverage" in c.lower()),None)
        return f"Final HV coverage: {float(df[col].iloc[-1]):.2f}" if col else "Final HV coverage: —"

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
        from matplotlib.ticker import PercentFormatter
        fig, ax = plt.subplots(figsize=(10, 5.5))
        y = pd.to_numeric(df[ycol], errors="coerce")
        ax.plot(pd.to_numeric(df[xcol], errors="coerce"), y, linewidth=2, marker="o", markersize=5)
        ax.set(xlabel="TS-EMO Iterations", ylabel="Hypervolume Coverage (%)", title="Loaded TS-EMO Hypervolume Coverage")
        ax.title.set_fontsize(19)
        ax.xaxis.label.set_size(17); ax.yaxis.label.set_size(17)
        ax.tick_params(labelsize=15)
        ax.grid(True, linestyle="--", alpha=.7)
        ax.yaxis.set_major_formatter(PercentFormatter(decimals=2))
        ax.set_ylim(0, min(100, max(5, float(y.max())*1.1)))
        fig.tight_layout()
        return image_from_figure(fig)

    def shap_drag_list(items, input_id):
        return ui.tags.div(*[ui.tags.div("☰  " + display_name(name), draggable="true", **{"data-value": str(name)},
                         style="cursor:grab;padding:9px 12px;margin:5px 0;border:1px solid #cbd5e1;border-radius:999px;background:#f8fafc;")
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
    @reactive.event(input.shap_file)
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
            ax.set_yticks(range(len(vars_found)), [display_name(v) for v in vars_found])
            ax.invert_yaxis()
            ax.axvline(0, color="#64748b", lw=1)
            ax.grid(alpha=.13)
            ax.set_xlabel(f"SHAP value – impact on {display_name(name)}", fontsize=10, labelpad=12)
            ax.tick_params(axis="x", labelsize=9)
            formatter=ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-3,4))
            ax.xaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
            ax.xaxis.get_offset_text().set_fontsize(9)
            ax.set_title(display_name(name))
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
        return render.DataGrid(display_table(clean_variable_view(rounded_table(d))), filters=False)

    @render.data_frame
    def variable_preview():
        v=variables()
        if v is None: return None
        v=v.copy()
        for c in ("Lower Bound", "Upper Bound"):
            v[c]=["" if pd.isna(x) else f"{x:.{int(d)}f}" for x,d in zip(v[c],v["Decimals"])]
        return render.DataGrid(clean_variable_view(v.drop(columns=["Original Column"], errors="ignore")), filters=False, selection_mode="row")

    @render.data_frame
    def domain_preview():
        d = domain()
        if d is None: return None
        if d is not None and "Original Column" in d.columns:
            d = d.drop(columns=["Original Column"])
        return render.DataGrid(display_table(clean_variable_view(rounded_table(d))), filters=False)

    @render.data_frame
    def hv_preview():
        view = table_view(hv()).copy()
        for c in view.select_dtypes(include="number"):
            view[c] = view[c].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
        if view.empty: return None
        return render.DataGrid(display_table(view), filters=False)

    @render.data_frame
    def shap_preview():
        if shap() is None: return None
        return render.DataGrid(display_table(rounded_table(shap())), filters=False)

    # PCP data selection, objective constraints, diversity selection, plotting and exports.
    @reactive.calc
    def pcp_source():
        return pcp()

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
    @reactive.event(input.pcp_file)
    def _pcp_use_file():
        pcp_selected_state.set(None)
        pareto_state.set(None)
        try:
            df = read_csv(input.pcp_file())
            if df is None or df.empty:
                raise ValueError("No Pareto points found.")
            pareto_state.set(df)
            pcp_message.set("Pareto file loaded. Existing objective bounds are retained; review them before running PCP.")
        except Exception as exc:
            pcp_message.set(f"Pareto upload failed: {exc}")
            ui.notification_show(f"Pareto upload failed: {exc}", type="error")

    @render.text
    def pcp_source_status():
        df=pcp_source()
        return "Available Pareto points: " + (str(len(df)) if df is not None else "—")

    @reactive.effect
    def _pcp_update_legend():
        df=pcp_source()
        if df is None: return
        categorical=[c for c in df.columns if (pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)) and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        ui.update_selectize("pcp_legend", choices=categorical, selected=categorical)

    def pcp_variable_row(col):
        settings = variables()
        if settings is None: return None
        key = display_name(col).casefold()
        for _, r in settings.iterrows():
            if any(display_name(r[k]).casefold() == key or display_name(str(r[k]).split("_-_")[0]).casefold() == key for k in ("Original Column", "Display Name")):
                return r
        return None

    def pcp_digits(col):
        r = pcp_variable_row(col)
        return int(r["Decimals"]) if r is not None else 4

    def pcp_label(col):
        r = pcp_variable_row(col)
        return display_name(r["Display Name"]) if r is not None else display_name(col)

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
                    col_widths=(6, 6)),
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
            ui.notification_show("Load a Pareto CSV in PCP first.",type="error")
            return
        eligible = pcp_filtered(df)
        if eligible.empty:
            pcp_selected_state.set(None)
            pcp_message.set(f"Loaded {len(df)} Pareto points, but none satisfy the current objective bounds. Review or clear bounds in Objective bounds & display settings.")
            return
        chosen=pcp_select(eligible,int(input.pcp_n() or 5))
        pcp_selected_state.set(chosen)
        pcp_message.set(f"PCP analysis complete: {len(chosen)} representative solutions selected from {len(df)} Pareto points.")

    @render.text
    def pcp_status():
        return pcp_message.get()

    @reactive.calc
    def pcp_selected():
        return pcp_selected_state.get()

    @render.text
    def pcp_constraints_status():
        df=pcp_selected()
        if df is None: return "Run PCP Analysis."
        valid=pcp_filtered(df)
        return f"{len(valid)} of {len(df)} selected solutions satisfy the current objective bounds."

    @render.ui
    def pcp_filters_ui():
        df=pcp_selected()
        if df is None: return ui.p("Categorical filters appear after selecting solutions.")
        categorical=[c for c in df.columns if (pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)) and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        return ui.div(*[ui.input_select(f"pcp_cat_{j}",display_name(c),
                choices={"__all__":"All",**{str(v):str(v) for v in df[c].dropna().unique()}},
                selected="__all__", width=f"{min(340,max(130, 9*max([len(str(c))]+[len(str(v)) for v in df[c].dropna().unique()])+55))}px") for j,c in enumerate(categorical)], style="display:flex;flex-wrap:wrap;justify-content:flex-start;gap:12px;")

    def pcp_table_data():
        df=pcp_selected()
        if df is None: return None
        df=pcp_filtered(df) if input.pcp_constrained() else df.copy()
        categorical=[c for c in df.columns if (pd.api.types.is_string_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)) and df[c].nunique(dropna=True)<=30 and not is_pcp_helper(c)]
        for j,c in enumerate(categorical):
            try: val=input[f"pcp_cat_{j}"]()
            except Exception: continue
            if val and val!="__all__": df=df[df[c].astype(str)==val]
        if "Utopia distance" not in df.columns:
            objectives = [c for c in pcp_objectives() if c in df.columns]
            if objectives:
                source = pcp_source()
                if source is None: source = df
                norm = np.empty((len(df), len(objectives)))
                for j, c in enumerate(objectives):
                    ref = pd.to_numeric(source[c] if c in source else df[c], errors="coerce")
                    vals = pd.to_numeric(df[c], errors="coerce")
                    lo, hi = ref.min(), ref.max()
                    z = np.clip((vals-lo)/(hi-lo), 0, 1) if hi > lo else np.full(len(df), .5)
                    if pcp_direction(c) == "Minimize": z = 1-z
                    norm[:, j] = np.nan_to_num(z, nan=.5)
                df["Utopia distance"] = np.linalg.norm(1-norm, axis=1)
        return df.drop(columns=[c for c in df.columns if is_pcp_helper(c) and c != "Row #" and c != "Utopia distance"], errors="ignore")

    def pcp_utopia_point():
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
            rows.append(ui.tags.tr(ui.tags.td(pcp_label(c)), ui.tags.td(f"{ideal:.{decimals}f}"),
                                   ui.tags.td(f"[{direction.lower()}]")))
        return ui.div(ui.strong("Utopia point"),
                      ui.tags.table(ui.tags.tbody(*rows), class_="table table-sm"),
                      class_="section-card")

    @reactive.effect
    @reactive.event(input.utopia_open)
    def _show_utopia():
        ui.modal_show(ui.modal(pcp_utopia_point(), title="Utopia point", easy_close=True))

    @reactive.effect
    @reactive.event(input.pcp_size_open)
    def _open_plot_size():
        w,h = pcp_size.get()
        ui.modal_show(ui.modal(ui.div(ui.input_slider("pcp_width", "Width (inches)", 8, 24, w, step=.5),
            ui.input_slider("pcp_height", "Height (inches)", 3, 12, h, step=.5),
            ui.input_slider("pcp_xpad", "Horizontal padding (axis units)", 0, 1, pcp_padding.get()[0], step=.05),
            ui.input_slider("pcp_ypad", "Vertical padding (normalized units)", 0, .5, pcp_padding.get()[1], step=.025), class_="pcp-size-controls"),
            title="PCP plot dimensions", footer=ui.div(ui.input_action_button("pcp_size_apply", "Apply"), class_="pcp-size-footer"), easy_close=True))

    @reactive.effect
    @reactive.event(input.pcp_size_apply)
    def _apply_plot_size():
        pcp_size.set((float(input.pcp_width()), float(input.pcp_height())))
        pcp_padding.set((float(input.pcp_xpad()), float(input.pcp_ypad())))
        ui.modal_remove()

    @reactive.effect
    @reactive.event(input.constraints_open)
    def _show_failures():
        df = pcp_selected()
        constraints = [(c,key,float(b)) for c in pcp_objectives()
            for key,b in pcp_bound_settings.get().get(c,{}).items()
            if key in ("low","high") and b is not None and df is not None and c in df.columns]
        rows = []
        for i, (_, row) in enumerate(df.iterrows() if df is not None else []):
            cells = []; failed = False
            for c,key,b in constraints:
                v = pd.to_numeric(row[c], errors="coerce")
                ok = pd.notna(v) and (v >= b if key == "low" else v <= b)
                failed |= not ok
                d = pcp_digits(c)
                op = ("≥" if ok else "<") if key == "low" else ("≤" if ok else ">")
                cells.append(ui.tags.td(ui.span("Missing" if pd.isna(v) else f"{v:.{d}f}", style=f"font-weight:700;color:{'#16803c' if ok else '#c62828'}"),
                    f" {op} {b:.{d}f}" if pd.notna(v) else f" (required {'≥' if key=='low' else '≤'} {b:.{d}f})"))
            if failed: rows.append(ui.tags.tr(ui.tags.td(str(row.get("Row #", i+1))), *cells))
        headers = ["Solution"] + [f"{pcp_label(c)} ({'minimum' if key=='low' else 'maximum'})" for c,key,b in constraints]
        content = ui.div(ui.tags.table(ui.tags.thead(ui.tags.tr(*[ui.tags.th(x) for x in headers])), ui.tags.tbody(*rows), class_="table", style="--bs-table-bg:white;--bs-table-striped-bg:white;"), style="overflow-x:auto;") if rows else ui.p("No selected solutions fail the current bounds." if df is not None else "No selected solutions loaded.")
        ui.modal_show(ui.modal(content, title="Constraint failures", size="xl", easy_close=True))

    @render.data_frame
    def pcp_preview():
        df=pcp_table_data()
        if df is None or df.empty: return None
        view = rounded_table(df)
        for col in pcp_objectives():
            if col in view.columns and col in df.columns:
                digits = pcp_digits(col)
                view[col] = pd.to_numeric(df[col].head(len(view)),errors="coerce").map(
                    lambda v: "" if pd.isna(v) else f"{v:.{digits}f}")
        styles = []
        full = pcp_selected()
        if full is not None and "Row #" in full and "Row #" in view:
            colors = pcp_solution_colors(full)
            lookup = dict(zip(full["Row #"], colors))
            for i, number in enumerate(df["Row #"].head(len(view))):
                styles.append({"rows":[i], "cols":[view.columns.get_loc("Row #")], "style":{"color":lookup.get(number,"#333"),"font-weight":"bold"}})
            view["Row #"] = df["Row #"].head(len(view)).map(str)
        return render.DataGrid(display_table(view),filters=False,styles=styles)

    def pcp_solution_colors(df):
        from matplotlib.colors import to_hex
        source = pcp_source()
        if source is None: source = df
        cols = [c for c in pcp_objectives() if c in df]
        norm = np.zeros((len(df),len(cols)))
        for j,c in enumerate(cols):
            ref = pd.to_numeric(source[c] if c in source else df[c],errors="coerce")
            lo,hi = ref.min(),ref.max()
            z = np.clip((pd.to_numeric(df[c],errors="coerce")-lo)/(hi-lo),0,1) if hi>lo else np.full(len(df),.5)
            if pcp_direction(c)=="Minimize": z=1-z
            norm[:,j]=np.nan_to_num(z,nan=.5)
        dist=np.linalg.norm(1-norm,axis=1)
        if not len(dist): return []
        t=(dist-dist.min())/(np.ptp(dist) or 1)
        return [to_hex((v,0,1-v)) for v in t]

    @render.ui
    def pcp_plot_view():
        df=pcp_selected()
        if df is None or df.empty: return ui.p("Run PCP Analysis or load a selected-solutions CSV to see the plot.")
        fig=pcp_figure(df)
        buf=io.StringIO()
        fig.savefig(buf, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg=buf.getvalue().split("<svg",1)[1]
        labels=[]
        for i, (_, row) in enumerate(df.iterrows()):
            entry = {"title": f"Solution {row.get('Row #', i+1)}", "inputs": [], "outputs": []}
            objectives = set(pcp_objectives())
            for c,v in row.items():
                if is_pcp_helper(c): continue
                value = f"{v:.{pcp_digits(c)}f}" if isinstance(v,(int,float,np.number)) and pd.notna(v) else str(v)
                entry["outputs" if c in objectives else "inputs"].append(f"{pcp_label(c)}: {value}")
            labels.append(entry)
        return ui.div(ui.p("Hover over a line to inspect it. Click to pin; click again to unpin."), ui.HTML("<svg"+svg), ui.div(class_="pcp-leader", style="display:none;position:absolute;height:1px;background:#475569;transform-origin:left center;pointer-events:none;z-index:9;"), ui.div(class_="pcp-tip", style="font-family:inherit;font-size:12px;text-align:center;line-height:1.5;cursor:move;display:none;position:absolute;z-index:10;background:white;border:1px solid #93b4d8;border-radius:8px;padding:12px;pointer-events:none;white-space:pre-wrap;max-width:90%;"), class_="pcp-interactive", style=f"position:relative;width:{pcp_size.get()[0]*80:.0f}px;max-width:100%;", **{"data-labels":json.dumps(labels)})

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
        fig,ax=plt.subplots(figsize=pcp_size.get())
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
        from matplotlib.lines import Line2D
        cats = [c for c in (input.pcp_legend() or []) if c in df.columns]
        groups = [tuple(str(row[c]) for c in cats) for _,row in df.iterrows()]
        unique = list(dict.fromkeys(groups))
        markers = ['o','s','^','d','v','>','<','p','h','X','*']
        styles = ['-','--',':','-.']
        for i,row in enumerate(values):
            g = unique.index(groups[i])
            ax.plot(x,row,color=color[i],alpha=.83,lw=1.6,gid=f"pcpline{i}", marker=markers[g%len(markers)] if cats else None, linestyle=styles[g%4] if cats else '-', markersize=5)
        if cats:
            ax.legend(handles=[Line2D([0],[0],color='#555',marker=markers[i%len(markers)],linestyle=styles[i%4],label=' + '.join(g)) for i,g in enumerate(unique)],loc="lower center",bbox_to_anchor=(.5,.015),ncol=min(4,len(unique)),fontsize=9,framealpha=.9)
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
        xp, yp = pcp_padding.get()
        ax.set_xlim(-xp,len(cols)-1+xp);ax.set_ylim(-yp,1+yp)
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
            "emulator": ("Emulator Training", [("Upload emulator_model.zip or .7z", bool(input.emulator_zip())), ("Predict outputs", prediction_data.get() is not None)], "Prediction uses direct CPU inference from an approved saved emulator; training is disabled."),
            "tsemo": ("TS-EMO Analysis", [("Load HV_coverage_data.csv", hv() is not None)], "The per_NB file uses normalised output units; it is not a separate optimisation run."),
            "shap": ("SHAP Analysis", [("Load SHAP_beeswarm_data.xlsx", shap() is not None)], "SHAP generation is unavailable."),
            "pcp": ("PCPs", [("Load Pareto CSV (original or per_NB)", pcp() is not None), ("Run PCP analysis", pcp_selected() is not None)], "Set objective bounds, select representative solutions, inspect parallel coordinates and export your selection."),
        }
        title, actions, note = steps.get(tab, steps["data"])
        content = [ui.h3(title)]
        for label, done in actions:
            content.append(ui.div(("✓  " if done else "○  ") + label, class_="help-done" if done else "help-pending"))
        content.append(ui.p(note, class_="hint"))
        await session.send_custom_message("noop", {}) if False else None
        ui.modal_show(ui.modal(*content, title="Workflow guide", easy_close=True, footer=ui.modal_button("Close")))


app = App(app_ui, server)
