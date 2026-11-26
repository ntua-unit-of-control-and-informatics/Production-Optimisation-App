<h1>Production Optimization App</h1>

The Production Optimization App provides an interactive workflow for experimental
data processing, surrogate modelling, multi-objective optimization, SHAP-based
interpretability and Pareto solution analysis.

The application is available in two formats: a Shiny web application and a
standalone Windows desktop application.

<h2> Shiny Web Application — v0.1 
<a href="https://01a212b3-bba7-53ad-74d3-72f73ff5b453.share.connect.posit.cloud">
<img src="shiny_logo.png" width="50" height="50" align="absmiddle" alt="Launch Shiny App">
</a> </h2>

The Shiny application can be accessed directly through a web browser without
requiring any local installation. Click the Shiny logo above to launch the app.

<h2> Windows Desktop Application (.exe) </h2>

The desktop application is distributed as a standalone Windows executable,
compiled using Nuitka. All required Python libraries and runtime dependencies
are bundled within the executable. No separate installation of Python, Anaconda
or additional packages is required.

During the first launch, the application extracts its bundled dependencies
into a local cache directory on the user's computer. Consequently, the initial
startup may take longer and require additional disk space. Subsequent launches
reuse the extracted dependencies, reducing startup time.

The cached files can be safely deleted while the application is closed.
They will be automatically recreated the next time the executable is launched.

### System Requirements

- Windows 10 or Windows 11 (64-bit).
- Sufficient available disk space for the executable and its extracted dependencies.
- Sufficient RAM for surrogate model training and SHAP analysis.
