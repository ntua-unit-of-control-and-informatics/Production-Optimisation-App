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

<h2>Windows Desktop Application (.exe)</h2>

<p>
  <a href="https://github.com/ntua-unit-of-control-and-informatics/Production-Optimisation-App/releases/download/v1.0.21/Production.Optimisation.Demo.exe">
    <img src="https://img.shields.io/badge/Download-EXE%20file-0078D4?style=for-the-badge&logo=windows&logoColor=white" alt="Download EXE file">
  </a>
</p>

The desktop application is distributed as a standalone Windows executable,
compiled using Nuitka. All required Python libraries and runtime dependencies
are bundled within the executable. No separate installation of Python, Anaconda
or additional packages is required.

<h3>Installation and usage</h3>

1. Click the **Download EXE file** button above.
2. Double-click the downloaded executable to launch the application.
3. Allow additional time for the initial startup.

During the first launch, the application extracts its bundled dependencies
into a local cache directory on the user's computer. Consequently, the initial
startup may take longer and require additional disk space. Subsequent launches
reuse the extracted dependencies, reducing startup time.

The cached files can be safely deleted while the application is closed.
They will be automatically recreated the next time the executable is launched.

<h3>System requirements</h3>

- Windows 10 or Windows 11 (64-bit).
- Sufficient available memory and disk space.
- No dedicated GPU required.

<h3>Release information</h3>

For release notes, available functionality and current limitations,
visit the <a href="https://github.com/ntua-unit-of-control-and-informatics/Production-Optimisation-App/releases">GitHub Releases</a> page.

### System Requirements

- Windows 10 or Windows 11 (64-bit).
- Sufficient available disk space for the executable and its extracted dependencies.
- Sufficient RAM for surrogate model training and SHAP analysis.
