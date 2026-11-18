# Full reproduction (Windows PowerShell). Every LLM call is cached, so a re-run is fast.
#   powershell -ExecutionPolicy Bypass -File scripts\run_all.ps1
$ErrorActionPreference = "Continue"
$py = ".\.venv\Scripts\python.exe"
$log = "results\run_all.log"
New-Item -ItemType Directory -Force results | Out-Null

function Log($text) { $text | Out-File -FilePath $log -Append -Encoding utf8 }

function Step($name, [string[]]$cmd) {
    Log "`n===== $(Get-Date -Format s) $name ====="
    & $py @cmd 2>&1 | ForEach-Object { Log "$_" }
    Log "===== $(Get-Date -Format s) end $name (exit $LASTEXITCODE) ====="
}

Step "tests" @("-m", "pytest")
Step "crosscheck" @("-m", "scripts.run_experiments", "--steps", "crosscheck", "--tag", "crosscheck")
if ($env:SKIP_GENERATE -ne "1") {
    Step "generate synthetic (qwen3.5:4b writes CVs)" @("-m", "scripts.generate_synthetic")
}
Step "4b synthetic+fairness" @("-m", "scripts.run_experiments", "--steps", "synthetic,fairness", "--model", "qwen3.5:4b", "--runs", "3", "--fairness-jobs", "J1")
Step "4b real data" @("-m", "scripts.run_experiments", "--steps", "real", "--model", "qwen3.5:4b", "--real-n", "20")
Step "0.8b synthetic (model-size comparison)" @("-m", "scripts.run_experiments", "--steps", "synthetic", "--model", "qwen3.5:0.8b", "--runs", "1", "--no-explain")
Step "notebook" @("-m", "scripts.build_notebook")
Step "execute notebook" @("-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace", "notebooks/results.ipynb")
Log "ALL DONE $(Get-Date -Format s)"
