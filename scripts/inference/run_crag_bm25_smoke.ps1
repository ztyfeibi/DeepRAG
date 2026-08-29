$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$srcDir = Join-Path $projectRoot "src"
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$modelDir = Join-Path $projectRoot "hf_models"
$dataPath = Join-Path $projectRoot "data\crag"
$sampleIdsFile = Join-Path $projectRoot "data\eval_subsets\crag_smoke_20.json"
$outputRoot = Join-Path $projectRoot "results\bm25_crag_smoke"

if (-not (Test-Path $pythonExe)) { throw "Python virtual environment not found: $pythonExe" }
if (-not (Test-Path $sampleIdsFile)) { throw "Fixed CRAG sample manifest not found: $sampleIdsFile" }

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:ES_HOST = "10.65.1.100"

Push-Location $srcDir
try {
    $runArgs = @(
        "--method", "srag-sftv2",
        "--model_name_or_path", $modelDir,
        "--remote_url", "http://10.65.1.110:8005/v1",
        "--follow_up_remote_url", "http://10.65.1.110:8005/v1",
        "--dataset", "hotpotqa",
        "--data_path", $dataPath,
        "--split", "test_with_id",
        "--sample_ids_file", $sampleIdsFile,
        "--record_retrieval_context",
        "--num_processes", "1",
        "--generate_max_length", "4096",
        "--fewshot", "8",
        "--output_dir", $outputRoot,
        "--temperature", "0.0",
        "--retriever", "BM25",
        "--retrieve_topk", "3",
        "--es_index_name", "crag"
    )
    & $pythonExe main.py @runArgs
    if ($LASTEXITCODE -ne 0) { throw "CRAG BM25 smoke inference failed with exit code $LASTEXITCODE" }
}
finally {
    Pop-Location
}
