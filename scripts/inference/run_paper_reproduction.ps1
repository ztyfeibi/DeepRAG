param(
    [string[]]$Dataset = @("hotpotqa", "2wiki", "webquestions", "popqa", "crag", "musique"),
    [int]$Sample = 1000,
    [int]$NumProcesses = 1,
    [switch]$EvaluateOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$srcDir = Join-Path $projectRoot "src"
$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$modelDir = Join-Path $projectRoot "hf_models"
$dataRoot = Join-Path $projectRoot "data"

if (-not (Test-Path $pythonExe)) {
    throw "Python virtual environment not found: $pythonExe"
}

$configs = @{
    "hotpotqa" = [PSCustomObject]@{ Loader = "hotpotqa"; DataDir = "hotpotqa"; Split = "dev"; EsIndex = "wiki" }
    "2wiki" = [PSCustomObject]@{ Loader = "2wikimultihopqa"; DataDir = "wikihop\data"; Split = "dev"; EsIndex = "wiki" }
    "webquestions" = [PSCustomObject]@{ Loader = "hotpotqa"; DataDir = "webquestions"; Split = "test_with_id"; EsIndex = "wiki" }
    "popqa" = [PSCustomObject]@{ Loader = "hotpotqa"; DataDir = "popqa"; Split = "test_with_id"; EsIndex = "wiki" }
    "crag" = [PSCustomObject]@{ Loader = "hotpotqa"; DataDir = "crag"; Split = "test_with_id"; EsIndex = "crag" }
    "musique" = [PSCustomObject]@{ Loader = "2wikimultihopqa"; DataDir = "musique"; Split = "dev"; EsIndex = "wiki" }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:ES_HOST = "10.65.1.100"

Push-Location $srcDir
try {
    foreach ($datasetName in $Dataset) {
        if (-not $configs.ContainsKey($datasetName)) {
            throw "Unsupported dataset: $datasetName"
        }

        $config = $configs[$datasetName]
        $outputRoot = Join-Path $projectRoot "results\paper_reproduction\$datasetName"
        $dataPath = Join-Path $dataRoot $config.DataDir

        if (-not $EvaluateOnly) {
            $runArgs = @(
                "--method", "srag-sftv2",
                "--model_name_or_path", $modelDir,
                "--remote_url", "http://10.65.1.110:8005/v1",
                "--follow_up_remote_url", "http://10.65.1.110:8005/v1",
                "--dataset", $config.Loader,
                "--data_path", $dataPath,
                "--split", $config.Split,
                "--sample", "$Sample",
                "--num_processes", "$NumProcesses",
                "--generate_max_length", "4096",
                "--fewshot", "8",
                "--output_dir", $outputRoot,
                "--temperature", "0.0",
                "--retriever", "BM25",
                "--retrieve_topk", "3",
                "--es_index_name", $config.EsIndex
            )

            & $pythonExe main.py @runArgs
            if ($LASTEXITCODE -ne 0) {
                throw "$datasetName inference failed with exit code $LASTEXITCODE"
            }
        }

        # Ignore interrupted directories that never produced the merged output.
        $completedRun = Get-ChildItem -Path $outputRoot -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                $outputFile = Join-Path $_.FullName "output.jsonl"
                if (Test-Path $outputFile) {
                    [PSCustomObject]@{
                        RunDir = $_.FullName
                        CompletedAt = (Get-Item $outputFile).LastWriteTime
                    }
                }
            } |
            Sort-Object CompletedAt -Descending |
            Select-Object -First 1

        if (-not $completedRun) {
            throw "$datasetName has no completed output.jsonl under $outputRoot"
        }

        & $pythonExe evaluate_ans.py --dir $completedRun.RunDir
        if ($LASTEXITCODE -ne 0) {
            throw "$datasetName evaluation failed with exit code $LASTEXITCODE"
        }

        Get-Content (Join-Path $completedRun.RunDir "metrics.json") -Raw -Encoding utf8
    }
}
finally {
    Pop-Location
}
