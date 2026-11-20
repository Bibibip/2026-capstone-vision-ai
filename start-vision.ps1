$ErrorActionPreference = "Stop"

$projectRoot = $PSScriptRoot
Set-Location $projectRoot

# Vision 환경변수
$env:VISION_AUDIO_MODE = "exhibition_rules"
$env:SILENT_SIGNAL_COORDINATE_MODE = "front"
$env:REAR_APPROACH_CAMERA_MODE = "front"

$env:VISION_SILENT_SIGNAL_SAME_HAND_ONLY = "false"
$env:VISION_SILENT_SIGNAL_OPERATOR_ONLY = "false"

$env:STREAMING_POC_CALLBACK_HOSTS = "api.omniguardy.cloud"

Remove-Item Env:STREAMING_POC_CALLBACK_BEARER_TOKEN -ErrorAction SilentlyContinue

# 저장된 암호화 키 복원
$keyFile = Join-Path $HOME ".omniguardy-callback-key.dpapi"
$secure = Get-Content $keyFile | ConvertTo-SecureString

$env:STREAMING_POC_CALLBACK_KEY = [System.Net.NetworkCredential]::new("", $secure).Password

Remove-Variable secure

Write-Host "Vision 환경변수 설정 완료"
Write-Host "FastAPI 8010 서버 실행 중..."

# 가상환경 Python으로 실행
& "$projectRoot\.venv\Scripts\python.exe" -m uvicorn experiments.streaming_poc.streaming_api_poc.app:app --host 0.0.0.0 --port 8010
