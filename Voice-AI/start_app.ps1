# Start all services in background

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Definition

Write-Host "Starting LiveKit Server..."
Start-Process -FilePath "$ScriptPath\livekit-server.exe" -ArgumentList "--dev" -WorkingDirectory $ScriptPath -NoNewWindow

Write-Host "Starting Backend Agent..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "uv run python src/agent.py dev" -WorkingDirectory "$ScriptPath\backend" -NoNewWindow

Write-Host "Starting Frontend..."
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "npx pnpm dev" -WorkingDirectory "$ScriptPath\frontend" -NoNewWindow

Write-Host "Services started in the same window."
