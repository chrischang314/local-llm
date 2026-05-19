param(
    [string]$TaskName = "Local LLM CHRIS-PC-2 Worker Controller",
    [string]$WorkerName = "chris-pc-2",
    [string]$Namespace = "local-llm",
    [string]$SwitchDeployment = "chris-pc-2-ollama-switch",
    [int]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"

$scriptPath = Resolve-Path (Join-Path $PSScriptRoot "local-ollama-worker-controller.ps1")
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-WorkerName", "`"$WorkerName`"",
    "-Namespace", "`"$Namespace`"",
    "-SwitchDeployment", "`"$SwitchDeployment`"",
    "-PollSeconds", $PollSeconds
) -join " "

try {
    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument $arguments `
        -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Description "Watches the Kubernetes $SwitchDeployment deployment and starts/stops the CHRIS-PC-2 local Ollama worker." `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $TaskName
    Get-ScheduledTask -TaskName $TaskName | Select-Object TaskName,State
} catch {
    $startupDir = [Environment]::GetFolderPath("Startup")
    $shortcutPath = Join-Path $startupDir "$TaskName.lnk"
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = "powershell.exe"
    $shortcut.Arguments = $arguments
    $shortcut.WorkingDirectory = [string]$repoRoot
    $shortcut.WindowStyle = 7
    $shortcut.Description = "Local LLM optional worker controller for $WorkerName"
    $shortcut.Save()

    Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $arguments `
        -WorkingDirectory $repoRoot `
        -WindowStyle Hidden

    [pscustomobject]@{
        InstallMode = "StartupShortcut"
        Shortcut = $shortcutPath
        State = "Started"
        Reason = "Scheduled task registration failed: $($_.Exception.Message)"
    }
}
