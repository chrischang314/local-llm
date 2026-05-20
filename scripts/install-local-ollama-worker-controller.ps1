param(
    [string]$TaskName = "Local LLM CHRIS-PC-2 Worker Controller",
    [string]$WorkerName = "chris-pc-2",
    [string]$Namespace = "local-llm",
    [string]$SwitchDeployment = "chris-pc-2-ollama-switch",
    [string]$ComposeFile = "docker-compose.pc-worker.yml",
    [string]$Service = "chris-pc-2-ollama",
    [string]$Container = "local-llm-chris-pc-2-ollama",
    [string]$Volume = "local_llm_chris_pc_2_ollama",
    [string]$DockerConfig = ".docker-worker",
    [string]$DockerDesktopTaskName = "Start Docker Desktop",
    [string]$TaskUserId = "",
    [switch]$InstallDockerDesktopStartupTask,
    [int]$PollSeconds = 10
)

$ErrorActionPreference = "Stop"

function Register-DockerDesktopStartupTask {
    param(
        [string]$TaskName,
        [string]$UserId
    )

    $dockerDesktop = Join-Path $env:ProgramFiles "Docker\Docker\Docker Desktop.exe"
    if (-not (Test-Path $dockerDesktop)) {
        Write-Warning "Docker Desktop executable not found at $dockerDesktop"
        return
    }

    $dockerAction = New-ScheduledTaskAction `
        -Execute "cmd.exe" `
        -Argument "/c start `"`" `"$dockerDesktop`""
    $dockerTrigger = New-ScheduledTaskTrigger -AtLogOn
    $dockerPrincipal = New-ScheduledTaskPrincipal `
        -UserId $UserId `
        -LogonType Interactive `
        -RunLevel Limited
    $dockerSettings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $dockerAction `
        -Trigger $dockerTrigger `
        -Principal $dockerPrincipal `
        -Settings $dockerSettings `
        -Description "Starts Docker Desktop for the local Ollama worker controller." `
        -Force | Out-Null

    Start-ScheduledTask -TaskName $TaskName
}

$scriptPath = Resolve-Path (Join-Path $PSScriptRoot "local-ollama-worker-controller.ps1")
$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$arguments = @(
    "-NoProfile",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$scriptPath`"",
    "-WorkerName", "`"$WorkerName`"",
    "-Namespace", "`"$Namespace`"",
    "-SwitchDeployment", "`"$SwitchDeployment`"",
    "-ComposeFile", "`"$ComposeFile`"",
    "-Service", "`"$Service`"",
    "-Container", "`"$Container`"",
    "-Volume", "`"$Volume`"",
    "-DockerConfig", "`"$DockerConfig`"",
    "-PollSeconds", $PollSeconds
) -join " "

try {
    if (-not $TaskUserId) {
        $TaskUserId = "$env:COMPUTERNAME\$env:USERNAME"
    }

    if ($InstallDockerDesktopStartupTask) {
        Register-DockerDesktopStartupTask `
            -TaskName $DockerDesktopTaskName `
            -UserId $TaskUserId
    }

    $action = New-ScheduledTaskAction `
        -Execute "powershell.exe" `
        -Argument $arguments `
        -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -AtLogOn
    $principal = New-ScheduledTaskPrincipal `
        -UserId $TaskUserId `
        -LogonType Interactive `
        -RunLevel Limited
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "Watches the Kubernetes $SwitchDeployment deployment and starts/stops the $WorkerName local Ollama worker." `
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
