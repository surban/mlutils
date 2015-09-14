
$ErrorActionPreference = 'Stop'

$snapin = Get-PSSnapin | Where { $_.Name -eq 'Microsoft.HPC' } 
if (-not $snapin) { Add-PSSnapin Microsoft.HPC }

function Get-ProjectAndConfigName($path)
{
    if ($path -match "([^\\]+)\\cfgs\\(.+)$")
    {
        $project = $Matches[1]
        $config = $Matches[2]
        return $project, $config
    }
    else
    {
        throw "Cannot determine project name and config name from $path"
    }
}

function Get-ProjectBasePath($path)
{
    if ($path -match "(.+)\\cfgs")
    {
        return $Matches[1]
    } 
    else 
    {
        throw "Cannot determine project base path from $path"
    }
}

function Perform-Substitutions($value, $basedir, $cfgdir)
{
    $value = $value.Replace('$BASEDIR$', $basedir)
    $value = $value.Replace('$CFGDIR$', $cfgdir)
    return $value
}

function ToUNCPath($path)
{
    if ($path.StartsWith('\\')) { return $path }

    $drive = Split-Path -Qualifier $path
    $logicalDisk = Get-WmiObject Win32_LogicalDisk -Filter "DriveType = 4 AND DeviceID = '$drive'"
    return $path.Replace($drive, $logicalDisk.ProviderName)
}


function New-HpcJobFromDirectory
{
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$false)] [string] $directory = $(Get-Location).Path
    )
    Process
    {
        $directory = $(Resolve-Path $directory).ProviderPath
        $directory = ToUNCPath $directory

        # load configuration file
        $cfgpath = Join-Path $directory "job.psd1"
        $cfg = Get-Content $cfgpath | Out-String | Invoke-Expression 
        if (-not $cfg.RunCommand) { throw "RunCommand must be specified in job.cfg" }
        $RunCommand = Perform-Substitutions $cfg.RunCommand $basedir $directory

        # find relative path to cfg directory
        $project, $cfgname = Get-ProjectAndConfigName($directory)
        $basedir = Get-ProjectBasePath($directory)

        # create job
        $job = New-HpcJob -Name "${project}: $cfgname" -Project $project 
        
        # set job properties from config file
        if ($cfg.TemplateName) { $job = Set-HpcJob -Job $job -TemplateName $cfg.TemplateName }
        if ($cfg.NodeGroups) { $job = Set-HpcJob -Job $job -NodeGroupOp Intersect -NodeGroups $cfg.NodeGroups }
        if ($cfg.ExcludedNodes) { $job = Set-HpcJob -Job $job -AddExcludedNodes $cfg.ExcludedNodes }
        if ($cfg.EstimatedProcessMemory) { $job = Set-HpcJob -Job $job -EstimatedProcessMemory $cfg.EstimatedProcessMemory }
        if ($cfg.JobEnv) 
        { 
            $JobEnv = $cfg.JobEnv | ForEach-Object {Perform-Substitutions $_ $basedir $directory}
            $job = Set-HpcJob -Job $job -JobEnv $JobEnv 
        }

        # create tasks from subdirectories
        $dirs = Get-ChildItem $directory -Directory
        foreach ($dir in $dirs)
        {
            $_, $cfginstname = Get-ProjectAndConfigName($dir.FullName)

            $stdout = Join-Path $dir.FullName "output.txt"
            $stderr = Join-Path $dir.FullName "output.txt"

            $task = Add-HpcTask -Job $job -Name $dir.Name -WorkDir $basedir -Stdout $stdout -Stderr $stderr `
                -CommandLine "$RunCommand $cfginstname" -Rerunnable $true
        }

        return $job
    }
}

function Submit-HpcJobFromDirectory
{
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$false)] [string] $directory = $(Get-Location).Path
    )
    Process
    {
        $job = New-HpcJobFromDirectory $directory 
        Submit-HpcJob -Job $job
    }
}


function Clear-Checkpoints
{
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$false)] [string] $directory = $(Get-Location).Path,
        [Parameter(Mandatory=$false)] [string] $filename = "checkpoint.dat"
    )
    Process
    {
        $dirs = Get-ChildItem $directory -Directory
        foreach ($dir in $dirs)
        {
            $path = Join-Path $dir $filename
            if (Test-Path $path) 
            {
                Remove-Item -Force $path
            }
        }        
    }
}

Export-ModuleMember -Function New-HpcJobFromDirectory
Export-ModuleMember -Function Submit-HpcJobFromDirectory
Export-ModuleMember -Function Clear-Checkpoints


