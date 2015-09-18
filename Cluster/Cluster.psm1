﻿
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
    <#
    .SYNOPSIS
    Creates a HPC job with options specified in a job description file and tasks defined by subdirectories.

    .DESCRIPTION
    This function creates a HPC job.
    
    Job parameters are read from the job.psd1 (can be changed using the -JobFilename parameter) file in the specified directory.
    The RunCommand parameter specifies the command that is used to execute each task. 
    The following optional parameteres are passed to the HPC scheduler: TemplateName, NodeGroups, ExcludedNodes, EstimatedProcessMemory, JobEnv.

    For each subdirectory of the specified directory an HPC task is added to the job. The task command is generated by appending the configuration path (relative to the project root) to RunCommand specified in job.psd1.
    Standard output and standard error are redirected to job.out (can be changed using the -OutputFilename parameter). Each task is started in the project root directory.

    The following substituions are performed:
    $CFGDIR$ is replaced by the directory containing job.psd1.
    $BASEDIR$ is replaced with the project base directory, i.e. the directory containing the cfgs subdirectory.
    All mapped network drives in paths are resolved to an absolute UNC path (i.e. \\srv-file\...).

    .PARAMETER Directory
    The directory that contains a job.psd1 file and subdirectories that correspond to tasks.
    If not specified the current directory is used.

    .PARAMETER JobFilename
    Job description filename. By default job.psd1 is used.

    .PARAMETER WithFinished
    By default subdirectories that contain a results.json file are skipped during task creation.
    If this parameter is specified, they are included.

    .PARAMETER OnlyFinished
    If this parameter is specified, tasks are created only from subdirectories that contain a results.json file.
    This is useful for jobs that use the results, for e.g. visualization and result analysis.

    .PARAMETER RunCommand
    Overrides the RunCommand specified in the job description file. If specfied, redirection of standard output and standard error is automatically disabled.

    .PARAMETER JobName
    Name of the HPC job. If not specified, it is automatically generated from the configuration and project name.

    .PARAMETER OutputFilename
    File into which standard output and standard error are redirected. If not specified, JobFilename with extension changed to .out is used.

    .PARAMETER NoRedirect
    Disables redirection of standard output and standard error to file.

    .OUTPUTS
    An HPC job object.

    .EXAMPLE
    PS Z:\dev\project> New-HpcJobFromDirectory -Directory cfgs\tst | Submit-HpcJob
    Id       Name          State           Owner                Priority        NumberOfTask
    --       ----          -----           -----                --------        ------------
    7046     project: tst  Queued          BRML\surban                          0

    This assumes that a job.psd1 file is located in Z:\dev\projects\cfgs\tst\job.psd1. For example, it might contain the following lines:

    @{
    RunCommand = 'python -u apps\myexp.py'
    JobEnv = @('THEANORC=$CFGDIR$\theanorc.txt', 
               'PYTHONPATH=$BASEDIR$',
               'INCLUDE=C:\Program Files\Microsoft SDKs\Windows\v7.0\include',
               'LIB=C:\Program Files\Microsoft SDKs\Windows\v7.0\lib\x64​')
    EstimatedProcessMemory = 1500
    }

    $CFGDIR$ will be replaced by \\srv-file\...\dev\project\cfgs\tst and $BASEDIR$ will be replaced by \\srv-file\...\dev\project.
    
    Assuming subdirectories 001, 002, ... exist in Z:\dev\projects\cfgs\tst, tasks with the command "python -u apps\myexp.py cfgs\tst\001", "python -u apps\myexp.py cfgs\tst\002", ... are created.
		
    #>
    [CmdletBinding(PositionalBinding=$False)]
    Param(
        [Parameter(Mandatory=$false)] [string] $Directory = $(Get-Location).Path,
        [Parameter(Mandatory=$false)] [string] $JobFilename = "job.psd1",
        [Parameter(Mandatory=$false)] [string] $RunCommand,
        [Parameter(Mandatory=$false)] [string] $JobName,
        [Parameter(Mandatory=$false)] [string] $OutputFilename,
        [Switch] $WithFinished,
        [Switch] $OnlyFinished,
        [Switch] $NoRedirect
    )
    Process
    {
        # convert to UNC path
        $Directory = $(Resolve-Path $Directory).ProviderPath
        $Directory = ToUNCPath $Directory

        # find relative path to cfg directory
        $project, $cfgname = Get-ProjectAndConfigName($Directory)
        $basedir = Get-ProjectBasePath($Directory)

        # load configuration file
        $cfgpath = Join-Path $Directory $JobFilename
        $cfg = Get-Content $cfgpath | Out-String | Invoke-Expression 
        if (-not $RunCommand)
        {
            if (-not $cfg.RunCommand) { throw "RunCommand must be specified either on the command line or in the job description file." }
            $RunCommand = $cfg.RunCommand 
            if (-not $JobName) { $JobName = "${project}: $cfgname\$JobFilename" }
        }
        else
        {
            $NoRedirect = $true
            if (-not $JobName) { $JobName = "${project}: $RunCommand $cfgname" }
        }
        $RunCommand = Perform-Substitutions $RunCommand $basedir $Directory

        if (-not $OutputFilename)
        {
            $OutputFilename = [io.path]::GetFileNameWithoutExtension($JobFilename) + ".out"
        }

        # create job
        $job = New-HpcJob -Name $JobName -Project $project 
        
        # set job properties from config file
        if ($cfg.TemplateName) { $job = Set-HpcJob -Job $job -TemplateName $cfg.TemplateName }
        if ($cfg.NodeGroups) { $job = Set-HpcJob -Job $job -NodeGroupOp Intersect -NodeGroups $cfg.NodeGroups }
        if ($cfg.ExcludedNodes) { $job = Set-HpcJob -Job $job -AddExcludedNodes $cfg.ExcludedNodes }
        if ($cfg.EstimatedProcessMemory) { $job = Set-HpcJob -Job $job -EstimatedProcessMemory $cfg.EstimatedProcessMemory }
        if ($cfg.JobEnv) 
        { 
            $JobEnv = $cfg.JobEnv | ForEach-Object {Perform-Substitutions $_ $basedir $Directory}
            $job = Set-HpcJob -Job $job -JobEnv $JobEnv 
        }

        # create tasks from subdirectories
        $dirs = Get-ChildItem $Directory -Directory
        foreach ($dir in $dirs)
        {
            $_, $cfginstname = Get-ProjectAndConfigName($dir.FullName)
            $resultspath = Join-Path $dir.FullName "results.json"
            
            if ($OnlyFinished)
            {
                if (-Not (Test-Path $resultspath)) { Write-Verbose "Skipping $cfginstname because no results are present."; continue }
            }
            elseif ((Test-Path $resultspath) -And -Not ($WithFinished)) 
            { 
                Write-Verbose "Skipping $cfginstname because results are present." 
                continue 
            }

            if (-not $NoRedirect) 
            {
                $outfile = Join-Path $dir.FullName $OutputFilename 
                $job = Add-HpcTask -Job $job -Name $dir.Name -WorkDir $basedir -Stdout $outfile -Stderr $outfile `
                    -CommandLine "$RunCommand $cfginstname" -Rerunnable $true                
            }
            else
            {
                $job = Add-HpcTask -Job $job -Name $dir.Name -WorkDir $basedir `
                    -CommandLine "$RunCommand $cfginstname" -Rerunnable $true
            }
        }

        return $job
    }
}


function Clear-Checkpoints
{
    <#
    .SYNOPSIS
    Removes all checkpoint files in the subdirectories of the specified directory.

    .DESCRIPTION
    This function deletes all checkpoints (by default stored in checkpoint.dat).
    
    .PARAMETER Directory
    The directory that contains subdirectories that correspond to tasks.
    If not specified the current directory is used.

    .PARAMETER Filename
    The filename of the checkpoint file (by default checkpoint.dat).

    .EXAMPLE
    PS Z:\dev\project> Clear-Checkpoints -Directory cfgs\tst | Submit-HpcJob
	
    #>
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$false)] [string] $Directory = $(Get-Location).Path,
        [Parameter(Mandatory=$false)] [string] $Filename = "checkpoint.dat"
    )

    $dirs = Get-ChildItem $Directory -Directory
    foreach ($dir in $dirs)
    {
        $path = Join-Path $dir $Filename
        if (Test-Path $path) 
        {
            Remove-Item -Force $path
        }
    }
}


function Get-HpcTaskResults
{
    <#
    .SYNOPSIS
    Gathers experiment results from JSON files.

    .DESCRIPTION
    This function reads all results.json files in the subdirectories of the specified directory.

    The following fields are interpreted in the results.json file:
    best_iter           - number of performed training iterations (returned as Iters)
    training_time       - training time in seconds (returned as Duration)
    best_val_loss       - best loss on validation set (returned as ValLoss)
    best_tst_loss       - best loss on test set (returned as TstLoss)
    cfg                 - JSON dictionary of used hyperparameters (all keys returned with original name)
    
    .PARAMETER Directory
    The directory that contains subdirectories that correspond to tasks.
    If not specified the current directory is used.

    .EXAMPLE
    PS Z:\dev\project> Get-HpcTaskResults -Directory cfgs\tst 
    Id         Iters  Duration   TstLoss    ValLoss
    --         -----  --------   -------    -------
    00066      1020   0.13:20    0.004678   0.004738
    00138      8990   0.16:24    0.006164   0.006342
    00136      9160   0.16:09    0.006224   0.006305
    00026      1810   0.03:34    0.006529   0.006643
    ...

    By default the hyperparameters are not displayed. Configurations are sorted by descending loss.

    .EXAMPLE
    PS Z:\dev\project> Get-HpcTaskResults -Directory cfgs\tst | Select *
    Id         Iters  Duration   TstLoss    ValLoss    step_rate  optimizer
    --         -----  --------   -------    -------    ---------  ---------
    00066      1020   0.13:20    0.004678   0.004738       0.001         gd  
    00138      8990   0.16:24    0.006164   0.006342       0.005         gd

    Pipe the output through Select * to see all hyperparameters.

    .EXAMPLE
    PS Z:\dev\project> Get-HpcTaskResults -Directory cfgs\tst | Select * | Out-GridView
    Displays a window with the ability to interactively filter and sort the results.
    #>
    [CmdletBinding()]
    Param(
        [Parameter(Mandatory=$false)] [string] $Directory = $(Get-Location).Path
    )

    $results = @()
    $dirs = Get-ChildItem $Directory -Directory
    foreach ($dir in $dirs)
    {
        $path = Join-Path $dir.FullName "results.json"
        if (Test-Path $path)
        {          
            try 
            { 
                $json = Get-Content $path | Out-String | ConvertFrom-Json
            }
            catch [System.ArgumentException]
            {
                Write-Verbose "Cannot parse JSON in ${path}: $_"
                continue 
            }
            if ($json.cfg) { $res = $json.cfg } else { $res = New-Object -TypeName PSObject }
            $res | Add-Member Id (Split-Path $dir -Leaf)
            if ($json.best_iter)  { $res | Add-Member Iters $json.best_iter }
            if ($json.training_time) { $res | Add-Member Duration (New-TimeSpan -Seconds $json.training_time) }
            if ($json.best_tst_loss) { $res | Add-Member TstLoss $json.best_tst_loss }
            if ($json.best_val_loss) { $res | Add-Member ValLoss $json.best_val_loss }
            $res = $res | Select-Object -Property Id, Iters, Duration, TstLoss, ValLoss, * -ErrorAction Ignore
            $res.PSObject.TypeNames.Insert(0, "Cluster.HpcTaskResult")
            $results += $res
        }
    }        
    $results | Sort-Object -Property TstLoss 
}

Export-ModuleMember -Function New-HpcJobFromDirectory
Export-ModuleMember -Function Get-HpcTaskResults
Export-ModuleMember -Function Clear-Checkpoints


