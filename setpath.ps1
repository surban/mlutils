$mydir = Split-Path -parent $PSCommandPath;
$env:PYTHONPATH = $env:PYTHONPATH + ";" + $mydir

