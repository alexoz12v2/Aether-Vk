param (
    [Parameter(Mandatory=$true, HelpMessage="Action to perform: 'create' or 'upload'")]
    [ValidateSet("create", "upload")]
    [string]$Action,

    [Parameter(Mandatory=$true, HelpMessage="Version number for the release, e.g. 1.0.0")]
    [string]$Version,

    [Parameter(Mandatory=$false, HelpMessage="Build type: 'official' or 'sxs'")]
    [ValidateSet("official", "sxs", "dev")]
    [string]$Type = "official",

    [Parameter(Mandatory=$false, HelpMessage="Target branch on GitHub")]
    [string]$TargetBranch = "main",

    [Parameter(Mandatory=$false, HelpMessage="Set to true if this is a hotfix for an older version")]
    [switch]$Hotfix,

    [Parameter(Mandatory=$false, HelpMessage="File to upload (required for 'upload' action)")]
    [string]$File,

    [Parameter(Mandatory=$false, HelpMessage="Bypass CI status check")]
    [switch]$IgnoreCI
)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "'gh' (GitHub CLI) is not installed. Please install it."
    exit 1
}

if ($Action -eq "create") {
    $IsPrerelease = $false
    if ($Type -eq "sxs" -or $Type -eq "dev") {
        $IsPrerelease = $true
        if (-not $Version.EndsWith("-sxs")) {
            $Version = "$Version-sxs"
        }
    }

    # Ensure version starts with 'v'
    if (-not $Version.StartsWith("v")) {
        $Version = "v$Version"
    }

    Write-Host "Checking version consistency..."
    $ReleaseOutput = gh release list --exclude-pre-releases -L 1 2>$null
    if ($ReleaseOutput) {
        $LatestRelease = ($ReleaseOutput -split '\s+')[0]
        $CleanNew = $Version -replace "^v", "" -replace "-.*", ""
        $CleanLatest = $LatestRelease -replace "^v", "" -replace "-.*", ""
        
        try {
            if ([version]$CleanNew -lt [version]$CleanLatest -and -not $Hotfix) {
                Write-Error "Requested version ($CleanNew) is lower than the latest release ($CleanLatest)."
                Write-Error "If this is a hotfix for an older version, use the -Hotfix switch."
                exit 1
            }
        } catch {
            Write-Warning "Failed to compare versions. Proceeding anyway..."
        }
    }

    Write-Host "Fetching latest commit on remote branch '$TargetBranch'..."
    $Commit = gh api repos/:owner/:repo/commits/$TargetBranch -q .sha

    if (-not $Commit) {
        Write-Error "Could not retrieve latest commit for branch '$TargetBranch'."
        exit 1
    }

    Write-Host "Checking CI status for commit $Commit..."
    try {
        $StatusJson = gh run list --commit $Commit --json conclusion 2>$null
        $StatusList = $StatusJson | ConvertFrom-Json
        if ($StatusList.Count -gt 0) {
            $Status = $StatusList[0].conclusion
            if ($Status -ne "success" -and $Status -ne $null) {
                if ($IgnoreCI) {
                    Write-Warning "CI pipeline failed (status: $Status), but -IgnoreCI is specified. Proceeding anyway..."
                } else {
                    Write-Error "CI pipeline for the target commit has not succeeded (status: $Status)."
                    Write-Error "Please ensure the commit passes CI before creating a release. Use -IgnoreCI to bypass."
                    exit 1
                }
            }
        } else {
            Write-Warning "Could not find CI runs for this commit. Proceeding anyway..."
        }
    } catch {
        Write-Warning "Failed to fetch CI status. Proceeding anyway..."
    }

    Write-Host "Creating GitHub release $Version targeting commit $Commit ($Type build)..."
    if ($IsPrerelease) {
        gh release create $Version --target $Commit --prerelease --generate-notes --title "Release $Version"
    } else {
        gh release create $Version --target $Commit --generate-notes --title "Release $Version"
    }

    Write-Host "Release $Version successfully created!"
    Write-Host "The GitHub Action release workflow should now be triggered to attach artifacts."

} elseif ($Action -eq "upload") {
    if (-not $File) {
        Write-Error "File parameter is required for 'upload' action."
        exit 1
    }
    
    # Ensure version starts with 'v'
    if (-not $Version.StartsWith("v")) {
        $Version = "v$Version"
    }

    Write-Host "Uploading $File to release $Version..."
    gh release upload $Version $File
    Write-Host "Upload complete!"
}
