param (
    [Parameter(Mandatory=$true, HelpMessage="Action to perform: 'create', 'refresh' or 'upload'")]
    [ValidateSet("create", "refresh", "upload")]
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

function Get-TargetCommitWithGreenCI {
    param([string]$Branch, [switch]$Bypass)
    Write-Host "Fetching latest commit on remote branch '$Branch'..."
    $Sha = gh api repos/:owner/:repo/commits/$Branch -q .sha
    if (-not $Sha) {
        Write-Error "Could not retrieve latest commit for branch '$Branch'."
        exit 1
    }
    Write-Host "Checking CI status for commit $Sha..."
    $StatusList = gh run list --commit $Sha --workflow test.yml --json conclusion,status 2>$null | ConvertFrom-Json
    if (-not $StatusList -or $StatusList.Count -eq 0) {
        if ($Bypass) {
            Write-Warning "No CI run found for $Sha, but -IgnoreCI is specified. Proceeding anyway..."
        } else {
            Write-Error "No CI run found for $Sha. Push the commit and wait for CI, or use -IgnoreCI to bypass."
            exit 1
        }
    } elseif ($StatusList[0].conclusion -ne "success") {
        $State = if ($StatusList[0].conclusion) { $StatusList[0].conclusion } else { $StatusList[0].status }
        if ($Bypass) {
            Write-Warning "CI for $Sha is '$State', but -IgnoreCI is specified. Proceeding anyway..."
        } else {
            Write-Error "CI for $Sha has not succeeded (state: $State). Wait for a green run, or use -IgnoreCI to bypass."
            exit 1
        }
    }
    return $Sha
}

if ($Action -eq "refresh") {
    # Rebuild an EXISTING release from the latest commit of $TargetBranch, keeping its tag name,
    # title and notes: every asset is deleted, the tag is moved to the new commit, and the release
    # workflow (release.yml) rebuilds and uploads the artifacts to the same release.
    if (($Type -eq "sxs" -or $Type -eq "dev") -and -not $Version.EndsWith("-sxs")) {
        $Version = "$Version-sxs"
    }
    if (-not $Version.StartsWith("v")) {
        $Version = "v$Version"
    }

    $Release = gh release view $Version --json tagName,assets,targetCommitish 2>$null | ConvertFrom-Json
    if (-not $Release) {
        Write-Error "Release $Version does not exist. Use -Action create for a new release."
        exit 1
    }

    $Commit = Get-TargetCommitWithGreenCI -Branch $TargetBranch -Bypass:$IgnoreCI

    $OldSha = gh api repos/:owner/:repo/git/ref/tags/$Version -q .object.sha
    Write-Host "Tag ${Version}: $OldSha -> $Commit"
    if ($OldSha -eq $Commit) {
        Write-Warning "The tag already points at ${Commit}: the artifacts are rebuilt from the same commit."
    }

    foreach ($Asset in $Release.assets) {
        Write-Host "Deleting asset $($Asset.name)..."
        gh release delete-asset $Version $Asset.name --yes
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Could not delete asset $($Asset.name)."
            exit 1
        }
    }

    Write-Host "Moving tag $Version to $Commit..."
    gh api -X PATCH repos/:owner/:repo/git/refs/tags/$Version -f sha=$Commit -F force=true | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Could not move tag $Version."
        exit 1
    }
    gh release edit $Version --target $Commit | Out-Null

    # The tag update normally starts release.yml (push event); start it by hand if it did not
    Write-Host "Waiting for the release workflow on $Commit..."
    $Run = $null
    for ($i = 0; $i -lt 12 -and -not $Run; $i++) {
        Start-Sleep -Seconds 10
        $Run = gh run list --workflow release.yml --commit $Commit --json databaseId,event,status -L 1 2>$null | ConvertFrom-Json | Select-Object -First 1
    }
    if (-not $Run) {
        Write-Host "No run started by the tag update: dispatching release.yml on $Version..."
        gh workflow run release.yml --ref $Version
        if ($LASTEXITCODE -ne 0) {
            Write-Error "Could not dispatch release.yml."
            exit 1
        }
    } else {
        Write-Host "Release workflow run $($Run.databaseId) ($($Run.event), $($Run.status))."
    }

    Write-Host "Release $Version refreshed: the workflow uploads the new artifacts to it."
    Write-Host "Watch it with: gh run list --workflow release.yml -L 3"
    exit 0
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
