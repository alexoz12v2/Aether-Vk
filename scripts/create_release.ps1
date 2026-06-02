param (
    [Parameter(Mandatory=$true, HelpMessage="Action to perform: 'create' or 'upload'")]
    [ValidateSet("create", "upload")]
    [string]$Action,

    [Parameter(Mandatory=$true, HelpMessage="Version number for the release, e.g. 1.0.0")]
    [string]$Version,

    [Parameter(Mandatory=$false, HelpMessage="File to upload (required for 'upload' action)")]
    [string]$File
)

if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "'gh' (GitHub CLI) is not installed. Please install it."
    exit 1
}

# Ensure version starts with 'v'
if (-not $Version.StartsWith("v")) {
    $Version = "v$Version"
}

if ($Action -eq "create") {
    Write-Host "Checking CI status for current commit..."
    $Commit = git rev-parse HEAD
    try {
        $StatusJson = gh run list --commit $Commit --json conclusion 2>$null
        $StatusList = $StatusJson | ConvertFrom-Json
        if ($StatusList.Count -gt 0) {
            $Status = $StatusList[0].conclusion
            if ($Status -ne "success" -and $Status -ne $null) {
                Write-Error "CI pipeline for the current commit has not succeeded (status: $Status)."
                Write-Error "Please ensure the commit passes CI before creating a release."
                exit 1
            }
        } else {
            Write-Warning "Could not find CI runs for this commit. Proceeding anyway..."
        }
    } catch {
        Write-Warning "Failed to fetch CI status. Proceeding anyway..."
    }

    Write-Host "Creating tag $Version..."
    git tag $Version

    Write-Host "Pushing tag $Version to origin..."
    git push origin $Version

    Write-Host "Creating GitHub release $Version..."
    gh release create $Version --generate-notes --title "Release $Version"

    Write-Host "Release $Version successfully created!"
    Write-Host "The GitHub Action release workflow should now be triggered to attach artifacts."

} elseif ($Action -eq "upload") {
    if (-not $File) {
        Write-Error "File parameter is required for 'upload' action."
        exit 1
    }
    Write-Host "Uploading $File to release $Version..."
    gh release upload $Version $File
    Write-Host "Upload complete!"
}
