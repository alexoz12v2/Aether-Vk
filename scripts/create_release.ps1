param (
    [Parameter(Mandatory=$true, HelpMessage="Version number for the release, e.g. 1.0.0")]
    [string]$Version
)

# Ensure version starts with 'v'
if (-not $Version.StartsWith("v")) {
    $Version = "v$Version"
}

Write-Host "Creating tag $Version..."
git tag $Version

Write-Host "Pushing tag $Version to origin..."
git push origin $Version

Write-Host "Release tag $Version successfully created and pushed!"
Write-Host "The GitHub Action release workflow should now be triggered."
