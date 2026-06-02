#!/bin/bash
set -e

ACTION=$1
VERSION=$2

if [ -z "$ACTION" ]; then
    echo "Usage: ./scripts/create_release.sh <create|upload> <version> [file]"
    echo "Examples:"
    echo "  ./scripts/create_release.sh create 1.0.0"
    echo "  ./scripts/create_release.sh upload 1.0.0 ./bin/publish/AetherVk_win-x64.msix"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "Error: 'gh' (GitHub CLI) is not installed. Please install it to proceed."
    exit 1
fi

if [ "$ACTION" == "create" ]; then
    if [ -z "$VERSION" ]; then
        echo "Error: Version not provided."
        echo "Usage: ./scripts/create_release.sh create <version>"
        exit 1
    fi

    # Ensure version starts with 'v'
    if [[ $VERSION != v* ]]; then
        VERSION="v$VERSION"
    fi

    echo "Checking CI status for current commit..."
    COMMIT=$(git rev-parse HEAD)
    STATUS=$(gh run list --commit "$COMMIT" --json conclusion -q '.[0].conclusion' || echo "")
    
    if [ "$STATUS" != "success" ] && [ -n "$STATUS" ] && [ "$STATUS" != "null" ]; then
        echo "Error: CI pipeline for the current commit has not succeeded (status: $STATUS)."
        echo "Please ensure the commit passes CI before creating a release."
        exit 1
    elif [ -z "$STATUS" ] || [ "$STATUS" == "null" ]; then
        echo "Warning: Could not find CI runs for this commit. Proceeding anyway..."
    fi

    echo "Creating tag $VERSION..."
    git tag "$VERSION"
    
    echo "Pushing tag $VERSION to origin..."
    git push origin "$VERSION"

    echo "Creating release $VERSION..."
    gh release create "$VERSION" --generate-notes --title "Release $VERSION"
    
    echo "Release $VERSION successfully created!"
    echo "The GitHub Action release workflow should now be triggered to attach artifacts."

elif [ "$ACTION" == "upload" ]; then
    FILE=$3
    if [ -z "$VERSION" ] || [ -z "$FILE" ]; then
        echo "Error: Version or file not provided."
        echo "Usage: ./scripts/create_release.sh upload <version> <file>"
        exit 1
    fi

    if [[ $VERSION != v* ]]; then
        VERSION="v$VERSION"
    fi

    echo "Uploading $FILE to release $VERSION..."
    gh release upload "$VERSION" "$FILE"
    echo "Upload complete!"
else
    echo "Invalid action: $ACTION. Must be 'create' or 'upload'."
    exit 1
fi
