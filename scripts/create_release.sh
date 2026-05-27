#!/bin/bash
set -e

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: ./scripts/create_release.sh <version>"
    echo "Example: ./scripts/create_release.sh 1.0.0"
    exit 1
fi

# Ensure version starts with 'v'
if [[ $VERSION != v* ]]; then
    VERSION="v$VERSION"
fi

echo "Creating tag $VERSION..."
git tag "$VERSION"

echo "Pushing tag $VERSION to origin..."
git push origin "$VERSION"

echo "Release tag $VERSION successfully created and pushed!"
echo "The GitHub Action release workflow should now be triggered."
