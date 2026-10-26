#!/bin/bash
set -e

ACTION=$1
VERSION=$2
TYPE=${3:-official} # official or sxs (side-by-side)
TARGET_BRANCH=${4:-main}
HOTFIX=${5:-false}
# SKIP_ALL_CHECKS=1 skips the version consistency and CI status checks
SKIP_ALL_CHECKS=${SKIP_ALL_CHECKS:-0}

if [ -z "$ACTION" ]; then
    echo "Usage: ./scripts/create_release.sh <create|upload> <version> [type] [target_branch] [hotfix]"
    echo "Examples:"
    echo "  ./scripts/create_release.sh create 1.0.0 official main"
    echo "  ./scripts/create_release.sh create 0.9.1 official main hotfix"
    echo "  ./scripts/create_release.sh create 1.0.0 sxs main"
    echo "  ./scripts/create_release.sh upload 1.0.0 ./bin/publish/AetherVk_win-x64.msix"
    echo "  SKIP_ALL_CHECKS=1 ./scripts/create_release.sh create 1.0.0 official main"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "Error: 'gh' (GitHub CLI) is not installed. Please install it to proceed."
    exit 1
fi

# Compares two X.Y.Z versions: returns 0 if equal, 1 if $1 > $2, 2 if $1 < $2
compare_versions() {
    if [[ $1 == $2 ]]; then return 0; fi
    local IFS=.
    local i ver1=($1) ver2=($2)
    # fill empty fields with zeros
    for ((i=${#ver1[@]}; i<3; i++)); do ver1[i]=0; done
    for ((i=${#ver2[@]}; i<3; i++)); do ver2[i]=0; done
    for ((i=0; i<3; i++)); do
        if ((10#${ver1[i]} > 10#${ver2[i]})); then return 1; fi
        if ((10#${ver1[i]} < 10#${ver2[i]})); then return 2; fi
    done
    return 0
}

if [ "$ACTION" == "create" ]; then
    if [ -z "$VERSION" ]; then
        echo "Error: Version not provided."
        echo "Usage: ./scripts/create_release.sh create <version> [type] [target_branch] [hotfix]"
        exit 1
    fi

    # Determine if it's a prerelease and update version tag
    IS_PRERELEASE=""
    if [ "$TYPE" == "sxs" ] || [ "$TYPE" == "dev" ]; then
        IS_PRERELEASE="--prerelease"
        if [[ $VERSION != *-sxs ]]; then
            VERSION="${VERSION}-sxs"
        fi
    elif [ "$TYPE" != "official" ]; then
        echo "Error: Type must be 'official' or 'sxs'"
        exit 1
    fi

    # Ensure version starts with 'v'
    if [[ $VERSION != v* ]]; then
        VERSION="v$VERSION"
    fi

    # The release tag is created on the latest commit of the target branch
    echo "Fetching latest commit on remote branch '$TARGET_BRANCH'..."
    COMMIT=$(gh api repos/:owner/:repo/commits/$TARGET_BRANCH -q .sha)

    if [ -z "$COMMIT" ]; then
        echo "Error: Could not retrieve latest commit for branch '$TARGET_BRANCH'."
        exit 1
    fi

    # Pre-release checks; all of them are skipped when SKIP_ALL_CHECKS=1
    if [ "$SKIP_ALL_CHECKS" == "1" ]; then
        echo "SKIP_ALL_CHECKS=1: skipping version consistency and CI status checks."
    else
        # Refuse a version lower than the latest published release, unless it is marked as a hotfix
        echo "Checking version consistency..."
        LATEST_RELEASE=$(gh release list --exclude-pre-releases -L 1 2>/dev/null | awk '{print $1}')
        if [ -n "$LATEST_RELEASE" ]; then
            CLEAN_NEW=$(echo "$VERSION" | sed -E 's/^v//' | sed -E 's/-.*//')
            CLEAN_LATEST=$(echo "$LATEST_RELEASE" | sed -E 's/^v//' | sed -E 's/-.*//')

            # `|| ...` keeps set -e from aborting when the new version is higher (return 1)
            COMP_RESULT=0
            compare_versions "$CLEAN_NEW" "$CLEAN_LATEST" || COMP_RESULT=$?

            if [ $COMP_RESULT -eq 2 ] && [ "$HOTFIX" != "hotfix" ] && [ "$HOTFIX" != "true" ]; then
                echo "Error: Requested version ($CLEAN_NEW) is lower than the latest release ($CLEAN_LATEST)."
                echo "If this is a hotfix for an older version, append 'hotfix' to the command line."
                exit 1
            fi
        fi

        # Block the release unless the commit's latest CI run succeeded; IGNORE_CI=1 overrides this
        echo "Checking CI status for commit $COMMIT..."
        STATUS=$(gh run list --commit "$COMMIT" --json conclusion -q '.[0].conclusion' || echo "")

        if [ "$STATUS" != "success" ] && [ -n "$STATUS" ] && [ "$STATUS" != "null" ]; then
            if [ "$IGNORE_CI" == "1" ] || [ "$IGNORE_CI" == "true" ]; then
                echo "Warning: CI pipeline failed (status: $STATUS), but IGNORE_CI is set. Proceeding anyway..."
            else
                echo "Error: CI pipeline for the target commit has not succeeded (status: $STATUS)."
                echo "Please ensure the commit passes CI before creating a release."
                echo "To override this, run with IGNORE_CI=1 ./scripts/create_release.sh ... (or SKIP_ALL_CHECKS=1)"
                exit 1
            fi
        elif [ -z "$STATUS" ] || [ "$STATUS" == "null" ]; then
            echo "Warning: Could not find CI runs for this commit. Proceeding anyway..."
        fi
    fi

    echo "Creating release $VERSION targeting commit $COMMIT ($TYPE build)..."
    gh release create "$VERSION" --target "$COMMIT" $IS_PRERELEASE --generate-notes --title "Release $VERSION"

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
