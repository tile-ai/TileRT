#!/bin/bash

# TileRT Linting Script
# Runs all code quality checks with proper exclusions
#
# Usage:
#   ./scripts/lint.sh          # Run all linting checks
#   ./scripts/lint.sh --help   # Show this help message (future feature)
#
# This script runs the same checks as the CI linting workflow:
# - isort (import sorting)
# - black (code formatting)
# - flake8 (style checking)
# - mypy (type checking)
# - bandit (security checking)
# - pyupgrade (syntax modernization)
# - codespell (spelling check)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run a linting tool
run_lint() {
    local tool_name="$1"
    local command="$2"

    print_status "Running $tool_name..."
    if eval "$command"; then
        print_status "$tool_name passed ✓"
    else
        print_error "$tool_name failed ✗"
        return 1
    fi
}

main() {
    print_status "Starting TileRT linting checks..."

    cd "$(dirname "$0")/.."

    local failed_checks=0

    # Import sorting - different behavior for CI vs local
    if [[ "${CI:-}" == "true" ]]; then
        # In CI: only check, don't auto-fix
        if ! run_lint "isort" "isort --check-only --settings-path pyproject.toml ."; then
            ((failed_checks++))
        fi
    else
        # Local development: auto-fix then verify
        print_status "Running import sorting (isort) with auto-fix..."
        isort --settings-path pyproject.toml .

        # Check if any changes were made
        if ! git diff --quiet; then
            print_warning "Import sorting made changes. Files have been automatically fixed."
            git --no-pager diff --stat
        fi

        # Verify import sorting
        if ! run_lint "isort verification" "isort --check-only --settings-path pyproject.toml ."; then
            ((failed_checks++))
        fi
    fi

    # Code formatting
    if ! run_lint "black" "black --check ."; then
        ((failed_checks++))
    fi

    # Style checking
    if ! run_lint "flake8" "flake8 ."; then
        ((failed_checks++))
    fi

    # Run mypy via pre-commit to match hook environment (no heavy deps like torch)
    if ! run_lint "mypy" "pre-commit run mypy --all-files"; then
        ((failed_checks++))
    fi

    # Security checking
    if ! run_lint "bandit" "bandit -c pyproject.toml -r ."; then
        ((failed_checks++))
    fi

    # Python syntax modernization
    if ! run_lint "pyupgrade" "find . -type f -name '*.py' -not -path './3rd-party/*' -exec pyupgrade --keep-percent-format --py311-plus {} +"; then
        ((failed_checks++))
    fi

    # Spelling check
    if ! run_lint "codespell" "codespell --toml pyproject.toml"; then
        ((failed_checks++))
    fi

    # Summary
    echo
    if [ $failed_checks -eq 0 ]; then
        print_status "All linting checks passed! 🎉"
        exit 0
    else
        print_error "$failed_checks linting check(s) failed ❌"
        exit 1
    fi
}

# Run main function
main "$@"
