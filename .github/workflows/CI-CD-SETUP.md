# CI/CD & Testing Setup Guide

This guide explains how to ensure tests pass before PRs are merged into your repository.

## 1. GitHub Actions Workflows (Automated Testing)

Two workflows have been created to automatically run tests on every PR and push:

### `tests.yml` - Test Suite
Runs on every push and PR to `main` or `develop` branches.

**What it does:**
- Tests across 3 operating systems (Ubuntu, Windows, macOS) with Python 3.12
- Runs full test suite with coverage reporting
- Uploads coverage to Codecov
- Fails if any tests fail

**Trigger conditions:**
- Push to `main` or `develop` branches
- Any pull request to `main` or `develop` branches

### `code-quality.yml` - Code Quality Checks
Runs on every push and PR to verify code style and documentation.

**What it does:**
- Checks code formatting with ruff
- Verifies imports are organized correctly
- Validates docstring compliance

## 2. Branch Protection Rules (Enforce Passing Tests)

To **require tests to pass before merging PRs**, you must set up branch protection rules:

### Steps to Enable Branch Protection:

1. **Go to GitHub Repository Settings**
   - Navigate to: Settings → Branches

2. **Add Branch Protection Rule**
   - Click "Add rule"
   - Pattern: `main` (or `develop`)

3. **Configure Required Status Checks**
   - ✅ Check "Require status checks to pass before merging"
   - ✅ Check "Dismiss stale pull request approvals when new commits are pushed"
   - ✅ Check "Require branches to be up to date before merging"

4. **Select Required Checks**
   Select all of the following:
   - `test (ubuntu-latest, 3.12)`
   - `test (windows-latest, 3.12)`
   - `test (macos-latest, 3.12)`
   - `quality (ubuntu-latest)`

5. **Optional: Additional Security**
   - ✅ "Require a pull request before merging"
   - ✅ "Require approvals" (set to 1 or more reviewers)
   - ✅ "Require review from Code Owners" (if using CODEOWNERS file)
   - ✅ "Include administrators" (enforce rules on admins too)

6. **Save**
   - Click "Create" or "Save changes"

### Result:
- ❌ PRs cannot be merged unless ALL tests pass
- ❌ PRs cannot be merged unless CI workflows complete successfully
- ✅ Developers must keep branches up to date with main
- ✅ All changes are tested before being merged

## 3. Pre-commit Hooks (Local Testing)

Run tests automatically before each commit on your local machine.

### Installation:

```bash
# Install pre-commit
pip install pre-commit

# Install the git hooks
pre-commit install
```

### Usage:

```bash
# Tests will run automatically before `git commit`
# If tests fail, the commit is blocked

git commit -m "My changes"
# → pre-commit hooks run
# → if failures: fix them, then stage and commit again

# To skip hooks (not recommended)
git commit -m "My changes" --no-verify
```

### Manual Hook Execution:

```bash
# Run hooks on all files
pre-commit run --all-files

# Run specific hook
pre-commit run ruff --all-files

# Update hooks
pre-commit autoupdate
```

## 4. Test Execution

### Run all tests locally:

```bash
# Install test dependencies
pip install -e ".[test]"

# Run all tests
pytest

# Run fast tests only (skip slow integration tests)
pytest -m "not slow"

# Run with coverage report
pytest --cov=packages --cov-report=html
# Open htmlcov/index.html in browser
```

### Run specific tests:

```bash
# Single test file
pytest tests/test_dataset_functions.py

# Single test class
pytest tests/test_model_functions.py::TestAdaptModelToData

# Single test method
pytest tests/test_model_functions.py::TestAdaptModelToData::test_adapt_model_output_classes

# Skip slow tests
pytest -m "not slow"
```

## 5. Typical PR Workflow

```bash
# 1. Create feature branch
git checkout -b feature/my-feature

# 2. Make changes and commit (pre-commit hooks will run)
git add .
git commit -m "Add new feature"

# 3. Local testing
pytest tests/

# 4. Push to GitHub
git push origin feature/my-feature

# 5. Create PR on GitHub
# → GitHub Actions workflows automatically trigger
# → Tests run across 3 operating systems
# → Coverage is reported
# → Branch protection rules prevent merge if tests fail

# 6. Address any test failures
# → Fix code
# → Commit and push
# → Tests re-run automatically

# 7. After tests pass and PR is approved
# → Merge PR (only possible if tests pass)
```

## 6. Monitoring Workflows

### View Workflow Status:

1. **On GitHub PR Page**
   - Scroll down to see "Checks" section
   - Shows status of each workflow
   - Click details to see full logs

2. **On GitHub Actions Tab**
   - Go to: Actions tab on main repo page
   - See all workflow runs
   - Click to view detailed logs

3. **CLI Status Check**
   ```bash
   # View workflow status (requires gh CLI)
   gh run list
   gh run view <run-id>
   ```

## 7. Customization

### Modify Workflows

Edit `.github/workflows/tests.yml` to:
- Add/remove Python versions
- Add/remove operating systems
- Change test commands
- Add new jobs

### Disable Workflows

To temporarily disable a workflow:
1. Go to Actions tab
2. Select workflow
3. Click "..." menu
4. Disable workflow

To re-enable, same process.

## 8. Common Issues

**Issue:** Tests pass locally but fail on CI
- **Solution:** Usually OS-specific (Windows path handling, etc.)
  - Check CI logs carefully
  - Test on the failing OS if possible
  - Use GitHub Actions local runner for debugging

**Issue:** Tests timeout on CI
- **Solution:** 
  - Increase timeout in workflow
  - Optimize slow tests
  - Mark very slow tests with `@pytest.mark.slow`

**Issue:** Flaky tests (pass sometimes, fail sometimes)
- **Solution:**
  - Review test for randomness or timing issues
  - Use `pytest-repeat` to test multiple times
  - Add proper synchronization

**Issue:** Branch protection preventing merges
- **Solution:**
  - Fix failing tests
  - Ensure branch is up to date
  - Check all required checks have passed

## 9. Viewing Coverage Reports

After tests pass with coverage reporting:

1. **On Codecov**
   - Visit codecov.io
   - Link GitHub repo
   - View coverage badges and reports

2. **Local HTML Report**
   ```bash
   pytest --cov=packages --cov-report=html
   # Open htmlcov/index.html
   ```

## 10. Next Steps

1. ✅ Workflows are already created (`.github/workflows/`)
2. **TODO:** Set up branch protection rules on GitHub
3. **TODO (Optional):** Install pre-commit hooks locally
4. **TODO:** Consider adding more CI checks (type checking, security scanning)

---

**Quick Links:**
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Branch Protection Rules](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-protected-branches)
- [Pre-commit Framework](https://pre-commit.com/)
- [Codecov Documentation](https://docs.codecov.io/)
