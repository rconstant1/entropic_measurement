## [v0.1.0] - 2025-08-03
### Added
- Official PyPI publication: entropic_measurement package is published on PyPI for pip install.
- Comprehensive Kaggle bias correction example and Wiki documentation.
- New Testing wiki page with guidelines, folder structure, and pytest usage.
- GitHub Discussions enabled for the repository.
- Trending GitHub topics/tags for discoverability (data-science, statistics, entropy, bias-correction, machine-learning).

### Changed
- Home wiki and README updated with PyPI and Testing information.
- Added resources/about section and direct links to PyPI and Wiki in repo metadata.
- Python CI workflow now limited to code (py), requirements, and pyproject.toml changes.
- CI now includes numpy, pandas, scikit-learn for reliable test runs.

### Fixed
- Addressed repeated CI failures by including missing dependencies in workflow.
- Corrected Kaggle Example Wiki link logic on Home page.


## [v0.1.0] - 2025-07-23
### Added
- Comprehensive type hints throughout the codebase for improved code clarity and IDE support
- Explicit error messages with detailed context for better debugging experience
- Robust input validation with comprehensive parameter checking
- Full test coverage ensuring code reliability and preventing regressions
- Continuous Integration (CI) pipeline for automated testing and quality assurance
- Enhanced tensor compatibility for seamless integration with scientific computing workflows
- Harmonized docstrings following consistent formatting standards across all modules
- **Advanced logging utility**: new EntropicLogger class with automatic timestamping, flexible record structure, and methods for efficient export to CSV or JSON
- `clear()` method to reset log entries on demand

### Changed
- Improved code documentation with standardized docstring format
- Enhanced error handling with more informative exception messages
- Strengthened input validation mechanisms for better data integrity
- **Upgraded logger export**: now ensures UTF-8 encoding, creates directories automatically, and supports robust field alignment for CSV exports

### Fixed
- Resolved compatibility issues with various tensor libraries
- Standardized function signatures and return types across the API
- Fixed missing or inconsistent fields during log export for heterogeneous entry structures

This entry now:
- Explicitly documents the **EntropicLogger** improvements as a high-value new feature
- Summarizes internal and external impacts (robust export, UTF-8, field alignment)
- Fits your changelog structure, making the logger.py refactor clear to users and contributors
