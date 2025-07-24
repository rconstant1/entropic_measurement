## [Unreleased]

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

You can paste this directly into your `CHANGELOG.md` for the next release.
