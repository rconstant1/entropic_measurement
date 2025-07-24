### Entropic Measurement Revolution

## Features

- Universal correction of entropic bias in any measurement process
- Calculation of Shannon and Kullback-Leibler (KL) entropies
- Traceability and export of entropic cost for each measurement (CSV or JSON logs)
- Extensible, object-oriented API compatible with scientific workflows
- Ready for integration with machine learning, experiments, open data, and more

## Project Structure
 ```
entropic_measurement/
├── __init__.py
├── measurement.py
├── entropy.py
├── correction.py
├── logger.py
└── utils.py
 ```

## Example Usage

from entropic_measurement.utils import measure_and_correct
from entropic_measurement.logger import EntropicLogger
 ```
Example data
observed_dist = [0.6, 0.4]
true_dist = [0.7, 0.3]
observed_value = 10.0

Bias correction
result = measure_and_correct(observed_value, observed_dist, true_dist, beta=1.0)
print("Result:", result)

Logging and exporting audit
logger = EntropicLogger()
logger.record(result)
logger.export("audit_log.csv", format="csv")
 ```

## Quick Start: Value Proposition vs. SciPy, Statsmodels & Others
 ```
| Task                              | SciPy/Statsmodels                | entropic_measurement              | Key Value Add                                   |
|-----------------------------------|----------------------------------|-----------------------------------|-------------------------------------------------|
| Compute mean & std                | Yes (`numpy.mean`, `numpy.std`)  | Indirect (basic stats)            | Standard statistics only                        |
| Robust to distribution shape      | No (assumes normality)           | Yes (explicit entropy correction) | Handles multi-modal, non-Gaussian, biased data  |
| Bias correction in measurement    | No                               | Yes (info-theoretic correction)   | Makes uncertainty sources explicit, corrects    |
| Traceable entropic cost export    | No                               | Yes (CSV/JSON logging)            | Audit-friendly, detailed reporting              |
| Designed for science & ML workflow| Limited (not traceable)          | Yes (object-oriented, extensible) | Open, composable, machine learning ready        |
| Entropy/KL metrics                | Partially (scipy.stats.entropy)  | Yes (dedicated methods, traceable)| Fully traceable & ready for bias audit          |
 ```
## When to Use Mean/Std vs. Entropy/Correction?- Use **mean/std** if:
  - Data is symmetric, unimodal, and low-uncertainty
  - Normality assumptions hold and you only need simple summary statistics
- Prefer **entropy/correction** if:
  - Data is multi-modal, skewed, or structurally uncertain
  - You need to detect, quantify, or correct bias/uncertainty in measurements
  - Reproducibility, transparency, and traceable uncertainty matter

## Applications

- **Physical and chemical experiments**: calibrate scientific measures with informational transparency
- **AI and data science**: detect, quantify and correct algorithmic bias robustly
- **Medical analysis**: ensure reproducibility and reliability in diagnostics
- **Industrial quality control**: documentation and certification with entropic measurement scores
- **Open science**: provide FAIR data with explicit uncertainty and bias info


## Documentation

**A Python library for explicit correction and traceability of informational bias in scientific and industrial measurement.**  
Inspired by the “Entropic Measurement Revolution” developed by Raphael Constantinis, this package enables robust, transparent, and reproducible correction of entropic bias using modern statistical entropy metrics.

Comprehensive documentation is available in the [Wiki](https://github.com/rconstant1/entropic_measurement/wiki).

- [Getting Started](https://github.com/rconstant1/entropic_measurement/wiki/Quick-Start-Guide)
- [Full Table of Contents](https://github.com/rconstant1/entropic_measurement/wiki)

## License

All code and methods are released in the **public domain (CC0)** – no copyright, patent, or exclusive ownership.

## Contributing

Contributions, feedback, and interdisciplinary collaboration are welcome.  
Standardize entropic metrology in your field and expand this library with new modules and applications!

