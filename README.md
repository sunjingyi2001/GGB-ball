 Attribute reduction for partially labeled hybrid data based on generalized granular-ball neighborhood rough set model and Student-t kernel

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official implementation of the attribute reduction algorithm designed for Partially Labeled Hybrid Information Systems (p-HDIS). 

Our method utilizes a generalized granular-ball neighborhood rough set model integrated with the Student-t kernel to construct adaptive boundaries, enabling the robust and effective handling of complex, irregularly shaped, and unevenly distributed hybrid data (categorical and numerical) with sparse labels.
📌 Key Features
Handling Hybrid Data: Effectively processes datasets containing both categorical and continuous numerical attributes.
Robust Distance Metric: Employs the heavy-tailed characteristic of the Student-t kernel to capture distant object correlations and avoids the gradient vanishing problem in high-dimensional spaces.
Adaptive Granular-Balls: Generates balls through non-geometric clustering without relying on fixed calculation centers or radii, flexibly approximating any manifold shape.
Semi-Supervised Attribute Reduction: Integrates dependency (supervised perspective) and generalized entropy (unsupervised perspective) to evaluate and guide the attribute reduction process.
🛠️ Environment Requirements

To run this code, you will need Python 3.8 or higher. Install the required dependencies using `pip`:

```bash
pip install pandas numpy openpyxl joblib scipy scikit-learn matplotlib seaborn
