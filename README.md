# GeoAI Nevada Hillside Letters

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains a GeoAI pipeline for detecting hillside letters in Nevada using high-resolution NAIP imagery and vector masks. The goal is to identify both prominent and subtle letters in various locations, including near ghost towns or remote areas.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Getting Started](#getting-started)  
  - [Prerequisites](#prerequisites)  
  - [Installation](#installation)  
- [Usage](#usage)  
- [Project Structure](#project-structure)  
- [License](#license)  
- [Contributing](#contributing)  

---

## Project Overview

Hillside letters ("mountain monograms") are large symbols or letters placed on hillsides. This project uses a combination of:

- **NAIP aerial imagery** (high-resolution RGB)  
- **Vector masks of known letters**  
- **Rasterization and image preprocessing**  
- **GeoAI workflows** for detection and training  

The goal is to build a dataset and model capable of automatically detecting hillside letters across Nevada, including less visible or partially obscured ones.

---

## Getting Started

### Prerequisites

This project is Python-based and uses several geospatial libraries:

- Python 3.13+
- [rasterio](https://rasterio.readthedocs.io/)
- [fiona](https://fiona.readthedocs.io/)
- [shapely](https://shapely.readthedocs.io/)
- [numpy](https://numpy.org/)
- Optional: Pixi for environment management

Ensure you have a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate