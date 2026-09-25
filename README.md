# Sentinel Multispectral Analysis

Python implementations of eight remote-sensing case studies in optical
image processing — spanning flood mapping, wildfire burn-scar detection,
land-cover classification, bathymetric retrieval, image fusion, floating-algae
detection, and vessel counting — using Sentinel-2, Sentinel-3, Landsat 8,
and Pléiades imagery.

Four of these case studies are written up in detail, with methodology,
results, and a discussion of limitations, in an accompanying paper:

**[Quantitative Retrieval and Classification from Multi-Sensor Optical
Imagery](./paper/Multi_Sensor_Optical_Imagery.pdf)** — bathymetric
estimation, multi-resolution image fusion, spectral classification, and
anomaly detection against a spatially varying background.

## Repository structure

### Case studies covered in the paper

| Folder | Paper section | Sensor | Task |
|---|---|---|---|
| [`Bathymetry`](./Bathymetry) | §3.1 | Pléiades | Water-depth retrieval via a blue/green band-ratio index, with NIR-based atmospheric/sun-glint correction |
| [`Image Merge`](./Image%20Merge) | §3.2 | Landsat 8 OLI | Pan-sharpening via the Brovey transform and HSV substitution |
| [`Camargue`](./Camargue) | §3.3 | Sentinel-2 MSI | Land-cover classification (soil/vegetation/water) via Euclidean distance and Spectral Angle Mapper |
| [`Algae`](./Algae) | §3.4 | Sentinel-3 OLCI | Floating-algae detection via the Maximum Chlorophyll Index (MCI) and background subtraction |

### Additional applications

| Folder | Sensor | Task |
|---|---|---|
| [`Flood`](./Flood) | Sentinel-2 MSI | Bi-temporal flood-extent mapping (Var, Nov. 2019) via NIR thresholding and image differencing |
| [`Wild fire`](./Wild%20fire) | Sentinel-2 MSI | Burn-scar mapping (Var, 2017) via NDVI thresholding and bi-temporal differencing |
| [`Port-Cros`](./Port-Cros) | Sentinel-2 MSI | Seasonal vessel-traffic counting via binarization and connected-component labelling |
| [`Atmospheric correction`](./Atmospheric%20correction) | — | *[describe: shared correction routine, or standalone TP?]* |
| [`Marine reflectance model`](./Marine%20reflectance%20model) | — | *[describe: shared model used by Bathymetry/Algae, or standalone?]* |
| [`Water components`](./Water%20components) | — | *[describe: what this isolates/computes]* |

> The three folders above are described only briefly here — see each
> folder's own README for details.

## Methodology

All case studies follow a common processing structure: multi-band image
loading and compositing, radiometric normalization, construction of a
spectral index or reference signature, and a final thresholding,
classification, or fusion step, interpreted and visualized
cartographically. Full methodological detail, results, and a discussion
of validation limitations for the four featured case studies are in the
[paper](./paper/Multi_Sensor_Optical_Imagery.pdf).

## Data sources

- **Sentinel-2 MSI** and **Sentinel-3 OLCI** imagery: [Copernicus Browser](https://browser.dataspace.copernicus.eu/)
- **Landsat 8 OLI** imagery: [USGS EarthExplorer](https://earthexplorer.usgs.gov/)
- **Pléiades** imagery: Airbus / provided through course materials

Raw imagery is not included in this repository (see `.gitignore`); each
folder's README notes the specific scene(s) used, where known.

## Requirements

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `opencv-python`, `scipy`, `matplotlib`.

## Background

This code was developed as practical coursework in aerospace image
processing at SeaTech Toulon (2024–2025). It is shared here as
supporting material for the accompanying paper.
