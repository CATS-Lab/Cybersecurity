# Security Strategy against Generalized Inter-Vehicle Cyberattacks in Car-following Scenarios for Connected and Autonomous Vehicles

## Introduction

This repository contains the source code and data for the following paper:

**Zhou, H., Ma, C., Cai, X., Ma, K., Li, X., \& Ran, B.** (2025). *Security strategy against generalized inter-vehicle cyberattacks in car-following scenarios for connected and autonomous vehicles.* Transportation Research Part C: Emerging Technologies.

Connected autonomous vehicles (CAVs) can improve safety and efficiency by sharing future driving intentions with nearby vehicles. However, overreliance on inter-vehicle communication may lead to critical safety risks when facing cyberattacks. This repository provides the implementation of a security-aware car-following control strategy under inter-vehicle cyberattacks, which dynamically adjusts its reliance on inter-vehicle information. The framework includes four control modes, a cyberattack misbehavior identification module, and two trajectory optimization models for connected and non-connected driving. We also include a cyberattack generation algorithm to simulate safety-critical attack scenarios and evaluate the proposed strategy's effectiveness.

## Usage

### Code

The project is developed using Python 3. Please ensure you have the corresponding environment set up.

The code related to our data processing and algorithm are included in folder **'src'**. As you proceed through all code, always verify the paths for both the input and output files. This ensures that everything runs smoothly.

### Data

The '**data**' folder contains the benchmark reference trajectories used in our paper. Some of the realistic AV trajectory are extracted from project [ULTra-AV](https://github.com/CATS-Lab/Filed-Experiment-Data-ULTra-AV).

## Developers

Developers - Hang Zhou (hzhou364@wisc.edu).

If you have any questions, please feel free to contact the CATS Lab at UW-Madison. We're here to help!
