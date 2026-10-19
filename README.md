# EN3150 Assignment 03: RealWaste

[EN3150_A03_Final.ipynb](EN3150_A03_Final.ipynb) is the final RealWaste notebook. It contains the preprocessing, custom CNNs, transfer learning, optimizer comparison and saved results.

RealWaste contains 4,752 images across nine classes. Inputs are RGB at 64 by 64 pixels, with a stratified 70/15/15 split and seed 42.

| Model | Parameters | Test accuracy |
| --- | ---: | ---: |
| Model A | 549,161 | 70.83% |
| Model B | 9,033 | 71.11% |
| MobileNetV2 | 2,235,401 | 77.56% |
| EfficientNet-B0 | 4,019,077 | 81.63% |

Training histories, metrics, plots and resource calculations are in [results/final](results/final). Python dependencies are listed in [requirements.txt](requirements.txt).

[EN3150_A03_new.ipynb](EN3150_A03_new.ipynb) contains the separate handwritten digits experiment. [EN3150_A03.ipynb](EN3150_A03.ipynb) is the original RealWaste notebook. Both are retained alongside the final version.
