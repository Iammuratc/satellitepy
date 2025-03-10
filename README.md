# Satellitepy
Satellitepy is a python module that focuses on fine-grained object recognition in high resolution optical satellite images. Our target objects are airplane, vessel, vehicle and helicopter. It provides handy tools to deal with many challenges that may arise during object recognition steps.

One of the advantages of Satellitepy is to train a DL model by using multi-task learning, e.g, classification, detection and segmentation. There are 16 tasks in our pipeline. Satellitepy allows each task to be handled individually and/or jointly.

# Datasets
We have trained/evaluated the SOtA models with the following datasets: [FineAir](https://huggingface.co/datasets/UniBwM-Informatik-IV/FineAir), DOTA (w. iSaid), Fair1M, XView, Rarepl. (Real+Synthetic), VEDAI, VHR-10, ShipRSImageNet, UCAS-AOD, DIOR, and Potsdam.

# How to use

## Installation

Please follow the steps in [the installation manual](docs/installing_satellitepy.md) to install satellitepy. After that, you can do the following:

1) You can use [create patches](docs/creating_patches.md) to create image patches to train/validate your models.
2) You can [train a model](docs/training_mtl_bbavector.md) on your [patches](docs/creating_patches.md).
3) You can [evaluate a pretrained model](docs/evaluating_pretrained_model.md) on the test images.

# Citations

* (IGARSS 2023) [Fine-Grained Airplane Recognition in Satellite Images based on Task Separation and Orientation Normalization](https://www.researchgate.net/publication/374876936_Fine-Grained_Airplane_Recognition_in_Satellite_Images_based_on_Task_Separation_and_Orientation_Normalization)

 * (CV4EO - WACV 2025) [FineAir: Finest-grained Airplanes in High-resolution Satellite Images](https://openaccess.thecvf.com/content/WACV2025W/CV4EO/html/Osswald_FineAir_Finest-grained_Airplanes_in_High-resolution_Satellite_Images_WACVW_2025_paper.html)