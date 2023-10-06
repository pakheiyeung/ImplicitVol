# ImplicitVol: Sensorless 3D Ultrasound Reconstruction with Deep Implicit Representation<!-- omit in toc -->
<!-- ![Figure](fig/head_gif.gif) -->

This repository contains the codes (in PyTorch) for the framework introduced in the following paper:

ImplicitVol: Sensorless 3D Ultrasound Reconstruction with Deep Implicit Representation
[[Paper]](https://arxiv.org/abs/2109.12108) [[Project Page]](https://pakheiyeung.github.io/ImplicitVol_wp/)

```
@article{yeung2021implicitvol,
	title = {ImplicitVol: Sensorless 3D Ultrasound Reconstruction with Deep Implicit Representation},
	author = {Yeung, Pak-Hei and Hesse, Linde and Aliasi, Moska and Haak, Monique and Xie, Weidi and Namburete, Ana IL and others},
	journal = {arXiv preprint arXiv:2109.12108},
	year = {2021},
}
```

## Contents<!-- omit in toc -->
- [Dependencies](#dependencies)
- [Reconstruct from an example volume](#reconstruct-from-an-example-volume)
- [Modify for your own images](#modify-for-your-own-images)
  

## Dependencies
- Python (3.7), other versions should also work
- PyTorch (1.12), other versions should also work 
- scipy
- skimage
- [nibabel](https://nipy.org/nibabel/)

## Reconstruct from an example volume
Due to data privacy of the ultrasound data we used in our study, in this repository, we use an example MRI volume `example/sub-feta001_T2w.nii.gz` from [FeTA](http://neuroimaging.ch/feta) for demonstration. When running `train.py`, here is the high-level description of what it does:
1. A set of 2D slices are sampled from the `example/sub-feta001_T2w.nii.gz`, with known plane locations in the volume. This mimics acquiring the 2D ultrasound vidoes and predicting their plane locations.
2. The set of 2D slices will be used to train the implicit representation model.
3. Novel views (i.e. 2D slices sampled from volume, perpendicular to the training slices) are generated from the trained model.

## Modify for your own images
To modify the codes for your own data, you may need to modify the following aspects:

### Plane localization<!-- omit in toc -->
1. If the plane location of each 2D image is known, you can skip this stage
2. Localizing the images in the 3D space using approaches such as [PlaneInVol](https://github.com/pakheiyeung/PlaneInVol).
	
### Reconstruction<!-- omit in toc -->
1. Save the 2D images and their corresponding plane location
2. Modify the `Dataset_volume_video` class in `dataset.py` to import the saved file.
