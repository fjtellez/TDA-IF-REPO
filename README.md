# A Topological Data Analysis Framework for Quantifying Necrosis in Glioblastomas

This is the repository for the paper: "A Topological Data Analysis Framework for Quantifying Necrosis in Glioblastomas". We introduce a shape descriptor that we call "interior function". This is a Topological Data Analysis (TDA) based descriptor that refines previous descriptors for image analysis. Using this concept, we define subcomplex lacunarity, a new index that quantifies geometric characteristics of necrosis in tumors such as conglomeration. Building on this framework, we propose a set of indices to analyze necrotic morphology and construct a diagram that captures the distinct structural and geometric properties of necrotic regions in tumors. We present an application of this framework in the study of MRIs of Glioblastomas. Using cluster analysis we identify four distinct subtypes of Glioblastomas that reflect geometric properties of necrotic regions.

## Data Availability

The results of this study are based on data from the TCGA cohort. The data used was obtained from the Smooth Euler Characteristic Transform GitHub repository: https://github.com/lorinanthony/SECT?tab=readme-ov-file. This dataset includes data that has already been preprocessed. Additionally, we include another dataset that underwent an extra preprocessing step, which was directly used for computing the indices proposed in this study.

We also provide an Excel spreadsheet containing the computed integral values of the interior function for all the images in the cohort, which were subsequently used to compute the indices. Moreover, an Excel file with the computed indices for the entire set of images is included. Finally, we provide a Python script with the functions used to compute the value of the interior function at a given point, which can be used to calculate the indices discussed in the paper if needed.

## Remarks

Much of the computational work in this study relies on Python libraries developed by other authors. Specifically, we used Cubical Ripser (https://github.com/shizuo-kaji/CubicalRipser_3dim) for computing cubical homology and Persim (https://github.com/scikit-tda/persim) for computing persistence landscapes.



