# Automated Nonlinearity Encoding (ANE)

Extrapolation to predict unseen data outside the training distribution is a common challenge in real-world scientific applications of physics and chemistry. However, the extrapolation capabilities of neural networks have not been extensively studied in machine learning. Although it has been recently revealed that neural networks become linear regression in extrapolation problems, a universally applicable method to support the extrapolation of neural networks in general regression settings has not been investigated. In this paper, we propose automated nonlinearity encoder (ANE) that is a data-agnostic embedding method to improve the extrapolation capabilities of neural networks by conversely linearizing the original input-to-target relationships without architectural modifications of prediction models. ANE achieved state-of-the-art extrapolation accuracies in extensive scientific applications of various data formats. As a real-world application, we applied ANE for high-throughput screening to discover novel solar cell materials, and ANE significantly improved the screening accuracy.

* Paper: Gyoung S. Na and Chanyoung Park, Nonlinearity Encoding for Extrapolation of Neural Networks, KDD 2022.


## Requirements
* Python $\geq$ 3.6
* Pytorch $\geq$ 1.9.0
* Pytorch Geometric $\geq$ 2.0.3
* Pymatgen $\geq$ 2022.0.11


## Dataset Sources
* **HOIP dataset:** Chiho Kim et al., A hybrid organic-inorganic perovskite dataset, Sci. Data, 2017.
* **Materials Project database:** Anubhav Jain et al., A materials genome approach to accelerating materials innovation. APL Mater., 2013.
* **MagNet NASA (MNN) dataset:** https://www.kaggle.com/arashnic/soalr-wind


## Run
Execute `experiment.py` in each task folder. The embedding and prediction results are saved in the `results` folder.


## License
ACM acknowledges that this contribution was authored or co-authored by an employee, contractor or affiliate of a national government. As such, the Government retains a nonexclusive, royalty-free right to publish or reproduce this article, or to allow others to do so, for Government purposes only
