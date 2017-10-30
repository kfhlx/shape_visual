# shape_visual

Applying incremental learning on 3D knee surfaces and reconstruction in C++ with VTK and ITK

Started with standard PCA to small set to get model containing mean vector, eigenvalue, eigenvector and coefficients.
The initial dataset can be discard and only the model are pass into Incremental PCA.
Learning is conducted in a subspace level.

Incremental PCA algorithm is taken from "Incremental and robust learning of subspace representations" by Skočaj, D. and Leonardis, A
