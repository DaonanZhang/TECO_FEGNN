![(Architecture of PEGNN versus FEGNN, enhanced with a feature encoder and a SOTA auxiliary learner.)](https://github.com/DaonanZhang/TECO_FEGNN/blob/master/FEGNN.png)

This is the official repository for the Bachelor Thesis by the TECO Institute from Karlsruhe Institute of Technology: Enhancing Positional Encoder Graph Neural Networks for 
Geographic Data with Auxiliary Information Integration (Author: Daonan Zhang, Supervising Staff: Chaofan Li, Responsible Supervisor: Prof. Dr. Michael Beigl)

The Processed Dataset can be found under the link: https://drive.google.com/file/d/1TKHP3fT7W71rv7UEMmdjd2PBztAmviML/view?usp=sharing

___

The folder Gaxlearn were implemented from [@Cite: Chen H, Wang X, Liu Y, et al. Module-aware optimization for auxiliary learning[J]. Advances in Neural Information Processing Systems, 2022, 35: 31827-31840.](https://github.com/AvivNavon/AuxiLearn)

Provide for each model a Test.ipynb file (For some model, the path of dataset and gauxlearn are up-to-date)

| name in folder | name in thesis       |
|----------------|----------------------|
| fe_al          | FEGNN                |
| al_only        | AL                   |
| Trsage         | Gsage                |
| Tsage_loss     | Denoising FE         |
