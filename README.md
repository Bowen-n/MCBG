# Malware Classification by Learning Semantic and Structural Features of Control Flow Graphs

This is the source code of our paper accepted by TrustCom2021: [Malware Classification by Learning Semantic and Structural Features of Control Flow Graphs](https://ieeexplore.ieee.org/document/9724385).

**Citation**
```
@inproceedings{wu2021malware,
  title={Malware Classification by Learning Semantic and Structural Features of Control Flow Graphs},
  author={Bolun Wu, Yuanhang Xu, Futai Zou},
  booktitle={2021 IEEE 20th International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
  pages={540--547},
  year={2021},
  organization={IEEE}
}
```
## Code Structure
```
.
├── README.md
└── src
    ├── bert_tidy     # BERT training code by Transformers
    ├── dataset       # parse BIG2015 to CFGs and BBs
    └── gnn           # gnn code
```

## Dataset
- [BIG2015](https://www.kaggle.com/c/malware-classification/data)

## Reference
- GNN
  - [GIN](https://arxiv.org/abs/1810.00826)
  - [DGCNN](https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf)
- NLP
  - [BERT](https://arxiv.org/abs/1810.04805)
- Instruction Normalization
  - [DEEPBINDIFF](https://www.ndss-symposium.org/wp-content/uploads/2020/02/24311-paper.pdf)
  - [SAFE](https://arxiv.org/pdf/1811.05296.pdf)
- Malware Classification
  - [Classifying Malware Represented as Control Flow Graphs using Deep Graph Convolutional Neural Network](http://www.cs.binghamton.edu/~ghyan/papers/dsn19.pdf)

