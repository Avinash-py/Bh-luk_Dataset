# IITMandi Smell20 Dataset
The IITMandi Smell20 dataset (in volt) comprises multiple olfactory classes, including Cardamom, Onion, Ginger, Chocolate, and others. Each class is represented through its sensor voltage responses, with 16 samples per class, capturing the characteristic odor signatures of each substance.


## Prerequisites

Data is expected at `helper/train_voltage`, `helper/validation_voltage`, `helper/test_voltage`, relative to the working directory.

```bash
cd ApprochA
ln -s ../helper helper (dataset put here)
```

## Running

```bash
cd ApprochA
python3 -m model_comp.main
```
## Citation 
@inproceedings{kushwaha2026bhaluk,
  author    = {Kushwaha, Avinash and Kulkarni, Prashant D. and Thakur, Richa and Roy Chowdhury, Shubhajit and Nigam, Aditya and Singh, Dinesh},
  title     = {{Bh{\=a}luk}: Learning the Unknown Basis of Human Olfactory System Using Deep Learning},
  booktitle = {Distributed Computing and Intelligent Technology},
  series    = {Lecture Notes in Computer Science},
  pages     = {343--357},
  publisher = {Springer},
  year      = {2026},
  doi       = {10.1007/978-3-032-16632-6_22}
}
