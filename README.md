# IITMandi Smell20 Dataset
The Bhāluk dataset comprises multiple olfactory classes, including Cardamom, Onion, Ginger, Chocolate, and others. Each class is represented through its sensor voltage responses, with 16 samples per class, capturing the characteristic odor signatures of each substance.


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
