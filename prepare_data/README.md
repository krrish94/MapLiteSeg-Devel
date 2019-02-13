# Data Preparation


`labelUtils.py` processes an annotated MapLite `pkl` file to generate data.

Dependencies:
* `numpy`, `imageio`, `scipy.stats`, `pandas`


Example usage:
```
python labelUtils.py -pkl path/to/maplite/annotated/file.pkl -outDir path/to/dir/that/stores/generated/data
```

**IMPORTANT**:

Once data is generated, create three files `train.txt`, `val.txt`, `test.txt` in the following format. Assuming you have 500 images in all, generated as `0.png`, `1.png`, ..., `499.png`, decide upon a train/val/test split. A sample  split could be to train on images 0-349, validate on images 350-399, and test on images 400-499.

#### train.txt

```
0
1
...
349
```

### val.txt

```
350
351
...
399
```

### test.txt

```
400
401
...
499
```
