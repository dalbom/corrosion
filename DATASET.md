Dataset Name: Corrosion Dataset

The active split CSV files live under:

```text
datasets/corrosion/splits/
  train.csv
  val.csv
  test.csv
```

Each split CSV has five columns: `filename`, `S11`, `S21`, `Phase11`, and `Phase21`.
Each measurement column is a single string containing 201 space-separated float values.

Target images live under:

```text
datasets/corrosion/images/
  <SAMPLE_INDEX>/
    <filename>.png
```

The target image for a measurement record is derived from the CSV `filename` column. The filename convention is:

```text
[DATE:MMDD]_[SAMPLE_INDEX]_[CORROSION_VALUE]_[real|augmented]
```

For example:

```text
filename:
  0525_61_30.89263840450541_augmented

target image:
  datasets/corrosion/images/61/0525_61_30.89263840450541_augmented.png
```

Raw S-parameter files, when needed for provenance or future preprocessing, live under:

```text
datasets/corrosion/sparameters/
  <SAMPLE_INDEX>/
    <filename>.txt
```
