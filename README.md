# FedAvg baseline on Edge-IIoTset (non-IID, Dirichlet)

Reproduces a FedAvg baseline on Edge-IIoTset with Dirichlet label-skew partitioning.
Reports accuracy, macro-F1, and communication cost per round (MB).

## Setup
Tested with Python 3.14, PyTorch, scikit-learn, pandas.
```
python -m venv venv
venv\Scripts\activate        # Linux/Mac: source venv/bin/activate
pip install -r requirements.txt
```

## Data
Download `DNN-EdgeIIoT-dataset.csv` (folder "Selected dataset for ML and DL") from the
Edge-IIoTset dataset by M. A. Ferrag on Kaggle and place it in `data/`.

## Run order
```
python src/preprocess.py                       # cleans data, 80/20 stratified split (seed 42)
python src/baseline.py                         # centralized MLP reference
python src/fedavg.py --alpha 0.5 --rounds 20 --seed 42
python src/fedavg.py --alpha 0.1 --rounds 20
python src/fedavg.py --alpha 1.0 --rounds 20
python src/fedavg.py --alpha 0.5 --rounds 20 --seed 1
python src/fedavg.py --alpha 0.5 --rounds 20 --seed 2
python src/report.py                           # table + results.png
python src/seeds.py                            # mean/std over seeds
```

## Setup details
- Model: MLP 46-128-64-15 (15,247 parameters)
- 10 clients, all participate every round, 20 rounds
- Local training: 1 epoch, SGD, lr 0.05, batch 512; server aggregates by sample count
- Partition: Dirichlet over class labels (alpha = 0.1, 0.5, 1.0)
- Features: 46 (identifiers, payloads and ports dropped), scaler fitted on train only
- Communication per round = parameters x 4 bytes x 2 (upload + download) x 10 clients = 1.22 MB

## Results (round 20)
| Setting | Accuracy | Macro-F1 | MB/round |
|---|---|---|---|
| Centralized (Adam, 5 epochs) | 0.955 | 0.754 | n/a |
| FedAvg alpha=1.0 (seed 42) | 0.941 | 0.606 | 1.22 |
| FedAvg alpha=0.5 (3 seeds) | 0.940 ± 0.001 | 0.595 ± 0.005 | 1.22 |
| FedAvg alpha=0.1 (seed 42) | 0.918 | 0.485 | 1.22 |

![results](results.png)

## Notes and limitations
- "Normal" is 73% of the data, so accuracy is high everywhere; macro-F1 shows the real effect.
- Lower alpha (more non-IID) reduces macro-F1 clearly (0.60 to 0.48).
- The centralized reference uses Adam and 5 epochs; FedAvg uses SGD for 20 rounds, so the gap is not only due to federation.
- Single train/test split; seeds vary the partition, initialization and batch order only.
- Communication cost is computed analytically, not measured on a network.