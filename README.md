# Product Clustering for an Online Retail Store

Explore product groupings in transactional retail data to support catalog and purchasing analysis.

## Why this project

This repository is part of my practical machine-learning portfolio. It focuses on a complete, understandable workflow rather than claiming production readiness.

## Dataset

The repository includes `Online Retail.xlsx.zip`. Document the original source and redistribution terms before sharing the archive.

## Approach

Data cleaning, categorical encoding, feature scaling, Elbow Method exploration, K-Means, and hierarchical clustering experiments.

### Features

Retail attributes including country, price, and quantity after preprocessing.

## Evaluation and current result

The notebook identifies 4 as the selected K-Means cluster count using the Elbow Method. Add a silhouette score and a short interpretation of each cluster to make the result easier to assess.

## Run locally

```bash
git clone https://github.com/MeehdiF/Product-Clustering-for-Online-Retail-Store.git
cd Product-Clustering-for-Online-Retail-Store
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python untitled.py
```

For notebook exploration, open the `.ipynb` file with Jupyter after installing the same dependencies.

## Limitations and next steps

The feature representation and encoding determine the clusters strongly. Future work should compare distance choices, use a documented aggregation level, and validate whether clusters lead to useful retail actions.

## Repository structure

- `README.md` — project context and reproducibility notes
- `requirements.txt` — Python dependencies used by the scripts
- `.ipynb` / `.py` files — analysis and model experiments

## License

See [`LICENSE`](LICENSE). Check the dataset's own terms separately; repository code licensing does not automatically license bundled data.
