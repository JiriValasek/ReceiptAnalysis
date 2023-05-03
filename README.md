# ReceiptAnalysis

Project for a receipt analysis of a dataset from kaggle. 

Full report (CS only) is [here](https://github.com/JiriValasek/ReceiptAnalysis/blob/main/report_cs.pdf)

## Requirements

- This project has not been optimized to run on a PC with any RAM size, 32Gb is thus recommended to analyze 100k of records.
- For more records, optimization or bigger RAM is necessary.
- Python v3.11.x (used through pyenv)

## Usage

1. Clone the repository
2. Register to Kaggle
3. Download dataset [eCommerce purchase history from electronics store](https://www.kaggle.com/datasets/mkechinov/ecommerce-purchase-history-from-electronics-store)
4. Place the dataset (kz.csv) into /data directory
5. Install poetry 
6. Install depencences `cd /path/to/ReceiptAnalysis && poetry install`
7. Run preprocessing `poetry run preprocessing`
8. Run clustering `poetry run clustering`
9. Generate associative rules `poetry run associative_rules`

## Outputs

- Saved numpy matrices are in /data
- Saved matplotlib figures are in /images
- Saved text outputs of the scripts are in /outputs
- Saved clusters and cluster rules are in /rules

## Used tutorials

- [Black, MyPy and Pylint](https://lynn-kwong.medium.com/use-black-mypy-and-pylint-to-make-your-python-code-more-professional-b594512f4362)
- [Pyenv and poetry](https://blog.pronus.io/en/posts/python/gerenciamento-de-versoes-ambiente-virtuais-e-dependencias-com-pyenv-e-poetry/)
