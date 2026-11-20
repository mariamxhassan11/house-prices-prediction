# Egyptian House Price Regression Project

A tabular regression project that estimates Egyptian property asking prices from listing attributes. The target variable is **Price**. The original CSV is preserved unchanged; cleaning is performed only in memory.

## Project structure

```text
Regression_Project/
├── data/original_dataset.csv
├── 01_EDA_and_Visualization.ipynb
├── 02_Preprocessing_and_Modeling.ipynb
├── 03_Streamlit_App/
│   ├── app.py
│   ├── best_model.pkl
│   └── model_metadata.json
├── 04_Project_Documentation.docx
├── requirements.txt
└── README.md
```

## Environment setup

Use Python 3.11 or newer in a virtual environment. From the project root:

```bash
python -m venv .venv
```

Windows CMD (important if multiple Python installations are installed):

```cmd
"C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe" -m pip install --upgrade pip
"C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe" -m pip install -r requirements.txt
```

Use the same Python executable for installation and launch. If your Python 3.11 installation is in a different location, replace the path above with that interpreter's full path. The plain `python` command may resolve to another installation (for example, MSYS2) that does not contain Streamlit.

Run both notebooks from top to bottom in Jupyter. The modeling notebook trains the regression models, performs GridSearchCV using training-only cross-validation, evaluates default and tuned pipelines on the same 20% test partition, and exports the selected pipeline and metadata.

## Launch the Streamlit application

From the project root, run:

```cmd
"C:\Users\Dell\AppData\Local\Programs\Python\Python311\python.exe" -m streamlit run "03_Streamlit_App/app.py"
```

If you use another Python installation, substitute its full executable path. Do not launch `app.py` with `python app.py`; Streamlit must be started with `-m streamlit run`.

## Using the application

- **Overview:** dataset and selected-model snapshot.
- **Data Insights:** target distribution, property-type counts, and city-level median prices.
- **Price Prediction:** enter the input features and request an estimate.
- **Model Performance:** compare default and tuned test results and review GridSearchCV results.

The app loads `best_model.pkl`; it does not retrain models. Predictions are estimates, not formal valuations.

## Modeling notes

The workflow compares Linear Regression, Lasso, Ridge, KNN, SVR, Decision Tree, Bagging, Random Forest, Extra Trees, Gradient Boosting, and AdaBoost regressors. Pipelines use numeric median imputation/scaling and categorical most-frequent imputation/one-hot encoding. A 3-fold shuffled CV search is performed on a reproducible training-only sample for practical runtime; selected pipelines are refitted on the full training partition. All models use the same 80/20 split (`random_state=42`). MAPE is omitted as a headline metric because relative errors can overemphasize low-priced listings across the broad target range.

Final evaluation numbers are written into `03_Streamlit_App/model_metadata.json` after the modeling notebook executes. See the Word documentation and Model Performance page for the actual run results.
