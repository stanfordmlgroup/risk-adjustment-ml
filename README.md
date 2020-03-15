## Machine Learning for Risk Adjustment

Code for the paper:

  > Incorporating Machine Learning and Social Determinants of Health Indicators into Prospective Risk Adjustment for Health Plan Payments
  >
  > Jeremy A. Irvin, Andrew A. Kondrich, Michael Ko, Pranav Rajpurkar, Behzad Haghgoo, Bruce E. Landon, Robert L. Phillips, Stephen Petterson, Andrew Y. Ng, Sanjay Basu 

Given demographic and diagnosis information (and optional ZIP code) of a patient, predict prospective annual healthcare spending using a trained LightGBM regression model as described in our paper.

## Usage

### Environment Setup
1. Please install [Anaconda](https://docs.conda.io/en/latest/miniconda.html) in order to create a Python environment.
2. Clone this repo (from the command-line, `git clone git@github.com:stanfordmlgroup/risk-adjustment-ml.git`.
3. Navigate to the cloned repo: `cd risk-adjustment-ml`.
4. Create the environment: `conda env create -f environment.yml`.
5. Activate the environment: `source activate ra-ml`.

### Preprocessing
We format the data in a CSV with columns: Patient (Integer), Age (Integer), Sex (Char: M/F), 5-digit ZIP code (Integer, optional), Cost (Double, optional) as well as a list of ICD-10 Diagnoses (Comma-delimited quote string). An optional "Zipcode" column is needed to run the SDH-based model. Each patient should be placed on their own row. The cost column is used only in training.

The following is a toy example CSV (see [the example](https://github.com/stanfordmlgroup/risk-adjustment-ml/blob/master/demo.csv)):

|Patient|Age|Sex|Zipcode|ICD10              |Cost |
|-------|---|---|-------|-------------------|-----|
|1      |46 |M  |95120  |F32.0,D64.0,D64.81 |89.0 |
|2      |64 |F  |85001  |L25.0              |1.0  |
|3      |56 |F  |72201  |R10.0,R10.30,R53.81|2.0  |
|4      |18 |M  |80201  |                   |200.0|
|5      |50 |F  |800    |N15.9,N30.00       |42.0 | 

### Training

You can train either a LightGBM or Linear regression model on your data using the following script and flags. This will also perform a hyperparameter search using [hyperopt](https://github.com/hyperopt/hyperopt). 

`python train.py --model [LinReg, LightGBM] --train_path path/to/training/data --test_path path/to/test/data --tuning_metric [mse, mae, huber] --regularizer [regular, l1, l2]`

If you wish to run the LightGBM model on GPU, add the `--gpu` flag. Once trained, the script will output the model's R2 and MAE scores on the test data. You can bottom code the predictions via `--bottom_coding [value]`. 

### Inference

We include the option to run our trained models on your data with the following script:

`python run.py --csv_path path/to/data --model_path path/to/saved/model --save_path path/to/save/output`

Including `--sdh` will automatically run the SDH-based model. This requires ZIP code data for each patient row. For example, to run the SDH-based model on the demo CSV,

`python run.py --csv_path demo.csv --model_path ml_model_sdh.txt --save_path costs.csv --sdh`
