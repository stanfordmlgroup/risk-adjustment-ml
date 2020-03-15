## Machine Learning for Risk Adjustment

Code for the paper:

  > Incorporating Machine Learning and Social Determinants of Health Indicators into Prospective Risk Adjustment for Health Plan Payments
  >
  > Jeremy A. Irvin, Andrew A. Kondrich, Michael Ko, Pranav Rajpurkar, Behzad Haghgoo, Bruce E. Landon, Robert L. Phillips, Stephen Petterson, Andrew Y. Ng, Sanjay Basu 

Given patient demographic and diagnoses information for a collection of patients, predict prospective insurance cost using a trained LightGBM Regression Model as described in our paper.

## Usage

### Environment Setup
1. Please have [Anaconda](https://docs.conda.io/en/latest/miniconda.html) installed to create a Python virtual environment.
2. Clone repo with `https://github.com/stanfordmlgroup/risk-adjustment-ml`.
3. Go into the cloned repo: `cd risk-adjustment-ml`.
4. Create the environment: `conda env create -f environment.yml`.
5. Activate the environment: `source activate ra-ml`.

### Preprocessing
We format the data in the form of a CSV with columns: Patient (Integer), Age (Integer), Sex (Char: M/F), 5-digit ZIP code (Integer, optional), as well as a list of ICD-10 Diagnoses (Comma-delimited quote string). An optional ZIP column is needed to run the SDH-based model. Each patient should be put on their own row. 

The following is a toy example CSV:


| Patient         | Age | Sex | Zipcode | ICD10                 | 
|-----------------|-----|-----|---------|-----------------------| 
| 1               | 46  | M   | 95120   | "F32.0,D64.0,D64.81"  | 
| 2               | 64  | F   | 85001   | "L25.0"               | 
| 3               | 56  | F   | 72201   | "R10.0,R10.30,R53.81" | 
| 4               | 18  | M   | 80201   | ""                    |
| 5               | 50  | F   | 800     | "N15.9,N30.00"        | 


### Evaluation

The script can be run in python using the following flags:

`python run.py --csv_path path/to/data --model_path path/to/saved/weights --save_path path/to/save/output`

Including `--sdh` will automatically run the SDH-based model. This requires zipcode data for each patient row. Model weights can be provided upon request.

