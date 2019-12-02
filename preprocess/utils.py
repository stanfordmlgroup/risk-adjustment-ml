""" Utils functions for general preprocessing tasks """
import pandas as pd
import numpy as np
import scipy as sp
import scipy.sparse as sparse
from datetime import datetime, date
from dateutil.relativedelta import relativedelta

# File paths.
ICD2CCS_PATH = "./preprocess/icd10cm_to_ccs.csv"
SDH_TABLE = "./preprocess/sdh_variables.csv"

# General constants.
PATIENT = "Patient"
# Cost adjustment applied so that predictive ratio
# was 1.0 on the top-coded set.
COST_ADJUSTMENT = -89.10665304360737

# Age and sex constants.
AGE_CUTOFFS = [25, 35, 45, 55, 60, 65]
NUM_AGE_BUCKETS = len(AGE_CUTOFFS)
AGE = "Age"
SEX = "Sex"

# Diagnosis constants.
ICD10 = "ICD10"
ICD10CM = "ICD10CM"
DIAG_NAN = "DIAGMAP_NO_DIAG"
DIAG_UNK = "DIAGMAP_KEY_ERROR_UNK"
CCS = "CCS"
# Dropped CCS codes based on frequency in the dataset.
DROPPED_CCS_CODES = [1, 12, 13, 16, 17, 18, 20, 21, 25, 27,
                     28, 30, 31, 32, 33, 34, 35, 37, 41, 43,
                     56, 76, 77, 78, 79, 85, 107, 116, 129, 132,
                     148, 156, 177, 178, 179, 180, 184, 185,
                     187, 188, 190, 192, 194, 207, 214, 215, 216,
                     218, 219, 220, 221, 222, 223, 224, 226, 227,
                     228, 234, 241, 242, 243, 248, 249, 656, 2601,
                     2602, 2604, 2605, 2606, 2608, 2609, 2610,
                     2612, 2613, 2614, 2615, 2616, 2618, 2619]

# SDH variable constants.
SDH2NORM = {"population_african": ("population_norm", "Population African American, %"),
            "population_asian": ("population_norm", "Population Asian, %"),
            "population_hispanic": ("population_norm", "Population Hispanic or Latino, %"),
            "population_native": ("population_norm", "Population American Indian and Alaska Native, %"),
            "population_white": ("population_norm", "Population White, %"),
            "parents": ("parents_norm", "Families with Single Parent, %"),
            "english": ("english_norm", "Population Speak English Less than \"Very Well\", %"),
            "household": (None, "Median Income in the Past 12 Months, $"),
            "highschool": ("education_norm", "Population Obtained High School Diploma, %"),
            "bachelors": ("education_norm", "Population Obtained Bachelor's Degree, %"),
            "poverty_50": ("poverty_norm", "Families Under 0.5 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "poverty_75": ("poverty_norm", "Families Between 0.5 and 0.74 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "poverty_99": ("poverty_norm", "Families Between 0.75 and 0.99 Ratio of Income to Poverty Level in the Past 12 Months, %"),
            "food": ("food_norm", "Families Received Food Stamps/Snap in the Past 12 months, %"),
            "woHealth": ("woHealth_norm", "Population Without Health Insurance Coverage, %"),
            "unemployment": ("unemployment_norm", "Population Unemployed, %"),
            "gini": (None, "Gini Index of Income Inequality")}
ZIPCODE = "Zipcode"
ZIP_UNK_KEY = -99999
ZIP_UNK = 'ZIP_UNK'
SDH_VARIABLES = list(SDH2NORM.keys()) + [ZIP_UNK]


def sex_age_bucketer(age, sex):
    """Assign an index of an age-sex bucket to an age and sex.
    F [0, 2) is index 0.
    F [2, 6) is index 1.
    ...
    M [0, 2) is index num_age_buckets
    M [2, 6) is index num_age_buckets + 1
    ...
    age is between 0 and 94, sex is either F (female) or M (male).
    """
    index = 0
    if sex == "F":
        sex_num = 0
    elif sex == "M":
        sex_num = 1
    else:
        raise ValueError(f"Sex {sex} not supported.")
    for age_cutoff in AGE_CUTOFFS:
        if age < age_cutoff:
            return index + sex_num * NUM_AGE_BUCKETS
        index += 1

    print("Warning: Age outside of [0, 65). Treating as final age bucket.")
    return (index - 1) + sex_num * NUM_AGE_BUCKETS


def onehot_sparseify(series, get_features=False):
    """
    Transform pandas series into a sparse matrix of onehot encoding.
    Maps NaN to an empty vector (by dropping it).
    :param series: Series with day (Categorical series if necessary)
    :param get_features: Boolean for returning feature names
    :return: A sparse representation of the series in a Series
    """
    one_hot = pd.get_dummies(series, sparse=True).to_sparse()
    one_hot = one_hot.reset_index(drop=True)
    sparse_matrix = sparse.csr_matrix(one_hot.to_coo())
    if get_features:
        return sparse_matrix, one_hot.columns.values
    else:
        return sparse_matrix


def get_sex_age_features(split):
    """Get age-sex features as done in 2017 risk adjusment model
    and Ash et al."""
    age_sex_cols = [AGE, SEX]
    sex_age_bucket = \
        split[age_sex_cols].apply(lambda row: sex_age_bucketer(row[AGE],
                                                               row[SEX]),
                                  axis=1)
    # There are 2*len(age sex bucket cutoffs). each cutoff defines one bucket,
    # and each sex has its buckets (hence the times 2).
    sex_age_d = pd.Categorical(sex_age_bucket,
                               categories=list(range(2 * NUM_AGE_BUCKETS)))

    sex_age_matrix = onehot_sparseify(sex_age_d)

    return sex_age_matrix


def load_icd2ccs(path):

    icd10_to_ccs = pd.read_csv(path)
    icd10_to_ccs = icd10_to_ccs.append({ICD10CM: DIAG_UNK,
                                        CCS: -1},
                                        ignore_index=True)

    dropped_ccs_row_inds = icd10_to_ccs[CCS].isin(DROPPED_CCS_CODES)
    icd10_to_ccs.drop(icd10_to_ccs.index[dropped_ccs_row_inds],
                      inplace=True)
    icd10_to_ccs.reset_index(inplace=True, drop=True)
    ccs_dummies = pd.get_dummies(icd10_to_ccs[CCS])
    ccs_codes = list(ccs_dummies)

    icd10_to_ccs[CCS] = list(sparse.csr_matrix(ccs_dummies))
    zeros_vector = sparse.csr_matrix(icd10_to_ccs[CCS].iloc[0].shape)
    icd10_to_ccs = icd10_to_ccs.append({ICD10CM: DIAG_NAN,
                                        CCS: zeros_vector},
                                       ignore_index=True)
    icd10_to_ccs = icd10_to_ccs.set_index(ICD10CM)

    assert len(set(icd10_to_ccs[CCS].apply(lambda x: x.shape))) == 1
    return icd10_to_ccs, ccs_codes


def get_diag_features(df):
    # Explode out ICD10.
    icd = df[ICD10].str.split(",")
    icd = icd.apply(pd.Series)
    icd[PATIENT] = df[PATIENT]
    icd = icd.melt(id_vars=PATIENT).drop(['variable'], axis=1)
    icd = icd.rename({'value': ICD10CM}, axis=1)
    icd.fillna(DIAG_NAN, inplace=True)
    icd = icd[~icd[ICD10CM].isna()]
    icd[ICD10CM] = icd[ICD10CM].str.replace(".", "")

    # Convert to CCS.
    icd2ccs, diag_names = load_icd2ccs(ICD2CCS_PATH)
    icd10codes = icd2ccs.index
    icd[ICD10CM].where(icd[ICD10CM].isin(icd10codes),
                       DIAG_UNK,
                       inplace=True)
    icd[CCS] = icd2ccs.loc[icd[ICD10CM]].reset_index(drop=True).values.flatten()

    # Aggregate within patients.
    icd = icd.groupby(PATIENT).sum()
    diag = sparse.vstack(icd[CCS])
    
    return diag


def load_and_normalize_sdh():
    """Loads the output of acs_query.R (at SDH_TABLE), aggregates from geoid to zip,
    then normalizes using the normalizers defined in constants (SDH2NORM).
    Additionally adds a row for unknown zip codes and a column to indicate this."""

    sdh_table = pd.read_csv(SDH_TABLE, dtype={'zip': str})
    sdh_table.rename(columns={'zip': ZIPCODE},
                     inplace=True)

    for sdh_var, (sdh_norm, _) in SDH2NORM.items():
        if sdh_norm:
            sdh_table[sdh_var] = sdh_table[sdh_var] / sdh_table[sdh_norm]

    sdh_table = sdh_table[list(SDH2NORM.keys()) + [ZIPCODE]]

    median_sdh = sdh_table.median()
    sdh_table.fillna(median_sdh, inplace=True)

    # Convert ZIP to int
    sdh_table[ZIPCODE] = sdh_table[ZIPCODE].astype(float).astype(int)

    # Add column where unknown zip is 1 and known zips are 0
    sdh_table[ZIP_UNK] = 0

    # Add row where unknown zip corresponds to median sdh
    median_df = pd.DataFrame(median_sdh).T
    median_df[ZIPCODE] = ZIP_UNK_KEY
    median_df[ZIP_UNK] = 1
    sdh_table = sdh_table.append(median_df).reset_index(drop=True)

    return sdh_table


def get_sdh_features(df):
    sdh_table = load_and_normalize_sdh()
    nf = ~df[ZIPCODE].isin(sdh_table[ZIPCODE])
    if nf.sum() != 0:
        print("Warning - the following patient(s) have ZIP " + 
              "codes without SDH variables:")
        unk_patient_zips = df[nf][[PATIENT, ZIPCODE]].apply(tuple,
                                                            axis=1).values
        print("Patient\tZipcode")
        for patient, zip in unk_patient_zips:
            print(f"{patient}\t{zip}")
        print("Using the median values for SDH variables across ZIP codes.")
        print()
    df.at[nf, ZIPCODE] = ZIP_UNK_KEY
    df = pd.merge(sdh_table, df, right_on=[ZIPCODE],
                  left_on=[ZIPCODE], how='right')

    sdh_features_list = []
    for var in SDH_VARIABLES:
        sdh_features_list.append(sparse.csr_matrix(df[var]).T)

    return sparse.hstack(sdh_features_list)
