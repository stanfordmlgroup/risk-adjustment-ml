import argparse
import pandas as pd
import lightgbm as lgb
from pathlib import Path

from preprocess import preprocess


def main(args):
    # Load the model.
    model_path = Path(args.model_path)
    model = lgb.Booster(model_file=args.model_path)

    # Load the data.
    csv_path = Path(args.csv_path)
    df = pd.read_csv(csv_path)
    data = preprocess(df, args.sdh)

    data_num_predictors = data.shape[1]
    model_num_predictors = model.num_feature()
    if model_num_predictors != data_num_predictors:
        raise ValueError(f"Model expects {model_num_predictors} predictors," +
                         f"Got {data_num_predictors} predictors.")

    # Run the model.
    prediction = np.clip(model.predict(data), 0, 400000) + COST_ADJUSTMENT
    patient_costs = list(zip(df[PATIENT], prediction))
    print("Patient\tCost")
    for patient, cost in patient_costs:
        print(f"{patient}\t{cost}")
    print()
    
    print(f"Saving predicted cost data to {args.save_path}.")
    costs_df = pd.DataFrame(patient_costs,
                            columns=[PATIENT, "Cost"])
    costs_df.to_csv(args.save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ML for Risk Adjustment Prediction')
    parser.add_argument('--csv_path',
                        required=True,
                        type=str,
                        help='Path to csv with patient information.',)
    parser.add_argument('--model_path',
                        default='ml_model_nosdh.txt',
                        type=str,
                        help='Path to model file.')
    parser.add_argument('--save_path',
                        default="costs.csv",
                        type=str,
                        help='Path to output save file.')
    parser.add_argument('--sdh',
                        action="store_true",
                        help='Run SDH model.')
    main(parser.parse_args())
