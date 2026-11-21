import os
import pandas as pd

def inspect_dataset():
    train_path = os.path.join("data", "student_query_train_5000.csv")
    test_path = os.path.join("data", "student_query_test_1000.csv")
    
    print("=" * 60)
    print("INSPECTING DATASET FILES")
    print("=" * 60)
    
    # 1. Train dataset
    if os.path.exists(train_path):
        train_df = pd.read_csv(train_path)
        print(f"\n[TRAIN DATASET]: {train_path}")
        print(f"Shape: {train_df.shape} (Rows: {train_df.shape[0]}, Columns: {train_df.shape[1]})")
        print(f"Columns: {list(train_df.columns)}")
        print("\nData Types:")
        print(train_df.dtypes)
        print("\nMissing Values:")
        print(train_df.isnull().sum())
        print(f"\nDuplicate Rows (overall): {train_df.duplicated().sum()}")
        print(f"Duplicate Query Texts: {train_df.duplicated(subset=['Student_Query']).sum() if 'Student_Query' in train_df.columns else 'N/A'}")
        
        for col in train_df.columns:
            if col != 'ID' and col != 'Student_Query':
                print(f"\nValue Counts for '{col}':")
                print(train_df[col].value_counts(dropna=False))
                print(f"Normalized Distribution for '{col}':")
                print(train_df[col].value_counts(normalize=True, dropna=False) * 100)
                
        print("\nSample 5 records from Train:")
        print(train_df.head(5).to_dict(orient="records"))
    else:
        print(f"ERROR: {train_path} not found!")

    # 2. Test dataset
    if os.path.exists(test_path):
        test_df = pd.read_csv(test_path)
        print(f"\n\n[TEST DATASET]: {test_path}")
        print(f"Shape: {test_df.shape} (Rows: {test_df.shape[0]}, Columns: {test_df.shape[1]})")
        print(f"Columns: {list(test_df.columns)}")
        print("\nData Types:")
        print(test_df.dtypes)
        print("\nMissing Values:")
        print(test_df.isnull().sum())
        print(f"\nDuplicate Rows (overall): {test_df.duplicated().sum()}")
        print(f"Duplicate Query Texts: {test_df.duplicated(subset=['Student_Query']).sum() if 'Student_Query' in test_df.columns else 'N/A'}")

        for col in test_df.columns:
            if col != 'ID' and col != 'Student_Query':
                print(f"\nValue Counts for '{col}':")
                print(test_df[col].value_counts(dropna=False))
                print(f"Normalized Distribution for '{col}':")
                print(test_df[col].value_counts(normalize=True, dropna=False) * 100)

        print("\nSample 5 records from Test:")
        print(test_df.head(5).to_dict(orient="records"))
    else:
        print(f"ERROR: {test_path} not found!")

if __name__ == "__main__":
    inspect_dataset()

