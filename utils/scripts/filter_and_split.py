import pandas as pd
import numpy as np
import random
import os

def filter_and_split_data(input_csv, output_dir='./output', seed=42):
    """
    Filter the input CSV, a summary statistics table output by DANCE,
    according to specific criteria and split it into train, validation and test sets. 
    
    Criteria:
    - at least 5 conformations per collection (nb_members >=5)
    - at least 2A pairwise max RMSD (rmsd_max >=2)
    - length of the query between 30 ans 1000 residues (30 <= ref_len <= 1000)
    - variance explained by the first mode at least 80% (ref_.var_1st >= 0.8)
    - at least 12 residues involved in the first principal component (ref_col_1st * ref_len >= 12)
    
    Args:
        input_csv: path to the input CSV file  
        output_dir: directory to save output files
        seed: seed for reproducibility -- PLEASE NOTE THIS IS NOT NECESSARILY THE SEED WE USED
    """
    # Create output directory if it does not exist 
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the data
    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # Print initial information
    initial_count = len(df)
    print(f"Initial number of collections: {initial_count}")
    
    # Apply filtering criteria
    print("Applying filtering criteria...")
    
    # Rename column of interest with % symbol
    df = df.rename(columns={
        "ref_%var_1st": "ref_var_1st",
    })
    
    # Make a copy of the data
    filtered_df = df.copy()
    
    # Filter 1: at least 5 conformations per collection
    filtered_df = filtered_df[filtered_df['nb_members'] >= 5]
    print(f"After filtering nb_members >= 5: {len(filtered_df)} remaining collections")
    
    # Filter 2: at least 2A pairwise max RMSD
    filtered_df = filtered_df[filtered_df['rmsd_max'] >= 2]
    print(f"After filtering rmsd_max >= 2: {len(filtered_df)} remaining collections")
    
    # Filter 3: length of the query between 30 ans 1000 residues
    filtered_df = filtered_df[(filtered_df['ref_len'] >= 30) & (filtered_df['ref_len'] <= 1000)]
    print(f"After filtering 30 <= ref_len <= 1000: {len(filtered_df)} remaining collections")
    
    # Filter 4: variance explained by the first mode at least 80%
    filtered_df = filtered_df[filtered_df['ref_var_1st'] >= 0.8]
    print(f"After filtering ref_var_1st >= 0.8: {len(filtered_df)} remaining collections")
    
    # Filter 5: at least 12 residues involved in the first principal component 
    filtered_df = filtered_df[filtered_df['ref_col_1st'] * filtered_df['ref_len'] >= 12]
    print(f"After filtering ref_col_1st * ref_len >= 12: {len(filtered_df)} remaining collections")
    
    final_count = len(filtered_df)
    if final_count == 0:
        print("None of the collections satisfies all criteria.")
        return
    
    print(f"\nFiltering step done. {final_count} collections over {initial_count} satisfy all criteria ({final_count/initial_count:.2%}).")
    
    # Splitting step 
    print("\nRandom splitting into train, validation and test (70%, 15%, 15%)...")
    
    random.seed(seed)
    all_ids = list(filtered_df['name_file'])
    # compute number of collections in each set
    train_size = int(0.7 * len(all_ids))
    val_size = int(0.15 * len(all_ids))
    # extract training examples
    train_ids = random.sample(all_ids, train_size)
    remaining_ids = list(set(all_ids) - set(train_ids))
    # extract validation examples in the remaining subset
    val_ids = random.sample(remaining_ids, val_size)
    # the rest is for testing
    test_ids = list(set(remaining_ids) - set(val_ids))
    # First divide between train and val+test
   # train_df, temp_df = train_test_split(filtered_df, test_size=0.3, random_state=seed)
    # Then divide between val and test
    #val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=seed)
    
    print(f"Training set: {len(train_ids)} collections ({len(train_ids)/final_count:.2%})")
    print(f"Validation set: {len(val_ids)} collections ({len(val_ids)/final_count:.2%})")
    print(f"Test set: {len(test_ids)} collections ({len(test_ids)/final_count:.2%})")
    
    # Save the splits
    train_output = os.path.join(output_dir, 'train.txt')
    val_output = os.path.join(output_dir, 'val.txt')
    test_output = os.path.join(output_dir, 'test.txt')
    filtered_output = os.path.join(output_dir, 'filtered_data.csv')
    
    with open(train_output, "w") as outfile:
        outfile.write("\n".join(train_ids))

    with open(val_output, "w") as outfile:
        outfile.write("\n".join(val_ids))

    with open(test_output, "w") as outfile:
        outfile.write("\n".join(test_ids))

    filtered_df.to_csv(filtered_output, index=False)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter the input CSV, a summary statistics table output by DANCE, according to specific criteria and split it into train, validation and test sets. ")
    parser.add_argument("input_csv", help="Path to the input CSV file ")
    parser.add_argument("--output_dir", default="./output", help="directory to save output files (by default: ./output)")
    parser.add_argument("--seed", type=int, default=42, help="seed for reproducibility (by default: 42)")
    
    args = parser.parse_args()
    
    filter_and_split_data(args.input_csv, args.output_dir, args.seed)