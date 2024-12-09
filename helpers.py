import numpy as np
import features
from itertools import product

## From the original notebook
def get_label(file_name):
  '''
  Function to retrieve output labels from filenames
  '''
  if 'ROC' in file_name:
    label=0
  elif 'LES' in file_name:
    label=1
  elif 'DCB' in file_name:
    label=2
  elif 'PRV' in file_name:
    label=3
  elif 'VLD' in file_name:
    label=4
  elif 'DTA' in file_name:
    label=5
  else:
    raise ValueError('invalid file name')
  return label

def save_features_to_npz(train_features, train_labels, 
                        test_clean_features, test_clean_labels, 
                        test_noisy_features, test_noisy_labels, 
                        output_file):
    """
    Saves train, test_clean, and test_noisy features and labels to an .npz file.

    Parameters:
        train_features (np.ndarray): Training feature matrix.
        train_labels (np.ndarray): Training labels.
        test_clean_features (np.ndarray): Clean test feature matrix.
        test_clean_labels (np.ndarray): Clean test labels.
        test_noisy_features (np.ndarray): Noisy test feature matrix.
        test_noisy_labels (np.ndarray): Noisy test labels.
        output_file (str): Path to the output .npz file.

    Returns:
        None
    """
    np.savez_compressed(
        output_file,
        train_features=train_features,
        train_labels=train_labels,
        test_clean_features=test_clean_features,
        test_clean_labels=test_clean_labels,
        test_noisy_features=test_noisy_features,
        test_noisy_labels=test_noisy_labels
    )
    print(f"Features and labels saved to {output_file}")

def create_filename_and_label(params, feature_label):
    parts = [f"{key}-{value}" for key, value in params.items()]
    combined_string = "_".join(parts)
    combined_string = f"{feature_label}_{combined_string}"
    return combined_string

def grid_search(param_grid, feature_extractor):
  keys, values = zip(*param_grid.items())
  for combination in product(*values):
    params = dict(zip(keys, combination))
    filename_and_label = create_filename_and_label(params, feature_extractor.__name__)
    
    train_df, y_train, clean_test_df, y_clean_test, noisy_test_df, y_noisy_test = features.prepare_datasets(
      feature_extractor=feature_extractor,
      feature_label=filename_and_label,
      extractor_params=params
    )
    
    save_features_to_npz(train_df, y_train, clean_test_df, y_clean_test, noisy_test_df, y_noisy_test, f"features/{filename_and_label}.npz")

def load_and_flatten_npz(file_path):
    """
    Loads an .npz file and flattens each feature set.

    Parameters:
        file_path (str): Path to the .npz file.

    Returns:
        dict: A dictionary with flattened feature sets and corresponding labels.
    """

    data = np.load(file_path)
    
    train_features = data["train_features"].reshape(data["train_features"].shape[0], -1)
    test_clean_features = data["test_clean_features"].reshape(data["test_clean_features"].shape[0], -1)
    test_noisy_features = data["test_noisy_features"].reshape(data["test_noisy_features"].shape[0], -1)
    
    train_labels = data["train_labels"]
    test_clean_labels = data["test_clean_labels"]
    test_noisy_labels = data["test_noisy_labels"]
    
    return {
        "train_features": train_features,
        "train_labels": train_labels,
        "test_clean_features": test_clean_features,
        "test_clean_labels": test_clean_labels,
        "test_noisy_features": test_noisy_features,
        "test_noisy_labels": test_noisy_labels
    }

def combine_and_save_features(combined_data_per_name, output_file):
    final_features = {}
    
    for key in combined_data_per_name:
        if 'labels' not in key:
            combined_array = np.concatenate(combined_data_per_name[key], axis=1)
            final_features[key] = combined_array
        else:
            final_features[key] = combined_data_per_name[key][0]
        
    np.savez(output_file, **final_features)
    print(f"Features saved to {output_file}")
