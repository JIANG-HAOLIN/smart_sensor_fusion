import numpy as np



def linearly_interpolated_array(x: np.array) -> np.array:
    # Find indices of non-NaN values
    non_nan_indices = np.where(~np.isnan(x))[0]

    # Find the first and last non-NaN indices
    first_non_nan_index = non_nan_indices[0]
    last_non_nan_index = non_nan_indices[-1]

    # Create an array with linearly interpolated values
    interpolated_values = np.interp(np.arange(len(x)), non_nan_indices, x[non_nan_indices])

    # Replace NaN values in the original array with interpolated values
    result_array = np.where(np.isnan(x), interpolated_values, x)
    # print(result_array.shape)
    return result_array


# original_array = np.array([np.nan, 23, np.nan, np.nan, np.nan, 32, np.nan, np.nan, np.nan, np.nan, 12, np.nan])
# linearly_interpolated_array(original_array)


