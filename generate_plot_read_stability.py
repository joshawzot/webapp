from db_operations import DB_CONFIG
from sqlalchemy import create_engine
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import io
import base64
import os
from io import BytesIO

def generate_plot_read_stability(table_names, database_name, form_data):
    # --- Combined setup from old and new versions ---
    engine_url = f"mysql+mysqlconnector://{DB_CONFIG['DB_USER']}:{DB_CONFIG['MYSQL_PASSWORD']}@{DB_CONFIG['DB_HOST']}/{database_name}"
    engine = create_engine(engine_url, pool_size=10, max_overflow=5)

    # Handle table_names if passed as a string
    if isinstance(table_names, str):
        table_names = table_names.split(',')
        table_names = [name.strip() for name in table_names]

    data_frames = {}
    arrays = []

    for table_name in table_names:
        df = pd.read_sql_table(table_name, con=engine)
        data_frames[table_name] = df
        arrays.append(df.values)

    if not arrays:
        raise ValueError("No valid data tables found or processed.")

    # --- Old code approach: stack along new axis and compute mean/noise ---
    combined_array = np.stack(arrays, axis=-1)
    array_mean = np.mean(combined_array, axis=2)
    noise = np.std(combined_array, axis=2)

    # Retrieve user inputs from form_data
    max_y_value = form_data.get('input_integer', 40)
    state_pattern = form_data.get('state_pattern')

    # Map state_pattern to file paths (old code logic)
    pattern_files = {
        "3x4_4states_debug": "/home/admin2/webapp_2/State_pattern_files/3x4_4states_debug.npy",
        "1296x64_rowbar_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_rowbar_4states.npy",
        "248x248_checkerboard_4states": "/home/admin2/webapp_2/State_pattern_files/248x248_checkerboard_4states.npy",
        "1296x64_Adrien_random_4states": "/home/admin2/webapp_2/State_pattern_files/1296x64_Adrien_random_4states.npy",
        "248x248_1state": "/home/admin2/webapp_2/State_pattern_files/248x248_1state.npy",
        "1296x64_1state": "/home/admin2/webapp_2/State_pattern_files/1296x64_1state.npy",
        "248x248_16states": "/home/admin2/webapp_2/State_pattern_files/248x248_16states.npy"
    }

    file_path = pattern_files.get(state_pattern)
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"Pattern file for '{state_pattern}' not found.")

    target = np.load(file_path)
    target = target[:, :]

    # Create scatter plot with color by target
    def plot_read_noise(noise_data, mean_data, target_data):
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        try:
            mean_flat = mean_data.flatten()
            noise_flat = noise_data.flatten()
            target_flat = target_data.flatten()

            if len(mean_flat) != len(noise_flat) or len(mean_flat) != len(target_flat):
                raise ValueError("Dimensions of array_mean, noise, and target do not match")

            norm = mcolors.Normalize(vmin=np.min(target_flat), vmax=np.max(target_flat))
            scatter = ax.scatter(mean_flat, noise_flat, c=target_flat, cmap='viridis', s=10, edgecolor='none')
            fig.colorbar(scatter, ax=ax, label='Target')

            ax.set_title("Read Noise")
            ax.set_ylabel("Sigma")
            ax.set_ylim(0, int(max_y_value))

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            scatter_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return scatter_base64
        finally:
            plt.close(fig)
            if 'buf' in locals():
                buf.close()

    # Create density plot
    def create_density_plot(x, y):
        # Create a new figure instance for this plot
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        try:
            sns.kdeplot(x=x, y=y, ax=ax, cmap='Blues', fill=True, cbar=True, thresh=0.0001)
            ax.set_title('2D Density Plot')
            ax.set_xlabel('Mean (Flattened Values)')
            ax.set_ylabel('Std Dev (Flattened Values)')
            ax.set_ylim(0, 6)  # Set y-axis limit from 0 to 6

            buf = BytesIO()
            fig.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            density_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            return density_base64
        finally:
            plt.close(fig)
            if 'buf' in locals():
                buf.close()

    # Generate both plots
    old_plot_base64 = plot_read_noise(noise, array_mean, target)

    # Compute data for density plot
    data_list = [df.values for df in data_frames.values()]
    combined_data_new = np.array(data_list)
    x = np.mean(combined_data_new, axis=0).flatten()
    y = np.std(combined_data_new, axis=0).flatten()
    new_plot_base64 = create_density_plot(x, y)

    return [old_plot_base64, new_plot_base64] 