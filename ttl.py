import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from fpdf import FPDF
from scipy.stats import zscore, kurtosis,probplot,pearsonr
from ydata_profiling import ProfileReport
from sklearn.impute import KNNImputer
import os
class analyisisToolkit:
    
    
    
    
    # Dummy function for correcting data types, replace with actual implementation
    def generate_business_report(df, report_filename="business_data_report.pdf"):
    
        # Initialize PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
    
        # Title and Introduction
        pdf.cell(200, 10, txt="Business Data Preparation Report", ln=True, align="C")
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            "This report summarizes the key steps taken to prepare the dataset for analysis. "
            "The objective is to ensure data quality, consistency, and readiness for business insights."
        ))
        pdf.ln(10)
    
        # Data Overview
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="1. Data Overview", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            f"Initial dataset contained {df.shape[0]} rows and {df.shape[1]} columns.\n"
            f"Summary of dataset:\n{df.describe(include='all').T.to_string()}\n"
        ))
        pdf.ln(10)
    
        # Key Cleaning Steps
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="2. Key Cleaning Steps", ln=True)
        pdf.set_font("Arial", size=12)
    
        # Missing Values
        missing = df.isnull().sum()
        missing_columns = missing[missing > 0]
        if not missing_columns.empty:
            pdf.multi_cell(0, 10, txt=(
                f"Missing values were detected in {len(missing_columns)} columns.\n"
                "Appropriate imputation techniques were applied based on data types and distributions."
            ))
        else:
            pdf.multi_cell(0, 10, txt="No missing values detected.")
        pdf.ln(10)
    
        # Duplicate Removal
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            pdf.multi_cell(0, 10, txt=f"Removed {duplicate_count} duplicate rows from the dataset.")
        else:
            pdf.multi_cell(0, 10, txt="No duplicate rows detected.")
        pdf.ln(10)
    
        # Outliers and Transformations
        pdf.cell(200, 10, txt="3. Outliers and Data Transformations", ln=True)
        outlier_columns = []  # Placeholder for any specific handling logic
        if outlier_columns:
            pdf.multi_cell(0, 10, txt=f"Outliers were detected and capped in the following columns: {outlier_columns}")
        else:
            pdf.multi_cell(0, 10, txt="No significant outliers detected.")
        pdf.ln(10)
    
        # Visualization Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="4. Data Insights and Visualizations", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt="Visualizations of key metrics and distributions are included below.")
        
        # Example: Correlation Heatmap
        correlation = df.select_dtypes(include=[np.number]).corr()
        if not correlation.empty:
            heatmap_path = "correlation_heatmap.png"
            sns.heatmap(correlation, annot=True, cmap="coolwarm")
            plt.savefig(heatmap_path)
            pdf.image(heatmap_path, x=10, w=180)
            plt.close()
            os.remove(heatmap_path)  # Clean up the saved image
    
        # Conclusion and Recommendations
        pdf.set_font("Arial", "B", 14)
        pdf.cell(200, 10, txt="5. Conclusion and Recommendations", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=(
            "The data has been cleaned and prepared for analysis. Key steps included handling missing values, "
            "removing duplicates, and normalizing numeric data. Next steps involve further exploration or predictive modeling."
        ))
    
        # Save Report
        pdf.output(report_filename)
        print(f"Report saved as {report_filename}")
    
    def plot_missing_values(df, handle_missing=False):
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if not missing.empty:
            missing.plot(kind='bar', color='skyblue')
            plt.title('Missing Values Per Column', fontsize=14)
            plt.ylabel('Count', fontsize=12)
            plt.xlabel('Columns', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
            if handle_missing:
                filling_method = input("Would you like to fill missing values? (mean/median/mode/drop): ").strip().lower()
                if filling_method == "mean":
                    df.fillna(df.mean(), inplace=True)
                elif filling_method == "median":
                    df.fillna(df.median(), inplace=True)
                elif filling_method == "mode":
                    df.fillna(df.mode().iloc[0], inplace=True)
                elif filling_method == "drop":
                    df.dropna(inplace=True)
                else:
                    print("Invalid method chosen. No changes applied.")
        else:
            print("No missing values to visualize.")
    def plot_correlation_heatmap(df):
        numeric_df = df.select_dtypes(include=[np.number])  # Select numeric columns
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]  # Drop low-variance columns
        if numeric_df.shape[1] > 1:
            corr = numeric_df.corr()
            plt.figure(figsize=(12, 8))  # Increase figure size for readability
            sns.heatmap(
                corr, 
                annot=True, 
                cmap="coolwarm", 
                fmt=".2f", 
                annot_kws={"size": 8},  # Adjust annotation font size
                linewidths=0.5,  # Add grid lines for clarity
                cbar_kws={'shrink': 0.75}  # Shrink color bar for better alignment
            )
            plt.title("Correlation Heatmap", fontsize=16)
            plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
            plt.tight_layout()  # Prevent labels from being cut off
            plt.show()
        else:
            print("Not enough valid numeric columns for a correlation heatmap.")
    def plot_skewness(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        skew_vals = df[numeric_cols].skew().dropna()
        if not skew_vals.empty:
            skew_vals = skew_vals[abs(skew_vals) > 0.5]  # Focus on highly skewed columns
            if not skew_vals.empty:
                skew_vals.plot(kind='bar', color='salmon')
                plt.title("Skewness of Numeric Columns (Skew > 1)")
                plt.ylabel("Skewness")
                plt.show()
            else:
                print("No significant skewness to display.")
        else:
            print("No numeric columns to visualize skewness.")
    def visualize_outliers(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            plt.figure(figsize=(8, 4))
            sns.boxplot(x=df[col], color='skyblue')
            plt.title(f"Box Plot for {col} (Outliers Highlighted)")
            plt.xlabel(col)
            plt.show()
            
    def handle_outliers(df, method='zscore', threshold=3):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs(zscore(df[col].dropna()))
                outliers = df[col][z_scores > threshold]
                print(f"Outliers in {col}: {outliers}")
                df[col] = df[col].where(z_scores <= threshold, np.nan)
            elif method == 'iqr':
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_condition = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                outliers = df[col][outlier_condition]
                print(f"Outliers in {col}: {outliers}")
                df[col] = df[col].where(~outlier_condition, np.nan)
        return df
        
    def plot_pairplot(df):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            sns.pairplot(df[numeric_cols])
            plt.show()
            
    def plot_distribution(series, title="Distribution Plot", figsize=(10, 6)):
        """
        Plot a distribution plot (histogram + KDE) for a numeric column.
    
        Parameters:
            series (pd.Series): The numeric column to plot.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
        """
        plt.figure(figsize=figsize)
        sns.histplot(series, kde=True, bins=30, color='skyblue')
        plt.title(title, fontsize=14)
        plt.xlabel(series.name, fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.show()
        
    def plot_violin_comparison(df, numeric_col, categorical_col, title="Violin Plot"):
        """
        Plot a violin plot to compare distributions across categories.
        """
        plt.figure(figsize=(20, 20))
        sns.violinplot(x=categorical_col, y=numeric_col, data=df, hue=categorical_col, palette="coolwarm", legend=False)
        plt.title(title)
        plt.xlabel(categorical_col)
        plt.ylabel(numeric_col)
        plt.show()
        
    def plot_swarm(df, numeric_col, categorical_col, title="Swarm Plot"):
        """
        Plot a swarm plot to visualize individual data points.
        """
        plt.figure(figsize=(14, 8))
        sns.swarmplot(x=categorical_col, y=numeric_col, data=df,  hue=categorical_col, palette="coolwarm", legend=False, size=3)
        plt.title(title)
        plt.xlabel(categorical_col)
        plt.ylabel(numeric_col)
        plt.show()
        
    def plot_missing_heatmap(df, title="Missing Values Heatmap"):
        """
        Plot a heatmap to visualize missing values in the dataset.
        """
        plt.figure(figsize=(14, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title(title)
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.show()
    
    def plot_qq(series, title="QQ Plot"):
        """
        Plot a QQ plot to check for normality.
        """
        # Convert the series to numeric, coercing errors to NaN
        series_numeric = pd.to_numeric(series, errors='coerce')
        # Drop NaN values (non-numeric data)
        series_numeric = series_numeric.dropna()
        if series_numeric.empty:
            print(f"Skipping QQ plot for '{series.name}': No numeric data available.")
            return
        # Plot the QQ plot
        plt.figure(figsize=(10, 8))
        probplot(series_numeric, dist="norm", plot=plt)
        plt.title(title)
        plt.show()    
        
    def plot_scatter_matrix(df, numeric_cols=None, title="Scatterplot Matrix", figsize=(20, 20), fontsize=10):
        """
        Plot a scatterplot matrix for numeric columns with increased size and clear labels.
        
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            numeric_cols (list): List of numeric column names to include in the scatterplot matrix.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
            fontsize (int): Font size for axis labels.
        """
        if numeric_cols is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Create the scatterplot matrix
        pd.plotting.scatter_matrix(df[numeric_cols], figsize=figsize, diagonal='kde')
        # Adjust layout and font size
        plt.suptitle(title, y=1.02, fontsize=fontsize + 2)  # Increase title font size
        plt.tight_layout()   
        # Set font size for axis labels
        for ax in plt.gcf().axes:
            ax.tick_params(axis='both', labelsize=fontsize)
            ax.set_xlabel(ax.get_xlabel(), fontsize=fontsize)
            ax.set_ylabel(ax.get_ylabel(), fontsize=fontsize)       
        plt.show()  

    def plot_pie(df, column, title="Pie Plot", figsize=(10, 10), autopct='%1.1f%%', startangle=90):
        """
        Plot a pie chart for a categorical column, including missing values.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data.
            column (str): The name of the categorical column to plot.
            title (str): Title of the plot.
            figsize (tuple): Size of the figure (width, height).
            autopct (str): Format string for displaying percentages on the pie chart.
            startangle (int): Angle at which the first slice starts (in degrees).
        """
        # Count the occurrences of each category
        counts = df[column].value_counts(dropna=False)  # Include missing values
        
        # Add a label for missing values if they exist
        if df[column].isnull().sum() > 0:
            counts['Missing'] = df[column].isnull().sum()
        
        # Plot the pie chart
        plt.figure(figsize=figsize)
        plt.pie(counts, labels=counts.index, autopct=autopct, startangle=startangle, colors=plt.cm.Paired.colors)
        plt.title(title, fontsize=14)
        plt.show()    
        # Normalization
    def normalize_data(df, method='minmax'):
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if method == 'minmax':
                for col in numeric_cols:
                    min_val = df[col].min()
                    max_val = df[col].max()
                    df[col] = (df[col] - min_val) / (max_val - min_val)
            elif method == 'standard':
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
            return df
        
    def plot_regression(df, x_col, y_col, title="Regression Plot"):
        """
        Plot a regression plot for two numeric columns.
        """
        plt.figure(figsize=(10, 8))
        sns.regplot(x=x_col, y=y_col, data=df, scatter_kws={'color': 'skyblue'}, line_kws={'color': 'red'})
        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.show()
    
    # Imputation
    def knn_impute(df):
        # Select only numeric columns for KNN imputation
        df_numeric = df.select_dtypes(include=['float64', 'int64'])
        # Initialize the KNNImputer
        imputer = KNNImputer(n_neighbors=5)
        # Apply the imputer to the numeric columns
        df_imputed_numeric = imputer.fit_transform(df_numeric)
        # Create a copy of the original dataframe and update the numeric columns with the imputed values
        df_imputed = df.copy()
        df_imputed[df_numeric.columns] = df_imputed_numeric
        return df_imputed

    def correct_data_types(df):
        corrections = []
        for col in df.columns:
            old_type = df[col].dtype
            
            # If the column is object type
            if old_type == 'object':
                # Check if all non-null values are integer-like
                if df[col].dropna().apply(lambda x: x.isdigit() if isinstance(x, str) else False).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
                # Check if all non-null values are float-like
                elif df[col].dropna().apply(lambda x: x.replace('.', '', 1).isdigit() if isinstance(x, str) else False).all():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
            
            # If the column is float type but contains integer-like values
            elif pd.api.types.is_float_dtype(df[col]):
                # Check if all non-null values are integer-like
                if df[col].dropna().apply(lambda x: x.is_integer() if not pd.isna(x) else False).all():
                    df[col] = df[col].astype('Int64')  # Nullable integer
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
            
            # If the column is numeric but not float, ensure proper handling of missing values
            elif pd.api.types.is_numeric_dtype(df[col]):
                if df[col].isna().sum() > 0 and old_type != 'float':
                    df[col] = df[col].astype('float')
                    new_type = df[col].dtype
                    corrections.append((col, old_type, new_type))
        
        return df, corrections    
    # Function to cap outliers
    def cap_outliers(series, lower_percentile=1, upper_percentile=99):
        if pd.api.types.is_numeric_dtype(series):
            lower_cap = series.quantile(lower_percentile / 100)
            upper_cap = series.quantile(upper_percentile / 100)
            return series.clip(lower=lower_cap, upper=upper_cap)
        else:
            return series
    
    # Function to remove outliers using IQR
    # def remove_outliers_iqr(series):
    #     Q1 = series.quantile(0.25)
    #     Q3 = series.quantile(0.75)
    #     IQR = Q3 - Q1
    #     lower_bound = Q1 - 1.5 * IQR
    #     upper_bound = Q3 + 1.5 * IQR
    #     # Replace outliers with NaN
    #     series = series.where((series >= lower_bound) & (series <= upper_bound), np.nan)
    #     return series
    # --- Outlier Handling Functions ---
    def remove_outliers_iqr(data, column):
        """Remove outliers using IQR."""
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def remove_outliers_zscore(data, column, threshold=3):
        """Remove outliers using Z-score."""
        z_scores = zscore(data[column])
        return data[(np.abs(z_scores) <= threshold)]
    
    def remove_outliers_percentile(data, column, lower_percentile=0.01, upper_percentile=0.99):
        """Remove outliers based on percentiles."""
        lower_bound = data[column].quantile(lower_percentile)
        upper_bound = data[column].quantile(upper_percentile)
        return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    
    def cap_outliers(data, column, lower_percentile=0.01, upper_percentile=0.99):
        """
        Cap outliers using percentile thresholds and return a new DataFrame.
        Converts the column to float64 before capping to avoid dtype issues.
        """
        if pd.api.types.is_numeric_dtype(data[column]):
            # Convert the column to float64 to ensure compatibility with clipped values
            data[column] = data[column].astype('float64')
            # Calculate the lower and upper bounds
            lower_bound = data[column].quantile(lower_percentile)
            upper_bound = data[column].quantile(upper_percentile)   
            # Cap the values using np.clip
            data.loc[:, column] = np.clip(data[column], lower_bound, upper_bound)
        return data        
    def log_transform(data, column):
        """Apply log transformation to reduce outlier impact."""
        data.loc[:, column] = np.log1p(data[column])  # Use .loc to avoid SettingWithCopyWarning
        return data
    
    def sqrt_transform(data, column):
        """Apply square root transformation to reduce outlier impact."""
        data.loc[:, column] = np.sqrt(data[column])  # Use .loc to avoid SettingWithCopyWarning
        return data
    
    def robust_scaling(data, column):
        """Scale data robustly, reducing the effect of outliers."""
        median = data[column].median()
        iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
        data.loc[:, column] = (data[column] - median) / iqr  # Use .loc to avoid SettingWithCopyWarning
        return data
    

    
    def describe_pluss(df):
        """
        Enhanced describe function with a dynamic interactive menu:
        - Users can explore the dataset and perform preprocessing tasks.
        - Allows replacing placeholders with NaN before handling missing values.
        - Implements dynamic menu updates based on used options.
        - Includes kurtosis calculation and outlier handling techniques.
        """
    
        # --- Outlier Handling Functions ---
        def remove_outliers_iqr(data, column):
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
        def remove_outliers_zscore(data, column, threshold=3):
            z_scores = zscore(data[column])
            return data[(np.abs(z_scores) <= threshold)]
        
        def remove_outliers_percentile(data, column, lower_percentile=0.01, upper_percentile=0.99):
            lower_bound = data[column].quantile(lower_percentile)
            upper_bound = data[column].quantile(upper_percentile)
            return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
        
        def cap_outliers(data, column, lower_percentile=0.01, upper_percentile=0.99):
            """
            Cap outliers using percentile thresholds and return a new DataFrame.
            Converts the column to float64 before capping to avoid dtype issues.
            """
            if pd.api.types.is_numeric_dtype(data[column]):
                # Convert the column to float64 to ensure compatibility with clipped values
                data[column] = data[column].astype('float64')
                # Calculate the lower and upper bounds
                lower_bound = data[column].quantile(lower_percentile)
                upper_bound = data[column].quantile(upper_percentile)   
                # Cap the values using np.clip
                data.loc[:, column] = np.clip(data[column], lower_bound, upper_bound)
            return data        
        
        def log_transform(data, column):
            data.loc[:, column] = np.log1p(data[column].astype(float))  # Ensure float dtype
            return data
        
        def robust_scaling(data, column):
            median = data[column].median()
            iqr = data[column].quantile(0.75) - data[column].quantile(0.25)
            data.loc[:, column] = (data[column].astype(float) - median) / iqr  # Ensure float dtype
            return data    
            
        def detect_outlier_handling_method(data, column):
                        skewness = data[column].skew()
                        kurtosis_val = kurtosis(data[column], nan_policy='omit')
                        n_outliers_iqr = len(data) - len(remove_outliers_iqr(data, column))
                        n_outliers_zscore = len(data) - len(remove_outliers_zscore(data, column))
                        n_outliers_percentile = len(data) - len(remove_outliers_percentile(data, column))
        
                        # Decision Logic
                        if abs(skewness) > 1:  # Highly skewed
                            if n_outliers_iqr > 0.05 * len(data):  # More than 5% outliers
                                return "Log Transformation"
                            else:
                                return "Cap using Percentiles"
                        elif abs(skewness) <= 1:  # Symmetric or moderately skewed
                            if n_outliers_zscore > 0.05 * len(data):  # Outliers based on normal distribution
                                return "Remove using Z-score"
                            elif n_outliers_percentile > 0.05 * len(data):  # Outliers based on percentiles
                                return "Remove using Percentiles"
                            else:
                                return "Robust Scaling"
                        else:
                            return "No Action Needed"
    
        # --- Correlation Analysis Function --
        def analyze_correlations(df, significance_level=0.05):
            """
            Analyze correlations between numeric variables in a DataFrame, allowing the user to filter results
            or manually choose columns for analysis.
            
            Parameters:
                df (pd.DataFrame): The DataFrame containing the data.
                significance_level (float): The threshold for statistical significance (default: 0.05).
            
            Returns:
                None (displays the correlation results and prints insights).
            """
            # Select only numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            # Check if there are at least two numeric columns
            if len(numeric_cols) < 2:
                print("Not enough numeric columns to calculate correlations.")
                return
            
            # Ask the user if they want to filter the results
            filter_choice = input("Do you want to filter the results? (yes/no): ").strip().lower()
            
            if filter_choice == "yes":
                # Display filtering options
                print("\nFilter Options:")
                print("1. Filter by Correlation Strength")
                print("2. Filter by Significance Level")
                print("3. Filter by Both")
                print("4. Display All Outcomes")
                filter_option = input("Choose an option (1/2/3/4): ").strip()
                
                # Initialize filters
                correlation_filter = None
                significance_filter = None
                
                # If the user chooses "Display All Outcomes", skip filtering
                if filter_option == "4":
                    print("\nDisplaying all correlation outcomes without filtering.")
                else:
                    # Filter by Correlation Strength
                    if filter_option in ["1", "3"]:
                        print("\nCorrelation Strength Options:")
                        print("1. Strong Positive (Correlation > 0.7)")
                        print("2. Moderate Positive (0.3 < Correlation <= 0.7)")
                        print("3. Weak or No Correlation (-0.3 <= Correlation <= 0.3)")
                        print("4. Moderate Negative (-0.7 < Correlation < -0.3)")
                        print("5. Strong Negative (Correlation <= -0.7)")
                        strength_choice = input("Choose a correlation strength (1/2/3/4/5): ").strip()
                        
                        if strength_choice == "1":
                            correlation_filter = lambda x: x > 0.7
                        elif strength_choice == "2":
                            correlation_filter = lambda x: 0.3 < x <= 0.7
                        elif strength_choice == "3":
                            correlation_filter = lambda x: -0.3 <= x <= 0.3
                        elif strength_choice == "4":
                            correlation_filter = lambda x: -0.7 < x < -0.3
                        elif strength_choice == "5":
                            correlation_filter = lambda x: x <= -0.7
                        else:
                            print("Invalid choice. No correlation filter applied.")
                    
                    # Filter by Significance Level
                    if filter_option in ["2", "3"]:
                        print("\nSignificance Level Options:")
                        print("1. Strong evidence (p-value < 0.001)")
                        print("2. Moderate evidence (0.001 <= p-value < 0.05)")
                        print("3. Weak evidence (0.05 <= p-value < 0.1)")
                        print("4. No evidence (p-value >= 0.1)")
                        significance_choice = input("Choose a significance level (1/2/3/4): ").strip()
                        
                        if significance_choice == "1":
                            significance_filter = lambda x: x < 0.001
                        elif significance_choice == "2":
                            significance_filter = lambda x: 0.001 <= x < 0.05
                        elif significance_choice == "3":
                            significance_filter = lambda x: 0.05 <= x < 0.1
                        elif significance_choice == "4":
                            significance_filter = lambda x: x >= 0.1
                        else:
                            print("Invalid choice. No significance filter applied.")
            
            else:
                # Allow the user to manually choose columns
                print("\nAvailable numeric columns:")
                for i, col in enumerate(numeric_cols):
                    print(f"{i + 1}. {col}")
                selected_indices = input("Enter the numbers of the columns you want to analyze (comma-separated): ").strip()
                selected_indices = [int(idx.strip()) - 1 for idx in selected_indices.split(",")]
                numeric_cols = [numeric_cols[idx] for idx in selected_indices]
            
            # Check if there are at least two numeric columns after filtering or selection
            if len(numeric_cols) < 2:
                print("\nNot enough numeric columns to calculate correlations. Please adjust your filters or select more columns.")
                return
            
            # Initialize a list to store correlation results
            results_list = []
            
            # Calculate correlations for all pairs of numeric columns
            for i, col1 in enumerate(numeric_cols):
                for j, col2 in enumerate(numeric_cols):
                    if i < j:  # Avoid duplicate pairs and self-correlations
                        # Calculate Pearson correlation and p-value
                        corr, p_value = pearsonr(df[col1], df[col2])
                        
                        # Determine significance
                        if p_value < 0.001:
                            significance = "Strong evidence"
                        elif p_value < 0.05:
                            significance = "Moderate evidence"
                        elif p_value < 0.1:
                            significance = "Weak evidence"
                        else:
                            significance = "No evidence"
                        
                        # Determine the meaning of the Pearson Correlation
                        if corr > 0.7:
                            meaning = "Strong positive linear correlation"
                        elif corr > 0.3:
                            meaning = "Moderate positive linear correlation"
                        elif corr > -0.3:
                            meaning = "Weak or no linear correlation"
                        elif corr > -0.7:
                            meaning = "Moderate negative linear correlation"
                        else:
                            meaning = "Strong negative linear correlation"
                        
                        # Append the results as a dictionary to the list
                        results_list.append({
                            'Variable 1': col1,
                            'Variable 2': col2,
                            'Pearson Correlation': corr,
                            'P-value': p_value,
                            'Significance': significance,
                            'Correlation Meaning': meaning
                        })
            
            # Convert the list of dictionaries to a DataFrame
            results = pd.DataFrame(results_list)
            
            # Apply filters to the results (if the user didn't choose "Display All Outcomes")
            if filter_choice == "yes" and filter_option != "4":
                if correlation_filter:
                    results = results[results['Pearson Correlation'].apply(correlation_filter)]
                if significance_filter:
                    results = results[results['P-value'].apply(significance_filter)]
            
            # Check if there are any results after filtering
            if len(results) == 0:
                print("\nNo results found after applying filters. Please adjust your filters.")
                return
            
            # Display the correlation results as a DataFrame
            print("\nCorrelation Analysis Results:")
            display(results)
            
            # Print insights for each pair of columns
            print("\nInsights:")
            for _, row in results.iterrows():
                var1, var2, corr, p_value, significance, meaning = row
                print(f"- {var1} and {var2}:")
                print(f"  - Pearson Correlation: {corr:.2f} ({meaning})")
                print(f"  - P-value: {p_value:.4f} ({significance} of correlation)")
                print(f"  - Interpretation: {get_correlation_interpretation(corr)}")
                print()        
        def get_correlation_interpretation(corr):
            """
            Provide a textual interpretation of the Pearson Correlation value.
            
            Parameters:
                corr (float): The Pearson Correlation coefficient.
            
            Returns:
                str: A textual interpretation of the correlation.
            """
            if corr > 0.7:
                return "A strong positive relationship exists. As one variable increases, the other tends to increase as well."
            elif corr > 0.3:
                return "A moderate positive relationship exists. As one variable increases, the other tends to increase slightly."
            elif corr > -0.3:
                return "A weak or no relationship exists. Changes in one variable do not significantly affect the other."
            elif corr > -0.7:
                return "A moderate negative relationship exists. As one variable increases, the other tends to decrease slightly."
            else:
                return "A strong negative relationship exists. As one variable increases, the other tends to decrease."


        # --- Helper Functions ---
        def display_menu(available_options):
            print("\nChoose an option from the menu:")
            for key, value in available_options.items():
                print(f"{key}. {value}")
    
        def display_column_menu(df):
            print("\nAvailable Columns:")
            for i, col in enumerate(df.columns, start=1):
                print(f"{i}. {col}")
            print(f"{len(df.columns) + 1}. All Columns")
            print(f"{len(df.columns) + 2}. Exit")
                    
        def get_column_choice(df):
            while True:
                choice = input(
                    f"Enter column numbers (1-{len(df.columns)}), "
                    f"'{len(df.columns) + 1}' for All Columns, "
                    f"'{len(df.columns) + 2}' to Exit, "
                    "or column names separated by commas (press Enter for All Columns): "
                ).strip()                  
                if not choice:
                    print("No columns selected. Please try again.")
                    continue
    
                choices = [ch.strip() for ch in choice.split(',')]
                selected_columns = []
                valid = True
    
                for ch in choices:
                    if ch.isdigit():
                        col_num = int(ch)
                        if 1 <= col_num <= len(df.columns):
                            selected_columns.append(df.columns[col_num - 1])
                        elif col_num == len(df.columns) + 1:
                            selected_columns = df.columns.tolist()
                            break
                        elif col_num == len(df.columns) + 2:
                            return "exit"
                        else:
                            print(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
                    else:
                        if ch in df.columns:
                            selected_columns.append(ch)
                        elif ch.lower() == "all":
                            selected_columns = df.columns.tolist()
                        elif ch.lower() == "exit":
                            return "exit"
                        else:
                            print(f"Invalid choice: {ch}. Please try again.")
                            valid = False
                            break
    
                if valid:
                    return selected_columns if len(selected_columns) > 1 else selected_columns[0]
    
        def replace_placeholders(df):
            default_placeholders = ['?', '-', 'None', 'N/A', '']
            print("\nDefault placeholders to replace:", default_placeholders)
            custom_placeholder = input("Enter any additional placeholder (leave blank to skip): ").strip()
            if custom_placeholder:
                default_placeholders.append(custom_placeholder)
            df.replace(default_placeholders, np.nan, inplace=True)
            print("\nPlaceholders have been replaced with NaN.")
    
        def update_summary(df):
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
    
            summary = df.describe(include='all').T
            summary['missing_count'] = df.isnull().sum()
            summary['duplicate_count'] = df.duplicated().sum()
            summary['mode'] = df.mode().iloc[0]
            summary['data_type'] = df.dtypes
            summary['skewness'] = np.nan
            summary['kurtosis'] = np.nan
            summary['variance'] = np.nan
            summary['outliers'] = "N/A"
            summary['missing_value_handling'] = "N/A"
            summary['outlier_handling'] = "N/A"
    
            summary['column_number'] = range(1, len(df.columns) + 1)
    
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    skew_val = df[col].skew(skipna=True)
                    kurtosis_val = kurtosis(df[col], nan_policy='omit')
                    variance_val = df[col].var(skipna=True)
                    summary.at[col, 'skewness'] = skew_val
                    summary.at[col, 'kurtosis'] = kurtosis_val
                    summary.at[col, 'variance'] = variance_val
    
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
                    summary.at[col, 'outliers'] = len(outliers)
    
                    recommendation = detect_outlier_handling_method(df, col)
                    summary.at[col, 'outlier_handling'] = recommendation
    
                    if summary.at[col, 'missing_count'] > 0:
                        if abs(skew_val) < 0.5:
                            summary.at[col, 'missing_value_handling'] = "Fill with Mean"
                        else:
                            summary.at[col, 'missing_value_handling'] = "Fill with Median"
                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    if summary.at[col, 'missing_count'] > 0:
                        summary.at[col, 'missing_value_handling'] = "Fill with Mode"
                    else:
                        summary.at[col, 'missing_value_handling'] = "No Action Needed"
                else:
                    summary.at[col, 'missing_value_handling'] = "Manual Inspection Needed"
    
            return summary
    
        def handle_user_choice(choice, df, available_options):
            nonlocal summary
            if choice == 1:
                summary = update_summary(df)
                display(summary)
            elif choice == 2:
                pd.set_option('display.max_columns', None)
                print("\nDisplaying df.head() with all columns:")
                display(df.head())
            elif choice == 3:
                replace_placeholders(df)
            elif choice == 4:
                summary = update_summary(df)
                for col in df.columns:
                    recommendation = summary.at[col, 'missing_value_handling']
                    if "Mean" in recommendation:
                        df[col] = df[col].astype(float)
                        df[col] = df[col].fillna(df[col].mean())
                    elif "Median" in recommendation:
                        df[col] = df[col].astype(float)
                        df[col] = df[col].fillna(df[col].median())
                    elif "Mode" in recommendation:
                        mode_val = df[col].mode()
                        if not mode_val.empty:
                            df[col] = df[col].fillna(mode_val[0])
                print("\nMissing values have been filled based on the recommendations.")
            elif choice == 5:
                duplicates_count = df.duplicated().sum()
                df.drop_duplicates(keep='first', inplace=True)
                print(f"\n{duplicates_count} duplicate rows have been removed.")
            elif choice == 6:
                df, corrections = analyisisToolkit.correct_data_types(df)
                if corrections:
                    print("\nData type corrections applied:")
                    for col, old_type, new_type in corrections:
                        print(f"- Column '{col}': {old_type} -> {new_type}")
                else:
                    print("\nAll columns already have correct data types.")
            elif choice == 7:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    if pd.api.types.is_categorical_dtype(df[col]):
                        continue  # Skip categorical columns
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                    min_val = df[col].min()
                    max_val = df[col].max()
                    if max_val - min_val != 0:
                        df.loc[:, col] = (df[col] - min_val) / (max_val - min_val)  # Normalize
                    else:
                        print(f"Warning: Column '{col}' has no variation; normalization not applied.")
                print("\nNumeric columns have been normalized.")                
            elif choice == 8:
                df_numeric = df.select_dtypes(include=['float64', 'int64','Int64'])
                imputer = KNNImputer(n_neighbors=5)
                df_imputed_numeric = imputer.fit_transform(df_numeric)
                df.loc[:, df_numeric.columns] = df_imputed_numeric
                print("\nMissing values have been imputed using KNN.")
            elif choice == 9:
                summary = update_summary(df)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        recommendation = summary.at[col, 'outlier_handling']
                        if recommendation == "Remove using IQR":
                            df = remove_outliers_iqr(df, col)  # Update the DataFrame
                        elif recommendation == "Remove using Z-score":
                            df = remove_outliers_zscore(df, col)  # Update the DataFrame
                        elif recommendation == "Remove using Percentiles":
                            df = remove_outliers_percentile(df, col)  # Update the DataFrame
                        elif recommendation == "Cap using Percentiles":
                            df = cap_outliers(df, col)  # Update the DataFrame
                        elif recommendation == "Log Transformation":
                            df = log_transform(df, col)  # Update the DataFrame
                        elif recommendation == "Robust Scaling":
                            df = robust_scaling(df, col)  # Update the DataFrame
                        elif recommendation == "No Action Needed":
                            print(f"No action needed for column '{col}'.")
                        else:
                            print(f"Unknown recommendation for column '{col}': {recommendation}")
                print("\nOutliers have been handled based on the recommendations.")
                summary = update_summary(df)               
            elif choice == 10:
                categorical_cols = df.select_dtypes(include=['object']).columns
                for col in categorical_cols:
                    df = pd.get_dummies(df, columns=[col], drop_first=True)
                print("\nCategorical variables have been encoded.")
            elif choice == 11:
                display_columns = input("Do You Want To display Columns ? (y/n)" ).strip()
                if display_columns == "y":
                    display_column_menu(df)   
                while True:      
                    visualize_choice = input(
                        "\nChoose visualizations:"
                        "\n1. Missing Values\n2. Correlation Heatmap\n3. Skewness\n4. Outliers\n5. Pairplot\n"
                        "6. Distribution Plot\n7. Violin Plot (Comparison)\n8. Swarm Plot\n9. Missing Values Heatmap\n"
                        "10. QQ Plot\n11. Scatterplot Matrix\n12. Regression Plot\n13. Pie Plot\n"
                        "Choose (1-13 or 'exit' to stop): ").strip().lower()
                    if visualize_choice == '1':
                        analyisisToolkit.plot_missing_values(df)
                    elif visualize_choice == '2':
                        analyisisToolkit.plot_correlation_heatmap(df)
                    elif visualize_choice == '3':
                        analyisisToolkit.plot_skewness(df)
                    elif visualize_choice == '4':
                        column_choice = get_column_choice(df)
                        if column_choice == "exit":
                            print("No action taken.")
                        else:
                            if isinstance(column_choice, list):
                                analyisisToolkit.visualize_outliers(df[column_choice])
                            else:
                                analyisisToolkit.visualize_outliers(df[[column_choice]])
                    elif visualize_choice == '5':
                        analyisisToolkit.plot_pairplot(df)
                    elif visualize_choice == '6':
                        column_choice = get_column_choice(df)
                        if column_choice == "exit":
                            print("No action taken.")
                        else:
                            if isinstance(column_choice, list):
                                # If multiple columns are selected, plot each one individually
                                for col in column_choice:
                                    analyisisToolkit.plot_distribution(df[col], title=f"Distribution Plot for {col}")
                            else:
                                # If a single column is selected, plot it directly
                                analyisisToolkit.plot_distribution(df[column_choice], title=f"Distribution Plot for {column_choice}")                                
                    elif visualize_choice == '7':
                        numeric_col = input("Enter numeric column name: ").strip()
                        categorical_col = input("Enter categorical column name: ").strip()
                        analyisisToolkit.plot_violin_comparison(df, numeric_col, categorical_col, title=f"Violin Plot: {numeric_col} by {categorical_col}")
                    elif visualize_choice == '8':
                        numeric_col = input("Enter numeric column name: ").strip()
                        categorical_col = input("Enter categorical column name: ").strip()
                        analyisisToolkit.plot_swarm(df, numeric_col, categorical_col, title=f"Swarm Plot: {numeric_col} by {categorical_col}")
                    elif visualize_choice == '9':
                        analyisisToolkit.plot_missing_heatmap(df, title="Missing Values Heatmap")
                    elif visualize_choice == '10':
                        column_choice = get_column_choice(df)
                        if column_choice == "exit":
                            print("No action taken.")
                        else:
                            analyisisToolkit.plot_qq(df[column_choice], title=f"QQ Plot for {column_choice}")
                    elif visualize_choice == '11':
                        analyisisToolkit.plot_scatter_matrix(df, title="Scatterplot Matrix")
                    elif visualize_choice == '12':
                        print("Choose the X-axis column:")
                        x_col = get_column_choice(df)
                        if x_col == "exit":
                            print("No action taken.")
                        else:
                            print("Choose the Y-axis column:")
                            y_col = get_column_choice(df)
                            if y_col == "exit":
                                print("No action taken.")
                            else:
                                analyisisToolkit.plot_regression(df, x_col, y_col, title=f"Regression Plot: {x_col} vs {y_col}")
                    elif visualize_choice == '13':
                        column_choice = get_column_choice(df)
                        if column_choice == "exit":
                            print("No action taken.")
                        else:
                            analyisisToolkit.plot_pie(df, column=column_choice, title=f"Pie Plot for {column_choice}")
                    elif visualize_choice == 'exit':
                        break
                    else:
                        print("Invalid choice, please try again.")     
       
            elif choice == 12:
                report_filename = input("Enter the filename for the report (default: business_data_report.html): ").strip()
                report_filename = report_filename if report_filename else "business_data_report.html"
                try:
                    profile = ProfileReport(df, title="Business Data Preparation Report", explorative=True)
                    profile.to_file(report_filename)
                    print(f"Interactive HTML report saved as {report_filename}")
                except Exception as e:
                    print(f"An error occurred while generating the HTML report: {e}")
            elif choice == 13:
                print("\nOutlier Handling Techniques:")
                print("1. Remove using IQR")
                print("2. Remove using Z-score")
                print("3. Remove using Percentiles")
                print("4. Cap using Percentiles")
                print("5. Log Transformation")
                print("6. Robust Scaling")
                outlier_choice_map = {
                    1: "Remove using IQR",
                    2: "Remove using Z-score",
                    3: "Remove using Percentiles",
                    4: "Cap using Percentiles",
                    5: "Log Transformation",
                    6: "Robust Scaling"
                }
                outlier_choice = int(input("Choose an outlier handling technique (1-6): ").strip())
                method = outlier_choice_map.get(outlier_choice, None)
                if method is None:
                    print("Invalid choice.")
                    return df
    
                column_choices = get_column_choice(df)
                if column_choices == "exit":
                    print("No action taken.")
                else:
                    if isinstance(column_choices, list):
                        for col in column_choices:
                            if pd.api.types.is_numeric_dtype(df[col]):
                                if method == "Remove using IQR":
                                    df = remove_outliers_iqr(df, col)
                                elif method == "Remove using Z-score":
                                    df = remove_outliers_zscore(df, col)
                                elif method == "Remove using Percentiles":
                                    df = remove_outliers_percentile(df, col)
                                elif method == "Cap using Percentiles":
                                    df = cap_outliers(df, col)
                                elif method == "Log Transformation":
                                    df = log_transform(df, col)
                                elif method == "Robust Scaling":
                                    df = robust_scaling(df, col)
                                else:
                                    print(f"Invalid method for column '{col}'. No action taken.")
                    else:
                        if pd.api.types.is_numeric_dtype(df[column_choices]):
                            if method == "Remove using IQR":
                                df = remove_outliers_iqr(df, column_choices)
                            elif method == "Remove using Z-score":
                                df = remove_outliers_zscore(df, column_choices)
                            elif method == "Remove using Percentiles":
                                df = remove_outliers_percentile(df, column_choices)
                            elif method == "Cap using Percentiles":
                                df = cap_outliers(df, column_choices)
                            elif method == "Log Transformation":
                                df = log_transform(df, column_choices)
                            elif method == "Robust Scaling":
                                df = robust_scaling(df, column_choices)
                            else:
                                print(f"Invalid method for column '{column_choices}'. No action taken.")
                    print(f"\nOutliers in selected columns have been handled.")
                    summary = update_summary(df)
                    display(summary)
            elif choice == 14:
                # Display numeric columns for the user to choose from
                analyze_correlations(df)
            else:
                print("Invalid choice. Please try again.")                
            if choice not in [1, 2, 11, 12, 13, 14]:
                available_options.pop(choice, None)
    
            return df
    
        # Initial setup
        summary = update_summary(df)
        always_available = {
            1: "Display Summary",
            2: "Display DataFrame Head",
            11: "Visualize Data (Missing Values, Correlation, Outliers, etc.)",
            12: "Generate Interactive HTML Report",
            13: "Handle Outliers with User Choice",
            14: "Analyze Correlations"
        }
        all_options = {
            1: "Display Summary",
            2: "Display DataFrame Head",
            3: "Replace Placeholders with NaN",
            4: "Apply Recommended Filling Techniques",
            5: "Remove Duplicates",
            6: "Check and Correct Data Types",
            7: "Normalize Numeric Columns",
            8: "Impute Missing Values (KNN)",
            9: "Apply Recommended Outlier Handling Techniques",
            10: "Encode Categorical Variables",
            11: "Visualize Data (Missing Values, Correlation, Outliers, etc.)",
            12: "Generate Interactive HTML Report",
            13: "Handle Outliers with User Choice",
            14: "Analyze Correlations"
        }
        available_options = all_options.copy()
    
        # Menu loop
        display_menu(available_options)
        while True:
            try:
                choice = int(input("\nEnter your choice: ").strip())
                df = handle_user_choice(choice, df, available_options)
    
                repeat_choice = int(input("\nEnter your choice (1: Choose again, 2: Redisplay menu, 0: Stop): ").strip())
                if repeat_choice == 1:
                    if not available_options:
                        print("All options have been used. Exiting.")
                        break
                elif repeat_choice == 2:
                    display_menu(available_options)
                elif repeat_choice == 0:
                    print("Exiting...")
                    break
            except ValueError:
                print("Invalid input. Please enter a number.")