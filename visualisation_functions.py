import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

def plot_multilca_impacts(df, impact_categories, colors, save_path=None):
    """
    Visualize LCA impacts for each category with specified colors and save the plot if a path is provided.

    Parameters:
    - df (pd.DataFrame): DataFrame with raw materials as index and impact categories as columns.
    - impact_categories (list of str): List of impact categories (column names) to plot.
    - colors (list of str): List of colors for each impact category.
    - save_path (str, optional): Path to save the plot image. If None, the plot will only display.
    """
    # Ensure 'Raw material' is a column by resetting the index
    df = df.reset_index()
    df.rename(columns={'index': 'Raw material'}, inplace=True)

    # Check that the number of colors matches the number of impact categories
    if len(colors) != len(impact_categories):
        raise ValueError("The number of colors must match the number of impact categories.")

    # Set up the figure with subplots
    plt.figure(figsize=(14, 10))

    # Create a bar plot for each impact category with specified colors
    for i, (impact, color) in enumerate(zip(impact_categories, colors), 1):
        plt.subplot(2, 2, i)
        plt.bar(df["Raw material"], df[impact], color=color, label=impact)
        plt.xticks(rotation=90)
        plt.ylabel(impact)
        plt.title(impact)
        plt.tight_layout()

    # Save the plot if a path is provided, otherwise display it
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_contribution_analysis(df, inventory_names, colors, save_dir="results"):
    """
    Plot contribution analysis for multiple inventories with consistent colors for impact categories
    and opacity based on score magnitude, sorted by contribution. Each plot is saved separately.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the contribution analysis data.
    - inventory_names (list of str): List of inventory names to filter and plot.
    - colors (list of str): List of colors for each impact category.
    - save_dir (str): Directory to save the plot images. Default is "results".
    """

    df = df.reset_index()

    # Loop through each inventory in the provided list
    for inventory_name in inventory_names:
        # Filter the DataFrame for the specified inventory
        df_inventory = df[df['Inventory'] == inventory_name]

        # List of unique impact categories for the inventory
        impact_categories = df_inventory['Impact Category'].unique()

        # Set up figure size based on the number of impact categories
        fig, axes = plt.subplots(len(impact_categories), 1, figsize=(10, len(impact_categories) * 2), sharex=True)

        # Create a color dictionary for consistent coloring across categories
        color_dict = {impact: colors[i % len(colors)] for i, impact in enumerate(impact_categories)}

        # Loop through each impact category and create a horizontal bar plot
        for i, impact_category in enumerate(impact_categories):
            # Filter data for each impact category and sort by percentage in descending order
            df_impact = df_inventory[df_inventory['Impact Category'] == impact_category]
            df_impact = df_impact.sort_values(by='percentage', ascending=False)

            # Normalize scores for opacity scaling
            max_score = df_impact['score'].max()
            normalized_scores = df_impact['score'] / max_score

            # Plot each reference product with varying opacity based on normalized score
            for ref_product, percentage, score in zip(df_impact['reference product'],
                                                      df_impact['percentage'],
                                                      normalized_scores):
                axes[i].barh(ref_product, percentage, color=color_dict[impact_category],
                             alpha=score, edgecolor='black')

            # Set plot labels and title
            axes[i].set_xlim(0, 100)
            axes[i].invert_yaxis()  # To have the highest contributions on top
            axes[i].set_title(f"{impact_category} - {inventory_name}", color=color_dict[impact_category])
            axes[i].set_xlabel("Contribution (%)")
            axes[i].set_ylabel("")

        plt.tight_layout()

        # Save each plot to the specified directory with a filename based on the inventory name
        save_path = f"{save_dir}/contribution_analysis_{inventory_name.replace(' ', '_')}.png"
        plt.savefig(save_path, bbox_inches="tight")
        print(f"Plot saved to {save_path}")
        plt.close(fig)  # Close the figure to free up memory


def plot_impacts_with_production(projected_df, production_df, impact_categories, save_dir="results",
                                         scenario_name=None):
    '''
    Function to plot projected impacts for each impact category as stacked bars with production volume as lines.

    :param projected_df: DataFrame with projected impact data
    :param production_df: DataFrame with production volume data
    :param impact_categories: List of impact categories to plot
    :param save_dir: Directory to save the plot image
    :param scenario_name: Optional scenario name to include in the title and filenames
    :return: None
    '''

    minerals = projected_df['Mineral'].unique()

    # Set up colors for each mineral
    colors = plt.cm.Set1.colors  # Use a colormap for consistent coloring

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Create a figure with 4 subplots (2x2 layout)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()  # Flatten to easily iterate over for each category

    for idx, category in enumerate(impact_categories):
        ax1 = axs[idx]

        # Prepare data for stacked bar chart
        years = projected_df['Year'].unique()
        bottom = np.zeros(len(years))  # Initialize bottom to zero for stacking

        for i, mineral in enumerate(minerals):
            subset = projected_df[projected_df['Mineral'] == mineral]
            # Ensure all years are represented (fill missing years with zero)
            values = [subset[subset['Year'] == year][category].values[0] if year in subset['Year'].values else 0 for
                      year in years]
            ax1.bar(years, values, bottom=bottom, color=colors[i % len(colors)], label=mineral)
            bottom += values  # Update bottom for stacking next mineral

        # Create a secondary y-axis for production volume
        ax2 = ax1.twinx()

        # Plot production volume as lines for each mineral
        for i, mineral in enumerate(minerals):
            production_subset = production_df[production_df['Mineral'] == mineral]
            ax2.plot(production_subset['Year'], production_subset['Value'], linestyle='--',
                     color=colors[i % len(colors)], alpha=0.6)

        # Set axis labels and title
        #ax1.set_xlabel("Year")
        #ax1.set_ylabel(f"{category} Impact")
        ax2.set_ylabel("Production Volume (kilotonnes)")
        ax1.set_title(f"{category} impact by mineral for {scenario_name}")

    # Add legends outside the subplots
    fig.legend([plt.Line2D([0], [0], color=color, lw=4) for color in colors[:len(minerals)]],
               minerals, loc="upper center", ncol=len(minerals), title="Minerals")
    fig.legend([plt.Line2D([0], [0], color='black', linestyle='--', lw=2, alpha=0.6)],
               ['Production'], loc="upper right", title="")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the legend

    # Save the combined figure
    filename = f"combined_impact_{scenario_name}.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Combined plot saved to {save_path}")
    plt.close(fig)  # Close the figure to free up memory


def plot_incremental_impacts(projected_impacts_existing, projected_impacts_potential,
                             production_existing, production_potential, impact_categories,
                             save_dir="results", scenario_name="incremental"):
    '''
    Function to calculate and plot incremental impacts as stacked bars and production volumes as lines.

    :param projected_impacts_existing: DataFrame with projected impact data for existing production scenario
    :param projected_impacts_potential: DataFrame with projected impact data for potential production scenario
    :param production_existing: DataFrame with production volume data for existing scenario
    :param production_potential: DataFrame with production volume data for potential scenario
    :param impact_categories: List of impact categories to plot
    :param save_dir: Directory to save the plot image
    :param scenario_name: Name to include in the filename
    :return: None
    '''

    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Calculate incremental impacts and production
    impact_diff_df = pd.merge(projected_impacts_potential, projected_impacts_existing,
                              on=['Year', 'Mineral'], suffixes=('_potential', '_existing'))
    production_diff_df = pd.merge(production_potential, production_existing,
                                  on=['Year', 'Mineral'], suffixes=('_potential', '_existing'))

    # Calculate the difference for each impact category and production volume
    for category in impact_categories:
        impact_diff_df[f"{category}_diff"] = impact_diff_df[f"{category}_potential"] - impact_diff_df[
            f"{category}_existing"]
    production_diff_df["Value_diff"] = production_diff_df["Value_potential"] - production_diff_df["Value_existing"]

    minerals = impact_diff_df['Mineral'].unique()

    # Set up colors for each mineral
    colors = plt.cm.Set1.colors  # Use a colormap for consistent coloring

    # Create a figure with 4 subplots (2x2 layout)
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    axs = axs.flatten()  # Flatten to easily iterate over for each category

    for idx, category in enumerate(impact_categories):
        ax1 = axs[idx]

        # Prepare data for stacked bar chart
        years = impact_diff_df['Year'].unique()
        bottom = np.zeros(len(years))  # Track the bottom for stacking bars

        for i, mineral in enumerate(minerals):
            subset = impact_diff_df[impact_diff_df['Mineral'] == mineral]
            # Align the data by year for stacking
            values = [
                subset[subset['Year'] == year][f"{category}_diff"].values[0] if year in subset['Year'].values else 0 for
                year in years]
            ax1.bar(years, values, bottom=bottom, color=colors[i % len(colors)], label=mineral)
            bottom += values  # Update bottom for the next mineral layer

        # Create a secondary y-axis for production volume (sum of minerals)
        ax2 = ax1.twinx()

        for i, mineral in enumerate(minerals):
            production_subset = production_diff_df[production_diff_df['Mineral'] == mineral]
            ax2.plot(production_subset['Year'], production_subset['Value_diff'],
                     linestyle="--", color=colors[i % len(colors)], alpha=0.6)

        # Set axis labels and title
        ax1.set_xlabel("Year")
        ax1.set_ylabel(f"{category} Incremental Impact")
        ax2.set_ylabel("Incremental Production Volume (kilotonnes)")
        ax1.set_title(f"{category} Incremental Impact from Potential Production")

    # Add legends outside the subplots for stacked bars
    fig.legend([plt.Line2D([0], [0], color=color, lw=2) for color in colors[:len(minerals)]],
               minerals, loc="upper center", ncol=len(minerals), title="Minerals")
    fig.legend([plt.Line2D([0], [0], color='black', linestyle="--", lw=2)],
               ['Incremental Production'], loc="upper right", title="")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make space for the legend

    # Save the combined figure
    filename = f"incremental_impact_{scenario_name}_stacked.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches="tight")
    print(f"Incremental impact stacked plot saved to {save_path}")
    plt.close(fig)  # Close the figure to free up memory
