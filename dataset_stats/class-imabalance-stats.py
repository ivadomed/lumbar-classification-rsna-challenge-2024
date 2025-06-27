import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(file_path):
    return pd.read_csv(file_path)

def compute_proportions(df, category):
    category_cols = [col for col in df.columns if category in col]
    proportions = {}
    for col in category_cols:
        prop = df[col].value_counts(normalize=True)
        proportions[col] = prop
    all_values = df[category_cols].values.flatten()
    overall_proportions = pd.Series(all_values).value_counts(normalize=True)
    proportions['overall'] = overall_proportions
    return proportions

CLASS_COLORS = {
    'Normal/Mild': 'tab:blue',
    'Moderate': 'tab:orange',
    'Severe': 'tab:green'
}

def plot_and_save(proportions, category, bar_width, output_dir):
    data = pd.DataFrame(proportions).fillna(0)
    data.to_csv(f"{output_dir}/{category}_proportions.csv")
    fig_width = 4 if category == 'spinal_canal_stenosis' else 8
    plt.figure(figsize=(fig_width, 6))
    bottom = [0] * len(data.columns)
    for cls in CLASS_COLORS.keys():
        if cls in data.index:
            plt.bar(data.columns, data.loc[cls], bottom=bottom, label=cls, width=bar_width, color=CLASS_COLORS[cls])
            bottom += data.loc[cls].values
    plt.title(f'Class Proportions for {category}')
    plt.xlabel('Columns')
    plt.ylabel('Proportion')
    plt.xticks(rotation=45)
    plt.legend(title='Classes')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{category}_proportions.jpg")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str)
    parser.add_argument('--output', type=str)
    args = parser.parse_args()

    df = load_data(args.file_path)
    categories = ['spinal_canal_stenosis', 'neural_foraminal_narrowing', 'subarticular_stenosis']
    for category in categories:
        proportions = compute_proportions(df, category)
        plot_and_save(proportions, category, 0.4, args.output)

if __name__ == "__main__":
    main()
