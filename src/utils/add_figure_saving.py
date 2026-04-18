"""
Script to add figure saving functionality to Jupyter notebooks
"""
import json
import os
import re

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def modify_notebook(notebook_path, notebook_name):
    """
    Modify a notebook to save all figures to reports/figures
    and tables to reports/tables
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Create reports directory setup code
    setup_code = """
import os
# Setup directories for saving outputs
reports_dir = os.path.join(os.path.dirname(os.path.abspath('__file__')), 'reports')
figures_dir = os.path.join(reports_dir, 'figures')
tables_dir = os.path.join(reports_dir, 'tables')
os.makedirs(figures_dir, exist_ok=True)
os.makedirs(tables_dir, exist_ok=True)
print(f"Figures will be saved to: {figures_dir}")
print(f"Tables will be saved to: {tables_dir}")
"""

    modified = False
    cell_counter = 0

    for i, cell in enumerate(notebook['cells']):
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']

            # Check if this cell has plotting code (plt.show())
            if 'plt.show()' in source or 'display(' in source:
                # Add savefig call before plt.show()
                # Create figure filename based on notebook and counter
                base_name = notebook_name.lower().replace('.ipynb', '').replace('-', '_')

                # Determine figure type from context
                if 'sentiment' in source.lower():
                    fig_type = 'sentiment'
                elif 'topic' in source.lower() or 'lda' in source.lower():
                    fig_type = 'topic'
                elif 'cluster' in source.lower():
                    fig_type = 'cluster'
                elif 'evaluation' in source.lower() or 'eval' in source.lower():
                    fig_type = 'eval'
                else:
                    fig_type = f"fig_{cell_counter}"

                # Modify the cell to save figure
                save_code = f"""
# Save figure
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir, exist_ok=True)
plt.savefig(os.path.join(figures_dir, '{base_name}_{fig_type}.png'), dpi=300, bbox_inches='tight')
print(f"Saved figure: {base_name}_{fig_type}.png")
"""
                # Insert savefig before plt.show()
                if 'plt.show()' in source:
                    source = source.replace('plt.show()', save_code + '\nplt.show()')
                    modified = True

            # Check for to_csv or table saving
            if '.to_csv(' in source and 'tables_dir' not in source:
                # Modify to save to tables_dir
                csv_pattern = r"(\w+)\.to_csv\(['\"]([^'\"]+)['\"]"
                def replace_csv(match):
                    var_name = match.group(1)
                    filename = match.group(2)
                    if not filename.startswith('tables_dir') and not filename.startswith('/'):
                        return f"{var_name}.to_csv(os.path.join(tables_dir, '{filename}')"
                    return match.group(0)
                source = re.sub(csv_pattern, replace_csv, source)
                modified = True

            # Update the cell source
            if isinstance(cell['source'], list):
                cell['source'] = [source]
            else:
                cell['source'] = source

            cell_counter += 1

    # Add setup cell at the beginning (after the first markdown cell or at position 0)
    setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": setup_code.strip().split('\n')
    }

    # Insert after first cell or at beginning
    insert_pos = 1 if notebook['cells'] and notebook['cells'][0]['cell_type'] == 'markdown' else 0
    notebook['cells'].insert(insert_pos, setup_cell)
    modified = True

    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1)
        print(f"Modified: {notebook_path}")
    else:
        print(f"No changes needed: {notebook_path}")

    return modified

# Process all notebooks
notebooks_to_process = [
    'notebooks/01_eda/EDA-ThiruvelAP.ipynb',
    'notebooks/01_eda/EDA-Fahmi.ipynb',
    'notebooks/01_eda/EDA-Ukhash.ipynb',
    'notebooks/01_eda/EDA-Xiang.ipynb',
    'notebooks/01_eda/EDA-Zhaoqi.ipynb',
    'notebooks/02_clustering/ClusterningAnalysis.ipynb',
    'notebooks/04_pipeline/TopicModelling.ipynb',
    'notebooks/04_pipeline/SentimentAnalysis.ipynb',
    'notebooks/04_pipeline/EmbeddingsVisualization.ipynb',
    'notebooks/05_evaluation/Evaluation.ipynb',
]

for nb_path in notebooks_to_process:
    if os.path.exists(nb_path):
        try:
            nb_name = os.path.basename(nb_path)
            modify_notebook(nb_path, nb_name)
        except Exception as e:
            print(f"Error processing {nb_path}: {e}")
    else:
        print(f"Not found: {nb_path}")
