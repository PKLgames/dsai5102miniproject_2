"""Export current report tables and supporting numerical records."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'));import schema as C
s=json.loads((R/'results/analysis.json').read_text());out=R/'results/tables';out.mkdir(exist_ok=True)
names={'full_forest':'All indicators with random forest','sparse_forest':'Sparse selection with random forest','full_logistic':'All indicators with logistic regression','sparse_logistic':'Sparse selection with logistic regression','three_forest':'Nested three-variable selection','without_income':'Random forest without income variables'}
tables={
'Table1_classification':pd.DataFrame([{'Procedure':names[k],**v} for k,v in s['classification'].items()]),
'Table2_cluster_comparison':pd.DataFrame(s['clusters']),
'TableA1_indicators':pd.DataFrame([{'Indicator':v[1],'Official series code':v[0],'Log transform':'Yes' if v[4] else 'No'} for v in C.INDICATORS]),
'Country_rankings':pd.read_csv(R/'results/rankings.csv'),
'Principal_component_variance':pd.DataFrame({'component':range(1,len(s['evr'])+1),'variance_share':s['evr']}),
'Principal_component_loadings':pd.DataFrame(list(s['loadings'].items()),columns=['indicator','first_component_loading']),
'Sparse_selection_path':pd.DataFrame(s['curve']),
'Selected_indicators_and_stability':pd.DataFrame([{'indicator':k,'selected_on_full_sample':k in s['selected'],'bootstrap_selection_frequency':v} for k,v in s['selection_frequency'].items()]),
'Cluster_profiles':pd.read_csv(R/'results/cluster_profiles.csv').rename(columns={'Unnamed: 0':'cluster_zero_based'}),
'Cluster_membership':pd.DataFrame(list(s['cluster_membership'].items()),columns=['iso3','cluster_zero_based']),
'Window_rank_comparison':pd.read_csv(R/'results/figure_window_ranks.csv'),
'Indicator_missingness':pd.read_csv(R/'results/figure_missingness.csv').rename(columns={'Unnamed: 0':'indicator'}),
'Outer_fold_indicator_counts':pd.DataFrame(s['folds']),
}
for stem,df in tables.items():
 df.to_csv(out/(stem+'.csv'),index=False)
 def esc(value):
  value=f'{value:.6f}' if isinstance(value,float) else str(value)
  return value.replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')
 lines=[r'\begin{tabular}{'+'l'*len(df.columns)+'}',r'\hline',' & '.join(map(esc,df.columns))+r' \\',r'\hline']
 lines += [' & '.join(map(esc,row))+r' \\' for row in df.itertuples(index=False,name=None)]
 lines += [r'\hline',r'\end{tabular}']
 (out/(stem+'.tex')).write_text('\n'.join(lines)+'\n')
(out/'README.md').write_text('''# Current report tables

Table1_classification, Table2_cluster_comparison and TableA1_indicators correspond to Tables 1, 2 and A1 in the current report. Other files support its figures and detailed numerical checks. Each table has CSV and LaTeX versions. CSVs retain numerical precision; the report rounds classification and cluster scores to three decimals.

These tables replace the earlier project's table outputs. Do not restore old result tables or the old three-component robustness result: those do not describe the revised analysis. Cluster identifiers in supporting files start at zero; figure group labels start at one.

The nine-variable descriptive candidate and the 17–18-variable training-fold classification subsets are different results; they must not share a single fixed-list performance claim.
''')
assert len(tables['Table1_classification'])==6 and len(tables['TableA1_indicators'])==28
print('Exported',len(tables),'current tables in CSV and LaTeX.')
