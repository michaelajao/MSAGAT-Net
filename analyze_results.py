import pandas as pd
import numpy as np

# Read Japan results
jp = pd.read_csv('report/results/japan/all_results.csv')
# Read Australia results
au = pd.read_csv('report/results/australia-covid/all_results.csv')

print('='*80)
print('COMPREHENSIVE JAPAN RESULTS (47 nodes)')
print('='*80)

models = ['MSTAGAT-Net', 'EpiSIG-Net-V3', 'EpiSIG-Net-V5', 'EpiSIG-Net', 'EpiDelay-Net-Full']
horizons = [3, 5, 10, 15]

for h in horizons:
    print(f'\n--- Horizon {h} ---')
    print(f"{'Model':<20} {'n':>4} {'MAE':>10} {'RMSE':>12} {'R2':>8} {'PCC':>8}")
    print('-'*64)
    
    for model in models:
        data = jp[(jp['model'] == model) & (jp['horizon'] == h)]
        if model == 'MSTAGAT-Net':
            data = data[(data['ablation'] == 'none') | (data['ablation'].isna())]
        
        if len(data) > 0:
            mae = data['mae'].mean()
            rmse = data['rmse'].mean()
            r2 = data['R2'].mean()
            pcc = data['pcc'].mean()
            n = len(data)
            print(f'{model:<20} {n:>4} {mae:>10.2f} {rmse:>12.2f} {r2:>8.4f} {pcc:>8.4f}')

print('\n' + '='*80)
print('AUSTRALIA-COVID RESULTS (8 nodes)')
print('='*80)

horizons_au = [3, 7, 14]
for h in horizons_au:
    print(f'\n--- Horizon {h} ---')
    print(f"{'Model':<20} {'n':>4} {'MAE':>10} {'RMSE':>12} {'R2':>8} {'PCC':>8}")
    print('-'*64)
    
    for model in models:
        data = au[(au['model'] == model) & (au['horizon'] == h)]
        if model == 'MSTAGAT-Net':
            data = data[(data['ablation'] == 'none') | (data['ablation'].isna())]
        
        if len(data) > 0:
            mae = data['mae'].mean()
            rmse = data['rmse'].mean()
            r2 = data['R2'].mean()
            pcc = data['pcc'].mean()
            n = len(data)
            print(f'{model:<20} {n:>4} {mae:>10.2f} {rmse:>12.2f} {r2:>8.4f} {pcc:>8.4f}')

print('\n' + '='*80)
print('SUMMARY: v3 vs v5 Comparison')
print('='*80)

# Japan summary
print('\nJAPAN (large graph, 47 nodes):')
for h in [5, 10, 15]:
    v3 = jp[(jp['model'] == 'EpiSIG-Net-V3') & (jp['horizon'] == h)]
    v5 = jp[(jp['model'] == 'EpiSIG-Net-V5') & (jp['horizon'] == h)]
    if len(v3) > 0 and len(v5) > 0:
        v3_mae, v5_mae = v3['mae'].mean(), v5['mae'].mean()
        v3_r2, v5_r2 = v3['R2'].mean(), v5['R2'].mean()
        winner = 'v3' if v3_mae < v5_mae else 'v5'
        print(f"  h={h}: v3 MAE={v3_mae:.1f} R2={v3_r2:.4f} | v5 MAE={v5_mae:.1f} R2={v5_r2:.4f} -> {winner} wins")

# Australia summary
print('\nAUSTRALIA (small graph, 8 nodes):')
for h in [3, 7, 14]:
    v3 = au[(au['model'] == 'EpiSIG-Net-V3') & (au['horizon'] == h)]
    v5 = au[(au['model'] == 'EpiSIG-Net-V5') & (au['horizon'] == h)]
    if len(v3) > 0:
        v3_mae, v3_r2 = v3['mae'].mean(), v3['R2'].mean()
        if len(v5) > 0:
            v5_mae, v5_r2 = v5['mae'].mean(), v5['R2'].mean()
            winner = 'v3' if v3_mae < v5_mae else 'v5'
            print(f"  h={h}: v3 MAE={v3_mae:.1f} R2={v3_r2:.4f} | v5 MAE={v5_mae:.1f} R2={v5_r2:.4f} -> {winner} wins")
        else:
            print(f"  h={h}: v3 MAE={v3_mae:.1f} R2={v3_r2:.4f} | v5 (no data)")
