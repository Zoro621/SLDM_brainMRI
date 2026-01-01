from pathlib import Path
import json

models = [
    ('FLAIR', 'logs_brats/vae_history.json', 'logs_brats/diffusion_history.json', 31938),
    ('T1ce', 'logs_brats_t1ce/vae_history.json', 'logs_brats_t1ce/diffusion_history.json', 8149),
    ('Kaggle', 'logs_kaggle_full/vae_history.json', 'logs_kaggle_full/diffusion_history.json', 1616)
]

print('=' * 120)
print(' ' * 40 + 'COMPREHENSIVE MODEL COMPARISON TABLE')
print('=' * 120)
print()

# Create markdown table
table_data = []

for name, vae_path, diff_path, dataset_size in models:
    row = {'Model': name, 'Dataset Size': dataset_size}
    
    # VAE metrics
    if Path(vae_path).exists():
        with open(vae_path, 'r') as f:
            vae_hist = json.load(f)
        
        row['VAE Epochs'] = len(vae_hist.get('train_loss', []))
        
        if 'train_loss' in vae_hist and len(vae_hist['train_loss']) > 0:
            row['VAE Train Loss'] = f"{vae_hist['train_loss'][-1]:.6f}"
        else:
            row['VAE Train Loss'] = 'N/A'
        
        if 'val_loss' in vae_hist and len(vae_hist['val_loss']) > 0:
            row['VAE Best Val'] = f"{min(vae_hist['val_loss']):.6f}"
            row['VAE Final Val'] = f"{vae_hist['val_loss'][-1]:.6f}"
        else:
            row['VAE Best Val'] = 'N/A'
            row['VAE Final Val'] = 'N/A'
        
        if 'lr' in vae_hist and len(vae_hist['lr']) > 0:
            row['VAE Final LR'] = f"{vae_hist['lr'][-1]:.6f}"
        else:
            row['VAE Final LR'] = 'N/A'
    else:
        row['VAE Epochs'] = 'N/A'
        row['VAE Train Loss'] = 'N/A'
        row['VAE Best Val'] = 'N/A'
        row['VAE Final Val'] = 'N/A'
        row['VAE Final LR'] = 'N/A'
    
    # Diffusion metrics
    if Path(diff_path).exists():
        with open(diff_path, 'r') as f:
            diff_hist = json.load(f)
        
        # Diffusion saves every N steps, so actual steps = len * save_interval
        train_losses = diff_hist.get('train_loss', [])
        row['Diff Steps'] = len(train_losses) if train_losses else 'N/A'
        
        if train_losses and len(train_losses) > 0:
            row['Diff Initial'] = f"{train_losses[0]:.4f}"
            row['Diff Final'] = f"{train_losses[-1]:.4f}"
            row['Diff Reduction'] = f"{((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%"
        else:
            row['Diff Initial'] = 'N/A'
            row['Diff Final'] = 'N/A'
            row['Diff Reduction'] = 'N/A'
        
        if 'val_loss' in diff_hist and len(diff_hist['val_loss']) > 0:
            row['Diff Best Val'] = f"{min(diff_hist['val_loss']):.4f}"
        else:
            row['Diff Best Val'] = 'N/A'
    else:
        row['Diff Steps'] = 'N/A'
        row['Diff Initial'] = 'N/A'
        row['Diff Final'] = 'N/A'
        row['Diff Reduction'] = 'N/A'
        row['Diff Best Val'] = 'N/A'
    
    table_data.append(row)

# Print table header
print(f"{'Model':<10} {'Dataset':<10} {'VAE':<8} {'VAE Train':<12} {'VAE Best':<12} {'VAE Final':<12} {'Final':<10}")
print(f"{'':<10} {'Size':<10} {'Epochs':<8} {'Loss':<12} {'Val Loss':<12} {'Val Loss':<12} {'LR':<10}")
print('-' * 120)

# Print table rows
for row in table_data:
    print(f"{row['Model']:<10} {str(row['Dataset Size']):<10} {str(row['VAE Epochs']):<8} "
          f"{row['VAE Train Loss']:<12} {row['VAE Best Val']:<12} {row['VAE Final Val']:<12} "
          f"{row['VAE Final LR']:<10}")

print()
print(f"{'Model':<10} {'Diff':<10} {'Diff':<12} {'Diff':<12} {'Diff Loss':<12} {'Diff Best':<12}")
print(f"{'':<10} {'Steps':<10} {'Initial':<12} {'Final':<12} {'Reduction':<12} {'Val Loss':<12}")
print(f"{'':<10} {'(logged)':<10} {'Loss':<12} {'Loss':<12} {'':<12} {'':<12}")
print('-' * 120)

for row in table_data:
    print(f"{row['Model']:<10} {str(row['Diff Steps']):<10} {row['Diff Initial']:<12} "
          f"{row['Diff Final']:<12} {row['Diff Reduction']:<12} {row['Diff Best Val']:<12}")

print()
print('=' * 120)
print()

# Summary comparison
print('KEY FINDINGS:')
print()
print('1. Dataset Quality Impact:')
print(f"   - FLAIR: {models[0][3]:,} images (sparse, tumor-focused)")
print(f"   - T1ce:  {models[1][3]:,} images (contrast-enhanced, slightly richer)")
print(f"   - Kaggle: {models[2][3]:,} images (full anatomical scans)")
print()
print('2. VAE Reconstruction Quality:')
for row in table_data:
    if row['VAE Best Val'] != 'N/A':
        print(f"   - {row['Model']}: {row['VAE Best Val']}")
print()
print('3. Diffusion Model Performance:')
for row in table_data:
    if row['Diff Best Val'] != 'N/A':
        quality = 'Poor' if float(row['Diff Best Val']) > 5.0 else 'Fair' if float(row['Diff Best Val']) > 2.0 else 'Good' if float(row['Diff Best Val']) > 0.5 else 'Excellent'
        steps_info = f"{row['Diff Steps']} logged steps"
        if row['Model'] == 'FLAIR':
            steps_info += " (~100K actual)"
        elif row['Model'] == 'T1ce':
            steps_info += " (~200K actual)"
        elif row['Model'] == 'Kaggle':
            steps_info += " (~100K actual)"
        print(f"   - {row['Model']}: {row['Diff Best Val']} ({quality}) - {steps_info}")
print()
print('4. Training Efficiency:')
for row in table_data:
    if row['Diff Reduction'] != 'N/A':
        print(f"   - {row['Model']}: {row['Diff Reduction']} loss reduction")
print()
print('=' * 120)

# Save as JSON
output_file = 'model_comparison_results.json'
with open(output_file, 'w') as f:
    json.dump(table_data, f, indent=2)
print(f'\nComparison data saved to: {output_file}')
