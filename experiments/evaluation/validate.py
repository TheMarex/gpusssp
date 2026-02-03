import pandas as pd


def validate_distances(variant_dfs):
    if len(variant_dfs) < 2:
        raise ValueError("Need at least 2 variants to compare")
    
    variant_names = list(variant_dfs.keys())
    dfs = list(variant_dfs.values())
    
    merge_keys = ['from_node_id', 'to_node_id', 'rank']
    
    df_merged = dfs[0][merge_keys + ['distance']].copy()
    df_merged = df_merged.rename(columns={'distance': f'distance_{variant_names[0]}'})
    
    for i in range(1, len(variant_names)):
        df_temp = dfs[i][merge_keys + ['distance']].copy()
        df_temp = df_temp.rename(columns={'distance': f'distance_{variant_names[i]}'})
        df_merged = pd.merge(df_merged, df_temp, on=merge_keys)
    
    distance_cols = [f'distance_{name}' for name in variant_names]
    
    mismatches = []
    for i in range(len(variant_names)):
        for j in range(i + 1, len(variant_names)):
            col_i = distance_cols[i]
            col_j = distance_cols[j]
            
            errors = df_merged[df_merged[col_i] != df_merged[col_j]]
            
            if len(errors) > 0:
                mismatches.append({
                    'variant_a': variant_names[i],
                    'variant_b': variant_names[j],
                    'count': len(errors),
                    'examples': errors[merge_keys + [col_i, col_j]].head(10)
                })
    
    if len(mismatches) > 0:
        print(f"⚠️  Found {len(mismatches)} pairwise mismatches:")
        for mismatch in mismatches:
            print(f"\n  {mismatch['variant_a']} vs {mismatch['variant_b']}: {mismatch['count']} mismatches")
            print(mismatch['examples'])
        return False
    else:
        print(f"✓ All distances match across {len(variant_names)} variants")
        return True
