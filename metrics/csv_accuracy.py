import pandas as pd

def is_point_in_patch(x, y, z, patch_slices):
    p_slice = patch_slices[0]
    i_slice = patch_slices[1]
    l_slice = patch_slices[2]
    return (
        p_slice[0] <= x <= p_slice[1] and
        i_slice[0] <= y <= i_slice[1] and
        l_slice[0] <= z <= l_slice[1]
    )

def test_point_combinations(P, I, L, new_P, new_I, new_L, patch_slices):
    combos = [
        (P, I, L), (new_P, new_I, new_L), (new_P, I, new_L),
        (new_P, new_I, L), (P, new_I, new_L), (P, I, new_L),
        (new_P, I, L), (P, new_I, L)
    ]
    for coords in combos:
        if is_point_in_patch(*coords, patch_slices):
            return True, 'original'
    return False, None



def get_patch_slices(row):
    return   (
        (float(row['patch_L_min']), float(row['patch_L_max'])),
        (float(row['patch_P_min']), float(row['patch_P_min'])),
        (float(row['patch_I_min']), float(row['patch_I_min']))
    )

def run_cross_check(df):
    results = []

    grouped = df.groupby(['study_id', 'level'])
    print(grouped)
    for (study_id, level), group in grouped:
        if len(group) < 2:
            continue  # Skip if we don't have both sides

        row_left = group[group['condition'].str.contains("Left", case=False)]
        row_right = group[group['condition'].str.contains("Right", case=False)]

        if row_left.empty or row_right.empty:
            continue

        row_left = row_left.iloc[0]
        row_right = row_right.iloc[0]

        # --- Extract Left info ---
        LP, LI, LL = float(row_left['point_P']), float(row_left['point_I']), float(row_left['point_L'])
        Lnew_P, Lnew_I, Lnew_L = float(row_left['new_point_P']), float(row_left['new_point_I']), float(row_left['new_point_L'])
        L_patch = get_patch_slices(row_left)

        # --- Extract Right info ---
        RP, RI, RL = float(row_right['point_P']), float(row_right['point_I']), float(row_right['point_L'])
        Rnew_P, Rnew_I, Rnew_L = float(row_right['new_point_P']), float(row_right['new_point_I']), float(row_right['new_point_L'])
        R_patch = get_patch_slices(row_right)

        # Test Left in Left patch
        ok, method = test_point_combinations(LP, LI, LL, Lnew_P, Lnew_I, Lnew_L, L_patch)
   
            
        if not ok:
            ok, method = test_point_combinations(LP, LI, LL, Lnew_P, Lnew_I, Lnew_L, R_patch)
            if ok: method = "cross"

        results.append({
            "study_id": study_id,
            "level": level,
            "side": "Left",
            "inside_patch": ok,
            "method": method if ok else "not found"
        })

        # Test Right in Right patch
        ok, method = test_point_combinations(RP, RI, RL, Rnew_P, Rnew_I, Rnew_L, R_patch)
   
           
        if not ok:
            ok, method = test_point_combinations(RP, RI, RL, Rnew_P, Rnew_I, Rnew_L, L_patch)
            if ok: method = "cross"

        results.append({
            "study_id": study_id,
            "level": level,
            "side": "Right",
            "inside_patch": ok,
            "method": method if ok else "not found"
        })

    return pd.DataFrame(results)

# === Run the check ===

# Load the CSV
df = pd.read_csv("accuracy.csv")  # Replace with your path

# Run the cross-check logic
result_df = run_cross_check(df)

# Show results
print(result_df)

# Optional: Save it
result_df.to_csv("point_patch_check_results.csv", index=False)
