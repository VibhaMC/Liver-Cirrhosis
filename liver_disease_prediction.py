def Model(patient_data):
    # Standard ranges for liver function tests
    standard_ranges = {
        'total_bilirubin': (0.1, 1.2),
        'direct_bilirubin': (0.0, 0.3),
        'alkaline_phosphatase': (44, 147),
        'sgpt': (7, 55),  # ALT
        'sgot': (8, 48),  # AST
        'total_proteins': (6.3, 8.2),
        'albumin': (3.5, 5.5)
    }

    # Initialize counter for out-of-range values
    out_of_range_count = 0

    # Check each patient's value against the standard range
    for key, value in patient_data.items():
        if key in standard_ranges:
            min_val, max_val = standard_ranges[key]
            if value < min_val or value > max_val:
                out_of_range_count += 1

    # Predict liver cirrhosis if more than 5 values are out of range
    if out_of_range_count > 5:
        return 1
    else:
        return 0

