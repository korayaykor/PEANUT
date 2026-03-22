#!/usr/bin/env python3
"""Patch collect_25ep.py to add semantic legend to map visualizations."""

with open("/nav/collect_25ep.py", "r") as f:
    content = f.read()

# 1) Add legend-related code after CATEGORY_NAMES
legend_code = '''

# Map layer legends (index -> color from palette, label)
MAP_LAYERS = [
    (0, "Unknown"),
    (1, "Obstacle"),
    (2, "Explored"),
]


def add_legend(img_rgb, detected_indices):
    """Add a semantic legend bar below the image."""
    color_pal = [int(x * 255.) for x in color_palette]
    H, W = img_rgb.shape[:2]

    entries = []
    for idx, label in MAP_LAYERS:
        r, g, b = color_pal[idx*3], color_pal[idx*3+1], color_pal[idx*3+2]
        entries.append(((r, g, b), label))
    for cat_i, cat_name in enumerate(CATEGORY_NAMES):
        pal_idx = cat_i + 5
        r, g, b = color_pal[pal_idx*3], color_pal[pal_idx*3+1], color_pal[pal_idx*3+2]
        marker = " *" if (cat_i + 1) in detected_indices else ""
        entries.append(((r, g, b), cat_name + marker))

    num_cols = 7
    num_rows = (len(entries) + num_cols - 1) // num_cols
    row_h = 30
    col_w = max(W // num_cols, 160)
    legend_w = max(W, col_w * num_cols)
    legend_h = num_rows * row_h + 10
    swatch_size = 16

    legend = np.ones((legend_h, legend_w, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1

    for i, (color_rgb, label) in enumerate(entries):
        row = i // num_cols
        col = i % num_cols
        x = col * col_w + 8
        y = row * row_h + 8

        cv2.rectangle(legend, (x, y), (x + swatch_size, y + swatch_size),
                      (color_rgb[2], color_rgb[1], color_rgb[0]), -1)
        cv2.rectangle(legend, (x, y), (x + swatch_size, y + swatch_size),
                      (0, 0, 0), 1)

        cv2.putText(legend, label, (x + swatch_size + 5, y + swatch_size - 2),
                    font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

    if legend_w != W:
        center_pad = (legend_w - W) // 2
        padded = np.ones((H, legend_w, 3), dtype=np.uint8) * 255
        padded[:, center_pad:center_pad+W] = img_rgb
        img_rgb = padded

    sep = np.ones((2, legend_w, 3), dtype=np.uint8) * 128
    result = np.vstack([img_rgb, sep, legend])
    return result

'''

# Insert legend code after CATEGORY_NAMES block
old_cat = "    'tv_monitor', 'fireplace', 'bathtub', 'mirror', 'other'\n]\n\n\ndef save_semantic_map"
new_cat = "    'tv_monitor', 'fireplace', 'bathtub', 'mirror', 'other'\n]" + legend_code + "\ndef save_semantic_map"
assert old_cat in content, "Could not find CATEGORY_NAMES block!"
content = content.replace(old_cat, new_cat)

# 2) Add detected_cats computation and legend to full map save
old_full = '    cv2.imwrite(\n        os.path.join(save_dir, f"{prefix}_semmap.png"),\n        sem_vis[:, :, ::-1]  # RGB -> BGR for cv2\n    )\n    \n    # Also save a cropped version'

new_full = '''    # Find which categories are detected
    detected_cats = set(np.unique(sem_label)) - {0}

    # Add legend to full map
    sem_vis_with_legend = add_legend(sem_vis, detected_cats)
    cv2.imwrite(
        os.path.join(save_dir, f"{prefix}_semmap.png"),
        sem_vis_with_legend[:, :, ::-1]  # RGB -> BGR for cv2
    )
    
    # Also save a cropped version'''

assert old_full in content, "Could not find full map save block!"
content = content.replace(old_full, new_full)

# 3) Add legend to cropped map save
old_crop = '        cropped = sem_vis[rmin:rmax+1, cmin:cmax+1]\n        cv2.imwrite(\n            os.path.join(save_dir, f"{prefix}_semmap_cropped.png"),\n            cropped[:, :, ::-1]\n        )'

new_crop = '''        cropped = sem_vis[rmin:rmax+1, cmin:cmax+1]
        cropped_with_legend = add_legend(cropped, detected_cats)
        cv2.imwrite(
            os.path.join(save_dir, f"{prefix}_semmap_cropped.png"),
            cropped_with_legend[:, :, ::-1]
        )'''

assert old_crop in content, "Could not find cropped map save block!"
content = content.replace(old_crop, new_crop)

with open("/nav/collect_25ep.py", "w") as f:
    f.write(content)

print("Patched collect_25ep.py with semantic legend successfully!")
