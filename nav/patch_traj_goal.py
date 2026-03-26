#!/usr/bin/env python3
"""Patch collect_25ep.py to overlay agent trajectory, position, and goal marker on semantic maps."""

with open('/nav/collect_25ep.py', 'r') as f:
    code = f.read()

# 1. Add overlay color constants after MAP_LAYERS
code = code.replace(
    '''MAP_LAYERS = [
    (0, "Unknown"),
    (1, "Obstacle"),
    (2, "Explored"),
]''',
    '''MAP_LAYERS = [
    (0, "Unknown"),
    (1, "Obstacle"),
    (2, "Explored"),
]

# Overlay colors (RGB)
TRAJECTORY_COLOR = (0, 180, 255)    # Cyan-blue for agent path
AGENT_POS_COLOR = (0, 220, 0)      # Green for current agent position
GOAL_MARKER_COLOR = (255, 50, 50)  # Red for goal object location'''
)

# 2. Modify add_legend to accept extra_entries
code = code.replace(
    'def add_legend(img_rgb, detected_indices):',
    'def add_legend(img_rgb, detected_indices, extra_entries=None):'
)
# Insert extra_entries append after the category loop
code = code.replace(
    '''        entries.append(((r, g, b), cat_name + marker))

    num_cols = 7''',
    '''        entries.append(((r, g, b), cat_name + marker))
    if extra_entries:
        entries.extend(extra_entries)

    num_cols = 7'''
)

# 3. Replace the visualization section in save_semantic_map
code = code.replace(
    '''    sem_vis = sem_vis.convert("RGB")
    sem_vis = np.flipud(np.array(sem_vis))
    
    # Find which categories are detected
    detected_cats = set(np.unique(sem_label)) - {0}

    # Add legend to full map
    sem_vis_with_legend = add_legend(sem_vis, detected_cats)''',
    '''    sem_vis = sem_vis.convert("RGB")
    sem_vis = np.array(sem_vis)

    # Overlay agent trajectory (channel 3)
    traj_mask = full_map[3] > 0.5
    sem_vis[traj_mask] = TRAJECTORY_COLOR

    # Overlay agent current position (channel 2) on top of trajectory
    pos_mask = full_map[2] > 0.5
    sem_vis[pos_mask] = AGENT_POS_COLOR

    # Flip vertically for display
    sem_vis = np.flipud(sem_vis)

    # Build extra legend entries
    extra_legend = [
        (TRAJECTORY_COLOR, "Agent Trajectory"),
        (AGENT_POS_COLOR, "Agent Position"),
    ]

    # Mark goal object location on the map
    goal_cat = state.goal_cat  # 0-indexed category
    if 0 <= goal_cat < len(CATEGORY_NAMES):
        goal_pixels = (sem_label == (goal_cat + 1))
        if goal_pixels.any():
            rows_g, cols_g = np.where(goal_pixels)
            cy = int(np.mean(rows_g))
            cx = int(np.mean(cols_g))
            # Flip row coordinate to match flipped image
            cy_flip = sem_vis.shape[0] - 1 - cy
            # Draw target marker (circle with dot)
            cv2.circle(sem_vis, (cx, cy_flip), 15, GOAL_MARKER_COLOR, 2)
            cv2.circle(sem_vis, (cx, cy_flip), 6, GOAL_MARKER_COLOR, -1)
        extra_legend.append((GOAL_MARKER_COLOR, "Goal: " + target_name))

    # Find which categories are detected
    detected_cats = set(np.unique(sem_label)) - {0}

    # Add legend to full map
    sem_vis_with_legend = add_legend(sem_vis, detected_cats, extra_legend)'''
)

# 4. Also pass extra_legend to the cropped version's add_legend call
code = code.replace(
    '''        cropped_with_legend = add_legend(cropped, detected_cats)''',
    '''        cropped_with_legend = add_legend(cropped, detected_cats, extra_legend)'''
)

with open('/nav/collect_25ep.py', 'w') as f:
    f.write(code)

print("Patch applied successfully.")
