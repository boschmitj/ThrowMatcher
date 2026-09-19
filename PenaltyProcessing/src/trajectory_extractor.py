"""
Extract full trajectory data for shots by cross-referencing position tracking file.
This enriches shots.csv with complete ballistic information.

Note: This is a standalone/experimental analysis module (uses pandas/numpy)
that predates the main penalty pipeline in ``penalty_processing.py``. It loads
a shots CSV together with a single position-tracking CSV and extracts
trajectories around each shot based on raw millisecond timestamps. It also
contains heuristics to detect whether a shot is a 7 m penalty.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class TrajectoryExtractor:
    """Match shots to position tracking data and extract complete trajectories."""
    
    def __init__(self, shots_file, positions_file):
        """Load both datasets.

        Args:
            shots_file: Path to a shots CSV (semicolon-delimited).
            positions_file: Path to a single position-tracking CSV (semicolon-delimited).
        """
        print("Loading shots data...")
        self.shots_df = pd.read_csv(shots_file, sep=';')
        
        print("Loading position tracking data (this may take a moment)...")
        self.positions_df = pd.read_csv(positions_file, sep=';')
        
        # Convert timestamps to a common millisecond integer representation.
        self.shots_df['ts_ms'] = self.shots_df['timestamp_ms'].astype(int)
        self.positions_df['ts_ms'] = pd.to_numeric(
            self.positions_df['ts in ms'].astype(str).str.replace(',', ''),
            errors='coerce'
        ).astype(int)
        
        # Keep only rows whose "full name" contains "Ball".
        self.ball_data = self.positions_df[
            self.positions_df['full name'].str.contains('Ball', case=False, na=False)
        ].sort_values('ts_ms').reset_index(drop=True)
        
        print(f"Loaded {len(self.shots_df)} shots and {len(self.ball_data)} ball position points")
    
    def extract_shot_trajectory(self, shot_idx, time_window_ms=2000):
        """
        Extract full trajectory for a shot.
        
        Args:
            shot_idx: Index of shot in shots_df
            time_window_ms: Time window before/after shot to extract (ms)
            
        Returns:
            dict with trajectory data or None if no data found
        """
        shot = self.shots_df.iloc[shot_idx]
        shot_time = int(shot['ts_ms'])
        
        # Select ball points whose timestamp falls within the window around
        # the shot timestamp.
        time_range = (
            (self.ball_data['ts_ms'] >= shot_time - time_window_ms) &
            (self.ball_data['ts_ms'] <= shot_time + time_window_ms)
        )
        
        trajectory_points = self.ball_data[time_range].copy()
        
        # Need at least two points to derive any trajectory information.
        if len(trajectory_points) < 2:
            return None
        
        # Relative time (in ms) of each trajectory point compared to the shot.
        trajectory_points['time_from_shot_ms'] = trajectory_points['ts_ms'] - shot_time
        
        # Keep only the columns we care about and drop rows with missing positions.
        trajectory_points = trajectory_points[[
            'ts_ms', 'time_from_shot_ms', 'x in m', 'y in m', 'z in m',
            'speed in m/s', 'acceleration in m/s2', 'direction of movement in deg'
        ]].dropna(subset=['x in m', 'y in m', 'z in m'])
        
        if len(trajectory_points) < 2:
            return None
        
        # The release is assumed to be the last point at or before the shot
        # timestamp; if there is none, fall back to the first point.
        release_idx = (trajectory_points['time_from_shot_ms'] <= 0).sum() - 1
        if release_idx < 0:
            release_idx = 0
        
        return {
            'shot_id': shot['id'],
            'player_id': shot['player_id'],
            'shot_type': shot['shot_type'],  # 'penalty' or other
            'shot_category': shot['shot_category'],
            'shot_time_ms': shot_time,
            'shot_success': shot['success'],
            'distance_m': shot['distance'],
            'speed_at_release_ms': shot['speed_ball'],
            'shot_position_x': shot['shot_position_x'],
            'shot_position_y': shot['shot_position_y'],
            'trajectory_points': trajectory_points,
            'release_point': trajectory_points.iloc[release_idx].to_dict() if release_idx >= 0 else None,
            'max_speed': trajectory_points['speed in m/s'].max(),
            'max_acceleration': trajectory_points['acceleration in m/s2'].max(),
            'flight_duration_ms': (trajectory_points['ts_ms'].max() - trajectory_points['ts_ms'].min()),
            'num_tracking_points': len(trajectory_points)
        }
    
    def detect_7m_penalty(self, shot_idx):
        """
        Detect if a shot is a 7m penalty based on trajectory characteristics.
        
        Returns:
            dict with detection results
        """
        traj = self.extract_shot_trajectory(shot_idx, time_window_ms=3000)
        if traj is None:
            return {'is_7m_penalty': False, 'reason': 'No trajectory data', 'trajectory_data': None}
        
        shot = self.shots_df.iloc[shot_idx]
        points = traj['trajectory_points']
        
        is_penalty_type = shot['shot_category'] == 'penalty'
        is_7m_distance = 6.5 <= shot['distance'] <= 7.5  # 7m line typically 7m
        
        # Check for straight trajectory (low angle variance)
        if len(points) > 2:
            angle_variance = points['direction of movement in deg'].std()
            is_straight = angle_variance < 15  # Low variance = straight
        else:
            is_straight = False
        
        # High speed is expected
        is_high_speed = traj['max_speed'] > 12
        
        # Check if from penalty spot area (x ≈ 7m)
        shot_x = shot['shot_position_x']
        is_7m_spot = 6.5 <= abs(shot_x) <= 7.5
        
        result = {
            'is_7m_penalty': is_penalty_type and is_7m_distance and is_high_speed,
            'penalty_type': shot['shot_category'],
            'distance': shot['distance'],
            'shot_position_x': shot_x,
            'is_7m_distance': is_7m_distance,
            'is_7m_spot': is_7m_spot,
            'max_speed': traj['max_speed'],
            'is_high_speed': is_high_speed,
            'angle_variance': points['direction of movement in deg'].std() if len(points) > 2 else None,
            'is_straight': is_straight,
            'trajectory_data': traj
        }
        
        return result
    
    def get_penalty_shots(self):
        """Get all penalty shots from the data."""
        return self.shots_df[self.shots_df['shot_category'] == 'penalty'].reset_index(drop=True)
    
    def analyze_7m_penalties(self, limit=None):
        """Analyze all 7m penalty shots.

        Runs ``detect_7m_penalty`` on every penalty shot (optionally limited
        to the first ``limit`` rows) and aggregates the results into a
        DataFrame. Trajectory data is excluded from the returned table; only
        the summary metrics are kept.
        """
        penalties = self.get_penalty_shots()
        if limit:
            penalties = penalties.head(limit)
        
        print(f"\nAnalyzing {len(penalties)} penalty shots...")
        results = []
        
        for idx in penalties.index:
            detection = self.detect_7m_penalty(idx)
            if 'trajectory_data' in detection and detection['trajectory_data'] is not None:
                # Pull summary metrics out of the trajectory data and drop the
                # bulky trajectory point table from the final result.
                traj = detection.pop('trajectory_data')
                detection['distance'] = traj['distance_m']
                detection['max_speed'] = traj['max_speed']
                detection['max_acceleration'] = traj['max_acceleration']
                results.append(detection)
            else:
                # Still include penalty shots without trajectory data
                shot = self.shots_df.iloc[idx]
                if 'trajectory_data' in detection:
                    detection.pop('trajectory_data')
                detection['distance'] = shot['distance']
                detection['max_speed'] = shot['speed_ball']
                results.append(detection)
        
        return pd.DataFrame(results) if results else pd.DataFrame()


# USAGE EXAMPLE
if __name__ == '__main__':
    # Example usage: load a shots file together with one position file and
    # analyze the penalty shots found in it.
    extractor = TrajectoryExtractor(
        '/home/josh/BA/Download_CSV/shots.csv',
        '/home/josh/BA/Download_CSV/HC_Erlangen_vs_HSV_Hamburg_2_phases_positions.csv'
    )
    
    # Analyze penalty shots
    penalties_analysis = extractor.analyze_7m_penalties(limit=10)
    print("\n=== 7M PENALTY DETECTION ===")
    if len(penalties_analysis) > 0:
        cols = ['distance', 'is_7m_distance', 'max_speed', 'is_7m_penalty']
        available_cols = [c for c in cols if c in penalties_analysis.columns]
        print(penalties_analysis[available_cols].head(10))
    
    # Get detailed trajectory for first shot with data
    print("\n=== DETAILED TRAJECTORY EXAMPLE ===")
    for idx in extractor.get_penalty_shots().head(5).index:
        traj = extractor.extract_shot_trajectory(idx, time_window_ms=3000)
        if traj and traj['num_tracking_points'] > 2:
            print(f"\nShot ID: {traj['shot_id']}")
            print(f"Distance: {traj['distance_m']:.2f}m")
            print(f"Shot Type: {traj['shot_category']}")
            print(f"Max Speed: {traj['max_speed']:.2f} m/s")
            print(f"Max Acceleration: {traj['max_acceleration']:.2f} m/s²")
            print(f"Flight Duration: {traj['flight_duration_ms']}ms")
            print(f"Tracking Points: {traj['num_tracking_points']}")
            print(f"Success: {traj['shot_success']}")
            print(f"\nTrajectory Points (first 5):")
            print(traj['trajectory_points'][['time_from_shot_ms', 'x in m', 'y in m', 'z in m', 'speed in m/s']].head())
            break
