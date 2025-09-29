import pandas as pd
import numpy as np
import time
from typing import Dict
import sys
import os

def original_apply_score_bounds_to_ratings(scores: pd.Series, score_to_rating_bounds: Dict[str, tuple]) -> pd.Series:
    """Original inefficient implementation (for benchmarking)"""
    # Create a manual mapping by checking each score against the bounds
    ratings = pd.Series(index=scores.index, dtype=object)
    
    # Sort ratings by their bounds for consistent application
    sorted_ratings = sorted(
        [(rating, bounds) for rating, bounds in score_to_rating_bounds.items() if rating != 'D'],
        key=lambda x: x[1][0]  # Sort by lower bound
    )
    
    # Apply ratings one by one based on their bounds
    for score_idx, score in enumerate(scores):
        # Default to the highest rating if no match is found
        assigned_rating = sorted_ratings[-1][0]
        
        # Check each rating's bounds
        for rating, (min_bound, max_bound) in sorted_ratings:
            if min_bound <= score < max_bound:
                assigned_rating = rating
                break
            # Handle edge case for the maximum value
            elif score == max_bound and max_bound == sorted_ratings[-1][1][1]:
                assigned_rating = rating
                break
                
        ratings.iloc[score_idx] = assigned_rating
    
    return ratings

def optimized_apply_score_bounds_to_ratings(scores: pd.Series, score_to_rating_bounds: Dict[str, tuple], default_rating='D') -> pd.Series:
    """
    Vectorized implementation for efficient score-to-rating mapping with large datasets.
    """
    # Filter out default rating if present and sort ratings by their bounds
    rating_bounds = [(rating, bounds) for rating, bounds in score_to_rating_bounds.items() 
                     if rating != default_rating]
    rating_bounds.sort(key=lambda x: x[1][0])  # Sort by lower bound
    
    # Create arrays of bounds and corresponding ratings
    lower_bounds = np.array([bounds[0] for _, bounds in rating_bounds])
    upper_bounds = np.array([bounds[1] for _, bounds in rating_bounds])
    ratings_list = [rating for rating, _ in rating_bounds]
    
    # Create a result series initialized with the highest rating
    # (will be used for scores that don't fall in any defined range)
    result = pd.Series(ratings_list[-1], index=scores.index)
    
    # For each rating (starting from the lowest), assign it to scores within its bounds
    # We process in reverse order so later (higher) ratings overwrite earlier ones
    scores_array = scores.values
    for i in range(len(ratings_list)-1, -1, -1):
        mask = (scores_array >= lower_bounds[i]) & (scores_array < upper_bounds[i])
        result[mask] = ratings_list[i]
    
    # Handle the edge case for exact match of the maximum bound
    # Only assign the rating if the score equals the max bound of the highest rating
    max_value_mask = scores_array == upper_bounds[-1]
    if np.any(max_value_mask):
        result[max_value_mask] = ratings_list[-1]
        
    return result

def benchmark_rating_assignment(n_samples=1000000):
    """Benchmark the performance of original vs optimized implementation"""
    print(f"Benchmarking with {n_samples:,} samples...")
    
    # Generate random scores
    np.random.seed(42)
    scores = pd.Series(np.random.beta(2, 5, size=n_samples))
    
    # Define rating bounds
    score_to_rating_bounds = {
        '1': (0.0, 0.05),
        '2': (0.05, 0.1),
        '3': (0.1, 0.2),
        '4': (0.2, 0.3),
        '5': (0.3, 0.5),
        '6': (0.5, 0.7),
        '7': (0.7, 0.9),
        '8': (0.9, 1.0),
        'D': (0, 0)  # Default rating
    }
    
    # Benchmark original implementation
    print("Testing original implementation...")
    start_time = time.time()
    original_ratings = original_apply_score_bounds_to_ratings(scores, score_to_rating_bounds)
    original_time = time.time() - start_time
    print(f"Original implementation time: {original_time:.4f} seconds")
    
    # Benchmark optimized implementation
    print("Testing optimized implementation...")
    start_time = time.time()
    optimized_ratings = optimized_apply_score_bounds_to_ratings(scores, score_to_rating_bounds)
    optimized_time = time.time() - start_time
    print(f"Optimized implementation time: {optimized_time:.4f} seconds")
    
    # Calculate speedup
    speedup = original_time / optimized_time
    print(f"Speedup: {speedup:.2f}x faster")
    
    # Verify results match
    ratings_match = original_ratings.equals(optimized_ratings)
    print(f"Results match: {ratings_match}")
    
    if not ratings_match:
        # Compare rating distributions
        original_counts = original_ratings.value_counts().sort_index()
        optimized_counts = optimized_ratings.value_counts().sort_index()
        print("\nRating distribution comparison:")
        print("Rating  | Original  | Optimized | Difference")
        print("--------|-----------|-----------|----------")
        for rating in sorted(set(original_counts.index) | set(optimized_counts.index)):
            orig_count = original_counts.get(rating, 0)
            opt_count = optimized_counts.get(rating, 0)
            diff = opt_count - orig_count
            print(f"{rating:7} | {orig_count:9,} | {opt_count:9,} | {diff:+10,}")

if __name__ == "__main__":
    # Test with different sample sizes
    for n in [1000, 10000, 100000, 1000000]:
        benchmark_rating_assignment(n)
        print("\n" + "-"*50 + "\n")