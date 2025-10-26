#!/usr/bin/env python3
"""
Train the Bayesian Team Predictor on the full Gen9OU modern_replays dataset.
This will download ~1M teams if not already cached and train the model.
"""

import os
import sys
import time
# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from bayesian.team_predictor import BayesianTeamPredictor


def main():
    """Train the predictor on the full dataset."""
    print("=" * 60)
    print("TRAINING BAYESIAN TEAM PREDICTOR ON FULL DATASET")
    print("=" * 60)
    
    # Set up environment
    cache_dir = os.getenv('METAMON_CACHE_DIR', '/tmp/metamon_cache')
    print(f"Using cache directory: {cache_dir}")
    
    # Initialize predictor
    predictor = BayesianTeamPredictor()
    
    # Start training
    start_time = time.time()
    
    try:
        predictor.load_and_train(force_retrain=False)  # Use cache if available
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*60}")
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"{'='*60}")
        print(f"Training time: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Total teams processed: {predictor.total_teams:,}")
        print(f"Unique species found: {len(predictor.species_counts):,}")
        print(f"Model cached at: {predictor.cache_path}")
        
        # Show some statistics
        print(f"\nTop 10 most common Pokemon:")
        for species, count in predictor.species_counts.most_common(10):
            percentage = (count / predictor.total_teams) * 100
            print(f"  {species}: {count:,} ({percentage:.1f}%)")
        
        # Test predictions
        print(f"\n{'='*40}")
        print("TESTING PREDICTIONS")
        print(f"{'='*40}")
        
        # Test case 1: Popular core
        test_revealed = ["Kingambit", "Gholdengo"]
        print(f"\nGiven revealed Pokemon: {test_revealed}")
        
        predictions = predictor.predict_unrevealed_pokemon(test_revealed, max_predictions=5)
        if predictions:
            print("Most likely unrevealed teammates:")
            for species, prob in predictions:
                print(f"  {species}: {prob:.4f}")
            
            # Test config prediction
            test_species = predictions[0][0]
            config = predictor.predict_pokemon_config(test_species, test_revealed)
            print(f"\nPredicted config for {test_species}:")
            for key, value in config.items():
                if key not in ['probability', 'species']:
                    print(f"  {key}: {value}")
            print(f"  confidence: {config.get('probability', 0):.4f}")
        
        # Test case 2: Less common core
        test_revealed2 = ["Zapdos", "Pecharunt"]
        print(f"\n{'-'*40}")
        print(f"Given revealed Pokemon: {test_revealed2}")
        
        predictions2 = predictor.predict_unrevealed_pokemon(test_revealed2, max_predictions=5)
        if predictions2:
            print("Most likely unrevealed teammates:")
            for species, prob in predictions2:
                print(f"  {species}: {prob:.4f}")
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        print(f"Processed {predictor.total_teams} teams before interruption.")
        
    except Exception as e:
        print(f"\nTraining failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
