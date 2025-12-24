"""
ACEF Complete System Test

This script demonstrates the full ACEF pipeline working with
the Facets Assignment.csv dataset.
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from data.facet_registry import create_registry_from_csv
from data.preprocessor import Conversation, Turn, ConversationPreprocessor
from models.evaluator import ConversationEvaluator, EvaluationConfig


def main():
    print("=" * 60)
    print("ACEF - Complete System Integration Test")
    print("=" * 60)
    
    # 1. Load facets from CSV
    print("\n[1] Loading Facets from CSV...")
    csv_path = Path(__file__).parent.parent / "Facets Assignment.csv"
    registry = create_registry_from_csv(str(csv_path))
    
    summary = registry.summary()
    print(f"    ✓ Loaded {summary['total_facets']} facets")
    print(f"    Categories: {summary['by_category']}")
    print(f"    Observability: {summary['by_observability']}")
    
    # 2. Show sample facets
    print("\n[2] Sample Facets from Dataset:")
    for i, facet in enumerate(list(registry.facets.values())[:10]):
        print(f"    {facet.facet_id}. {facet.original_name}")
        print(f"       → {facet.name} ({facet.category.value}, {facet.observability.value})")
    
    # 3. Create test conversation
    print("\n[3] Creating Test Conversation...")
    conversation = Conversation(
        conversation_id="integration_test_001",
        turns=[
            Turn(turn_id=1, speaker="user", 
                 text="I'm feeling really stressed and anxious about my calculus exam tomorrow."),
            Turn(turn_id=2, speaker="assistant", 
                 text="I understand how stressful exams can be. Let me help you prepare."),
            Turn(turn_id=3, speaker="user", 
                 text="Thank you, I've been studying for weeks but still feel unprepared."),
            Turn(turn_id=4, speaker="assistant", 
                 text="It sounds like you've put in great effort. Let's focus on key concepts together.")
        ]
    )
    print(f"    ✓ Created conversation with {len(conversation.turns)} turns")
    
    # 4. Preprocess conversation
    print("\n[4] Preprocessing Conversation...")
    preprocessor = ConversationPreprocessor()
    processed = preprocessor.preprocess(conversation)
    print(f"    ✓ Preprocessed. Risk profile: {processed.risk_profile}")
    
    # 5. Initialize evaluator
    print("\n[5] Initializing Evaluator...")
    config = EvaluationConfig()
    evaluator = ConversationEvaluator(registry, config)
    print(f"    ✓ Evaluator ready with {len(registry.facets)} facets")
    
    # 6. Select sample facets for evaluation
    print("\n[6] Selecting Facets for Evaluation...")
    # Get first 20 observable facets
    observable = registry.get_observable_facets()[:20]
    facet_ids = [f.facet_id for f in observable]
    print(f"    ✓ Selected {len(facet_ids)} facets for evaluation")
    
    # 7. Run evaluation
    print("\n[7] Running Evaluation...")
    result = evaluator.evaluate_conversation(conversation, facet_ids)
    print(f"    ✓ Evaluated {result.total_turns} turns × {result.total_facets_evaluated} facets")
    print(f"    Summary: {result.summary}")
    
    # 8. Show sample results
    print("\n[8] Sample Evaluation Results (Turn 1):")
    turn_1_scores = result.get_turn_scores(1)
    for facet_id, score in list(turn_1_scores.items())[:10]:
        print(f"    {score.facet_name}: {score.label} (conf={score.confidence:.2f})")
    
    # 9. Save results
    print("\n[9] Saving Results...")
    output_path = Path(__file__).parent / "data" / "test_results.json"
    with open(output_path, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    print(f"    ✓ Results saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("✓ COMPLETE SYSTEM INTEGRATION TEST PASSED!")
    print("=" * 60)
    print(f"\nAPI Server: http://localhost:8000")
    print(f"Swagger UI: http://localhost:8000/docs")
    print(f"Total Facets: {summary['total_facets']}")
    

if __name__ == "__main__":
    main()
