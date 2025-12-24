"""
ACEF Streamlit Web Interface

Visual interface for conversation evaluation with:
- Facet registry browser
- Conversation input
- Heatmap visualization
- Confidence overlays
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.facet_registry import create_registry_from_csv, FacetCategory
from data.preprocessor import Conversation, Turn, ConversationPreprocessor
from models.evaluator import ConversationEvaluator, EvaluationConfig

# Page config
st.set_page_config(
    page_title="ACEF - Conversation Evaluator",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .score-very-low { background-color: #ef4444; color: white; padding: 2px 8px; border-radius: 4px; }
    .score-low { background-color: #f97316; color: white; padding: 2px 8px; border-radius: 4px; }
    .score-neutral { background-color: #6b7280; color: white; padding: 2px 8px; border-radius: 4px; }
    .score-high { background-color: #22c55e; color: white; padding: 2px 8px; border-radius: 4px; }
    .score-very-high { background-color: #10b981; color: white; padding: 2px 8px; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_registry():
    """Load facet registry from CSV."""
    csv_path = Path(__file__).parent.parent / "Facets Assignment.csv"
    return create_registry_from_csv(str(csv_path))


@st.cache_resource
def load_evaluator(_registry):
    """Initialize the evaluator."""
    config = EvaluationConfig()
    return ConversationEvaluator(_registry, config)


def render_header():
    """Render the main header."""
    st.markdown('<h1 class="main-header">🎯 ACEF - Conversation Evaluation Framework</h1>', unsafe_allow_html=True)
    st.markdown("---")


def render_sidebar(registry):
    """Render the sidebar with facet browser."""
    with st.sidebar:
        st.header("📊 Facet Registry")
        
        summary = registry.summary()
        st.metric("Total Facets", summary["total_facets"])
        
        # Category filter
        st.subheader("Browse by Category")
        categories = list(summary["by_category"].keys())
        selected_cat = st.selectbox("Select Category", ["All"] + categories)
        
        # Show facets
        if selected_cat == "All":
            facets = list(registry.facets.values())[:50]
        else:
            facets = [f for f in registry.facets.values() if f.category.value == selected_cat][:50]
        
        st.subheader(f"Facets ({len(facets)} shown)")
        for f in facets[:20]:
            with st.expander(f"{f.facet_id}. {f.name}"):
                st.write(f"**Original:** {f.original_name}")
                st.write(f"**Category:** {f.category.value}")
                st.write(f"**Observability:** {f.observability.value}")
                st.write(f"**Signal Type:** {f.signal_type.value}")


def render_conversation_input():
    """Render conversation input section."""
    st.subheader("💬 Enter Conversation")
    
    # Sample conversations
    sample_conversations = {
        "Emotional Support": [
            {"speaker": "user", "text": "I'm feeling really stressed about my exam tomorrow."},
            {"speaker": "assistant", "text": "I understand exam stress can be overwhelming. Would you like some tips?"},
            {"speaker": "user", "text": "Yes, I've been studying for weeks but still feel unprepared."},
            {"speaker": "assistant", "text": "It sounds like you've put in significant effort. Let's focus on key concepts."}
        ],
        "Problem Solving": [
            {"speaker": "user", "text": "I need help organizing a team project with 15 people."},
            {"speaker": "assistant", "text": "Let's break this down. What's the main deliverable?"},
            {"speaker": "user", "text": "We're building a mobile app for fitness tracking."},
            {"speaker": "assistant", "text": "I'd suggest dividing into specialized sub-teams: frontend, backend, and QA."}
        ],
        "Custom": []
    }
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        template = st.selectbox("Load Template", list(sample_conversations.keys()))
    
    with col2:
        conversation_id = st.text_input("Conversation ID", value="conv_001")
    
    # Turn inputs
    if template != "Custom" and sample_conversations[template]:
        turns_data = sample_conversations[template]
        st.info(f"Loaded {len(turns_data)} turns from template")
    else:
        turns_data = []
    
    st.subheader("Conversation Turns")
    
    num_turns = st.number_input("Number of turns", min_value=1, max_value=10, value=max(len(turns_data), 2))
    
    turns = []
    for i in range(int(num_turns)):
        cols = st.columns([1, 4])
        with cols[0]:
            default_speaker = turns_data[i]["speaker"] if i < len(turns_data) else ("user" if i % 2 == 0 else "assistant")
            speaker = st.selectbox(f"Speaker {i+1}", ["user", "assistant"], 
                                   index=0 if default_speaker == "user" else 1,
                                   key=f"speaker_{i}")
        with cols[1]:
            default_text = turns_data[i]["text"] if i < len(turns_data) else ""
            text = st.text_area(f"Turn {i+1}", value=default_text, key=f"text_{i}", height=80)
        
        if text:
            turns.append({"turn_id": i+1, "speaker": speaker, "text": text})
    
    return conversation_id, turns


def render_facet_selector(registry):
    """Render facet selection."""
    st.subheader("🎯 Select Facets to Evaluate")
    
    col1, col2 = st.columns(2)
    
    with col1:
        selection_mode = st.radio("Selection Mode", ["Quick Select", "Custom"])
    
    with col2:
        if selection_mode == "Quick Select":
            quick_options = {
                "Top 10 Facets": list(range(1, 11)),
                "Top 20 Facets": list(range(1, 21)),
                "Top 50 Facets": list(range(1, 51)),
                "Emotional Facets": [f.facet_id for f in registry.facets.values() 
                                     if f.category == FacetCategory.EMOTIONAL][:20],
                "Behavioral Facets": [f.facet_id for f in registry.facets.values() 
                                      if f.category == FacetCategory.BEHAVIORAL][:20],
            }
            selected_option = st.selectbox("Quick Select", list(quick_options.keys()))
            facet_ids = quick_options[selected_option]
        else:
            facet_options = {f"{f.facet_id}. {f.name}": f.facet_id for f in list(registry.facets.values())[:100]}
            selected = st.multiselect("Select Facets", list(facet_options.keys()), default=list(facet_options.keys())[:10])
            facet_ids = [facet_options[s] for s in selected]
    
    st.info(f"Selected {len(facet_ids)} facets for evaluation")
    return facet_ids


def create_heatmap(result_df):
    """Create a heatmap visualization of scores."""
    # Pivot for heatmap
    pivot_df = result_df.pivot(index="facet_name", columns="turn_id", values="score")
    
    fig = px.imshow(
        pivot_df,
        labels=dict(x="Turn", y="Facet", color="Score"),
        x=[f"Turn {i}" for i in pivot_df.columns],
        y=pivot_df.index,
        color_continuous_scale="RdYlGn",
        aspect="auto"
    )
    
    fig.update_layout(
        title="Facet Scores Heatmap",
        height=max(400, len(pivot_df) * 25)
    )
    
    return fig


def create_confidence_chart(result_df):
    """Create confidence bar chart."""
    avg_conf = result_df.groupby("facet_name")["confidence"].mean().reset_index()
    avg_conf = avg_conf.sort_values("confidence", ascending=True)
    
    fig = px.bar(
        avg_conf,
        x="confidence",
        y="facet_name",
        orientation="h",
        title="Average Confidence by Facet",
        color="confidence",
        color_continuous_scale="Blues"
    )
    
    fig.update_layout(height=max(400, len(avg_conf) * 25))
    
    return fig


def create_score_distribution(result_df):
    """Create score distribution pie chart."""
    score_counts = result_df["label"].value_counts()
    
    colors = {
        "very_low": "#ef4444",
        "low": "#f97316", 
        "neutral": "#6b7280",
        "high": "#22c55e",
        "very_high": "#10b981"
    }
    
    fig = px.pie(
        values=score_counts.values,
        names=score_counts.index,
        title="Score Distribution",
        color=score_counts.index,
        color_discrete_map=colors
    )
    
    return fig


def render_results(result):
    """Render evaluation results."""
    st.header("📊 Evaluation Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Turns", result.total_turns)
    with col2:
        st.metric("Facets Evaluated", result.total_facets_evaluated)
    with col3:
        st.metric("Total Scores", result.summary.get("total_scores", 0))
    with col4:
        st.metric("Avg Confidence", f"{result.summary.get('avg_confidence', 0):.2f}")
    
    st.markdown("---")
    
    # Build results dataframe
    rows = []
    for turn_id, facet_scores in result.scores.items():
        for facet_id, score in facet_scores.items():
            rows.append({
                "turn_id": turn_id,
                "facet_id": score.facet_id,
                "facet_name": score.facet_name,
                "score": score.score,
                "label": score.label,
                "confidence": score.confidence
            })
    
    result_df = pd.DataFrame(rows)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["🔥 Heatmap", "📈 Charts", "📋 Table", "📄 JSON"])
    
    with tab1:
        if len(result_df) > 0:
            fig = create_heatmap(result_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No results to display")
    
    with tab2:
        if len(result_df) > 0:
            col1, col2 = st.columns(2)
            with col1:
                fig = create_score_distribution(result_df)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = create_confidence_chart(result_df)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        if len(result_df) > 0:
            # Style the dataframe
            def color_score(val):
                colors = {
                    "very_low": "background-color: #fee2e2",
                    "low": "background-color: #ffedd5",
                    "neutral": "background-color: #f3f4f6",
                    "high": "background-color: #dcfce7",
                    "very_high": "background-color: #d1fae5"
                }
                return colors.get(val, "")
            
            styled_df = result_df.style.applymap(color_score, subset=["label"])
            st.dataframe(styled_df, use_container_width=True, height=400)
    
    with tab4:
        st.json(result.to_dict())


def main():
    """Main application."""
    render_header()
    
    # Load registry
    try:
        registry = load_registry()
        evaluator = load_evaluator(registry)
    except Exception as e:
        st.error(f"Failed to load registry: {e}")
        return
    
    # Sidebar
    render_sidebar(registry)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        conversation_id, turns = render_conversation_input()
    
    with col2:
        facet_ids = render_facet_selector(registry)
    
    st.markdown("---")
    
    # Evaluate button
    if st.button("🚀 Evaluate Conversation", type="primary", use_container_width=True):
        if not turns:
            st.error("Please enter at least one turn")
            return
        
        with st.spinner("Evaluating conversation..."):
            # Create conversation object
            turn_objects = [
                Turn(turn_id=t["turn_id"], speaker=t["speaker"], text=t["text"])
                for t in turns
            ]
            
            conversation = Conversation(
                conversation_id=conversation_id,
                turns=turn_objects
            )
            
            # Evaluate
            try:
                result = evaluator.evaluate_conversation(conversation, facet_ids)
                st.success("✅ Evaluation complete!")
                
                # Store in session for persistence
                st.session_state["last_result"] = result
                
                render_results(result)
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
    
    # Show last result if exists
    elif "last_result" in st.session_state:
        render_results(st.session_state["last_result"])


if __name__ == "__main__":
    main()
