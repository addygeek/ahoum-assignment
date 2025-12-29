"""ACEF Streamlit Web Interface"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
from pathlib import Path

# Fix imports and paths
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
import os

# Robust .env loading
env_path = project_root.parent / ".env"
try:
    load_dotenv(env_path)
except UnicodeDecodeError:
    try:
        load_dotenv(env_path, encoding='utf-16')
    except Exception as e:
        print(f"Warning: Could not load .env file: {e}")
except Exception as e:
    print(f"Warning: Could not load .env file: {e}")

from data.facet_registry import create_registry_from_csv, FacetCategory
from data.preprocessor import Conversation, Turn, ConversationPreprocessor
from models.evaluator import ConversationEvaluator, EvaluationConfig

st.set_page_config(
    page_title="ACEF - Conversation Evaluator",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .scenario-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #667eea;
        margin: 0.5rem 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Sample conversations for different scenarios
SAMPLE_CONVERSATIONS = {
    "Mental Health Support": {
        "description": "User seeking emotional support for anxiety",
        "turns": [
            {"speaker": "user", "text": "I've been having panic attacks lately and I don't know what to do. They come out of nowhere."},
            {"speaker": "assistant", "text": "I'm really sorry to hear you're experiencing panic attacks. That sounds incredibly difficult. Can you tell me more about when they usually happen?"},
            {"speaker": "user", "text": "Usually at work or when I'm in crowded places. My heart races and I feel like I can't breathe."},
            {"speaker": "assistant", "text": "Those are common triggers. Have you tried any breathing techniques? I can guide you through a simple 4-7-8 breathing exercise that many find helpful."},
            {"speaker": "user", "text": "I haven't tried that. I usually just try to leave the situation as fast as possible."},
            {"speaker": "assistant", "text": "That's a natural response called avoidance. While it provides temporary relief, learning coping techniques can help you feel more in control. Would you like to try the breathing exercise together?"}
        ]
    },
    "Technical Problem Solving": {
        "description": "Debugging a complex software issue",
        "turns": [
            {"speaker": "user", "text": "My Python script keeps crashing with a memory error when processing large files."},
            {"speaker": "assistant", "text": "Memory errors with large files usually indicate the data is being loaded entirely into RAM. Are you using pandas to read the file?"},
            {"speaker": "user", "text": "Yes, I'm using pd.read_csv() on a 10GB file."},
            {"speaker": "assistant", "text": "That's the issue. Try using chunked reading with pd.read_csv(file, chunksize=100000). This processes the file in smaller batches."},
            {"speaker": "user", "text": "That makes sense. But I need to aggregate data across all chunks. How do I do that?"},
            {"speaker": "assistant", "text": "You can initialize an empty result and iterate through chunks: results = []; for chunk in pd.read_csv(file, chunksize=100000): results.append(chunk.groupby('column').sum()). Then pd.concat(results) at the end."},
            {"speaker": "user", "text": "Brilliant! That worked perfectly. The script now runs without crashing."},
            {"speaker": "assistant", "text": "Excellent! For even larger files, consider using Dask or Vaex which are designed for out-of-core computing."}
        ]
    },
    "Customer Service Complaint": {
        "description": "Handling an upset customer with billing issue",
        "turns": [
            {"speaker": "user", "text": "I've been charged twice for the same order and no one is helping me! This is ridiculous!"},
            {"speaker": "assistant", "text": "I sincerely apologize for the double charge and the frustration this has caused. Let me look into this immediately. Can you provide your order number?"},
            {"speaker": "user", "text": "Order #45892. I've been trying to get this resolved for 3 days now."},
            {"speaker": "assistant", "text": "I can see the duplicate charge on order #45892. I'm processing a full refund for $89.99 right now. It will appear in 3-5 business days."},
            {"speaker": "user", "text": "Finally! But why did it take so long? This should have been fixed immediately."},
            {"speaker": "assistant", "text": "You're absolutely right, and I apologize for the delay. I'm also adding a $20 credit to your account for the inconvenience. Is there anything else I can help with?"}
        ]
    },
    "Educational Tutoring": {
        "description": "Teaching a complex concept with patience",
        "turns": [
            {"speaker": "user", "text": "I don't understand recursion at all. Can you explain it?"},
            {"speaker": "assistant", "text": "Recursion is when a function calls itself. Think of it like Russian nesting dolls - each doll contains a smaller version of itself."},
            {"speaker": "user", "text": "But how does it know when to stop?"},
            {"speaker": "assistant", "text": "Great question! Every recursive function needs a 'base case' - a condition that stops the recursion. Without it, it would loop forever."},
            {"speaker": "user", "text": "Can you give me a simple example?"},
            {"speaker": "assistant", "text": "Sure! Factorial: factorial(5) = 5 * factorial(4) = 5 * 4 * factorial(3)... until factorial(1) = 1 (base case). So 5! = 5*4*3*2*1 = 120."},
            {"speaker": "user", "text": "Oh! So each call waits for the next one to finish?"},
            {"speaker": "assistant", "text": "Exactly! They stack up like a tower of plates. Each call waits for its 'child' call to return, then uses that result."}
        ]
    },
    "Conflict Resolution": {
        "description": "Mediating a workplace disagreement",
        "turns": [
            {"speaker": "user", "text": "My colleague keeps taking credit for my ideas in meetings. I'm furious."},
            {"speaker": "assistant", "text": "That's understandably frustrating. Being overlooked for your contributions is demoralizing. Have you spoken to this colleague directly?"},
            {"speaker": "user", "text": "No, I don't want to create drama. But it's affecting my motivation."},
            {"speaker": "assistant", "text": "Addressing it doesn't have to be dramatic. You could say something like 'I noticed my idea about X was presented as yours. Can we discuss this?'"},
            {"speaker": "user", "text": "What if they get defensive?"},
            {"speaker": "assistant", "text": "Stay calm and focus on facts, not accusations. Use 'I felt' statements. If it continues, document instances and involve your manager."}
        ]
    },
    "Crisis Intervention": {
        "description": "Handling a potentially sensitive situation",
        "turns": [
            {"speaker": "user", "text": "I feel like giving up. Nothing seems to matter anymore."},
            {"speaker": "assistant", "text": "I hear you, and I'm concerned about how you're feeling. When you say 'giving up,' can you help me understand what you mean?"},
            {"speaker": "user", "text": "I'm just so tired of everything. Work, relationships, life."},
            {"speaker": "assistant", "text": "It sounds like you're carrying a heavy burden. These feelings are valid, but please know you don't have to face this alone. Have you talked to anyone else about this?"},
            {"speaker": "user", "text": "No. I don't want to burden anyone."},
            {"speaker": "assistant", "text": "Your feelings are not a burden. Would you consider reaching out to a counselor or crisis helpline? They're trained to help and available 24/7. In the US, you can text HOME to 741741."}
        ]
    },
    "Creative Brainstorming": {
        "description": "Helping generate innovative ideas",
        "turns": [
            {"speaker": "user", "text": "I need unique ideas for a team building event. Budget is $500 for 20 people."},
            {"speaker": "assistant", "text": "Here are some creative options: 1) Escape room challenge, 2) Cooking competition, 3) Scavenger hunt around the city, 4) Volunteer day at local charity."},
            {"speaker": "user", "text": "The escape room sounds fun but might be too expensive. What about something outdoors?"},
            {"speaker": "assistant", "text": "Outdoor options: 1) DIY Olympics with relay races and games, 2) Hiking followed by picnic, 3) Beach cleanup + bonfire, 4) Geocaching adventure."},
            {"speaker": "user", "text": "I love the DIY Olympics idea! How would we organize that?"},
            {"speaker": "assistant", "text": "Split into 4 teams of 5. Events: sack race, water balloon toss, tug of war, trivia challenge. Prizes for winners, participation medals for all. Budget: supplies ~$100, food ~$300, prizes ~$100."}
        ]
    },
    "Negative Interaction": {
        "description": "Poor customer service example",
        "turns": [
            {"speaker": "user", "text": "I've been waiting 2 hours for support. This is unacceptable!"},
            {"speaker": "assistant", "text": "Well, we're very busy. You'll just have to wait like everyone else."},
            {"speaker": "user", "text": "But I have an urgent issue with my account being locked!"},
            {"speaker": "assistant", "text": "That's not my department. You should have called the right number."},
            {"speaker": "user", "text": "Can you at least transfer me to someone who can help?"},
            {"speaker": "assistant", "text": "I guess. Hold on."}
        ]
    }
}

def find_csv_path():
    # Helper to check paths
    def check(p):
        if p and Path(p).exists():
            print(f"Found CSV at: {p}")
            return str(p)
        return None

    # 1. Environment variable
    env_csv = os.getenv("FACETS_CSV_PATH")
    if found := check(env_csv):
        return found
        
    # 2. Heuristic paths
    paths = [
        Path("/mount/src/facenet-assignment/Facets Assignment.csv"),
        project_root.parent / "Facets Assignment.csv",
        project_root / "Facets Assignment.csv",
        Path("Facets Assignment.csv"),
    ]
    for p in paths:
        if found := check(p):
            return found
            
    # Default fallback even if not found
    fallback = project_root.parent / "Facets Assignment.csv"
    print(f"Warning: CSV not found. Defaulting to {fallback}")
    return str(fallback)

@st.cache_resource
def load_registry():
    return create_registry_from_csv(find_csv_path())

@st.cache_resource
def load_evaluator(_registry):
    return ConversationEvaluator(_registry, EvaluationConfig())

def render_header():
    st.markdown('<h1 class="main-header">ACEF - Conversation Evaluation Framework</h1>', unsafe_allow_html=True)
    st.markdown("**Automated evaluation of conversations across 399 psychological and behavioral facets**")

def render_sidebar(registry):
    with st.sidebar:
        st.header("📊 System Info")
        summary = registry.summary()
        
        col1, col2 = st.columns(2)
        col1.metric("Facets", summary["total_facets"])
        col2.metric("Categories", len(summary["by_category"]))
        
        st.divider()
        st.subheader("Browse Facets")
        
        categories = list(summary["by_category"].keys())
        selected_cat = st.selectbox("Category", ["All"] + categories)
        
        if selected_cat == "All":
            facets = list(registry.facets.values())[:30]
        else:
            facets = [f for f in registry.facets.values() if f.category.value == selected_cat][:30]
        
        for f in facets[:15]:
            with st.expander(f"{f.facet_id}. {f.name}"):
                st.caption(f"Category: {f.category.value}")
                st.caption(f"Observability: {f.observability.value}")

def render_scenario_selector():
    st.subheader("📝 Select Scenario")
    
    cols = st.columns(4)
    scenarios = list(SAMPLE_CONVERSATIONS.keys())
    
    selected = None
    for i, scenario in enumerate(scenarios):
        with cols[i % 4]:
            if st.button(scenario, use_container_width=True, key=f"btn_{scenario}"):
                selected = scenario
    
    # Show selected or default
    if "selected_scenario" not in st.session_state:
        st.session_state.selected_scenario = "Mental Health Support"
    
    if selected:
        st.session_state.selected_scenario = selected
    
    scenario = st.session_state.selected_scenario
    data = SAMPLE_CONVERSATIONS[scenario]
    
    st.markdown(f'<div class="scenario-card"><strong>{scenario}</strong>: {data["description"]}</div>', unsafe_allow_html=True)
    
    return scenario, data["turns"]

def render_conversation(turns):
    st.subheader("💬 Conversation")
    
    for i, turn in enumerate(turns):
        if turn["speaker"] == "user":
            st.chat_message("user").write(turn["text"])
        else:
            st.chat_message("assistant").write(turn["text"])
    
    return [{"turn_id": i+1, **t} for i, t in enumerate(turns)]

def render_facet_selector(registry):
    st.subheader("🎯 Facets to Evaluate")
    
    all_facet_ids = [f.facet_id for f in registry.facets.values()]
    
    options = {
        "Core 20": list(range(1, 21)),
        "Top 50": list(range(1, 51)),
        "Top 100": list(range(1, 101)),
        "Emotional (20)": [f.facet_id for f in registry.facets.values() if f.category == FacetCategory.EMOTIONAL][:20],
        "Behavioral (20)": [f.facet_id for f in registry.facets.values() if f.category == FacetCategory.BEHAVIORAL][:20],
        "Safety (all)": [f.facet_id for f in registry.facets.values() if f.category == FacetCategory.SAFETY],
        "Cognitive (all)": [f.facet_id for f in registry.facets.values() if f.category == FacetCategory.COGNITIVE],
        f"ALL {len(all_facet_ids)} FACETS": all_facet_ids,
    }
    
    selected = st.selectbox("Select facet group", list(options.keys()))
    facet_ids = options[selected]
    
    if len(facet_ids) > 100:
        st.warning(f"⚠️ Evaluating {len(facet_ids)} facets may take longer")
    else:
        st.caption(f"Evaluating {len(facet_ids)} facets")
    
    return facet_ids

def create_radar_chart(result_df):
    # Average score by category
    avg_by_facet = result_df.groupby("facet_name")["score"].mean().head(8)
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=avg_by_facet.values,
        theta=avg_by_facet.index,
        fill='toself',
        name='Avg Score'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 4])),
        title="Score Radar (Top 8 Facets)",
        height=400
    )
    return fig

def create_turn_comparison(result_df):
    turn_avg = result_df.groupby("turn_id")["score"].mean().reset_index()
    turn_avg["turn"] = turn_avg["turn_id"].apply(lambda x: f"Turn {x}")
    
    fig = px.bar(turn_avg, x="turn", y="score", title="Average Score by Turn",
                 color="score", color_continuous_scale="RdYlGn", range_color=[0, 4])
    fig.update_layout(height=350)
    return fig

def render_analysis(result, scenario_name):
    st.header("📈 Analysis Results")
    
    # Build dataframe
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
    
    df = pd.DataFrame(rows)
    
    # Key metrics
    cols = st.columns(4)
    cols[0].metric("Total Turns", result.total_turns)
    cols[1].metric("Facets Evaluated", result.total_facets_evaluated)
    cols[2].metric("Avg Score", f"{df['score'].mean():.2f}")
    cols[3].metric("Avg Confidence", f"{df['confidence'].mean():.2%}")
    
    st.divider()
    
    # Visualizations
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Heatmap", "Turn Analysis", "Radar", "Distribution", "Raw Data"])
    
    with tab1:
        pivot = df.pivot(index="facet_name", columns="turn_id", values="score")
        fig = px.imshow(pivot, labels=dict(color="Score"), 
                       x=[f"Turn {i}" for i in pivot.columns],
                       color_continuous_scale="RdYlGn", aspect="auto")
        fig.update_layout(title=f"Score Heatmap - {scenario_name}", height=max(400, len(pivot)*25))
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = create_turn_comparison(df)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            # Confidence by turn
            conf_turn = df.groupby("turn_id")["confidence"].mean().reset_index()
            fig = px.line(conf_turn, x="turn_id", y="confidence", 
                         title="Confidence by Turn", markers=True)
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        fig = create_radar_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            score_counts = df["label"].value_counts()
            colors = {"very_low": "#ef4444", "low": "#f97316", "neutral": "#6b7280", 
                     "high": "#22c55e", "very_high": "#10b981"}
            fig = px.pie(values=score_counts.values, names=score_counts.index,
                        title="Score Distribution", color=score_counts.index,
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.histogram(df, x="score", nbins=5, title="Score Histogram")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.dataframe(df, use_container_width=True, height=400)
        st.download_button("Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
    
    # Insights
    st.subheader("💡 Key Insights")
    
    high_scores = df[df["score"] >= 3]["facet_name"].value_counts().head(5)
    low_scores = df[df["score"] <= 1]["facet_name"].value_counts().head(5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Strengths (High Scores):**")
        for facet, count in high_scores.items():
            st.markdown(f"- {facet}: {count} turns")
    with col2:
        st.markdown("**Areas to Improve (Low Scores):**")
        for facet, count in low_scores.items():
            st.markdown(f"- {facet}: {count} turns")

def main():
    render_header()
    
    try:
        registry = load_registry()
        evaluator = load_evaluator(registry)
    except Exception as e:
        st.error(f"Failed to load: {e}")
        return
    
    render_sidebar(registry)
    
    st.divider()
    
    # Scenario selection
    scenario_name, turns_data = render_scenario_selector()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        turns = render_conversation(turns_data)
    
    with col2:
        facet_ids = render_facet_selector(registry)
        st.divider()
        evaluate_btn = st.button("🚀 Evaluate", type="primary", use_container_width=True)
    
    st.divider()
    
    if evaluate_btn:
        with st.spinner("Analyzing conversation..."):
            turn_objects = [Turn(turn_id=t["turn_id"], speaker=t["speaker"], text=t["text"]) for t in turns]
            conversation = Conversation(conversation_id=scenario_name.replace(" ", "_"), turns=turn_objects)
            
            try:
                result = evaluator.evaluate_conversation(conversation, facet_ids)
                st.session_state["last_result"] = result
                st.session_state["last_scenario"] = scenario_name
                st.success("Analysis complete!")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                return
    
    if "last_result" in st.session_state:
        render_analysis(st.session_state["last_result"], st.session_state.get("last_scenario", ""))

if __name__ == "__main__":
    main()
