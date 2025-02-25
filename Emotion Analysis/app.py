import streamlit as st
import plotly.graph_objects as go
from SeparateApproach.main import SeparateEmotionAnalyzer
from SinglePromptApproach.main import EmotionAnalyzer

st.set_page_config(
    page_title="Emotion Analysis",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        /* Global dark theme */
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        /* Text area styling */
        .stTextArea textarea {
            background-color: #262730;
            color: #FAFAFA;
            border: none;
            border-radius: 4px;
            padding: 10px;
            font-size: 16px;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #FF4B4B;
            color: white;
            border: none;
            border-radius: 4px;
            padding: 0.5rem 2rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #FF6B6B;
            border: none;
        }
        
        /* Metric styling */
        div[data-testid="stMetricValue"] {
            font-size: 48px;
            color: #FAFAFA;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 14px;
            color: #FAFAFA;
        }
        
        /* Container styling */
        .plot-container {
            background-color: #262730;
            border-radius: 4px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }
        
        .json-viewer {
            background-color: #262730;
            border-radius: 4px;
            padding: 1.5rem;
            margin-top: 1rem;
            font-family: monospace;
        }
        
        /* Headers */
        h1, h2, h3, h4 {
            color: #FAFAFA;
            font-weight: 500;
        }
        
        /* Remove default streamlit padding */
        .main > div {
            padding: 1rem 3rem;
        }
        
        /* Custom divider */
        .divider {
            border-bottom: 1px solid #333;
            margin: 1rem 0;
        }
        
        /* Theme scores */
        .theme-score {
            background-color: #262730;
            padding: 0.5rem 1rem;
            border-radius: 4px;
            margin: 0.25rem 0;
        }

        /* Toggle container */
        .toggle-container {
            display: flex;
            justify-content: flex-end;
            padding: 1rem 0;
        }
        
        /* Radio buttons */
        div[data-testid="stRadio"] > div {
            background-color: #262730;
            padding: 0.5rem;
            border-radius: 4px;
        }
        .stRadio > label {
            color: #FAFAFA !important;
        }
    </style>
""", unsafe_allow_html=True)

def create_emotion_radar(emotions_data):
    primary = emotions_data.get('primary', {})
    secondary = emotions_data.get('secondary', {})
    
    fig = go.Figure()

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                showline=False,
                color='#666',
                gridcolor='#333',
                tickfont=dict(color='#666')
            ),
            angularaxis=dict(
                color='#666',
                gridcolor='#333'
            ),
            bgcolor='#262730'
        ),
        paper_bgcolor='#262730',
        plot_bgcolor='#262730',
        font=dict(color='#FAFAFA'),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)',
            font=dict(color='#FAFAFA')
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    if primary:
        fig.add_trace(go.Scatterpolar(
            r=[primary.get('intensity', 0)],
            theta=[primary.get('emotion', '')],
            fill='toself',
            name='Primary',
            fillcolor='rgba(255, 75, 75, 0.1)',
            line=dict(color='#FF4B4B', width=2)
        ))
    
    if secondary:
        fig.add_trace(go.Scatterpolar(
            r=[secondary.get('intensity', 0)],
            theta=[secondary.get('emotion', '')],
            fill='toself',
            name='Secondary',
            fillcolor='rgba(54, 162, 235, 0.1)',
            line=dict(color='#36A2EB', width=2)
        ))
    
    return fig

# Initialize analyzers in session state
if 'separate_analyzer' not in st.session_state:
    st.session_state.separate_analyzer = SeparateEmotionAnalyzer()
if 'single_analyzer' not in st.session_state:
    st.session_state.single_analyzer = EmotionAnalyzer()

st.title("Emotion Analysis")

# Add approach selector
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    approach = st.radio(
        "",
        ["Single Prompt", "Separate Prompts"],
        horizontal=True,
        key="approach"
    )

    feedback = st.text_area("", placeholder="Enter your feedback here...", height=100)
    analyze_button = st.button("Analyze")

if analyze_button and feedback:
    try:
        with st.spinner("Analyzing feedback..."):
            if approach == "Single Prompt":
                analyzer = st.session_state.single_analyzer
                final_result = analyzer.analyze_text(feedback)
            else:
                analyzer = st.session_state.separate_analyzer
                emotion_result = analyzer.analyze_emotions(feedback)
                if emotion_result:
                    topic_result = analyzer.analyze_topics(feedback)
                    if topic_result:
                        adorescore_result = analyzer.calculate_adorescore(feedback, topic_result)
                        if adorescore_result:
                            final_result = analyzer.generate_final_output(
                                emotion_result, topic_result, adorescore_result
                            )
                        else:
                            final_result = None
                    else:
                        final_result = None
                else:
                    final_result = None
            
            if final_result:
                st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                
                # Create main content columns
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<div class="plot-container">', unsafe_allow_html=True)
                    if 'emotions' in final_result:
                        fig = create_emotion_radar(final_result['emotions'])
                        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    if 'adorescore' in final_result:
                        st.metric("Adorescore", f"+{final_result['adorescore']['overall']}")
                    
                    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
                    
                    st.markdown("##### Analysis JSON")
                    st.markdown('<div class="json-viewer">', unsafe_allow_html=True)
                    st.json(final_result)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    if 'themes' in final_result:
                        st.markdown("##### Top Themes")
                        for theme, score in final_result['themes'].items():
                            st.markdown(
                                f'<div class="theme-score">{theme}: {score}</div>',
                                unsafe_allow_html=True
                            )
            else:
                st.error("Failed to generate analysis")
    except Exception as e:
        st.error(f"An error occurred during analysis: {str(e)}")
elif analyze_button:
    st.warning("Please enter some feedback to analyze.") 
