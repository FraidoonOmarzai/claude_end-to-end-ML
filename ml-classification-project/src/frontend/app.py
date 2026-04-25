# IMPORTANT: st.set_page_config() MUST be the first Streamlit command!
# That's why we import and call it immediately, before anything else.

import streamlit as st

# =============================================================================
# PAGE CONFIG - MUST BE FIRST!
# =============================================================================
st.set_page_config(
    page_title="Iris Classifier",
    page_icon="🌸",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Now we can import other modules
# =============================================================================
import requests
import json
from typing import Dict, Optional
import time

"""
Streamlit Frontend for Iris Classification
==========================================

This module creates an interactive web UI for the ML model.

STUDY NOTE: What is Streamlit?
------------------------------
Streamlit is a Python library that turns scripts into web apps.
- No HTML/CSS/JavaScript needed
- Just Python!
- Auto-reloads on code changes
- Great for ML demos and dashboards

Key Concepts:
- Widgets: Input elements (sliders, buttons, text inputs)
- Layout: Columns, sidebar, containers
- State: Session state for persistence
- Caching: @st.cache_data for performance

Run this app:
    streamlit run src/frontend/app.py --server.port 8501

Or use:
    make frontend
"""

# API Configuration
API_URL = "http://localhost:8000"

# Iris species information for display
SPECIES_INFO = {
    "setosa": {
        "emoji": "🌸",
        "color": "#FF6B6B",
        "description": "Iris Setosa - Small petals, easily distinguishable",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/Kosaciec_szczecinkowaty_Iris_setosa.jpg/220px-Kosaciec_szczecinkowaty_Iris_setosa.jpg"
    },
    "versicolor": {
        "emoji": "💜",
        "color": "#4ECDC4",
        "description": "Iris Versicolor - Medium-sized, blue flag iris",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Iris_versicolor_3.jpg/220px-Iris_versicolor_3.jpg"
    },
    "virginica": {
        "emoji": "💐",
        "color": "#45B7D1",
        "description": "Iris Virginica - Large petals, southern blue flag",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9f/Iris_virginica.jpg/220px-Iris_virginica.jpg"
    }
}


# =============================================================================
# API FUNCTIONS
# =============================================================================
"""
STUDY NOTE: Connecting to Backend
---------------------------------
The frontend communicates with FastAPI using HTTP requests.
This separation allows:
- Independent scaling
- Different technologies
- API reuse (mobile apps, other UIs)
"""


def check_api_health() -> Dict:
    """
    Check if the API is running and healthy.

    Returns:
        Dict with status information or error
    """
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"status": "offline", "error": "Cannot connect to API"}
    except Exception as e:
        return {"status": "error", "error": str(e)}


def get_model_info() -> Optional[Dict]:
    """
    Get information about the loaded model.

    Returns:
        Dict with model info or None if unavailable
    """
    try:
        response = requests.get(f"{API_URL}/model/info", timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def predict(features: Dict) -> Optional[Dict]:
    """
    Make a prediction using the API.

    Parameters:
        features: Dict with sepal_length, sepal_width, petal_length, petal_width

    Returns:
        Prediction result or None if failed
    """
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=features,
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the API is running (make api)")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
    return None


# =============================================================================
# UI COMPONENTS
# =============================================================================
"""
STUDY NOTE: Streamlit Widgets
-----------------------------
Streamlit provides many input widgets:
- st.slider(): Numeric slider
- st.number_input(): Numeric input box
- st.text_input(): Text input
- st.selectbox(): Dropdown
- st.button(): Clickable button
- st.checkbox(): Checkbox

Each widget returns its current value.
When a value changes, Streamlit reruns the script.
"""


def render_header():
    """Render the page header."""
    st.title("🌸 Iris Flower Classifier")
    st.markdown("""
    This app predicts the species of an Iris flower based on its measurements.

    **How to use:**
    1. Adjust the flower measurements using the sliders
    2. Click "Predict" to classify the flower
    3. View the prediction and confidence scores
    """)
    st.divider()


def render_sidebar():
    """
    Render the sidebar with API status and info.

    STUDY NOTE: Sidebar
    -------------------
    st.sidebar contains elements that appear in the left sidebar.
    Good for: navigation, settings, status info
    """
    with st.sidebar:
        st.header("⚙️ System Status")

        # API Health Check
        health = check_api_health()

        if health.get("status") == "healthy":
            st.success("✅ API Online")
            st.caption(f"Model: {health.get('model_name', 'N/A')}")
            st.caption(f"Version: {health.get('version', 'N/A')}")
        elif health.get("status") == "degraded":
            st.warning("⚠️ API Degraded - Model not loaded")
        else:
            st.error("❌ API Offline")
            st.caption("Run: `make api` to start")

        st.divider()

        # Model Information
        st.header("📊 Model Info")
        model_info = get_model_info()

        if model_info:
            st.write(f"**Model:** {model_info['model_name']}")
            st.write(f"**Version:** {model_info['version']}")
            st.write("**Features:**")
            for feat in model_info['feature_names']:
                st.caption(f"  • {feat}")
            st.write("**Classes:**")
            for cls in model_info['target_names']:
                info = SPECIES_INFO.get(cls, {})
                st.caption(f"  {info.get('emoji', '•')} {cls}")
        else:
            st.caption("Model info unavailable")

        st.divider()

        # About section
        st.header("ℹ️ About")
        st.markdown("""
        **Iris Dataset**

        Classic ML dataset with 150 samples
        of iris flowers from 3 species.

        [Learn more →](https://en.wikipedia.org/wiki/Iris_flower_data_set)
        """)


def render_input_form() -> Dict:
    """
    Render the input form for flower measurements.

    STUDY NOTE: Columns Layout
    --------------------------
    st.columns() creates side-by-side columns.
    col1, col2 = st.columns(2)  # Two equal columns
    col1, col2 = st.columns([2, 1])  # 2:1 ratio

    Returns:
        Dict with the input features
    """
    st.header("📏 Flower Measurements")

    # Create two columns for inputs
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sepal")
        sepal_length = st.slider(
            "Sepal Length (cm)",
            min_value=4.0,
            max_value=8.0,
            value=5.4,
            step=0.1,
            help="Length of the sepal (outer part of the flower)"
        )
        sepal_width = st.slider(
            "Sepal Width (cm)",
            min_value=2.0,
            max_value=4.5,
            value=3.4,
            step=0.1,
            help="Width of the sepal"
        )

    with col2:
        st.subheader("Petal")
        petal_length = st.slider(
            "Petal Length (cm)",
            min_value=1.0,
            max_value=7.0,
            value=4.7,
            step=0.1,
            help="Length of the petal (colorful inner part)"
        )
        petal_width = st.slider(
            "Petal Width (cm)",
            min_value=0.1,
            max_value=2.5,
            value=1.4,
            step=0.1,
            help="Width of the petal"
        )

    return {
        "sepal_length": sepal_length,
        "sepal_width": sepal_width,
        "petal_length": petal_length,
        "petal_width": petal_width
    }


def render_prediction_result(result: Dict):
    """
    Render the prediction result with visualization.

    STUDY NOTE: Metrics and Progress
    --------------------------------
    st.metric(): Display a big number with label
    st.progress(): Show a progress bar
    st.bar_chart(): Simple bar chart

    Parameters:
        result: Prediction result from API
    """
    predicted_class = result["predicted_class"]
    confidence = result["confidence"]
    probabilities = result["probabilities"]

    species_info = SPECIES_INFO.get(predicted_class, {})

    st.divider()
    st.header("🎯 Prediction Result")

    # Main prediction display
    col1, col2 = st.columns([2, 1])

    with col1:
        # Large prediction display
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {species_info.get('color', '#666')}22, {species_info.get('color', '#666')}44);
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid {species_info.get('color', '#666')};
        ">
            <h2 style="margin:0; color: {species_info.get('color', '#666')};">
                {species_info.get('emoji', '🌸')} {predicted_class.upper()}
            </h2>
            <p style="margin:5px 0 0 0; opacity: 0.8;">
                {species_info.get('description', '')}
            </p>
        </div>
        """, unsafe_allow_html=True)

        # Confidence metric
        st.metric(
            label="Confidence",
            value=f"{confidence:.1%}",
            delta=None
        )

    with col2:
        # Confidence indicator
        if confidence >= 0.9:
            st.success("High Confidence ✓")
        elif confidence >= 0.7:
            st.warning("Medium Confidence")
        else:
            st.error("Low Confidence ⚠️")

    # Probability breakdown
    st.subheader("📊 Class Probabilities")

    # Create probability bars
    for species, prob in sorted(probabilities.items(), key=lambda x: -x[1]):
        info = SPECIES_INFO.get(species, {})
        col1, col2, col3 = st.columns([1, 3, 1])

        with col1:
            st.write(f"{info.get('emoji', '•')} {species}")
        with col2:
            st.progress(prob)
        with col3:
            st.write(f"{prob:.1%}")


def render_sample_buttons() -> Optional[Dict]:
    """
    Render buttons to load sample data.

    STUDY NOTE: Session State
    -------------------------
    st.session_state persists data across reruns.
    Useful for: form data, user preferences, counters

    Returns:
        Sample data dict if a button was clicked, None otherwise
    """
    st.subheader("🧪 Try Sample Data")
    st.caption("Click a button to load typical measurements for each species")

    col1, col2, col3 = st.columns(3)

    samples = {
        "setosa": {
            "sepal_length": 5.0,
            "sepal_width": 3.4,
            "petal_length": 1.5,
            "petal_width": 0.2
        },
        "versicolor": {
            "sepal_length": 5.9,
            "sepal_width": 2.8,
            "petal_length": 4.2,
            "petal_width": 1.3
        },
        "virginica": {
            "sepal_length": 6.6,
            "sepal_width": 3.0,
            "petal_length": 5.5,
            "petal_width": 2.1
        }
    }

    with col1:
        if st.button("🌸 Setosa", use_container_width=True):
            return samples["setosa"]

    with col2:
        if st.button("💜 Versicolor", use_container_width=True):
            return samples["versicolor"]

    with col3:
        if st.button("💐 Virginica", use_container_width=True):
            return samples["virginica"]

    return None


def render_batch_prediction():
    """
    Render batch prediction interface.

    STUDY NOTE: Expander
    --------------------
    st.expander() creates a collapsible section.
    Good for: advanced options, additional info
    """
    with st.expander("📦 Batch Prediction (Advanced)"):
        st.markdown("""
        Enter multiple samples as JSON array:
        ```json
        [
            {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
            {"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}
        ]
        ```
        """)

        json_input = st.text_area(
            "JSON Input",
            height=150,
            placeholder='[{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}]'
        )

        if st.button("🚀 Predict Batch", use_container_width=True):
            if json_input:
                try:
                    samples = json.loads(json_input)
                    response = requests.post(
                        f"{API_URL}/predict/batch",
                        json={"samples": samples},
                        timeout=30
                    )

                    if response.status_code == 200:
                        results = response.json()
                        st.success(f"✅ Processed {results['count']} samples")

                        # Display results as table
                        for i, pred in enumerate(results['predictions']):
                            info = SPECIES_INFO.get(pred['predicted_class'], {})
                            st.write(
                                f"{i+1}. {info.get('emoji', '•')} "
                                f"**{pred['predicted_class']}** "
                                f"({pred['confidence']:.1%})"
                            )
                    else:
                        st.error(f"API Error: {response.text}")

                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON format")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")


# =============================================================================
# MAIN APPLICATION
# =============================================================================
"""
STUDY NOTE: Streamlit Execution Model
-------------------------------------
Streamlit reruns the ENTIRE script on every interaction.
This is different from traditional apps!

Flow:
1. User moves slider
2. Streamlit reruns app.py from top to bottom
3. Widgets remember their values
4. UI updates with new values

This is why we structure code as functions and call them in main().
"""


def main():
    """Main application entry point."""

    # Initialize session state for features
    if "features" not in st.session_state:
        st.session_state.features = {
            "sepal_length": 5.4,
            "sepal_width": 3.4,
            "petal_length": 4.7,
            "petal_width": 1.4
        }

    # Render page components
    render_header()
    render_sidebar()

    # Check for sample button clicks
    sample = render_sample_buttons()
    if sample:
        st.session_state.features = sample
        st.rerun()  # Rerun to update sliders

    st.divider()

    # Render input form
    features = render_input_form()

    st.divider()

    # Predict button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        predict_clicked = st.button(
            "🔮 Predict Species",
            type="primary",
            use_container_width=True
        )

    # Make prediction
    if predict_clicked:
        with st.spinner("Analyzing flower measurements..."):
            result = predict(features)

            if result:
                render_prediction_result(result)

    # Batch prediction section
    st.divider()
    render_batch_prediction()

    # Footer
    st.divider()
    st.caption(
        "Built with Streamlit & FastAPI | "
        "[View API Docs](http://localhost:8000/docs)"
    )


# =============================================================================
# RUN APPLICATION
# =============================================================================

if __name__ == "__main__":
    main()
