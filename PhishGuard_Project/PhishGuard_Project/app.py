import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
import streamlit.components.v1 as components

# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))

from src.feature_extractor import extract_url_features

st.set_page_config(
    page_title="PhishGuard | Phishing URL Analysis",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# 2. PROFESSIONAL LIGHT THEME
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --navy: #183653;
        --blue: #416F9E;
        --text: #263445;
        --muted: #718096;
        --border: #E2E8F0;
        --background: #F7F9FC;
        --green: #287D61;
        --red: #B94B4B;
    }
    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: "Inter", "Segoe UI", sans-serif;
    }
    [data-testid="stSidebar"] {
        background: #F0F4F8;
        border-right: 1px solid var(--border);
    }
    [data-testid="stSidebar"] > div:first-child { padding-top: 1.6rem; }
    .block-container {
        max-width: 1500px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3 {
        color: var(--navy);
        letter-spacing: -0.3px;
        font-weight: 650;
    }
    h1 { font-size: 2rem; }
    h2 { font-size: 1.45rem; }
    h3 { font-size: 1.12rem; }
    p, li { color: var(--text); line-height: 1.65; }
    .brand-name {
        color: var(--navy);
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.7px;
        margin-bottom: 0.1rem;
    }
    .brand-subtitle, .page-subtitle {
        color: var(--muted);
        font-size: 0.88rem;
        margin-bottom: 1.3rem;
    }
    .page-subtitle { margin-top: -0.45rem; font-size: 0.96rem; }
    .section-label {
        color: var(--blue);
        font-size: 0.73rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .info-card {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 7px;
        padding: 16px 18px;
        margin-bottom: 11px;
    }
    .info-card-title { color: var(--navy); font-weight: 650; }
    .info-card-text { color: var(--muted); font-size: 0.86rem; }
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid var(--border);
        border-radius: 7px;
        padding: 14px 11px;
        min-width: 0;
        overflow: visible;
    }
    [data-testid="stMetricLabel"] {
        color: var(--muted);
        font-size: 0.78rem;
        white-space: normal;
    }
    [data-testid="stMetricValue"] {
        color: var(--navy);
        font-weight: 700;
        font-size: clamp(1.12rem, 1.65vw, 1.75rem);
        white-space: nowrap;
        overflow: visible;
    }
    div.stButton > button, div.stFormSubmitButton > button {
        background: var(--navy);
        color: #FFFFFF;
        border: 1px solid var(--navy);
        border-radius: 5px;
        font-weight: 600;
        min-height: 2.6rem;
    }
    div.stButton > button:hover, div.stFormSubmitButton > button:hover {
        background: var(--blue);
        color: #FFFFFF;
        border-color: var(--blue);
    }
    .footer-text {
        color: var(--muted);
        font-size: 0.78rem;
        text-align: center;
    }
    hr { border-color: var(--border); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 3. LOAD MODEL AND ARTIFACTS
# ============================================================

@st.cache_resource
def load_model():
    """Load the portable native XGBoost JSON model."""
    from xgboost import XGBClassifier

    model_path = MODELS / "xgboost_model.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    classifier = XGBClassifier()
    classifier.load_model(str(model_path))
    return classifier


@st.cache_resource
def load_explainer():
    return shap.TreeExplainer(load_model())


@st.cache_data
def load_json(filename):
    path = ARTIFACTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_csv(filename):
    path = ARTIFACTS / filename
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return pd.read_csv(path)


# Fail clearly if an expected project file is missing.
model = load_model()
metrics = load_json("evaluation_metrics.json")
dataset_summary = load_json("dataset_summary.json")
comparison = load_csv("model_comparison.csv")
threshold_results = load_csv("threshold_analysis.csv")
shap_importance = load_csv("shap_feature_importance.csv")
class_distribution = load_csv("class_distribution.csv")
top_tlds = load_csv("top_tlds.csv")

with (MODELS / "feature_names.json").open("r", encoding="utf-8") as file:
    feature_data = json.load(file)

if isinstance(feature_data, dict):
    FEATURE_NAMES = feature_data.get(
        "features", feature_data.get("feature_names", [])
    )
else:
    FEATURE_NAMES = feature_data

with (MODELS / "threshold.json").open("r", encoding="utf-8") as file:
    threshold_config = json.load(file)

THRESHOLD = float(
    threshold_config.get("threshold", 0.40)
    if isinstance(threshold_config, dict)
    else threshold_config
)

if not FEATURE_NAMES:
    st.error("The feature_names.json file does not contain any features.")
    st.stop()

# ============================================================
# 4. SIDEBAR AND COMMON HELPERS
# ============================================================

st.sidebar.markdown(
    "<div class='brand-name'>PhishGuard</div>", unsafe_allow_html=True
)
st.sidebar.markdown(
    "<div class='brand-subtitle'>Explainable Phishing URL Detection</div>",
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Live URL Scanner",
        "Explainability",
        "Dataset Insights",
        "Model Evaluation",
        "Methodology & Limitations",
    ],
)

st.sidebar.divider()
st.sidebar.markdown("**MODEL CONFIGURATION**")
st.sidebar.caption("Classification algorithm")
st.sidebar.write("XGBoost")
st.sidebar.caption("Input characteristics")
st.sidebar.write(f"{len(FEATURE_NAMES)} URL-derived features")
st.sidebar.caption("Decision threshold")
st.sidebar.write(f"{THRESHOLD:.2f}")
st.sidebar.divider()
st.sidebar.caption("Academic research prototype")


def make_prediction(url):
    features = extract_url_features(url)
    input_df = pd.DataFrame([features], columns=FEATURE_NAMES)

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)
    phishing_index = classes.index(0)
    legitimate_index = classes.index(1)

    phishing_probability = float(probabilities[phishing_index])
    legitimate_probability = float(probabilities[legitimate_index])
    prediction = (
        "Phishing"
        if phishing_probability >= THRESHOLD
        else "Legitimate"
    )

    return (
        features,
        input_df,
        prediction,
        phishing_probability,
        legitimate_probability,
    )


def get_local_shap_values(input_df):
    explainer = load_explainer()
    values = explainer.shap_values(input_df)

    if isinstance(values, list):
        values = values[1]

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[:, :, 1]

    values = values.reshape(-1)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} SHAP values, got {len(values)}."
        )
    return values


def page_heading(title, subtitle):
    st.title(title)
    st.markdown(
        f"<div class='page-subtitle'>{subtitle}</div>",
        unsafe_allow_html=True,
    )


def section_heading(title, description=None):
    st.subheader(title)
    if description:
        st.caption(description)


# ============================================================
# 5. SANKEY WORKFLOW GRAPH
# ============================================================

def create_flow_graph(prediction, phishing_probability, legitimate_probability):
    labels = [
        "Input URL",
        "17 URL Features",
        "XGBoost Model",
        "Phishing",
        "Legitimate",
    ]
    node_colors = [
        "#A9C4DF", "#BFD2E5", "#527EBA", "#D58B8B", "#65A78C"
    ]
    outcome_colors = [
        "#D58B8B" if prediction == "Phishing" else "rgba(213,139,139,0.30)",
        "#65A78C" if prediction == "Legitimate" else "rgba(101,167,140,0.30)",
    ]

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=27,
                    thickness=23,
                    line=dict(color="#DCE5EF", width=1),
                    label=labels,
                    color=node_colors,
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=[0, 1, 2, 2],
                    target=[1, 2, 3, 4],
                    value=[
                        1,
                        1,
                        max(phishing_probability, 0.001),
                        max(legitimate_probability, 0.001),
                    ],
                    color=[
                        "rgba(93,135,178,0.30)",
                        "rgba(93,135,178,0.38)",
                        outcome_colors[0],
                        outcome_colors[1],
                    ],
                    hovertemplate=(
                        "%{source.label} → %{target.label}"
                        "<br>Flow: %{value:.3f}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title=dict(text="Prediction Workflow", font=dict(size=17, color="#183653")),
        font=dict(size=12, color="#34465A"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        height=390,
        margin=dict(l=15, r=15, t=65, b=15),
    )
    return fig


# ============================================================
# 6. INTERACTIVE FEATURE NETWORK
# ============================================================

def create_interactive_graph_html(
    url,
    features,
    prediction,
    phishing_probability,
    legitimate_probability,
    local_shap_values,
):
    """Build an interactive Vis Network graph for the current scan."""

    feature_data = []
    for index, name in enumerate(FEATURE_NAMES):
        shap_value = float(local_shap_values[index])

        if shap_value > 0:
            contribution = "Pushes toward legitimate (class 1)"
            color = "#65A78C"
        elif shap_value < 0:
            contribution = "Pushes toward phishing (class 0)"
            color = "#D58B8B"
        else:
            contribution = "No measurable contribution"
            color = "#B8C5D3"

        feature_data.append(
            {
                "id": f"feature_{index}",
                "name": name,
                "value": str(features[name]),
                "shap": shap_value,
                "contribution": contribution,
                "color": color,
            }
        )

    payload = {
        "url": url,
        "prediction": prediction,
        "phishing_probability": phishing_probability,
        "legitimate_probability": legitimate_probability,
        "threshold": THRESHOLD,
        "features": feature_data,
    }

    # JSON is inserted as data, not executable source.
    payload_json = json.dumps(payload, ensure_ascii=True).replace("</", "<\\/")

    html_template = r"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
* { box-sizing: border-box; }
body {
    margin: 0; padding: 12px; font-family: Arial, sans-serif;
    color: #263445; background: #FFFFFF;
}
.toolbar {
    display: flex; flex-wrap: wrap; gap: 8px;
    align-items: center; margin-bottom: 12px;
}
.toolbar button {
    background: #FFFFFF; border: 1px solid #CBD5E1;
    color: #183653; padding: 8px 12px; border-radius: 5px;
    cursor: pointer; font-size: 12px;
}
.toolbar button.active {
    background: #183653; color: #FFFFFF; border-color: #183653;
}
.toolbar label {
    font-size: 12px; color: #526477;
    display: flex; gap: 6px; align-items: center;
}
#network {
    height: 510px; width: 100%; border: 1px solid #E2E8F0;
    border-radius: 6px; background: #FFFFFF;
}
.details {
    margin-top: 12px; padding: 14px 16px;
    background: #F7F9FC; border: 1px solid #E2E8F0;
    border-radius: 6px; min-height: 105px;
}
.details h3 { margin: 0 0 8px; color: #183653; font-size: 15px; }
.details p {
    font-size: 13px; line-height: 1.6; margin: 4px 0;
    color: #526477; overflow-wrap: anywhere;
}
.legend {
    display: flex; flex-wrap: wrap; gap: 16px;
    margin: 10px 0; font-size: 11px; color: #64748B;
}
.dot {
    display: inline-block; width: 9px; height: 9px;
    border-radius: 50%; margin-right: 5px;
}
#status { color: #64748B; font-size: 11px; margin-left: auto; }
</style>
</head>
<body>
<div class="toolbar">
    <button id="allBtn" class="active">Show all features</button>
    <button id="topBtn">Top 8 contributors</button>
    <button id="resetBtn">Reset layout</button>
    <label><input type="checkbox" id="physicsToggle" checked>
        Enable node movement</label>
    <span id="status">17 features</span>
</div>
<div class="legend">
    <span><i class="dot" style="background:#8EAFD0"></i>URL input</span>
    <span><i class="dot" style="background:#65A78C"></i>Pushes toward legitimate</span>
    <span><i class="dot" style="background:#D58B8B"></i>Pushes toward phishing</span>
    <span><i class="dot" style="background:#527EBA"></i>Model</span>
</div>
<div id="network"></div>
<div class="details" id="details">
    <h3>Explore the prediction graph</h3>
    <p>Select a node to inspect its details. Select a feature node to see its
    extracted value and SHAP contribution.</p>
</div>

<script>
const data = __PAYLOAD__;
const featureNodes = data.features.map((feature) => ({
    id: feature.id,
    label: feature.name,
    color: {
        background: feature.color,
        border: "#FFFFFF",
        highlight: {background: feature.color, border: "#183653"}
    },
    size: 17,
    font: {size: 11, color: "#263445", face: "Arial"},
    title: feature.name +
        "\nExtracted value: " + feature.value +
        "\nSHAP contribution: " + feature.shap.toFixed(5) +
        "\n" + feature.contribution
}));

const nodes = new vis.DataSet([
    {
        id: "url", label: "Submitted URL", shape: "box",
        color: "#EAF1F8", font: {color: "#183653", size: 13},
        title: data.url, margin: 12
    },
    ...featureNodes,
    {
        id: "model", label: "XGBoost Model", shape: "box",
        color: "#527EBA", font: {color: "#FFFFFF", size: 13},
        margin: 12, title: "XGBoost classifier\n17 input features"
    },
    {
        id: "output", label: data.prediction, shape: "box",
        color: data.prediction === "Phishing" ? "#F5DEDE" : "#DCEFE6",
        font: {color: "#183653", size: 13}, margin: 12,
        title: "Prediction: " + data.prediction +
            "\nPhishing score: " +
            (data.phishing_probability * 100).toFixed(2) + "%" +
            "\nLegitimate score: " +
            (data.legitimate_probability * 100).toFixed(2) + "%"
    }
]);

const edgesArray = [];
data.features.forEach((feature) => {
    edgesArray.push({
        id: "url_" + feature.id, from: "url", to: feature.id,
        arrows: "to", color: {color: "#CBD6E2", opacity: 0.8}, width: 1
    });
    edgesArray.push({
        id: "model_" + feature.id, from: feature.id, to: "model",
        arrows: "to", color: {color: "#CBD6E2", opacity: 0.8}, width: 1
    });
});
edgesArray.push({
    id: "model_output", from: "model", to: "output", arrows: "to",
    color: {color: "#527EBA"}, width: 2
});
const edges = new vis.DataSet(edgesArray);

const network = new vis.Network(
    document.getElementById("network"),
    {nodes: nodes, edges: edges},
    {
        autoResize: true,
        nodes: {borderWidth: 1, borderWidthSelected: 3, shadow: false},
        edges: {
            smooth: {enabled: true, type: "continuous", roundness: 0.25},
            selectionWidth: 2
        },
        physics: {
            enabled: true,
            stabilization: {iterations: 180},
            barnesHut: {
                gravitationalConstant: -3500, centralGravity: 0.25,
                springLength: 145, springConstant: 0.035, damping: 0.2
            }
        },
        interaction: {
            hover: true, tooltipDelay: 150, dragNodes: true,
            dragView: true, zoomView: true, navigationButtons: true,
            keyboard: true
        }
    }
);

function escapeHTML(value) {
    return String(value).replace(/[&<>"']/g, function(char) {
        return {
            "&": "&amp;", "<": "&lt;", ">": "&gt;",
            '"': "&quot;", "'": "&#039;"
        }[char];
    });
}

function showDetails(nodeId) {
    const panel = document.getElementById("details");
    if (!nodeId) {
        panel.innerHTML =
            "<h3>Explore the prediction graph</h3>" +
            "<p>Select a node to inspect its details.</p>";
        return;
    }
    if (nodeId === "url") {
        panel.innerHTML =
            "<h3>Submitted URL</h3><p><b>URL:</b> " +
            escapeHTML(data.url) + "</p>" +
            "<p>This is the input supplied to the feature extractor.</p>";
        return;
    }
    if (nodeId === "model") {
        panel.innerHTML =
            "<h3>XGBoost Classification Model</h3>" +
            "<p><b>Input features:</b> " + data.features.length + "</p>" +
            "<p><b>Decision threshold:</b> " +
            data.threshold.toFixed(2) + "</p>" +
            "<p>The model receives the extracted feature vector.</p>";
        return;
    }
    if (nodeId === "output") {
        panel.innerHTML =
            "<h3>Classification Result: " +
            escapeHTML(data.prediction) + "</h3>" +
            "<p><b>Phishing score:</b> " +
            (data.phishing_probability * 100).toFixed(2) + "%</p>" +
            "<p><b>Legitimate score:</b> " +
            (data.legitimate_probability * 100).toFixed(2) + "%</p>" +
            "<p><b>Threshold:</b> " + data.threshold.toFixed(2) + "</p>";
        return;
    }
    const feature = data.features.find(item => item.id === nodeId);
    if (feature) {
        panel.innerHTML =
            "<h3>" + escapeHTML(feature.name) + "</h3>" +
            "<p><b>Extracted value:</b> " +
            escapeHTML(feature.value) + "</p>" +
            "<p><b>SHAP contribution:</b> " +
            feature.shap.toFixed(6) + "</p>" +
            "<p><b>Interpretation:</b> " +
            escapeHTML(feature.contribution) + "</p>" +
            "<p>SHAP values explain contribution to the model's raw output " +
            "for class 1 (legitimate), not a direct percentage change " +
            "in probability.</p>";
    }
}

network.on("click", function(params) {
    if (params.nodes.length > 0) showDetails(params.nodes[0]);
});
network.on("deselectNode", function() { showDetails(null); });

function setVisibleFeatures(topOnly) {
    const sorted = [...data.features].sort(
        (a, b) => Math.abs(b.shap) - Math.abs(a.shap)
    );
    const visibleIds = new Set(
        (topOnly ? sorted.slice(0, 8) : data.features).map(item => item.id)
    );

    nodes.update(data.features.map(feature => ({
        id: feature.id, hidden: !visibleIds.has(feature.id)
    })));

    const edgeUpdates = data.features.flatMap(feature => ([
        {id: "url_" + feature.id, hidden: !visibleIds.has(feature.id)},
        {id: "model_" + feature.id, hidden: !visibleIds.has(feature.id)}
    ]));
    edges.update(edgeUpdates);

    document.getElementById("status").textContent =
        topOnly ? "Showing 8 of " + data.features.length + " features"
                : "Showing all " + data.features.length + " features";
    network.fit({animation: true});
}

document.getElementById("allBtn").onclick = function() {
    this.classList.add("active");
    document.getElementById("topBtn").classList.remove("active");
    setVisibleFeatures(false);
};
document.getElementById("topBtn").onclick = function() {
    this.classList.add("active");
    document.getElementById("allBtn").classList.remove("active");
    setVisibleFeatures(true);
};
document.getElementById("resetBtn").onclick = function() {
    network.setOptions({physics: {enabled: true}});
    network.stabilize();
    network.fit({animation: true});
    document.getElementById("physicsToggle").checked = true;
};
document.getElementById("physicsToggle").onchange = function() {
    network.setOptions({physics: {enabled: this.checked}});
};
network.once("stabilizationIterationsDone", function() {
    network.fit({animation: true});
});
</script>
</body>
</html>
"""
    return html_template.replace("__PAYLOAD__", payload_json)


# ============================================================
# 7. APPLICATION HEADER
# ============================================================

st.markdown("<div class='brand-name'>PhishGuard</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='page-subtitle'>Explainable and Robust Machine Learning "
    "Framework for Phishing URL Detection</div>",
    unsafe_allow_html=True,
)

# ============================================================
# 8. OVERVIEW
# ============================================================

if page == "Overview":
    page_heading(
        "Project Overview",
        "An explainable machine learning approach to structural phishing URL analysis.",
    )
    st.markdown(
        "PhishGuard analyzes URL-derived structural characteristics and applies "
        "an XGBoost classifier to distinguish potentially phishing URLs from "
        "legitimate URLs."
    )

    test = metrics["test"]
    validation = metrics["validation"]

    st.markdown(
        "<div class='section-label'>Held-out test performance</div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy", f"{test['accuracy'] * 100:.2f}%")
    c2.metric("Phishing Precision", f"{test['phishing_precision'] * 100:.2f}%")
    c3.metric("Phishing Recall", f"{test['phishing_recall'] * 100:.2f}%")
    c4.metric("Phishing F1-score", f"{test['phishing_f1'] * 100:.2f}%")
    c5.metric("ROC-AUC", f"{test['roc_auc']:.4f}")

    st.divider()
    section_heading("Dataset Summary")
    a, b, c, d = st.columns([1.3, 1.3, 1.3, 0.9])
    a.metric("Unique URLs", "235,370")
    b.metric("Unique Domains", "175,509")
    c.metric("Phishing URLs", "100,520")
    d.metric("URL Features", "17")

    st.divider()
    section_heading(
        "Validation and Test Performance",
        "Comparison of model performance across the two evaluation sets.",
    )
    metric_comparison = pd.DataFrame(
        {
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
            "Validation": [
                validation["accuracy"],
                validation["phishing_precision"],
                validation["phishing_recall"],
                validation["phishing_f1"],
                validation["roc_auc"],
            ],
            "Test": [
                test["accuracy"],
                test["phishing_precision"],
                test["phishing_recall"],
                test["phishing_f1"],
                test["roc_auc"],
            ],
        }
    )
    fig = px.bar(
        metric_comparison,
        x="Metric",
        y=["Validation", "Test"],
        barmode="group",
        color_discrete_sequence=["#A9C4DF", "#527EBA"],
        labels={"value": "Score", "variable": "Evaluation set"},
    )
    fig.update_layout(
        yaxis_tickformat=".0%",
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
        margin=dict(t=25, b=20),
    )
    st.plotly_chart(
        fig, use_container_width=True,
        config={"displayModeBar": True, "scrollZoom": True, "responsive": True},
    )
    st.info(
        "The test set was separated by registered domain. These results describe "
        "performance on the evaluated dataset and do not guarantee real-world detection."
    )

# ============================================================
# 9. LIVE URL SCANNER
# ============================================================

elif page == "Live URL Scanner":
    page_heading(
        "Live URL Scanner",
        "Analyze a URL using the trained classifier and inspect its feature-level explanation.",
    )
    st.markdown("**Enter a website URL**")

    with st.form("url_scan_form"):
        url = st.text_input(
            "URL", placeholder="https://example.com", label_visibility="collapsed"
        )
        scan = st.form_submit_button("Analyze URL", use_container_width=True)

    if scan:
        if not url.strip():
            st.warning("Please enter a URL.")
        else:
            try:
                (
                    features,
                    input_df,
                    prediction,
                    phishing_probability,
                    legitimate_probability,
                ) = make_prediction(url)

                st.session_state["last_scan"] = {
                    "url": url,
                    "features": features,
                    "input_df": input_df,
                    "prediction": prediction,
                    "phishing_probability": phishing_probability,
                    "legitimate_probability": legitimate_probability,
                }
            except Exception as error:
                st.error(f"Unable to analyze this URL: {error}")

    if "last_scan" in st.session_state:
        result = st.session_state["last_scan"]
        st.divider()
        section_heading("Classification Result")

        if result["prediction"] == "Phishing":
            st.error("Model classification: PHISHING")
        else:
            st.success("Model classification: LEGITIMATE")

        c1, c2, c3 = st.columns(3)
        c1.metric("Phishing Score", f"{result['phishing_probability'] * 100:.2f}%")
        c2.metric("Legitimate Score", f"{result['legitimate_probability'] * 100:.2f}%")
        c3.metric("Decision Threshold", f"{THRESHOLD:.2f}")

        st.caption(
            "Scores are outputs of the trained model. They are not independently "
            "verified security probabilities."
        )
        st.divider()

        section_heading(
            "Prediction Workflow",
            "Explore how the submitted URL moves through feature extraction and classification.",
        )
        st.markdown("**1. Sankey workflow visualization**")
        st.caption(
            "Displays the flow from URL input to feature extraction, classification "
            "and the two possible outcomes. Branch widths reflect model scores."
        )
        sankey_fig = create_flow_graph(
            result["prediction"],
            result["phishing_probability"],
            result["legitimate_probability"],
        )
        st.plotly_chart(
            sankey_fig,
            use_container_width=True,
            key="phishguard_sankey",
            config={
                "displayModeBar": True,
                "scrollZoom": True,
                "responsive": True,
            },
        )

        st.markdown("**2. Interactive feature network**")
        st.caption(
            "Click a node to inspect it, drag nodes to rearrange the network, "
            "zoom and pan, or show only the eight largest absolute SHAP contributors."
        )

        try:
            local_values = get_local_shap_values(result["input_df"])
            graph_html = create_interactive_graph_html(
                result["url"],
                result["features"],
                result["prediction"],
                result["phishing_probability"],
                result["legitimate_probability"],
                local_values,
            )
            components.html(graph_html, height=700, scrolling=False)
        except Exception as error:
            st.warning(f"Interactive feature graph unavailable: {error}")

        st.divider()
        section_heading(
            "Extracted URL Features",
            "The 17 structural characteristics supplied to the XGBoost model.",
        )
        feature_table = pd.DataFrame(
            {
                "Feature": FEATURE_NAMES,
                "Extracted Value": [result["features"][name] for name in FEATURE_NAMES],
            }
        )
        st.dataframe(feature_table, use_container_width=True, hide_index=True)

        st.divider()
        section_heading(
            "Local SHAP Explanation",
            "Feature contributions for this individual prediction.",
        )
        try:
            local_values = get_local_shap_values(result["input_df"])
            local_df = pd.DataFrame(
                {"Feature": FEATURE_NAMES, "SHAP Contribution": local_values}
            )
            local_df["Absolute Contribution"] = local_df["SHAP Contribution"].abs()
            local_df = local_df.sort_values(
                "Absolute Contribution", ascending=True
            ).tail(10)

            fig = px.bar(
                local_df,
                x="SHAP Contribution",
                y="Feature",
                orientation="h",
                color="SHAP Contribution",
                color_continuous_scale=["#D58B8B", "#F1F3F6", "#65A78C"],
                labels={"SHAP Contribution": "Contribution to model output"},
            )
            fig.add_vline(x=0, line_dash="dash", line_color="#65758B")
            fig.update_layout(
                paper_bgcolor="white",
                plot_bgcolor="white",
                coloraxis_showscale=False,
                margin=dict(t=25, b=20),
            )
            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displayModeBar": True, "responsive": True},
            )
            st.caption(
                "Positive SHAP contributions push the model's raw output toward "
                "class 1 (legitimate); negative contributions push it toward "
                "class 0 (phishing). These are raw model-output contributions, "
                "not percentage-point changes in probability."
            )
        except Exception as error:
            st.warning(f"SHAP explanation unavailable: {error}")

# ============================================================
# 10. GLOBAL EXPLAINABILITY
# ============================================================

elif page == "Explainability":
    page_heading(
        "Model Explainability",
        "Understanding which URL characteristics influence the trained classifier.",
    )
    section_heading(
        "Global Feature Importance",
        "Mean absolute SHAP values calculated using a validation sample of 1,000 URLs.",
    )
    importance_sorted = shap_importance.sort_values(
        "Mean Absolute SHAP", ascending=True
    )
    fig = px.bar(
        importance_sorted,
        x="Mean Absolute SHAP",
        y="Feature",
        orientation="h",
        color_discrete_sequence=["#739BC8"],
    )
    fig.update_layout(
        paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=25, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        section_heading("SHAP Summary Plot")
        st.image(
            str(ARTIFACTS / "shap_summary.png"), use_container_width=True
        )
    with col2:
        section_heading("SHAP Importance Plot")
        st.image(str(ARTIFACTS / "shap_bar.png"), use_container_width=True)
    st.caption(
        "SHAP values describe model behaviour and feature contributions. "
        "They do not establish causation."
    )

# ============================================================
# 11. DATASET INSIGHTS
# ============================================================

elif page == "Dataset Insights":
    page_heading(
        "Dataset Insights",
        "Descriptive analysis of the PhiUSIIL dataset used during model development.",
    )
    col1, col2 = st.columns(2)

    with col1:
        section_heading("Class Distribution")
        fig = px.pie(
            class_distribution,
            names="Class",
            values="Count",
            color="Class",
            color_discrete_map={"Phishing": "#D58B8B", "Legitimate": "#65A78C"},
            hole=0.42,
        )
        fig.update_layout(
            paper_bgcolor="white", margin=dict(t=25, b=20), legend_title_text=""
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_heading("Top 15 TLDs")
        fig = px.bar(
            top_tlds.sort_values("URL_Count", ascending=True),
            x="URL_Count",
            y="TLD",
            orientation="h",
            color_discrete_sequence=["#739BC8"],
        )
        fig.update_layout(
            paper_bgcolor="white", plot_bgcolor="white", margin=dict(t=25, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section_heading("Phishing and Legitimate URLs Across TLDs")
    tld_long = top_tlds.melt(
        id_vars=["TLD"],
        value_vars=["Phishing_Count", "Legitimate_Count"],
        var_name="Class",
        value_name="URL Count",
    )
    tld_long["Class"] = tld_long["Class"].replace(
        {"Phishing_Count": "Phishing", "Legitimate_Count": "Legitimate"}
    )
    fig = px.bar(
        tld_long,
        x="TLD",
        y="URL Count",
        color="Class",
        barmode="group",
        color_discrete_map={"Phishing": "#D58B8B", "Legitimate": "#739BC8"},
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend_title_text="",
        margin=dict(t=25, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "These are historical dataset statistics, not live domain reputation "
        "checks or real-time threat intelligence."
    )

# ============================================================
# 12. MODEL EVALUATION
# ============================================================

elif page == "Model Evaluation":
    page_heading(
        "Model Evaluation",
        "Comparison of candidate models, threshold selection and held-out test performance.",
    )
    section_heading("Model Comparison")
    fig = px.bar(
        comparison,
        x="Model",
        y="Phishing F1",
        color="Model",
        color_discrete_sequence=["#739BC8", "#A7C7E7", "#C4D7E8"],
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        margin=dict(t=25, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    comparison_display = comparison.copy()
    format_map = {
        "Accuracy": "{:.2%}",
        "Phishing Precision": "{:.2%}",
        "Phishing Recall": "{:.2%}",
        "Phishing F1": "{:.2%}",
        "ROC-AUC": "{:.4f}",
        "Training Time (sec)": "{:.2f}",
    }
    available_formats = {
        key: value for key, value in format_map.items()
        if key in comparison_display.columns
    }
    st.dataframe(
        comparison_display.style.format(available_formats),
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    section_heading(
        "Classification Threshold Analysis",
        "The selected threshold was determined using validation data.",
    )
    fig = go.Figure()
    for column in ["Phishing Precision", "Phishing Recall", "Phishing F1"]:
        fig.add_trace(
            go.Scatter(
                x=threshold_results["Threshold"],
                y=threshold_results[column],
                mode="lines+markers",
                name=column,
            )
        )
    fig.add_vline(
        x=THRESHOLD,
        line_dash="dash",
        line_color="#527EBA",
        annotation_text="Selected threshold",
    )
    fig.update_layout(
        paper_bgcolor="white",
        plot_bgcolor="white",
        xaxis_title="Classification threshold",
        yaxis_title="Score",
        yaxis_range=[0, 1],
        margin=dict(t=40, b=25),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    section_heading(
        "Held-Out Test Confusion Matrix",
        "Rows represent actual classes; columns represent predicted classes.",
    )
    cm = np.array(metrics["test"]["confusion_matrix"])
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        x=["Predicted Phishing", "Predicted Legitimate"],
        y=["Actual Phishing", "Actual Legitimate"],
        color_continuous_scale="Blues",
        aspect="auto",
    )
    cm_fig.update_layout(paper_bgcolor="white", margin=dict(t=25, b=25))
    st.plotly_chart(cm_fig, use_container_width=True)
    st.caption("Class 0 = Phishing; Class 1 = Legitimate.")

# ============================================================
# 13. METHODOLOGY AND LIMITATIONS
# ============================================================

elif page == "Methodology & Limitations":
    page_heading(
        "Methodology and Limitations",
        "Research workflow, experimental decisions and scope of the application.",
    )
    section_heading("Research Pipeline")
    steps = [
        ("01", "Dataset", "PhiUSIIL Phishing URL Dataset."),
        ("02", "Data Cleaning", "Duplicate URL audit, label validation and preprocessing."),
        ("03", "Feature Extraction", "17 deterministic URL-derived structural features."),
        ("04", "Data Splitting", "Training, validation and test sets separated by registered domain."),
        ("05", "Model Training", "Logistic Regression, Random Forest and XGBoost."),
        ("06", "Model Selection", "XGBoost selected using validation performance."),
        ("07", "Threshold Optimization", "Classification threshold selected using validation data."),
        ("08", "Explainability", "Global and local SHAP analysis."),
        ("09", "Application", "Interactive Streamlit dashboard with URL scanning and visualizations."),
    ]
    for number, title, description in steps:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="section-label">{number}</div>
                <div class="info-card-title">{title}</div>
                <div class="info-card-text">{description}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    section_heading("Model Limitations")
    limitations = [
        "The scanner uses URL-derived structural features only.",
        "It does not fetch or inspect live webpage HTML.",
        "It does not perform live DNS, TLS certificate or domain reputation checks.",
        "A legitimate prediction does not guarantee that a website is safe.",
        "A phishing prediction is a model classification, not a verified security incident.",
        "Dataset-derived TLD and domain statistics are historical, not real-time intelligence.",
        "Independent external dataset evaluation is recommended before real-world use.",
    ]
    for limitation in limitations:
        st.markdown(f"- {limitation}")

    st.divider()
    section_heading("Dataset and Evaluation Notes")
    for note in metrics.get("notes", []):
        st.markdown(f"- {note}")

    st.info(
        "PhishGuard is an academic research prototype. It should not be used "
        "as the sole basis for security decisions."
    )

# ============================================================
# 14. FOOTER
# ============================================================

st.divider()
st.markdown(
    """
    <div class="footer-text">
        PhishGuard | Explainable Machine Learning for Phishing URL Detection
        <br>
        Academic Research Project
    </div>
    """,
    unsafe_allow_html=True,
)
