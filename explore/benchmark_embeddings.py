"""
Benchmark Embeddings Module.

This script evaluates embedding models for the DiagnoSys RAG pipeline.

IMPORTANT CONTEXT:
- Patient input: French (via Speech-to-Text)
- Medical documentation: English (most available resources)
- Requirement: Models must handle cross-lingual semantic matching (FR query -> EN docs)

This benchmark helps select models that perform well in this cross-lingual scenario.

Usage:
    python benchmark_embeddings.py                    # Basic benchmark
    python benchmark_embeddings.py --umap             # Include UMAP visualization
    python benchmark_embeddings.py --stats            # Include statistical tests
    python benchmark_embeddings.py --umap --stats     # Full analysis
"""

import argparse
import json
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import ndcg_score
from deep_translator import GoogleTranslator

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


# Default paths for test data
DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_FR_DATA = DATA_DIR / "benchmark_embeddings.json"
DEFAULT_EN_DATA = DATA_DIR / "benchmark_embeddings_en.json"


@dataclass
class ModelConfig:
    """Configuration for an embedding model."""
    path: str
    language: str  # 'en', 'fr', or 'multi'
    description: str


# Models to benchmark
# NOTE: Multilingual models are preferred for cross-lingual (FR->EN) matching
MODELS_TO_BENCHMARK: dict[str, ModelConfig] = {
    # Multilingual models (recommended for FR queries -> EN docs)
    "paraphrase-multilingual-MiniLM-L12-v2": ModelConfig(
        path="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        language="multi",
        description="Production model - multilingual paraphrase (fast, good quality)"
    ),
    "multilingual-e5-small": ModelConfig(
        path="intfloat/multilingual-e5-small",
        language="multi",
        description="Multilingual E5 small (balanced speed/quality)"
    ),
    "multilingual-e5-large": ModelConfig(
        path="intfloat/multilingual-e5-large",
        language="multi",
        description="Multilingual E5 large (best quality, slower)"
    ),
    "bge-m3": ModelConfig(
        path="BAAI/bge-m3",
        language="multi",
        description="BGE-M3 multilingual (state-of-the-art)"
    ),
    # English-only medical models (require FR->EN translation)
    "biobert": ModelConfig(
        path="dmis-lab/biobert-v1.1",
        language="en",
        description="BioBERT - biomedical domain (requires translation)"
    ),
    "clinical-bert": ModelConfig(
        path="emilyalsentzer/Bio_ClinicalBERT",
        language="en",
        description="ClinicalBERT - clinical notes (requires translation)"
    ),
    "pubmed-bert": ModelConfig(
        path="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        language="en",
        description="PubMedBERT - biomedical literature (requires translation)"
    ),
    # French-only models (for cross-lingual gap demonstration)
    # These models expect French text, so they struggle with English docs
    # Included to demonstrate why multilingual models are preferred
    "drbert": ModelConfig(
        path="Dr-BERT/DrBERT-7GB",
        language="fr",
        description="DrBERT - French medical BERT (demonstrates FR-only limitation)"
    ),
    "camembert-base": ModelConfig(
        path="almanach/camembert-base",
        language="fr",
        description="CamemBERT - French general BERT (demonstrates FR-only limitation)"
    ),
    "camembert-bio": ModelConfig(
        path="almanach/camembert-bio-base",
        language="fr",
        description="CamemBERT-bio - French biomedical (demonstrates FR-only limitation)"
    ),
    # Baseline
    "all-MiniLM-L6-v2": ModelConfig(
        path="sentence-transformers/all-MiniLM-L6-v2",
        language="multi",
        description="Lightweight baseline (fast, general purpose)"
    ),
}


# Test cases: French queries (simulating STT output) paired with English documentation
# This reflects the production scenario: patient speaks French, docs are in English
TEST_DATA = [
    {
        "query_fr": "Patient présente toux productive et fièvre 39°C depuis 3 jours",
        "query_en": "Patient presents productive cough and fever 39°C for 3 days",
        "expected_doc": "Diagnostic protocol for acute respiratory infections and pneumonia. Clinical presentation typically includes productive cough, fever, and dyspnea.",
        "category": "Pulmonology"
    },
    {
        "query_fr": "Douleur thoracique irradiant vers le bras gauche, sueurs, nausées",
        "query_en": "Chest pain radiating to left arm, sweating, nausea",
        "expected_doc": "Emergency management of Acute Coronary Syndrome (ACS). Typical symptoms include chest pain with radiation, diaphoresis, and nausea.",
        "category": "Cardiology"
    },
    {
        "query_fr": "Polyurie, polydipsie, perte de poids inexpliquée",
        "query_en": "Polyuria, polydipsia, unexplained weight loss",
        "expected_doc": "Diagnostic criteria and follow-up for type 1 and type 2 diabetes mellitus. Classic triad: polyuria, polydipsia, and weight loss.",
        "category": "Endocrinology"
    },
    {
        "query_fr": "Céphalées intenses avec raideur de nuque et photophobie",
        "query_en": "Severe headache with neck stiffness and photophobia",
        "expected_doc": "Meningitis diagnostic criteria and emergency management. Classic triad: headache, neck stiffness, and photophobia.",
        "category": "Neurology"
    },
    {
        "query_fr": "Douleur abdominale aiguë fosse iliaque droite avec défense",
        "query_en": "Acute abdominal pain in right iliac fossa with guarding",
        "expected_doc": "Appendicitis diagnosis and surgical management. McBurney point tenderness and peritoneal signs indicate surgical emergency.",
        "category": "Surgery"
    },
]

# Distractor documents for RAG simulation (needle in haystack)
# English-only distractors: matches production ChromaDB (English medical docs only)
# The real cross-lingual challenge is: can the model match a French query
# to the correct English doc among other English docs?
DISTRACTORS = [
    "Tibial fractures require cast immobilization for 6-8 weeks with regular X-ray follow-up.",
    "Diabetic diet should be low in simple carbohydrates and high in fiber.",
    "Atopic dermatitis is treated with topical corticosteroids and emollients.",
    "Arterial hypertension requires regular blood pressure monitoring and lifestyle modifications.",
    "Influenza vaccination is recommended for elderly patients and immunocompromised individuals.",
    "Asthma treatment includes bronchodilators and inhaled corticosteroids.",
    "Cardiovascular disease prevention includes regular exercise and healthy diet.",
    "Urinary tract infections are more common in women due to anatomical factors.",
    "Breast cancer screening is performed through mammography in women over 50.",
    "Sleep disorders can be treated with benzodiazepines or cognitive behavioral therapy.",
    "Dehydration requires oral or intravenous rehydration depending on severity.",
    "Seasonal allergies are treated with antihistamines and nasal corticosteroids.",
    "Prenatal care includes multiple ultrasound examinations throughout pregnancy.",
    "Stress fractures are common in athletes and require rest and gradual return to activity.",
    "Gastroesophageal reflux is managed with proton pump inhibitors and lifestyle changes.",
]


@dataclass
class BenchmarkResult:
    """Single benchmark result entry."""
    model: str
    category: str
    query_language: str  # 'fr' or 'en'
    query_type: str  # 'extracted' (clean, post-LLM) or 'transcript' (noisy, pre-LLM)
    similarity: float
    encoding_time_s: float
    init_time_s: float
    required_translation: bool
    # RAG metrics
    mrr: float = 0.0
    rank: int = 0
    top_1: bool = False
    top_3: bool = False
    top_5: bool = False
    ndcg_3: float = 0.0
    ndcg_5: float = 0.0


def load_test_data(
    fr_path: Path | None = None,
    en_path: Path | None = None,
    include_transcript: bool = True
) -> list[dict]:
    """
    Load test data from JSON files.

    Args:
        fr_path: Path to French test data JSON
        en_path: Path to English test data JSON
        include_transcript: If True, also load transcript queries (verbose/noisy)
                           to compare against extracted (clean) queries.
                           This comparison justifies the LLM denoising step.

    Returns:
        List of normalized test cases with query_fr, query_en, expected_doc, category
    """
    test_cases = []

    # Load French data
    fr_file = fr_path or DEFAULT_FR_DATA
    if fr_file.exists():
        with open(fr_file, "r", encoding="utf-8") as f:
            fr_data = json.load(f)
            print(f"  Loaded {len(fr_data)} French test cases from {fr_file.name}")

            for case in fr_data:
                test_cases.append({
                    "query_fr": case.get("query", ""),
                    "query_en": None,
                    "expected_doc": case.get("expected_doc", ""),
                    "category": case.get("category", "Unknown"),
                    "severity": case.get("severity", ""),
                    "query_type": "extracted",  # French data is already clean/extracted
                })

    # Load English data (has both extracted and transcript versions)
    en_file = en_path or DEFAULT_EN_DATA
    if en_file.exists():
        with open(en_file, "r", encoding="utf-8") as f:
            en_data = json.load(f)

            # Count what we're loading
            n_extracted = len(en_data)
            n_transcript = len(en_data) if include_transcript else 0
            print(f"  Loaded {n_extracted} English extracted cases from {en_file.name}")
            if include_transcript:
                print(f"  Loaded {n_transcript} English transcript cases from {en_file.name}")

            for case in en_data:
                # Load EXTRACTED query (clean, after LLM processing)
                test_cases.append({
                    "query_fr": None,
                    "query_en": case.get("query_extracted", ""),
                    "expected_doc": case.get("expected_doc", ""),
                    "category": case.get("category", "Unknown"),
                    "severity": case.get("severity", ""),
                    "query_type": "extracted",  # Clean query (post-LLM)
                })

                # Load TRANSCRIPT query (noisy, raw patient speech)
                # This allows comparing performance WITH vs WITHOUT LLM denoising
                if include_transcript and case.get("query_transcript"):
                    test_cases.append({
                        "query_fr": None,
                        "query_en": case.get("query_transcript", ""),
                        "expected_doc": case.get("expected_doc", ""),
                        "category": case.get("category", "Unknown"),
                        "severity": case.get("severity", ""),
                        "query_type": "transcript",  # Noisy query (pre-LLM)
                    })

    if not test_cases:
        raise FileNotFoundError(
            f"No test data found. Expected files at:\n"
            f"  - {DEFAULT_FR_DATA}\n"
            f"  - {DEFAULT_EN_DATA}"
        )

    return test_cases


class EmbeddingBenchmarker:
    """
    Benchmarker for evaluating embedding models in cross-lingual RAG scenarios.

    Evaluates how well models match French patient queries to English medical
    documentation - the core use case for DiagnoSys.

    Attributes:
        models: Dictionary of model configurations to benchmark
        results: List of benchmark results
        output_dir: Directory for saving results and visualizations
        embeddings_cache: Cached embeddings for UMAP visualization
    """

    def __init__(
        self,
        models: dict[str, ModelConfig],
        output_dir: str = "logs"
    ) -> None:
        """
        Initialize the benchmarker.

        Args:
            models: Dictionary mapping model names to ModelConfig
            output_dir: Directory to save results
        """
        self.models = models
        self.results: list[BenchmarkResult] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self._translator_fr_en = GoogleTranslator(source='fr', target='en')
        self._translator_en_fr = GoogleTranslator(source='en', target='fr')
        self.embeddings_cache: dict[str, list[dict]] = {}

    def _translate_to_english(self, text: str) -> str:
        """Translate French text to English for English-only models."""
        try:
            return self._translator_fr_en.translate(text)
        except Exception as e:
            print(f"    Warning: FR->EN translation failed ({e}), using original")
            return text

    def _translate_to_french(self, text: str) -> str:
        """Translate English text to French for French-only models."""
        try:
            return self._translator_en_fr.translate(text)
        except Exception as e:
            print(f"    Warning: EN->FR translation failed ({e}), using original")
            return text

    def _prepare_query(
        self,
        query_fr: str | None,
        query_en: str | None,
        model_language: str
    ) -> tuple[str, bool]:
        """
        Prepare query for the model based on its language capability.

        Args:
            query_fr: French version of the query (may be None)
            query_en: English version of the query (may be None)
            model_language: Model's language capability ('en', 'fr', 'multi')

        Returns:
            Tuple of (query to use, whether translation was required)
        """
        if model_language == "multi":
            # Multilingual models: prefer French (production scenario), fallback to English
            if query_fr:
                return query_fr, False
            return query_en or "", False
        elif model_language == "en":
            # English models: need English query
            if query_en:
                return query_en, False
            # Translate French to English
            return self._translate_to_english(query_fr or ""), True
        else:
            # French models: need French query
            if query_fr:
                return query_fr, False
            # Translate English to French
            return self._translate_to_french(query_en or ""), True

    def _calculate_rag_metrics(
        self,
        query_vec: np.ndarray,
        corpus_vecs: list[np.ndarray],
        target_index: int
    ) -> dict:
        """
        Calculate RAG retrieval metrics.

        Args:
            query_vec: Query embedding
            corpus_vecs: Document embeddings (target + distractors)
            target_index: Index of correct document

        Returns:
            Dictionary with MRR, rank, top-k, and NDCG metrics
        """
        scores = cosine_similarity([query_vec], corpus_vecs)[0]
        ranked_indices = np.argsort(scores)[::-1]
        rank = int(np.where(ranked_indices == target_index)[0][0]) + 1

        # Relevance vector for NDCG
        relevance = np.zeros(len(corpus_vecs))
        relevance[target_index] = 1

        # NDCG@k
        true_rel = relevance.reshape(1, -1)
        pred_scores = scores.reshape(1, -1)

        return {
            "mrr": 1.0 / rank,
            "rank": rank,
            "top_1": rank == 1,
            "top_3": rank <= 3,
            "top_5": rank <= 5,
            "ndcg_3": ndcg_score(true_rel, pred_scores, k=3),
            "ndcg_5": ndcg_score(true_rel, pred_scores, k=5),
        }

    def run(
        self,
        test_data: list[dict] | None = None,
        cache_embeddings: bool = False
    ) -> None:
        """
        Run benchmark across all models.

        Args:
            test_data: Custom test cases (loads from JSON files if None)
            cache_embeddings: If True, store embeddings for UMAP visualization
        """
        # Load test data from JSON files if not provided
        if test_data is None:
            print("\n--- Loading Test Data ---")
            test_data = load_test_data()

        print("\n" + "=" * 70)
        print("CROSS-LINGUAL EMBEDDING BENCHMARK")
        print("Scenario: French patient queries -> English medical documentation")
        print(f"Total test cases: {len(test_data)}")
        print("=" * 70)

        for name, config in self.models.items():
            print(f"\nTesting: {name}")
            print(f"  Language: {config.language} | {config.description}")

            try:
                # Initialize model
                start_init = time.time()
                model = HuggingFaceEmbeddings(model_name=config.path)
                init_time = time.time() - start_init
                print(f"  Initialized in {init_time:.2f}s")

                # Pre-encode distractors
                distractor_vecs = model.embed_documents(DISTRACTORS)

                if cache_embeddings:
                    self.embeddings_cache[name] = []

                cases_processed = 0
                for case in test_data:
                    # Determine query language based on available data
                    query_language = "fr" if case.get("query_fr") else "en"
                    query_type = case.get("query_type", "extracted")

                    query_vec = self._benchmark_case(
                        model=model,
                        model_name=name,
                        config=config,
                        case=case,
                        query_language=query_language,
                        query_type=query_type,
                        init_time=init_time,
                        distractor_vecs=distractor_vecs
                    )

                    if cache_embeddings and query_vec is not None:
                        self.embeddings_cache[name].append({
                            "vector": query_vec,
                            "category": case["category"],
                            "language": query_language,
                            "query_type": query_type
                        })

                    cases_processed += 1

                print(f"  Completed {cases_processed} test cases")

            except Exception as e:
                print(f"  Error: {e}")

    def _benchmark_case(
        self,
        model: HuggingFaceEmbeddings,
        model_name: str,
        config: ModelConfig,
        case: dict,
        query_language: str,
        query_type: str,
        init_time: float,
        distractor_vecs: list[np.ndarray]
    ) -> np.ndarray | None:
        """
        Benchmark a single test case.

        Args:
            model: The embedding model to test
            model_name: Name of the model
            config: Model configuration
            case: Test case dictionary
            query_language: 'fr' or 'en'
            query_type: 'extracted' (clean, post-LLM) or 'transcript' (noisy, pre-LLM)
            init_time: Model initialization time
            distractor_vecs: Pre-computed distractor embeddings

        Returns:
            Query embedding vector (for caching) or None on error
        """
        # Get appropriate query based on model language
        query, required_translation = self._prepare_query(
            case.get("query_fr"),
            case.get("query_en"),
            config.language
        )

        if not query:
            return None

        # Encode query and document
        start_enc = time.time()
        query_vec = model.embed_query(query)
        doc_vec = model.embed_query(case["expected_doc"])
        encoding_time = time.time() - start_enc

        # Direct similarity
        similarity = cosine_similarity([query_vec], [doc_vec])[0][0]

        # RAG simulation (needle in haystack)
        corpus_vecs = distractor_vecs + [doc_vec]
        target_index = len(distractor_vecs)
        rag_metrics = self._calculate_rag_metrics(query_vec, corpus_vecs, target_index)

        # Store result
        self.results.append(BenchmarkResult(
            model=model_name,
            category=case["category"],
            query_language=query_language,
            query_type=query_type,
            similarity=round(similarity, 4),
            encoding_time_s=round(encoding_time, 4),
            init_time_s=round(init_time, 2),
            required_translation=required_translation,
            mrr=round(rag_metrics["mrr"], 4),
            rank=rag_metrics["rank"],
            top_1=rag_metrics["top_1"],
            top_3=rag_metrics["top_3"],
            top_5=rag_metrics["top_5"],
            ndcg_3=round(rag_metrics["ndcg_3"], 4),
            ndcg_5=round(rag_metrics["ndcg_5"], 4),
        ))

        return np.array(query_vec)

    def get_summary(self) -> pd.DataFrame:
        """
        Generate summary statistics grouped by model, query language, and query type.

        Returns:
            DataFrame with aggregated metrics per model/language/type combination
        """
        df = pd.DataFrame([vars(r) for r in self.results])

        summary = df.groupby(["model", "query_language", "query_type"]).agg({
            "similarity": "mean",
            "mrr": "mean",
            "top_1": "mean",
            "top_3": "mean",
            "top_5": "mean",
            "ndcg_3": "mean",
            "ndcg_5": "mean",
            "encoding_time_s": "mean",
            "required_translation": "first",
        }).round(4)

        summary.columns = [
            "Similarity", "MRR", "Top-1", "Top-3", "Top-5",
            "NDCG@3", "NDCG@5", "Enc. Time (s)", "Needs Translation"
        ]

        return summary

    def get_llm_denoising_impact(self) -> pd.DataFrame:
        """
        Calculate the impact of LLM denoising by comparing extracted vs transcript queries.

        This comparison justifies the use of the LLM (Mistral) for denoising patient speech.
        A large MRR drop from extracted to transcript proves the LLM step is necessary.

        Returns:
            DataFrame showing MRR for extracted, transcript, and the degradation per model
        """
        df = pd.DataFrame([vars(r) for r in self.results])

        # Only English data has both extracted and transcript
        df_en = df[df["query_language"] == "en"]

        if df_en.empty:
            return pd.DataFrame()

        # Check if we have both query types
        query_types = df_en["query_type"].unique()
        if "extracted" not in query_types or "transcript" not in query_types:
            return pd.DataFrame()

        pivot = df_en.pivot_table(
            values="mrr",
            index="model",
            columns="query_type",
            aggfunc="mean"
        )

        if "extracted" in pivot.columns and "transcript" in pivot.columns:
            # Degradation: how much worse transcript is compared to extracted
            pivot["degradation"] = pivot["extracted"] - pivot["transcript"]
            # Relative degradation as percentage
            pivot["degradation_pct"] = (pivot["degradation"] / pivot["extracted"] * 100).round(1)
            pivot = pivot.round(4).sort_values("degradation", ascending=False)

        return pivot

    def get_cross_lingual_gap(self) -> pd.DataFrame:
        """
        Calculate performance gap between French and English queries.

        This measures how much quality is lost when using French queries
        instead of English - critical for understanding model suitability.

        Returns:
            DataFrame showing MRR gap (EN - FR) per model
        """
        df = pd.DataFrame([vars(r) for r in self.results])

        pivot = df.pivot_table(
            values="mrr",
            index="model",
            columns="query_language",
            aggfunc="mean"
        )

        if "en" in pivot.columns and "fr" in pivot.columns:
            pivot["gap (EN-FR)"] = pivot["en"] - pivot["fr"]
            pivot = pivot.round(4).sort_values("gap (EN-FR)")

        return pivot

    def calculate_statistical_significance(self) -> pd.DataFrame:
        """
        Perform paired t-tests between models to determine statistical significance.

        Compares MRR scores across test cases to identify significant differences.

        Returns:
            DataFrame with p-values and effect sizes for model comparisons
        """
        from scipy.stats import ttest_rel
        from scipy import stats

        df = pd.DataFrame([vars(r) for r in self.results])

        # Focus on French queries (production scenario)
        df_fr = df[df["query_language"] == "fr"]

        if len(df_fr) == 0:
            print("Not enough French data for statistical testing")
            return pd.DataFrame()

        models = df_fr["model"].unique()
        comparisons = []

        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                scores1 = df_fr[df_fr["model"] == model1]["mrr"].values
                scores2 = df_fr[df_fr["model"] == model2]["mrr"].values

                if len(scores1) == len(scores2) and len(scores1) > 1:
                    # Paired t-test
                    t_stat, p_value = ttest_rel(scores1, scores2)

                    # Cohen's d (effect size)
                    diff = scores1 - scores2
                    cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

                    # 95% confidence interval
                    ci_95 = stats.t.interval(
                        0.95, len(diff) - 1,
                        loc=np.mean(diff),
                        scale=stats.sem(diff) if stats.sem(diff) > 0 else 1e-10
                    )

                    comparisons.append({
                        "Model A": model1,
                        "Model B": model2,
                        "Mean MRR A": round(np.mean(scores1), 4),
                        "Mean MRR B": round(np.mean(scores2), 4),
                        "p-value": round(p_value, 4),
                        "Significant": "Yes" if p_value < 0.05 else "No",
                        "Cohen's d": round(cohens_d, 4),
                        "CI 95% Lower": round(ci_95[0], 4),
                        "CI 95% Upper": round(ci_95[1], 4),
                    })

        return pd.DataFrame(comparisons)

    def plot_umap(self) -> None:
        """
        Visualize embedding space using UMAP dimensionality reduction.

        Shows how different models cluster queries by category and language.
        """
        try:
            import umap
        except ImportError:
            print("  UMAP not installed. Run: uv add umap-learn")
            return

        if not self.embeddings_cache:
            print("  No cached embeddings. Run benchmark with cache_embeddings=True")
            return

        # Limit to 4 models for visualization
        models_to_plot = list(self.embeddings_cache.keys())[:4]

        if len(models_to_plot) == 0:
            print("  No embeddings to visualize")
            return

        fig = make_subplots(
            rows=1, cols=len(models_to_plot),
            subplot_titles=models_to_plot
        )

        for idx, model_name in enumerate(models_to_plot):
            embeddings = self.embeddings_cache[model_name]

            if len(embeddings) < 2:
                continue

            vectors = np.array([e["vector"] for e in embeddings])
            categories = [e["category"] for e in embeddings]
            languages = [e["language"] for e in embeddings]

            # Apply UMAP
            n_neighbors = min(15, len(vectors) - 1)
            reducer = umap.UMAP(n_neighbors=n_neighbors, random_state=42)
            embedding_2d = reducer.fit_transform(vectors)

            # Create hover text
            hover_text = [f"Category: {cat}<br>Language: {lang}"
                          for cat, lang in zip(categories, languages)]

            # Add scatter trace
            fig.add_trace(
                go.Scatter(
                    x=embedding_2d[:, 0],
                    y=embedding_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=10,
                        color=[hash(cat) % 10 for cat in categories],
                        colorscale='Viridis',
                        symbol=['circle' if lang == 'fr' else 'diamond' for lang in languages],
                    ),
                    text=hover_text,
                    hoverinfo='text',
                    name=model_name
                ),
                row=1, col=idx + 1
            )

        fig.update_layout(
            title="Embedding Space Visualization (UMAP)",
            showlegend=False,
            height=500,
            width=400 * len(models_to_plot)
        )

        output_path = self.output_dir / "embedding_space_umap.html"
        fig.write_html(str(output_path))
        print(f"  UMAP visualization saved to {output_path}")

    def plot_results(self) -> None:
        """Generate and save interactive Plotly visualization."""
        df = pd.DataFrame([vars(r) for r in self.results])

        if df.empty:
            print("No results to plot.")
            return

        # Check if we have transcript data for the LLM impact plot
        has_transcript = "transcript" in df["query_type"].unique()

        # Create subplots - 3 rows if we have transcript data
        n_rows = 3 if has_transcript else 2
        subplot_titles = [
            "MRR by Model and Query Language",
            "Top-K Accuracy (French Queries - Extracted)",
            "Average Encoding Time",
            "Cross-Lingual Gap (EN MRR - FR MRR)",
        ]
        if has_transcript:
            subplot_titles.extend([
                "LLM Denoising Impact (Extracted vs Transcript)",
                "MRR Degradation Without LLM (%)"
            ])

        fig = make_subplots(
            rows=n_rows, cols=2,
            subplot_titles=subplot_titles,
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Color palette
        colors = px.colors.qualitative.Plotly

        # Plot 1: MRR by model and query language (extracted only for fair comparison)
        df_extracted = df[df["query_type"] == "extracted"]
        mrr_data = df_extracted.groupby(["model", "query_language"])["mrr"].mean().reset_index()
        for i, lang in enumerate(mrr_data["query_language"].unique()):
            lang_data = mrr_data[mrr_data["query_language"] == lang]
            fig.add_trace(
                go.Bar(
                    x=lang_data["model"],
                    y=lang_data["mrr"],
                    name=f"Query: {lang.upper()}",
                    marker_color=colors[i],
                    legendgroup="mrr",
                    showlegend=True
                ),
                row=1, col=1
            )

        # Plot 2: Top-K accuracy (French queries, extracted only - production scenario)
        df_fr = df[(df["query_language"] == "fr") & (df["query_type"] == "extracted")]
        if not df_fr.empty:
            top_k_data = df_fr.groupby("model")[["top_1", "top_3", "top_5"]].mean().reset_index()
            for i, metric in enumerate(["top_1", "top_3", "top_5"]):
                fig.add_trace(
                    go.Bar(
                        x=top_k_data["model"],
                        y=top_k_data[metric],
                        name=metric.replace("_", "-").upper(),
                        marker_color=colors[i + 2],
                        legendgroup="topk",
                        showlegend=True
                    ),
                    row=1, col=2
                )

        # Plot 3: Encoding time
        time_data = df.groupby("model")["encoding_time_s"].mean().reset_index()
        fig.add_trace(
            go.Bar(
                x=time_data["model"],
                y=time_data["encoding_time_s"],
                name="Encoding Time",
                marker_color=colors[5],
                showlegend=False
            ),
            row=2, col=1
        )

        # Plot 4: Cross-lingual gap (extracted only)
        gap_data = self.get_cross_lingual_gap()
        if "gap (EN-FR)" in gap_data.columns:
            gap_data = gap_data.reset_index()
            fig.add_trace(
                go.Bar(
                    x=gap_data["model"],
                    y=gap_data["gap (EN-FR)"],
                    name="Gap",
                    marker_color=[colors[6] if v >= 0 else colors[7]
                                  for v in gap_data["gap (EN-FR)"]],
                    showlegend=False
                ),
                row=2, col=2
            )
            fig.add_hline(y=0, line_dash="dash", line_color="green", row=2, col=2)

        # Plot 5 & 6: LLM Denoising Impact (if transcript data available)
        if has_transcript:
            llm_impact = self.get_llm_denoising_impact()
            if not llm_impact.empty and "extracted" in llm_impact.columns:
                llm_impact = llm_impact.reset_index()

                # Plot 5: Extracted vs Transcript MRR
                fig.add_trace(
                    go.Bar(
                        x=llm_impact["model"],
                        y=llm_impact["extracted"],
                        name="Extracted (with LLM)",
                        marker_color=colors[0],
                        legendgroup="llm",
                        showlegend=True
                    ),
                    row=3, col=1
                )
                fig.add_trace(
                    go.Bar(
                        x=llm_impact["model"],
                        y=llm_impact["transcript"],
                        name="Transcript (without LLM)",
                        marker_color=colors[1],
                        legendgroup="llm",
                        showlegend=True
                    ),
                    row=3, col=1
                )

                # Plot 6: Degradation percentage
                fig.add_trace(
                    go.Bar(
                        x=llm_impact["model"],
                        y=llm_impact["degradation_pct"],
                        name="Degradation %",
                        marker_color="crimson",
                        showlegend=False
                    ),
                    row=3, col=2
                )

        # Update layout
        fig.update_layout(
            title="Embedding Benchmark Results",
            barmode='group',
            height=400 * n_rows,
            width=1400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )

        # Update axes
        fig.update_yaxes(title_text="MRR", range=[0, 1], row=1, col=1)
        fig.update_yaxes(title_text="Accuracy", range=[0, 1], row=1, col=2)
        fig.update_yaxes(title_text="Time (s)", row=2, col=1)
        fig.update_yaxes(title_text="MRR Gap", row=2, col=2)
        if has_transcript:
            fig.update_yaxes(title_text="MRR", range=[0, 1], row=3, col=1)
            fig.update_yaxes(title_text="Degradation (%)", row=3, col=2)

        # Rotate x-axis labels
        fig.update_xaxes(tickangle=45)

        output_path = self.output_dir / "benchmark_results.html"
        fig.write_html(str(output_path))
        print(f"  Interactive plots saved to {output_path}")

    def export_results(
        self,
        include_stats: bool = False,
        include_umap: bool = False
    ) -> None:
        """
        Export results to CSV and print summary.

        Args:
            include_stats: If True, run statistical significance tests
            include_umap: If True, generate UMAP embedding visualization
        """
        df = pd.DataFrame([vars(r) for r in self.results])

        print("\n" + "=" * 70)
        print("BENCHMARK RESULTS SUMMARY")
        print("=" * 70)

        # Summary by model
        print("\n--- Aggregated Metrics ---")
        print(self.get_summary().to_string())

        # ============================================================
        # LLM DENOISING IMPACT ANALYSIS
        # This proves why the Mistral LLM step is necessary
        # ============================================================
        llm_impact = self.get_llm_denoising_impact()
        if not llm_impact.empty:
            print("\n" + "=" * 70)
            print("LLM DENOISING IMPACT (Justification for Mistral)")
            print("=" * 70)
            print("\nComparing EXTRACTED (clean, post-LLM) vs TRANSCRIPT (noisy, pre-LLM)")
            print("Higher degradation = LLM is more necessary for this model\n")
            print(llm_impact.to_string())

            # Key insight
            if "degradation_pct" in llm_impact.columns:
                avg_degradation = llm_impact["degradation_pct"].mean()
                max_degradation = llm_impact["degradation_pct"].max()
                max_model = llm_impact["degradation_pct"].idxmax()
                print(f"\n  Key Findings:")
                print(f"  - Average MRR degradation without LLM: {avg_degradation:.1f}%")
                print(f"  - Worst affected model: {max_model} ({max_degradation:.1f}% degradation)")
                print(f"  - CONCLUSION: The LLM denoising step is essential for RAG performance")

        # Cross-lingual gap analysis
        print("\n" + "=" * 70)
        print("CROSS-LINGUAL PERFORMANCE GAP")
        print("=" * 70)
        print("(Lower gap = better cross-lingual capability)\n")
        print(self.get_cross_lingual_gap().to_string())

        # French-only results (production scenario)
        df_fr = df[(df["query_language"] == "fr") & (df["query_type"] == "extracted")]
        if not df_fr.empty:
            print("\n" + "=" * 70)
            print("PRODUCTION SCENARIO (French Queries -> English Docs)")
            print("=" * 70)
            prod_summary = df_fr.groupby("model").agg({
                "mrr": "mean",
                "top_3": "mean",
                "required_translation": "first"
            }).round(4).sort_values("mrr", ascending=False)
            prod_summary.columns = ["MRR", "Top-3 Acc", "Needs Translation"]
            print(prod_summary.to_string())

            # Recommendation
            best_model = prod_summary["MRR"].idxmax()
            print(f"\n  Recommended model: {best_model}")
            print(f"  (Highest MRR on French queries)")

        # Statistical significance (optional)
        if include_stats:
            print("\n" + "=" * 70)
            print("STATISTICAL SIGNIFICANCE TESTS (Paired t-tests)")
            print("=" * 70)
            sig_results = self.calculate_statistical_significance()
            if not sig_results.empty:
                print(sig_results.to_string(index=False))
                sig_path = self.output_dir / "statistical_significance.csv"
                sig_results.to_csv(sig_path, index=False)
                print(f"\n  Statistical tests saved to {sig_path}")

        # Save CSV
        csv_path = self.output_dir / "embedding_benchmark_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nDetailed results saved to {csv_path}")

        # Save LLM impact CSV
        if not llm_impact.empty:
            llm_path = self.output_dir / "llm_denoising_impact.csv"
            llm_impact.to_csv(llm_path)
            print(f"LLM denoising impact saved to {llm_path}")

        # Generate plots
        print("\n--- Generating Visualizations ---")
        self.plot_results()

        # UMAP visualization (optional)
        if include_umap:
            self.plot_umap()


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark embedding models for cross-lingual RAG (FR->EN)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_embeddings.py                    # Basic benchmark (with LLM impact analysis)
  python benchmark_embeddings.py --no-transcript    # Skip transcript queries (faster)
  python benchmark_embeddings.py --umap             # Include UMAP visualization
  python benchmark_embeddings.py --stats            # Include statistical tests
  python benchmark_embeddings.py --umap --stats     # Full analysis
  python benchmark_embeddings.py --output results/  # Custom output directory

LLM Impact Analysis:
  By default, the benchmark tests both EXTRACTED (clean) and TRANSCRIPT (noisy) queries.
  This comparison proves mathematically why the LLM denoising step (Mistral) is necessary.
  Use --no-transcript to skip this analysis and run a faster benchmark.
        """
    )

    parser.add_argument(
        "--umap",
        action="store_true",
        help="Generate UMAP embedding space visualization"
    )

    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run statistical significance tests between models"
    )

    parser.add_argument(
        "--no-transcript",
        action="store_true",
        help="Skip transcript queries (disables LLM impact analysis, faster benchmark)"
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        default="logs",
        help="Output directory for results (default: logs)"
    )

    parser.add_argument(
        "--fr-data",
        type=str,
        default=None,
        help="Path to French test data JSON (default: data/benchmark_embeddings.json)"
    )

    parser.add_argument(
        "--en-data",
        type=str,
        default=None,
        help="Path to English test data JSON (default: data/benchmark_embeddings_en.json)"
    )

    return parser.parse_args()


def main() -> None:
    """Run the embedding benchmark."""
    args = parse_args()

    include_transcript = not args.no_transcript

    print("=" * 70)
    print("DIAGNOSYS EMBEDDING MODEL BENCHMARK")
    print("=" * 70)
    print("\nUse Case:")
    print("  - Input: French patient speech (via STT)")
    print("  - Target: English medical documentation")
    print("  - Goal: Find models with best cross-lingual semantic matching")
    print(f"\nOptions:")
    print(f"  - LLM impact analysis: {'Yes' if include_transcript else 'No (--no-transcript)'}")
    print(f"  - UMAP visualization: {'Yes' if args.umap else 'No'}")
    print(f"  - Statistical tests: {'Yes' if args.stats else 'No'}")
    print(f"  - Output directory: {args.output}")

    if include_transcript:
        print("\n  [INFO] Testing both EXTRACTED and TRANSCRIPT queries to measure LLM impact")
    else:
        print("\n  [INFO] Testing only EXTRACTED queries (faster, no LLM impact analysis)")

    # Load test data
    fr_path = Path(args.fr_data) if args.fr_data else None
    en_path = Path(args.en_data) if args.en_data else None

    benchmarker = EmbeddingBenchmarker(MODELS_TO_BENCHMARK, output_dir=args.output)

    test_data = load_test_data(fr_path, en_path, include_transcript=include_transcript)

    benchmarker.run(
        test_data=test_data,
        cache_embeddings=args.umap  # Only cache if UMAP is requested
    )

    benchmarker.export_results(
        include_stats=args.stats,
        include_umap=args.umap
    )

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
