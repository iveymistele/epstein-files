# Epstein Files Network Influence Pipeline


**Problem statement:** Identify which individuals are most structurally influential in Epstein file records by modeling name co-occurrence as a network and evaluating which centrality patterns remain stable across resampled document sets.

This notebook is the analysis and visualization pipeline for the project. It starts by querying the MongoDB document database into a dataframe, prepares the extracted names for analysis, builds a co-occurrence network, computes centrality metrics, uses bootstrap resampling to test the stability of the rankings, and produces a final visualization.

The goal is not to prove real-world influence. The goal is to identify structural influence within this document collection. In this project, an individual is treated as structurally influential if they repeatedly appear in records with many other named individuals and remain highly ranked when the document set is resampled.


## Analysis Rationale

The refined problem concerns the relationships between people in documents, so I implemented a network model. Each person is represented as a node. If two people are mentioned in the same document, an edge is placed between them. The edge weight counts how many documents contain that pair.

This approach fits the document model because each MongoDB document represents one email or file record with extracted metadata fields, including people mentioned. Instead of analyzing each document independently, the pipeline uses the document collection to create a secondary structure: a person-to-person co-occurrence network.

I use weighted degree as the main influence measure for ease of interpretation. An individual has a higher weighted degree when they co-occur frequently with many other people. I also compute degree centrality and PageRank as additional model-based network metrics that help to compare frequency-based influence with broader graph position.

The concept I took from previous ML classes is bootstrapping. I use bootstrapping to resample documents with replacement, rebuild the network many times, and test whether the same people remain central across slightly different versions of the dataset. This helps quantify uncertainty in the ranking rather than relying on one single network result.


## 1. Imports and Setup

This section imports the libraries used for database access, data preparation, network modeling, bootstrapping, and visualization. Logging is included so database and pipeline errors are recorded instead of silently failing.



```python
import os
import itertools
import logging
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.errors import PyMongoError

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

load_dotenv()

```




    True



## 2. Query MongoDB into a DataFrame

The first required pipeline step is to query MongoDB and place the document records into a dataframe. The database credentials are read from environment variables so that passwords are not stored in the notebook or GitHub repository.



```python
def load_email_documents():
    """Load Epstein email/file records from MongoDB into a pandas DataFrame."""
    mongo_user = os.getenv("MONGO_USER")
    mongo_pass = os.getenv("MONGO_PASS")
    mongo_cluster = os.getenv("MONGO_CLUSTER")

    if not all([mongo_user, mongo_pass, mongo_cluster]):
        raise ValueError(
            "Missing MongoDB credentials. Set MONGO_USER, MONGO_PASS, and MONGO_CLUSTER in a .env file."
        )

    mongo_uri = (
        f"mongodb+srv://{mongo_user}:{mongo_pass}@{mongo_cluster}/"
        "?retryWrites=true&w=majority"
    )

    try:
        client = MongoClient(mongo_uri)
        collection = client["epstein_db"]["emails"]
        records = list(collection.find({}))
        logging.info("Loaded %s records from MongoDB.", len(records))
        return pd.DataFrame(records)
    except PyMongoError as err:
        logging.exception("MongoDB query failed.")
        raise err

raw_df = load_email_documents()
raw_df.shape

```




    (2322, 35)



## 3. Data Preparation

Only fields relevant to the problem are kept. The most important field for the network model is `people_mentioned`, because it provides the extracted names used to create co-occurrence edges. Text fields are retained for context and for the supporting text model.



```python
relevant_columns = [
    "_id", "document_id", "email_text", "subject", "date",
    "people_mentioned", "participant_names", "organizations", "locations",
    "primary_topic", "tone", "evidence_strength",
    "participant_count", "attachment_count", "url_count",
]

available_columns = [col for col in relevant_columns if col in raw_df.columns]
df = raw_df[available_columns].copy()

for col in ["email_text", "subject", "primary_topic", "tone", "evidence_strength"]:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

for col in ["participant_count", "attachment_count", "url_count"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

if "email_text" in df.columns:
    df["email_length"] = df["email_text"].str.len()
    df["word_count"] = df["email_text"].str.split().str.len()

# Remove records without usable text for analysis.
df = df[df.get("email_text", "") != ""].copy()

df.shape

```




    (2322, 17)



### Name Cleaning Rationale

Names are cleaned before the network is built. I remove blank values, one-word strings, duplicates within a document, and generic labels that are not useful as people nodes. Duplicates within the same document are removed because the network represents whether two people co-occurred in a document, not how many times a name was repeated inside that same document.



```python
GENERIC_NAME_VALUES = {
    "unknown", "none", "nan", "redacted", "n/a", "email", "image", "attachment",
}

def clean_name_list(value):
    """Return a sorted list of unique, usable person names from one document field."""
    if not isinstance(value, list):
        return []

    cleaned = []
    for name in value:
        if not isinstance(name, str):
            continue
        name = " ".join(name.strip().split())
        if not name or name.lower() in GENERIC_NAME_VALUES:
            continue
        if len(name.split()) < 2:
            continue
        cleaned.append(name)

    return sorted(set(cleaned))

df["people_clean"] = df["people_mentioned"].apply(clean_name_list)
network_df = df[df["people_clean"].str.len() >= 2].copy()
docs = network_df["people_clean"].tolist()

print("Documents in dataframe:", len(df))
print("Documents usable for network:", len(docs))
print("Unique cleaned people:", len(set(itertools.chain.from_iterable(docs))))

```

    Documents in dataframe: 2322
    Documents usable for network: 2116
    Unique cleaned people: 3931


## 4. Supporting Text Model

This short modeling step uses TF-IDF and KMeans to group records by text similarity. This is not the main solution to the influence problem. It is included as a supporting model to give context for the document collection and to satisfy the project requirement that the pipeline implement a model.

The main analytical model is the co-occurrence network, but the text model helps check whether the records contain different kinds of document contexts. This matters because centrality may be affected by what kinds of records dominate the dataset.



```python
if "email_text" in network_df.columns and len(network_df) >= 3:
    vectorizer = TfidfVectorizer(stop_words="english", max_features=500, min_df=2)
    X_text = vectorizer.fit_transform(network_df["email_text"])

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    network_df["text_cluster"] = kmeans.fit_predict(X_text)

    cluster_summary = (
        network_df.groupby("text_cluster")
        .agg(
            document_count=("email_text", "count"),
            avg_word_count=("word_count", "mean"),
            avg_people_per_doc=("people_clean", lambda x: np.mean([len(v) for v in x]))
        )
        .round(2)
    )

    display(cluster_summary)
else:
    print("Not enough text records for KMeans clustering.")

```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>document_count</th>
      <th>avg_word_count</th>
      <th>avg_people_per_doc</th>
    </tr>
    <tr>
      <th>text_cluster</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>107</td>
      <td>364.83</td>
      <td>3.67</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1027</td>
      <td>1165.67</td>
      <td>9.96</td>
    </tr>
    <tr>
      <th>2</th>
      <td>982</td>
      <td>461.83</td>
      <td>3.92</td>
    </tr>
  </tbody>
</table>
</div>


## 5. Build the Name Co-occurrence Network

The co-occurrence graph is the main model used to solve the problem. Each cleaned person name becomes a node. Two names receive an edge when they appear in the same document. Repeated co-occurrence across documents increases the edge weight.



```python
def build_cooccurrence_graph(doc_name_lists, min_edge_weight=1):
    """Build a weighted undirected graph from lists of people mentioned per document."""
    edge_counter = Counter()
    node_counter = Counter()

    for names in doc_name_lists:
        unique_names = sorted(set(names))
        for name in unique_names:
            node_counter[name] += 1
        for person_a, person_b in itertools.combinations(unique_names, 2):
            edge_counter[(person_a, person_b)] += 1

    graph = nx.Graph()
    for name, count in node_counter.items():
        graph.add_node(name, document_frequency=count)
    for (person_a, person_b), weight in edge_counter.items():
        if weight >= min_edge_weight:
            graph.add_edge(person_a, person_b, weight=weight)
    return graph

G = build_cooccurrence_graph(docs)
print("Network nodes:", G.number_of_nodes())
print("Network edges:", G.number_of_edges())

```

    Network nodes: 3931
    Network edges: 90281


## 6. Compute Centrality Metrics

The centrality table ranks individuals by several network measures:

- `document_frequency`: number of documents where the person appears.
- `weighted_degree`: total weighted co-occurrence strength with other people.
- `degree_centrality`: share of other people this person is connected to.
- `pagerank`: graph-based centrality that gives more weight to connections with other central people.

Weighted degree is used as the primary metric because it directly matches the problem: structurally influential people should repeatedly co-occur with many other named individuals across the records.



```python
def compute_influence_metrics(graph):
    """Compute centrality metrics for each person in the co-occurrence graph."""
    if graph.number_of_nodes() == 0:
        return pd.DataFrame()

    weighted_degree = dict(graph.degree(weight="weight"))
    degree_centrality = nx.degree_centrality(graph)
    pagerank = nx.pagerank(graph, weight="weight")

    metrics = pd.DataFrame({
        "name": list(graph.nodes()),
        "document_frequency": [graph.nodes[n].get("document_frequency", 0) for n in graph.nodes()],
        "weighted_degree": [weighted_degree.get(n, 0) for n in graph.nodes()],
        "degree_centrality": [degree_centrality.get(n, 0) for n in graph.nodes()],
        "pagerank": [pagerank.get(n, 0) for n in graph.nodes()],
    })
    return metrics.sort_values("weighted_degree", ascending=False).reset_index(drop=True)

influence_df = compute_influence_metrics(G)
influence_df.head(15)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>document_frequency</th>
      <th>weighted_degree</th>
      <th>degree_centrality</th>
      <th>pagerank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jeffrey Epstein</td>
      <td>1750</td>
      <td>9646</td>
      <td>0.679898</td>
      <td>0.040331</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Donald Trump</td>
      <td>546</td>
      <td>2784</td>
      <td>0.260560</td>
      <td>0.011299</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Bill Clinton</td>
      <td>259</td>
      <td>2778</td>
      <td>0.209669</td>
      <td>0.008653</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Prince Andrew</td>
      <td>107</td>
      <td>1391</td>
      <td>0.087532</td>
      <td>0.004055</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ghislaine Maxwell</td>
      <td>96</td>
      <td>1113</td>
      <td>0.067430</td>
      <td>0.003217</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Barack Obama</td>
      <td>87</td>
      <td>1095</td>
      <td>0.158524</td>
      <td>0.004749</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Larry Summers</td>
      <td>100</td>
      <td>1066</td>
      <td>0.102799</td>
      <td>0.003689</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Woody Allen</td>
      <td>47</td>
      <td>1038</td>
      <td>0.130534</td>
      <td>0.002383</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bill Gates</td>
      <td>51</td>
      <td>1022</td>
      <td>0.079135</td>
      <td>0.002984</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Darren Indyke</td>
      <td>123</td>
      <td>904</td>
      <td>0.076590</td>
      <td>0.003372</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alan Dershowitz</td>
      <td>71</td>
      <td>887</td>
      <td>0.077354</td>
      <td>0.002639</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Peggy Siegal</td>
      <td>29</td>
      <td>875</td>
      <td>0.099746</td>
      <td>0.001678</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Steve Bannon</td>
      <td>130</td>
      <td>854</td>
      <td>0.071247</td>
      <td>0.003421</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Peter Thiel</td>
      <td>29</td>
      <td>808</td>
      <td>0.086514</td>
      <td>0.001657</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Kathy Ruemmler</td>
      <td>146</td>
      <td>804</td>
      <td>0.061578</td>
      <td>0.003246</td>
    </tr>
  </tbody>
</table>
</div>



## 7. Bootstrap Resampling for Stability

This is the uncertainty and stability step. The pipeline repeatedly resamples the document set with replacement. For each bootstrap sample, it rebuilds the co-occurrence network and recomputes weighted degree.

I included this because bootstrapping is a concept I learned in previous classes. It is useful here because the dataset is not a perfect record of all possible communications. If a person only looks central in one exact dataset but falls away when the documents are resampled, that ranking is less stable. If a person remains highly ranked across bootstrap samples, the centrality pattern is more reliable within this dataset.



```python
def bootstrap_centrality(doc_name_lists, n_boot=300, metric="weighted_degree", random_state=42):
    """Resample documents with replacement and recompute centrality for each sample."""
    rng = np.random.default_rng(random_state)
    n_docs = len(doc_name_lists)
    results = []

    for bootstrap_iter in range(n_boot):
        sample_idx = rng.choice(n_docs, size=n_docs, replace=True)
        sampled_docs = [doc_name_lists[i] for i in sample_idx]
        boot_graph = build_cooccurrence_graph(sampled_docs)
        boot_metrics = compute_influence_metrics(boot_graph)

        if metric not in boot_metrics.columns:
            raise ValueError(f"Metric {metric} not found in centrality table.")

        boot_metrics = boot_metrics[["name", metric]].copy()
        boot_metrics["bootstrap_iter"] = bootstrap_iter
        results.append(boot_metrics)

    return pd.concat(results, ignore_index=True)

def summarize_bootstrap_results(boot_df, metric="weighted_degree"):
    """Summarize bootstrap centrality estimates by person."""
    summary = (
        boot_df.groupby("name")[metric]
        .agg(
            mean="mean",
            std="std",
            q025=lambda x: x.quantile(0.025),
            median="median",
            q975=lambda x: x.quantile(0.975),
            times_observed="count"
        )
        .reset_index()
    )
    summary["interval_width"] = summary["q975"] - summary["q025"]
    summary["coefficient_of_variation"] = summary["std"] / summary["mean"].replace(0, np.nan)
    return summary.sort_values("mean", ascending=False).reset_index(drop=True)

boot_df = bootstrap_centrality(docs, n_boot=300, metric="weighted_degree")
boot_summary = summarize_bootstrap_results(boot_df, metric="weighted_degree")
boot_summary.head(15)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>mean</th>
      <th>std</th>
      <th>q025</th>
      <th>median</th>
      <th>q975</th>
      <th>times_observed</th>
      <th>interval_width</th>
      <th>coefficient_of_variation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jeffrey Epstein</td>
      <td>9661.010000</td>
      <td>357.204517</td>
      <td>8979.450</td>
      <td>9666.0</td>
      <td>10282.425</td>
      <td>300</td>
      <td>1302.975</td>
      <td>0.036974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bill Clinton</td>
      <td>2780.176667</td>
      <td>242.778810</td>
      <td>2329.975</td>
      <td>2784.5</td>
      <td>3263.200</td>
      <td>300</td>
      <td>933.225</td>
      <td>0.087325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Donald Trump</td>
      <td>2779.050000</td>
      <td>272.351142</td>
      <td>2270.375</td>
      <td>2766.0</td>
      <td>3278.725</td>
      <td>300</td>
      <td>1008.350</td>
      <td>0.098002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Prince Andrew</td>
      <td>1392.906667</td>
      <td>161.576301</td>
      <td>1075.475</td>
      <td>1386.5</td>
      <td>1742.500</td>
      <td>300</td>
      <td>667.025</td>
      <td>0.115999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ghislaine Maxwell</td>
      <td>1119.533333</td>
      <td>156.442117</td>
      <td>840.000</td>
      <td>1107.0</td>
      <td>1407.525</td>
      <td>300</td>
      <td>567.525</td>
      <td>0.139739</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Barack Obama</td>
      <td>1094.030000</td>
      <td>157.289039</td>
      <td>816.700</td>
      <td>1086.0</td>
      <td>1398.625</td>
      <td>300</td>
      <td>581.925</td>
      <td>0.143770</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Larry Summers</td>
      <td>1076.133333</td>
      <td>172.646435</td>
      <td>783.325</td>
      <td>1069.5</td>
      <td>1420.200</td>
      <td>300</td>
      <td>636.875</td>
      <td>0.160432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Woody Allen</td>
      <td>1021.403333</td>
      <td>271.154229</td>
      <td>564.600</td>
      <td>992.5</td>
      <td>1613.000</td>
      <td>300</td>
      <td>1048.400</td>
      <td>0.265472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bill Gates</td>
      <td>1008.883333</td>
      <td>162.414875</td>
      <td>713.475</td>
      <td>995.5</td>
      <td>1331.300</td>
      <td>300</td>
      <td>617.825</td>
      <td>0.160985</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Darren Indyke</td>
      <td>908.660000</td>
      <td>113.186491</td>
      <td>697.475</td>
      <td>903.0</td>
      <td>1125.500</td>
      <td>300</td>
      <td>428.025</td>
      <td>0.124564</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alan Dershowitz</td>
      <td>896.143333</td>
      <td>154.320420</td>
      <td>625.875</td>
      <td>885.5</td>
      <td>1235.575</td>
      <td>300</td>
      <td>609.700</td>
      <td>0.172205</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Peggy Siegal</td>
      <td>873.586667</td>
      <td>281.869053</td>
      <td>385.400</td>
      <td>868.0</td>
      <td>1510.775</td>
      <td>300</td>
      <td>1125.375</td>
      <td>0.322657</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Steve Bannon</td>
      <td>855.016667</td>
      <td>104.353639</td>
      <td>664.950</td>
      <td>844.0</td>
      <td>1075.575</td>
      <td>300</td>
      <td>410.625</td>
      <td>0.122049</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kathy Ruemmler</td>
      <td>813.556667</td>
      <td>97.690523</td>
      <td>642.275</td>
      <td>813.0</td>
      <td>1016.100</td>
      <td>300</td>
      <td>373.825</td>
      <td>0.120078</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Peter Thiel</td>
      <td>799.030000</td>
      <td>245.305002</td>
      <td>404.000</td>
      <td>791.5</td>
      <td>1352.000</td>
      <td>300</td>
      <td>948.000</td>
      <td>0.307003</td>
    </tr>
  </tbody>
</table>
</div>



## 8. Final Ranking Table

The final table combines the original centrality metrics with the bootstrap stability results. The most stable structurally influential people should have high centrality and relatively narrow bootstrap intervals.



```python
final_rankings = (
    influence_df.merge(
        boot_summary,
        on="name",
        how="left",
        suffixes=("_original", "_bootstrap")
    )
    .sort_values("mean", ascending=False)
    .reset_index(drop=True)
)

final_rankings.head(20)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>document_frequency</th>
      <th>weighted_degree</th>
      <th>degree_centrality</th>
      <th>pagerank</th>
      <th>mean</th>
      <th>std</th>
      <th>q025</th>
      <th>median</th>
      <th>q975</th>
      <th>times_observed</th>
      <th>interval_width</th>
      <th>coefficient_of_variation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Jeffrey Epstein</td>
      <td>1750</td>
      <td>9646</td>
      <td>0.679898</td>
      <td>0.040331</td>
      <td>9661.010000</td>
      <td>357.204517</td>
      <td>8979.450</td>
      <td>9666.0</td>
      <td>10282.425</td>
      <td>300</td>
      <td>1302.975</td>
      <td>0.036974</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Bill Clinton</td>
      <td>259</td>
      <td>2778</td>
      <td>0.209669</td>
      <td>0.008653</td>
      <td>2780.176667</td>
      <td>242.778810</td>
      <td>2329.975</td>
      <td>2784.5</td>
      <td>3263.200</td>
      <td>300</td>
      <td>933.225</td>
      <td>0.087325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Donald Trump</td>
      <td>546</td>
      <td>2784</td>
      <td>0.260560</td>
      <td>0.011299</td>
      <td>2779.050000</td>
      <td>272.351142</td>
      <td>2270.375</td>
      <td>2766.0</td>
      <td>3278.725</td>
      <td>300</td>
      <td>1008.350</td>
      <td>0.098002</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Prince Andrew</td>
      <td>107</td>
      <td>1391</td>
      <td>0.087532</td>
      <td>0.004055</td>
      <td>1392.906667</td>
      <td>161.576301</td>
      <td>1075.475</td>
      <td>1386.5</td>
      <td>1742.500</td>
      <td>300</td>
      <td>667.025</td>
      <td>0.115999</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ghislaine Maxwell</td>
      <td>96</td>
      <td>1113</td>
      <td>0.067430</td>
      <td>0.003217</td>
      <td>1119.533333</td>
      <td>156.442117</td>
      <td>840.000</td>
      <td>1107.0</td>
      <td>1407.525</td>
      <td>300</td>
      <td>567.525</td>
      <td>0.139739</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Barack Obama</td>
      <td>87</td>
      <td>1095</td>
      <td>0.158524</td>
      <td>0.004749</td>
      <td>1094.030000</td>
      <td>157.289039</td>
      <td>816.700</td>
      <td>1086.0</td>
      <td>1398.625</td>
      <td>300</td>
      <td>581.925</td>
      <td>0.143770</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Larry Summers</td>
      <td>100</td>
      <td>1066</td>
      <td>0.102799</td>
      <td>0.003689</td>
      <td>1076.133333</td>
      <td>172.646435</td>
      <td>783.325</td>
      <td>1069.5</td>
      <td>1420.200</td>
      <td>300</td>
      <td>636.875</td>
      <td>0.160432</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Woody Allen</td>
      <td>47</td>
      <td>1038</td>
      <td>0.130534</td>
      <td>0.002383</td>
      <td>1021.403333</td>
      <td>271.154229</td>
      <td>564.600</td>
      <td>992.5</td>
      <td>1613.000</td>
      <td>300</td>
      <td>1048.400</td>
      <td>0.265472</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Bill Gates</td>
      <td>51</td>
      <td>1022</td>
      <td>0.079135</td>
      <td>0.002984</td>
      <td>1008.883333</td>
      <td>162.414875</td>
      <td>713.475</td>
      <td>995.5</td>
      <td>1331.300</td>
      <td>300</td>
      <td>617.825</td>
      <td>0.160985</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Darren Indyke</td>
      <td>123</td>
      <td>904</td>
      <td>0.076590</td>
      <td>0.003372</td>
      <td>908.660000</td>
      <td>113.186491</td>
      <td>697.475</td>
      <td>903.0</td>
      <td>1125.500</td>
      <td>300</td>
      <td>428.025</td>
      <td>0.124564</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Alan Dershowitz</td>
      <td>71</td>
      <td>887</td>
      <td>0.077354</td>
      <td>0.002639</td>
      <td>896.143333</td>
      <td>154.320420</td>
      <td>625.875</td>
      <td>885.5</td>
      <td>1235.575</td>
      <td>300</td>
      <td>609.700</td>
      <td>0.172205</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Peggy Siegal</td>
      <td>29</td>
      <td>875</td>
      <td>0.099746</td>
      <td>0.001678</td>
      <td>873.586667</td>
      <td>281.869053</td>
      <td>385.400</td>
      <td>868.0</td>
      <td>1510.775</td>
      <td>300</td>
      <td>1125.375</td>
      <td>0.322657</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Steve Bannon</td>
      <td>130</td>
      <td>854</td>
      <td>0.071247</td>
      <td>0.003421</td>
      <td>855.016667</td>
      <td>104.353639</td>
      <td>664.950</td>
      <td>844.0</td>
      <td>1075.575</td>
      <td>300</td>
      <td>410.625</td>
      <td>0.122049</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Kathy Ruemmler</td>
      <td>146</td>
      <td>804</td>
      <td>0.061578</td>
      <td>0.003246</td>
      <td>813.556667</td>
      <td>97.690523</td>
      <td>642.275</td>
      <td>813.0</td>
      <td>1016.100</td>
      <td>300</td>
      <td>373.825</td>
      <td>0.120078</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Peter Thiel</td>
      <td>29</td>
      <td>808</td>
      <td>0.086514</td>
      <td>0.001657</td>
      <td>799.030000</td>
      <td>245.305002</td>
      <td>404.000</td>
      <td>791.5</td>
      <td>1352.000</td>
      <td>300</td>
      <td>948.000</td>
      <td>0.307003</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Graydon Carter</td>
      <td>10</td>
      <td>775</td>
      <td>0.087786</td>
      <td>0.001182</td>
      <td>778.186667</td>
      <td>278.623309</td>
      <td>287.900</td>
      <td>763.0</td>
      <td>1365.525</td>
      <td>300</td>
      <td>1077.625</td>
      <td>0.358042</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Michael Wolff</td>
      <td>182</td>
      <td>729</td>
      <td>0.046819</td>
      <td>0.003004</td>
      <td>729.856667</td>
      <td>75.509570</td>
      <td>595.475</td>
      <td>731.0</td>
      <td>881.825</td>
      <td>300</td>
      <td>286.350</td>
      <td>0.103458</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Larry Gagosian</td>
      <td>12</td>
      <td>713</td>
      <td>0.082188</td>
      <td>0.001053</td>
      <td>715.763333</td>
      <td>269.527223</td>
      <td>225.950</td>
      <td>701.5</td>
      <td>1326.125</td>
      <td>300</td>
      <td>1100.175</td>
      <td>0.376559</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Al Gore</td>
      <td>51</td>
      <td>689</td>
      <td>0.052926</td>
      <td>0.002170</td>
      <td>690.160000</td>
      <td>115.828935</td>
      <td>491.175</td>
      <td>692.0</td>
      <td>932.450</td>
      <td>300</td>
      <td>441.275</td>
      <td>0.167829</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lesley Groff</td>
      <td>50</td>
      <td>689</td>
      <td>0.102290</td>
      <td>0.002187</td>
      <td>681.623333</td>
      <td>177.971348</td>
      <td>388.850</td>
      <td>675.5</td>
      <td>1068.100</td>
      <td>300</td>
      <td>679.250</td>
      <td>0.261099</td>
    </tr>
  </tbody>
</table>
</div>



## 9. Visualization Rationale

The final visualization has two panels.

The left panel shows the co-occurrence network for the top-ranked people. This makes the network model visible: node size represents document frequency, edge width represents co-occurrence strength, and node color represents centrality.

The right panel shows the bootstrap stability of the weighted degree ranking. Each point is the bootstrapped mean weighted degree for one individual, and the error bars show the 95% bootstrap interval. This is the most important visualization for the problem because it shows not only who ranks highly, but whether the ranking remains stable across resampled document sets.



```python
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

def plot_network_and_stability(graph, boot_summary, output_path="epstein_network_and_stability.png", top_n=15):
    """Create the final publication-quality network and bootstrap stability visualization."""
    top_nodes = boot_summary.head(top_n)["name"].tolist()
    graph_plot = graph.subgraph(top_nodes).copy()
    plot_df = boot_summary.head(top_n).sort_values("mean")

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    ax = axes[0]
    pos = nx.spring_layout(graph_plot, seed=42, k=1.2)
    node_sizes = [graph_plot.nodes[n].get("document_frequency", 1) * 20 for n in graph_plot.nodes()]
    edge_widths = [graph_plot[u][v].get("weight", 1) * 0.5 for u, v in graph_plot.edges()]
    centrality = nx.degree_centrality(graph_plot)
    node_colors = [centrality[n] for n in graph_plot.nodes()]

    nx.draw_networkx_nodes(graph_plot, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph_plot, pos, width=edge_widths, alpha=0.4, ax=ax)
    nx.draw_networkx_labels(graph_plot, pos, font_size=8, ax=ax)
    ax.set_title("Co-occurrence Network: Top Individuals")


    # Create custom legend elements
    legend_elements = [

        # Node size meaning
        Line2D([0], [0], marker='o', color='w',
           label='Node size = frequency of appearance',
           markerfacecolor='gray', markersize=10),

        # Node color meaning
        Patch(facecolor=plt.cm.viridis(0.8),
          label='Node color = centrality (lighter = more central)'),

        # Edge width meaning
        Line2D([0], [0], color='black', lw=2,
           label='Edge width = co-occurrence strength'),

    ]

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        frameon=True,
        fontsize=9
    )
    ax.axis("off")

    ax = axes[1]
    ax.errorbar(
        x=plot_df["mean"],
        y=plot_df["name"],
        xerr=[plot_df["mean"] - plot_df["q025"], plot_df["q975"] - plot_df["mean"]],
        fmt="o",
        capsize=3
    )
    ax.set_xlabel("Bootstrapped Mean Weighted Degree")
    ax.set_ylabel("Individual")
    ax.set_title("Stability Across Bootstrapped Document Samples")
    ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Epstein Files: Structural Influence from Name Co-occurrence Network", fontsize=18)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

plot_network_and_stability(G, boot_summary, output_path="epstein_network_and_stability.png", top_n=15)

```


    
![png](pipeline_files/pipeline_21_0.png)
    


## 10. Analysis: Solving the Problem

This pipeline turns the document collection of Epstein's emails into a network of name co-occurrences and evaluates which people are central in that network.

The pipeline first queries the MongoDB document database into a dataframe, which connects the analysis directly to the document model dataset. It then cleans the `people_mentioned` metadata so each document contributes a list of usable person names. From those lists, the pipeline builds a weighted undirected graph. In this graph, nodes are individuals, edges are shared document appearances, and edge weights represent repeated co-occurrence across records.

The pipeline identifies structural influence by computing centrality metrics. The most important metric is weighted degree because it measures how strongly each person is connected to others through repeated co-occurrence. Degree centrality and PageRank are included as additional network metrics so the ranking is not based on only one view of graph position.

The pipeline also addresses uncertainty by using bootstrap resampling, which is the concept I carried over from previous classes. Instead of treating one observed network as final, the notebook repeatedly resamples the document set, rebuilds the network, and recomputes centrality. The resulting bootstrap means and 95% intervals show whether high-ranking individuals remain central across resampled document sets.

The final visualization directly supports the solution. The network panel shows the structure of relationships among highly ranked people, while the bootstrap panel shows the stability of their influence rankings. Together, the table and visualization answer the problem: the most structurally influential individuals are the names with high centrality scores whose rankings remain stable under document-level resampling.

The main limitation is that these results only describe structural influence within the released and processed Epstein file records. They should not be interpreted as proof of real-world influence, guilt, or involvement. The results depend on the released document subset and on the quality of the upstream metadata extraction.

