# DS 4320 Project 2: Mapping Influence in the Epstein Files

#### Executive Summary

Ivey Mistele

zyh4up

DOI: [![DOI](https://zenodo.org/badge/1215532260.svg)](https://doi.org/10.5281/zenodo.19862841)

[Press Release](https://github.com/iveymistele/epstein-files/blob/main/PRESS_RELEASE.md) 

[Pipeline](https://github.com/iveymistele/epstein-files/blob/main/pipeline.ipynb) 

[License: MIT](https://github.com/iveymistele/epstein-files/blob/main/LICENSE)


## Problem Definition

#### Initial General Problem

Understanding and analyzing hidden relationships and influence within complex social networks.

#### Refined Problem Statement

Identify which individuals are most structurally influential in Epstein file records by modeling name co-occurrence as a network and evaluating which centrality patterns remain stable across resampled document sets.

#### Project Motivation


Social networks are complex representations of human behavior, and using data to understand those patterns is an interesting application of data science. The Epstein files provide a unique dataset that captures interactions among a specific group of individuals whose relationships have been subject to public attention. While the context is controversial, the data itself offers an opportunity to explore how influence and connections appear within a real-world network. This makes it a compelling case for analyzing how relationships form and which individuals appear most central within a network.

#### Refinement Rationale

The Epstein files are a strong example of a publicly available dataset that reflects real-world human interactions. Instead of studying abstract or simulated networks, this project focuses on a dataset that has been widely examined and discussed, making it more meaningful to analyze. Refining the problem to identifying central individuals through co-occurrence allows for a more structured and measurable approach to understanding relationships within the data. This shift turns a broad idea about social networks into a specific, data-driven question that can be answered through analysis and visualization.

#### Press Release: (Hidden Networks Revealed in Epstein Case Data: A Small Number of Individuals Appear Across a Disproportionate Share of Epstein Case Records)[link]

## Domain Exposition

#### Terminology


| Term | Explanation |
|------|-------------|
| Email (Document) | A single email record in the dataset, treated as a document containing text and referenced individuals. |
| Sender | The individual who authored or sent the email. |
| Recipient | The individual(s) who received the email. |
| Entity (Person) | An individual identified within an email, either as sender, recipient, or mentioned in the text. |
| Named Entity Recognition (NER) | A process used to identify and extract names of individuals from unstructured email text. |
| Co-occurrence | The presence of two individuals within the same email, used to infer a potential relationship. |
| Node | A representation of an individual in the network. |
| Edge | A connection between two individuals, created when they appear in the same email. |
| Edge Weight | The number of emails in which two individuals co-occur, representing strength of connection. |
| Degree (Node Degree) | The number of unique individuals a person is connected to. |
| Degree Centrality | A metric based on how many direct connections an individual has. |
| Frequency | The number of emails in which an individual appears. |
| Top-K Analysis | Identifying the top K individuals based on a metric (e.g., most frequent or most connected). |
| Collection | A group of email documents stored in MongoDB. |
| Bias | Distortion in the dataset due to incomplete or selectively released emails. |
| Uncertainty | The limitation that co-occurrence in an email does not necessarily imply a direct relationship. |

#### Project Domain

This project lives in the domain of social network analysis, which focuses on understanding relationships and patterns of interaction between individuals. In this case, the network is constructed from email records associated with the Epstein case, representing a subset of individuals whose interactions have been subject to significant public attention. This dataset provides a unique look into a high-profile social network, often associated with elite or influential individuals. By treating emails as documents and individuals as entities within those documents, the project applies data science techniques to explore how connections form and which individuals appear most central within the network.

#### Background Readings

Link: https://myuva-my.sharepoint.com/:u:/g/personal/zyh4up_virginia_edu/IQBthTxJFZyOS7x-yOkOJtbNAVNGjXP0_atOeZccmCOFC-U?e=Sd2hul

#### Reading Summary


| # | Title | Description | Link |
|---|-------|-------------|------|
| 1 | The Epstein Files and the Anatomy of Hidden Social Networks | Explains how secrecy shapes what's observable in document-derived networks; warns against overclaiming centrality from partial data | https://myuva-my.sharepoint.com/:u:/r/personal/zyh4up_virginia_edu/Documents/HW%209%20Background%20Reading/manlius.substack.com.url?csf=1&web=1&e=FcvhUA |
| 2 | The Importance of Social Networks for Innovation and Productivity | Argues that network structure determines how information flows; motivates why studying social networks matters | https://myuva-my.sharepoint.com/:u:/r/personal/zyh4up_virginia_edu/Documents/HW%209%20Background%20Reading/ourworldindata.org.url?csf=1&web=1&e=bSzq88 |
| 3 | How Newsrooms Are Digging Into the Epstein Files | Reuters Institute piece on how BBC, NYT, and Guardian used AI and data tools to analyze the same document corpus; contrasts journalistic vs. computational approaches | https://myuva-my.sharepoint.com/:u:/r/personal/zyh4up_virginia_edu/Documents/HW%209%20Background%20Reading/reutersinstitute.politics.ox.ac.uk.url?csf=1&web=1&e=pzEjjB |
| 4 | Social Network Analysis 101 | Introductory guide to SNA concepts (nodes, edges, centrality, community structure) from Visible Network Labs | https://myuva-my.sharepoint.com/:u:/r/personal/zyh4up_virginia_edu/Documents/HW%209%20Background%20Reading/visiblenetworklabs.com.url?csf=1&web=1&e=2Iz0mu |
| 5 | Who is Jeffrey Epstein? | BBC background article on Epstein, his criminal history, and his social circle of powerful figures | https://myuva-my.sharepoint.com/:u:/r/personal/zyh4up_virginia_edu/Documents/HW%209%20Background%20Reading/www.bbc.com.url?csf=1&web=1&e=EyaQ30 |

## Data Creation 

#### Provenance

I acquired the raw data through a open-source ETL pipeline from DocETL (https://www.docetl.org/showcase/epstein-email-explorer). The dataset is made of structued JSON records containing information on emails and extracted metadata including participants and email addresses, topics, and summaries. The DocETL pipeline drew the data from the official House Oversight Committee Epstein correspondence. I downloaded the raw JSON file from the website and created a python script to upload the documents into my MongoDB cluster. This python script read each JSON object into a MongoDB collection as individual documents, each representing an email. The python script contained light transformations, such as creating a `timestamp` field from the original `date` and `time` fields. Because the data was already processed by the authors, I did not do any heavy transformations to acquire the raw data.

#### Code

| File | Description | Link |
|---|---|---|
|`upload_data.py`| Python script reading raw JSON containing emails & metadata and uploading to MongoDB. | https://github.com/iveymistele/epstein-files/blob/main/upload_data.py |

Note: another file containing the raw JSON data from the original source was used and referenced in `upload_data.py`. Because of the size of the file, it was not included in this repository. 

#### Rationale

One major judgement call I made in creating the data is to use a pre-transformed dataset from DocETL (Epstein Email Explorer). I chose to go down this route as opposed to download emails directly from the House Oversight Committee because the document files from the government website were more complicated to work with than the pre-transformed JSON records provided by the DocETL project. This way, I am able to build upon an existing project rather than start from scratch where a preexisting & reliable solution already exists. Chosing to download the information outside of the original source/release came with a tradeoff of uncertainty. As described before, the ETL steps taken in the pipeline potentially introduced algorithmic bias into the data. This was a worthy tradeoff for me however, as it saved lots of time and computational effort, and I am able to address these potential sources of bias through my analysis.

Other decisions I made came up in the data extraction process. Primarily, I chose to preserve the original text of the email as a field in my documents. I chose to do this so that the pre-derived fields from DocETL's pipeline can be cross-checked with the original text for verification. This step allows me to address/quantify uncertainty, and allows me to explore my own transformations/analysis to build on the existing work rather than just mirror it.

I also performed light transformations in reading the data and uploading it to MongoDB. I first clearly established a primary key by setting doc["_id"] (the syntax for Mongo's unique identifier) to doc["document_id"], the existing identifier in DocETL's dataset. This way, I did not need to generate a new ID for the primary key in Mongo.

I then created a new field `timestamp` from the exiting fields `date` and `time`. This way, analysis steps like sorting and filtering are easier, as the date and time are combined into one datetime string. I still kept the individual `date` and `time` fields in case I need to identify missing/incomplete information or need to cross reference the timestamp with original values in case of missing information. This allows me to address potential uncertainty that can come up when extracting the new field.

Next, I created two new fields, `participant_names` and `participant_emails`. These are created from the original `participants` field, and I only kept non-null values. By flattening these arrays, querying for names and emails is easier. This does assume that the extraction of participants from DocETL's pipeline is correct and did not introduce uncertainty, but this potential uncertainty will be addressed in downstream analysis.

Next, I added counting fields `participant_count`, `attachment_count`, and `url_count` which contain information regarding counts of each field within an email/document. These will be helpful in analysis by providing quantified summaries of each email/document. Like the participant names and emails, these derived features are dependent on the accuracy of the original data processing pipeline, and this is something that will need to be addressed in communicating final results.


#### Bias Identification

The primary source of bias introduction in the data collection is what the documents represent. The House Oversight Committee only released a subset of Epstein's emails. This means that whatever patterns are found in the data in terms of his relations are merely a reflection of the data released, and not necessarily representative of all of his communications and relationships (selection bias).

Another potential source of bias is introduced in the ETL pipeline itself used by DocETL to produce the raw JSON I transformed/loaded into my MongoDB. The techniques used to extract metadata, such as assigning topics, can introduce algorithmic bias. Subjects may be misidentified, or information may be lost in the pipeline process.

Last, the nature of the Epstein files introduces another source of potential bias. The emails contain redacted information, and the process by which the House Oversight Committee selected emails to release was not stated. This lack of transparency presents another form of bias in terms of not having a representative sample.

#### Bias Mitigation

These potential biases can be mitigated through how I define the takeaways from my analysis. Any patterns found/relationships identified between Epstein and his correspondents are representative of only the subset of Epstein's communications, and therefore cannot be treated as representative of his communications as a whole. I will address this by clearly stating the limitations of the results of analysis.

I will account for the potential algorithmic bias by keeping the original email text as a field in each collection. This way, I am able to verify/cross-reference metadata with the original text to ensure accuracy. This will also help identify information in documents which are missing this metadata/extracted fields. This way, I can also quantify uncertainty by comparing metadata (transformed fields) with original text and looking for missing values/inconsistent results. I will also be sure to document when a finding from analysis comes from extracted text, and indicate that it should not be taken as fact due to the potential bias.

## Metadata 

#### Implicit Schema 

- Each document in the collection represents one email.

- Each document must have a unique `_id`, which is taken from the original `document_id` to maintain consistency with the source data.

- Core fields expected in most documents include:
  - `email_text` (full email content)
  - `date`
  - `time`
  - `subject`

- A `timestamp` field is created by combining `date` and `time` to make time-based queries easier, while still keeping the original fields.

- Participant data is stored in two ways:
  - `participants`: a nested list of objects with `name` and/or `email`
  - `participant_names` and `participant_emails`: flattened arrays for easier querying

- Optional metadata fields may include:
  - `topics`
  - `primary_topic`
  - `organizations`
  - `locations`
  - `urls`
  - `attachment_names`
  - `summary`
  - May be missing or empty depending on the document.

- Derived numeric fields are included:
  - `participant_count`
  - `attachment_count`
  - `url_count`
  - Calculated from their corresponding arrays and stored as integers.

- Documents are allowed to have missing or null fields due to variability in the source data and extraction process.

- The original `email_text` is always preserved to allow for manual verification of extracted metadata and to reduce reliance on automated processing.

#### Data Summary 

| Metric                     | Value/Example            | Description |
|--------------------------|---------------------------|-------------|
| Total Documents          | 2,322                    | Total number of email documents stored in the collection |
| Document Type            | Email                     | Each document represents one email |
| Date Range               | 2000–2018 (approx.)       | Range of timestamps across emails |
| Fields per Document      | ~10–20                    | Includes core, optional, and derived fields |
| Core Fields              | email_text, date, subject | Primary attributes present in most documents |
| Optional Fields          | topics, organizations, urls, attachments | Metadata extracted via ETL, may be missing |
| Nested Fields            | participants              | Stored as list of objects (name, email) |
| Derived Fields           | participant_count, attachment_count, url_count | Computed from arrays for analysis |
| Text Field Size          | Variable                  | Email text length varies significantly across documents |
| Missing Data             | Present                   | Some fields may be null or empty depending on extraction quality |
| Data Source              | DocETL Epstein Dataset    | Preprocessed dataset from DocETL pipeline |
| Storage Model            | MongoDB Collection        | Document-oriented database (one email per document) |

#### Data Dictionary 

| Field Name            | Data Type        | Description | Example |
|----------------------|------------------|-------------|---------|
| _id                  | String           | Unique identifier for each email document (copied from document_id) | "HOUSE_OVERSIGHT_011277" |
| document_id          | String           | Original identifier from source dataset | "HOUSE_OVERSIGHT_011277" |
| email_text           | String           | Full raw email content used for verification and analysis | "From: ... To: ... Subject: ..." |
| date                 | String (YYYY-MM-DD) | Date the email was sent | "2016-12-09" |
| time                 | String (HH:MM:SS) | Time the email was sent | "15:46:19" |
| timestamp            | String (ISO-like) | Combined date and time field for easier querying | "2016-12-09T15:46:19" |
| subject              | String           | Subject line of the email | "RE: Financials: buy XLF call spreads" |
| is_email             | Boolean          | Indicates whether the document is classified as an email | true |
| participants         | Array (Objects)  | List of participants with name and/or email | [{"name": "Jeffrey Epstein", "email": "..." }] |
| participant_names    | Array (Strings)  | Flattened list of participant names for querying | ["Jeffrey Epstein", "Amanda Ens"] |
| participant_emails   | Array (Strings)  | Flattened list of participant email addresses | ["jeffrey@...", "amanda@..."] |
| participant_count    | Integer          | Number of participants in the email | 3 |
| attachment_names     | Array (Strings)  | List of attachment file names | ["image001.png"] |
| attachment_count     | Integer          | Number of attachments in the email | 1 |
| urls                 | Array (Strings)  | List of URLs found in the email text | ["http://example.com"] |
| url_count            | Integer          | Number of URLs in the email | 0 |
| organizations        | Array (Strings)  | Organizations mentioned in the email | ["JP Morgan"] |
| locations            | Array (Strings)  | Locations mentioned in the email | ["New York"] |
| people_mentioned     | Array (Strings)  | People referenced in the email text | ["Jeffrey Epstein"] |
| topics               | Array (Strings)  | List of topics assigned to the email by ETL process | ["financial", "business"] |
| primary_topic        | String           | Main topic classification for the email | "financial" |
| summary              | String           | Short summary of the email content | "Discussion about financial strategy..." |
| tone                 | String           | General tone classification of the email | "routine" |
| evidence_strength    | String           | Indicates level of evidentiary relevance | "none" |

#### Quantification of Uncertainty 

| Field | Example | Estimated Uncertainty | Numeric Estimate | Notes |
|------|---------|----------------------|-----------------|------|
| participant_count | 3 | missed participant extraction | ±1 (~33% at n=3) | ETL extraction risk |
| attachment_count | 1 | missing attachment metadata | ±1 (~100% at n=1) | high error for small counts |
| url_count | 2 | regex extraction misses | ±1 (~50%) | pattern matching uncertainty |
| topics | financial | topic classification error | ~10-15% | algorithmic labeling |
| organizations | JP Morgan | entity extraction error | ~5-10% | NER uncertainty |
| timestamp | 2016-12-09T... | parsing error/missing time | <1% | source incompleteness |

