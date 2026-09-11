Yes — your goal is exactly right: **don't memorize implementation/code; learn the architectural decision-making.**

Below are reference notes designed around:

> **What is it? → Why do we need it? → When would I choose it? → Tradeoffs → How does it affect system design?**

---

# Production RAG — Modern RAG Techniques Master Notes

## 🧭 Decision Framework

Before choosing any RAG technique, ask:

```text
1. What kind of data do I have?
2. Is it structured or unstructured?
3. How large is it?
4. How frequently does it change?
5. Are relationships important?
6. Is exact keyword matching important?
7. How complex are user queries?
8. How much latency can I afford?
9. How much can I spend per query?
10. What accuracy/recall do I need?
11. Do users have different permissions?
12. Can the whole relevant data fit into context?
```

### Core principle

> **Don't choose the most advanced RAG architecture. Choose the simplest architecture that satisfies the requirements.**

---

# 🥇 TIER 1 — MUST KNOW

---

# 1. Hybrid RAG

### Concept

Combine multiple retrieval methods:

```text
              Query
                ↓
       ┌────────┴────────┐
       ↓                 ↓
   Dense Search        BM25
   semantic            keyword
       ↓                 ↓
       └───────┬─────────┘
               ↓
              RRF
               ↓
          Final ranking
```

### Why?

Dense retrieval understands **meaning**.

BM25 understands **exact terms**.

Example:

```text
Query:
"CVE-2026-1234 affecting nginx"
```

BM25 is excellent for:

```text
CVE-2026-1234
nginx
```

Dense retrieval helps with:

```text
"security vulnerability in nginx"
```

### When choose?

Use Hybrid when:

* documents contain technical terms
* IDs/codes/names matter
* both semantic and exact matching are important
* retrieval quality matters more than absolute simplicity

### Tradeoffs

**Pros**

* Better recall
* Robust to different query types
* Handles semantic + exact search

**Cons**

* Two retrieval systems
* More infrastructure
* More latency
* More tuning
* Fusion strategy required

### Decision

```text
Normal semantic documents
→ Dense may be enough

Enterprise/technical knowledge
→ Hybrid usually safer
```

---

# 2. Reranking

### Concept

First-stage retrieval:

```text
Query
 ↓
Retrieve 50 candidates
```

Then:

```text
50 candidates
 ↓
Reranker
 ↓
Top 5
```

Reranker performs **deeper relevance evaluation**.

### Why?

Vector similarity is only an approximation of relevance.

Two chunks can have similar embeddings but only one actually answers the question.

### When choose?

Use when:

* retrieval recall is good
* but top results aren't ordered well
* high answer accuracy matters
* candidate set is manageable

### Tradeoff

```text
Without reranker
Fast
 ↓
Potentially weaker ranking

With reranker
Slower
 ↓
Better precision
```

### Important architecture

Don't rerank 10,000 documents.

```text
10,000
 ↓ ANN/BM25
100
 ↓ reranker
10
 ↓ LLM
```

### Decision

> **Retriever = high recall. Reranker = high precision.**

---

# 3. RAG Evaluation

### Concept

Measure RAG as separate components.

```text
Retrieval
   ↓
Generation
   ↓
End-to-end
```

### Why?

Suppose answer is wrong.

You need to know:

```text
Did retrieval fail?
        OR
Did LLM fail to use correct evidence?
```

### Retrieval metrics

| Metric      | Meaning                                  |
| ----------- | ---------------------------------------- |
| Recall@K    | Did we retrieve relevant evidence?       |
| Precision@K | How much retrieved evidence is relevant? |
| MRR         | How early is first relevant result?      |
| NDCG        | Quality of ranking                       |

### Generation metrics

* Faithfulness
* Groundedness
* Answer relevance
* Citation correctness
* Completeness

### When choose?

**Always.**

Production RAG without evaluation = you're guessing.

### Tradeoff

Evaluation itself requires:

* datasets
* labeling
* infrastructure
* compute
* maintenance

But without it you can't safely optimize.

### Critical idea

```text
Better retrieval
≠
Better answer automatically
```

---

# 4. Agentic RAG

### Concept

Traditional RAG:

```text
Query
 ↓
Retrieve
 ↓
Answer
```

Agentic RAG:

```text
Query
 ↓
Agent
 ↓
Plan
 ↓
Search
 ↓
Inspect evidence
 ↓
Enough?
 ├── YES → Answer
 └── NO
       ↓
   Rewrite query
       ↓
   Search again
```

### Why?

Some questions cannot be solved with one retrieval call.

Example:

> "Which vulnerabilities affect systems owned by companies acquired by X?"

Requires multiple steps.

### When choose?

Use for:

* multi-hop questions
* research
* investigations
* complex enterprise queries
* multiple data sources

### Don't use for

Simple:

> "What is our vacation policy?"

Traditional RAG is better.

### Tradeoffs

**Pros**

* Adaptive
* Multi-step reasoning
* Multiple searches
* Multiple tools
* Better for complex tasks

**Cons**

* Higher latency
* Higher cost
* More failure modes
* Harder debugging
* Non-deterministic behavior

### Decision

```text
Simple query
→ Traditional RAG

Complex query
→ Agentic RAG
```

---

# 5. Context Engineering

### Concept

Don't ask:

> "How much information can I retrieve?"

Ask:

> **"What information does the model actually need?"**

Pipeline:

```text
Retrieve
 ↓
Filter
 ↓
Rerank
 ↓
Deduplicate
 ↓
Compress
 ↓
Order
 ↓
Context
 ↓
LLM
```

### Why?

More context can cause:

* higher cost
* higher latency
* distraction
* redundant information
* weaker reasoning

### Techniques

* context compression
* summarization
* deduplication
* relevance filtering
* ordering
* token budgeting
* neighboring chunk retrieval
* evidence prioritization

### Tradeoff

Too little:

```text
Missing evidence
```

Too much:

```text
Noise
```

Goal:

> **Minimum sufficient context.**

---

# 6. Long Context vs RAG

### Long-context approach

```text
Large documents
 ↓
LLM context
 ↓
Answer
```

### RAG

```text
Large documents
 ↓
Retrieve relevant pieces
 ↓
LLM
```

### When long context wins

* small corpus
* temporary documents
* need broad context
* document fits comfortably
* retrieval quality is difficult

### When RAG wins

* huge corpus
* frequently changing knowledge
* many users
* permission filtering
* cost-sensitive systems
* targeted retrieval

### Tradeoff

| Long Context          | RAG                     |
| --------------------- | ----------------------- |
| Simpler               | More complex            |
| More context          | Less context            |
| Potentially expensive | More efficient at scale |
| No retrieval failure  | Retrieval can fail      |
| Good for small corpus | Good for large corpus   |

### Important

> **Large context window does not automatically make RAG obsolete.**

---

# 7. Tool / API / SQL Retrieval

### Concept

Not every question needs vector search.

Choose the correct information source.

```text
                 Query
                   ↓
                 Router
          ┌────────┼─────────┐
          ↓        ↓         ↓
        RAG       SQL       API
```

### Example

> "What is our revenue this quarter?"

Use:

```text
SQL
```

Not:

```text
Vector DB
```

### Example

> "What's the current account balance?"

Use:

```text
Bank API
```

Not RAG.

### Example

> "Explain our password policy."

Use:

```text
Document RAG
```

### Decision

```text
Unstructured knowledge → RAG
Structured data → SQL
Live state → API
Relationships → Graph
```

### Tradeoff

More specialized sources:

**Pros**

* More accurate
* Fresh
* Deterministic

**Cons**

* More integrations
* More routing complexity
* More permissions/security concerns

---

# 🥈 TIER 2 — STRONG DIFFERENTIATORS

---

# 8. GraphRAG

### Concept

Represent entities and relationships as a graph.

```text
Document
 ↓
Entities + Relationships
 ↓
Knowledge Graph
 ↓
Graph retrieval
 ↓
LLM
```

Example:

```text
Company A
   ↓ owns
System X
   ↓ runs
Service Y
   ↓ affected_by
CVE-Z
```

### Why?

Vector similarity answers:

> "What text looks similar?"

Graph retrieval can answer:

> "What entities are connected through these relationships?"

### Best for

* cybersecurity
* fraud
* finance
* supply chains
* organizations
* legal relationships
* scientific knowledge

### Tradeoffs

**Pros**

* Strong multi-hop reasoning
* Explicit relationships
* Explainable relationships

**Cons**

* Graph construction is difficult
* Entity extraction errors
* Graph maintenance
* More infrastructure
* Not necessary for simple documents

### Decision

```text
Relationships are central
→ GraphRAG

Relationships don't matter
→ Don't add graph complexity
```

---

# 9. Contextual Retrieval

### Concept

A chunk is enriched with its surrounding document context before indexing.

Normal:

```text
"She was promoted in 2024."
```

Contextual:

```text
Employee: Alice
Section: Promotions

Alice was promoted in 2024.
```

### Why?

Individual chunks can lose context.

### Useful when

* documents are long
* chunks depend heavily on surrounding context
* pronouns/references occur frequently
* enterprise documents have complex structure

### Tradeoff

**Pros**

* Better chunk meaning
* Better retrieval

**Cons**

* Extra processing
* More embedding/indexing cost
* More storage
* Need to regenerate when source changes

### Decision

If:

```text
chunk alone is ambiguous
```

→ contextual retrieval is valuable.

---

# 10. Query Planning / Decomposition

### Concept

Break a complex question into smaller retrieval tasks.

```text
"What are X's security risks compared to Y?"

             ↓

      Query Planner
        ↓    ↓    ↓
       X     Y   comparison
        ↓    ↓    ↓
       retrieve
          ↓
       combine
          ↓
        answer
```

### Why?

One query may mix several information needs.

### Best for

* multi-hop questions
* comparison
* research
* analytical questions

### Tradeoffs

**Pros**

* Better coverage
* Handles complex questions

**Cons**

* More LLM calls
* More latency
* More complexity
* Bad decomposition can make retrieval worse

### Decision

```text
Simple query → one query

Complex query → decomposition
```

---

# 11. Hierarchical RAG / RAPTOR

### Concept

Instead of flat chunks:

```text
Document
 ↓
Chunks
```

Create hierarchy:

```text
Document
 ↓
Sections
 ↓
Chunks
 ↓
Summaries
```

RAPTOR-style systems recursively create summaries/clusters that can also be retrieved.

### Why?

Some questions require **high-level understanding**, not a single paragraph.

Example:

> "What are the major themes of this 500-page report?"

Retrieving one paragraph isn't enough.

### Best for

* books
* large reports
* long technical documents
* global/summary questions

### Tradeoffs

**Pros**

* Supports local + global questions
* Better hierarchical understanding

**Cons**

* More indexing cost
* More storage
* Summary quality matters
* More complexity

### Decision

```text
"What does section 4 say?"
→ normal RAG

"What are the major conclusions of this 500-page report?"
→ hierarchical RAG can help
```

---

# 12. Late Chunking

### Normal

```text
Document
 ↓
Chunk
 ↓
Embed each chunk
```

### Late chunking

Conceptually:

```text
Full document
 ↓
Long-context embedding
 ↓
Create chunk representations
```

### Why?

The embedding process gets access to broader context before representing individual chunks.

### Benefit

Potentially preserves relationships/context across chunk boundaries.

### Tradeoffs

**Pros**

* Better contextual representations
* Can reduce chunk-boundary problems

**Cons**

* Requires suitable long-context embedding model
* More computational complexity
* Not always better
* Additional experimentation required

### Decision

Treat as an **advanced optimization**, not your default RAG architecture.

---

# 🥉 TIER 3 — KNOW CONCEPTUALLY

---

# 13. CAG — Cache-Augmented Generation

### Concept

Instead of retrieving every query:

```text
Knowledge
 ↓
Load/cache
 ↓
LLM context
```

### Why?

If knowledge is:

* small
* stable
* repeatedly queried

retrieval infrastructure may be unnecessary.

### Good for

```text
Small policy document
Product manual
Stable reference dataset
```

### Bad for

```text
Millions of documents
Frequently changing data
Strong per-user ACLs
```

### Tradeoff

**Simpler retrieval**

but:

**Context/token cost increases with knowledge size.**

---

# 14. ColBERT / Late Interaction

### Concept

Traditional dense retrieval:

```text
Document → one vector
Query → one vector
```

ColBERT-style retrieval represents/query-matches **multiple token-level vectors** and performs late interaction between them.

Think:

```text
Query tokens
     ↓
many vectors

Document tokens
     ↓
many vectors

      ↓

token-level relevance
```

### Why?

A single vector can lose fine-grained information.

Late interaction preserves more detail while remaining much cheaper than a full cross-encoder over every document.

### Position

Conceptually:

```text
Dense retrieval
      ↓
ColBERT
      ↓
Cross-encoder
```

Different speed/quality tradeoffs.

### Tradeoff

* Better fine-grained matching
* More storage
* More complex indexing
* More compute than simple dense retrieval

### Know

> **ColBERT is a retrieval architecture, not simply another embedding model.**

---

# 15. SPLADE / Learned Sparse Retrieval

### Concept

Traditional BM25 uses exact terms.

SPLADE learns sparse representations that expand important terms.

Conceptually:

```text
"car"
 ↓
learned sparse representation
 ↓
related semantic terms
```

It attempts to combine some semantic understanding with sparse retrieval.

### Why?

Want:

```text
keyword-like efficiency
+
semantic expansion
```

### Tradeoffs

**Pros**

* Sparse retrieval efficiency
* Better semantic matching than pure lexical search

**Cons**

* More specialized infrastructure
* Model/index complexity
* Less common than BM25 in basic systems

### Decision

For most production systems:

```text
Start BM25
 ↓
Hybrid
 ↓
Only explore SPLADE if retrieval demands it
```

---

# 16. Knowledge Graph + Vector Hybrid

### Concept

Combine:

```text
Vector retrieval
+
Knowledge graph
```

Example:

```text
Question
 ↓
Vector search → relevant documents
 ↓
Graph → relationships
 ↓
Combine evidence
 ↓
LLM
```

### Why?

Vector search answers:

> "What text is relevant?"

Graph answers:

> "How are these entities connected?"

### Best for

* cyber threat intelligence
* fraud
* financial networks
* enterprise architecture
* scientific knowledge

### Tradeoff

Much more powerful for relational problems.

But:

```text
More power
=
More data engineering
+
More maintenance
+
More failure modes
```

---

# 17. Multimodal RAG

### Concept

Retrieve more than text.

```text
Documents
 ├── Text
 ├── Images
 ├── Tables
 ├── Charts
 └── Diagrams
```

Then:

```text
Multimodal retrieval
 ↓
Multimodal LLM
```

### Example

Question:

> "What does the architecture diagram on page 42 show?"

Text-only RAG may fail.

Multimodal RAG can retrieve:

```text
page image
+
caption
+
text
```

### Best for

* PDFs
* engineering diagrams
* medical/scientific documents
* financial reports
* presentations
* image-heavy manuals

### Tradeoffs

* More expensive
* Harder parsing
* More storage
* More complex evaluation
* Vision model dependency

---

# 18. Self / Corrective RAG

### Concept

The system evaluates its own retrieval.

```text
Query
 ↓
Retrieve
 ↓
Grade evidence
 ↓
Relevant?
 ├── YES → Generate
 └── NO
       ↓
   Rewrite query
       ↓
   Retrieve again
```

### Why?

Traditional RAG assumes:

> "Top-k retrieval is good enough."

Corrective RAG asks:

> **"Is this evidence actually sufficient?"**

### Useful for

* difficult queries
* high accuracy systems
* agentic research
* uncertain retrieval

### Tradeoffs

**Pros**

* Adaptive
* Can recover from bad retrieval

**Cons**

* More model calls
* More latency
* More cost
* Evaluator itself can be wrong

### Decision

Use when:

```text
cost of wrong answer > cost of extra retrieval
```

---

# 🧠 THE MOST IMPORTANT DECISION TABLE

| Requirement                       | Good choice              |
| --------------------------------- | ------------------------ |
| Normal document Q&A               | Basic RAG                |
| Exact keywords + semantic meaning | **Hybrid RAG**           |
| Retrieval ranking isn't great     | **Reranker**             |
| Complex multi-step question       | **Agentic RAG**          |
| Huge context but small corpus     | **Long Context**         |
| Small + stable knowledge          | **CAG**                  |
| Structured numerical data         | **SQL**                  |
| Live information                  | **API/tool**             |
| Entity relationships              | **GraphRAG**             |
| Large books/reports               | **Hierarchical RAG**     |
| Ambiguous chunks                  | **Contextual Retrieval** |
| Complex question                  | **Query decomposition**  |
| Tables/images/diagrams            | **Multimodal RAG**       |
| Retrieval needs semantic + sparse | **SPLADE / hybrid**      |
| Fine-grained retrieval            | **ColBERT**              |
| Retrieval quality uncertain       | **Corrective RAG**       |
| Relationships + semantic search   | **Graph + Vector**       |

---

# 🔥 Production Decision Flow

This is the part I would **memorize**, not individual technologies:

```text
START
  ↓
What data?
  │
  ├── Structured → SQL
  │
  ├── Live state → API / Tool
  │
  └── Unstructured
         ↓
     How large?
         │
         ├── Small → Long Context / CAG
         │
         └── Large
              ↓
        Need exact matching?
              │
              ├── YES → Hybrid
              └── NO  → Dense
                         ↓
                   Retrieval quality?
                         ↓
                      Reranker
                         ↓
                Query complexity?
                    │
          ┌─────────┴─────────┐
          ↓                   ↓
       Simple              Complex
          ↓                   ↓
       RAG              Query Planning
                              ↓
                         Agentic RAG
                              ↓
                    Multiple information sources?
                              ↓
                      Router / Tools / APIs
                              ↓
                    Relationships important?
                         │
                         ├── YES → GraphRAG
                         └── NO
                              ↓
                     Context Engineering
                              ↓
                         LLM Answer
                              ↓
                          Evaluate
```

---

# 🧠 What You Should Actually Remember

Don't memorize 18 techniques independently.

Remember these **6 architectural dimensions**:

### 1. **Where do I get information?**

```text
RAG / SQL / API / Graph / Tools
```

### 2. **How do I find it?**

```text
Dense / BM25 / Hybrid / SPLADE / ColBERT
```

### 3. **How do I select the best evidence?**

```text
Reranking
MMR
Filtering
Contextual retrieval
```

### 4. **How do I handle difficult questions?**

```text
Query rewriting
Decomposition
Planning
Agentic RAG
Corrective RAG
```

### 5. **How do I represent complex knowledge?**

```text
Chunks
Hierarchy
Knowledge Graph
Multimodal data
```

### 6. **How do I know it works?**

```text
Recall
Precision
NDCG
Faithfulness
Groundedness
Answer relevance
Latency
Cost
Security
```

---

# 🎯 Your Learning Priority

For **your AI Systems Engineer goal**, I'd learn them in this order:

```text
Production RAG
      ↓
Hybrid Search
      ↓
Reranking
      ↓
Evaluation
      ↓
Context Engineering
      ↓
Long Context vs RAG
      ↓
SQL/API/Tool Retrieval
      ↓
Agentic RAG
      ↓
Query Planning
      ↓
GraphRAG
      ↓
Contextual Retrieval
      ↓
Hierarchical RAG
      ↓
Multimodal RAG
      ↓
Advanced retrieval
(ColBERT / SPLADE / Late Chunking)
```

**The key skill isn't knowing "RAPTOR, ColBERT, SPLADE, GraphRAG..." by name.**

It's being able to sit in a system-design interview and say:

> **"Given the data, query complexity, latency, cost, freshness, security and accuracy requirements, I'd choose X instead of Y because..."**

That is the **production-level RAG skill** you're actually trying to build.
