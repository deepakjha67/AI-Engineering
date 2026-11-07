# CME 295: Transformers & LLMs - Lecture 1 Notes


**Course Goals:**
* Understand how LLMs work
* Learn the transformer architecture
* Understand training methods, capabilities, and limitations

**Prerequisites:**
* Linear algebra basics (matrices, multiplication)
* Basic ML knowledge (loss functions, neural networks, embeddings)

---

## 🕰️ How We Got Here: The Evolution of NLP

### The 2010s: Pre-Transformer Era
Before 2017, the field looked completely different:
* One model per use case (not one model for everything)
* **RNNs** (Recurrent Neural Networks) dominated
* Separate models for:
  * Sentiment analysis (positive/negative/neutral)
  * Translation
  * Named entity recognition (names, locations)
* *Performance was OK but not amazing.*

### 2017: The Turning Point
The "Attention is All You Need" paper introduced the transformer architecture. 
**Why it mattered:**
* Scalable: More data + more compute + more parameters = much better performance
* This scaling property was far superior to previous approaches

### 2022: ChatGPT
* Released in November 2022
* First time the general public interacted with chatbots at scale
* Product-level turning point

### 2026: Current State
* Coding agents used daily
* Evolution from conversational AI to agentic behavior
* Post-training advances (policy distillation, AI agents)
* New paradigms beyond token-by-token prediction (diffusion approaches)

---

## 🔠 Tokenization: The Foundation

### What is Tokenization?
**Core Problem:** Models understand numbers, not text. We need to convert text into numbers.
**Tokenization** = Dividing text into indivisible units called tokens.

*Example:* `"A cute Teddy bear is reading"`
* Tokens: `["A", "cute", "Teddy bear", "is", "reading"]`
* *(Note: "Teddy bear" can be one token!)*

Each token will be represented by a vector of numbers (embedding).

### Vocabulary
**Vocabulary** = All possible tokens you can use
* Your choice of vocabulary affects sequence length (number of tokens)
* There's a trade-off between vocabulary size and sequence length

### Three Levels of Tokenization

**1. Word-Level Tokenization**
* **How it works:** Split by spaces. Each word = one token.
* **Pros:** Very simple, interpretable (each token maps to a word).
* **Cons:** Large vocabulary (many word variations), can't leverage similar roots ("bear" and "bears" get completely different representations), Out-of-vocabulary (OOV) risk.

**2. Character-Level Tokenization**
* **How it works:** Split by individual characters. Fixed alphabet.
* **Pros:** No OOV risk, small vocabulary, robust to misspellings and casing.
* **Cons:** Much longer sequences → slower computation, model complexity grows with sequence length, less interpretable.

**3. Subword Tokenization (Most Popular)**
* **How it works:** Split into meaningful subunits learned from data (e.g., `"reading"` → `"read" + "ing"`).
* **Pros:** Leverages word roots, learned from actual data, shorter sequences than character-level.
* **Cons:** Requires extra training step, depends heavily on training data quality.

### BPE (Byte Pair Encoding)
Most common subword tokenizer.
**How it works:**
1. Start with initial vocabulary (e.g., all letters)
2. Find most frequent pairs in training data
3. Merge them into new tokens
4. Repeat until desired vocabulary size

*Example:* `"an"` appears frequently → add `"an"` as a token.
**Key insight:** Common words get short token sequences; rare words get split into subwords. 
*(Important for multilingual work: Train tokenizer on corpus representing all your languages).*

### Special Tokens
Special tokens convey structural information:

| Token | Purpose |
| :--- | :--- |
| `[UNK]` (Unknown) | Represents out-of-vocabulary text |
| `[BOS]` (Beginning of Sequence) | Signals start of generation |
| `[EOS]` (End of Sequence) | Signals end of generation |
| `[PAD]` (Padding) | Makes all sequences same length (for efficiency) |

*Modern additions:* User/assistant tokens for chatbots.
*Note: Vocabulary size for modern LLMs is usually hundreds of thousands (e.g., ~200,000).*

---

## 🔢 Representing Tokens: From One-Hot to Word2Vec

### One-Hot Encoding (The Naive Way)
Each token = a vector with one `1` and the rest `0s`.
**Problem:** All tokens are orthogonal (dot product = $0$). There is no similarity information. `"Teddy bear"` and `"soft"` are just as related as `"Teddy bear"` and `"book"`.

### Word2Vec (2013)
**Idea:** Learn meaningful representations through a proxy task (a task that helps learn good representations).
**Two versions:**
1. **Continuous Bag of Words (CBOW):** Predict middle word from context
2. **Skip-gram:** Predict surrounding words from middle word

**How Word2Vec Works (Skip-gram example):**
1. Input word "a" as one-hot vector
2. Matrix multiply → hidden state
3. Matrix multiply → output probabilities (softmax)
4. Compare with actual next word ("cute")
5. Compute loss (cross-entropy) and update weights

**The Magic:** After training, the hidden representations capture meaning! (e.g., Paris is to France, what Berlin is to Germany).

**Limitations of Word2Vec:**
1. **No context awareness:** `"bank"` (river) vs. `"bank"` (financial) get the same vector.
2. **Ignores word order:** `"The child is hugging the Teddy bear"` vs. `"The Teddy bear is hugging the child"` have the exact same embeddings.
3. **OOV problem:** Can't represent words not seen in training.
4. **Scalability:** Representation size tied to vocabulary size.

---

## 🔄 RNNs (Recurrent Neural Networks)

### The Solution to Word Order
**Key idea:** Process tokens one at a time, maintaining a hidden state that encodes "meaning of sequence seen so far".
* Step 1: Input "A" → predict "cute", update hidden state
* Step 2: Input "cute" + hidden state → predict "Teddy bear", update hidden state

**RNN Use Cases:**
1. Classification (sentiment analysis)
2. Tagging (POS tagging)
3. Generation (Predict next word, feed it back as input)

### Limitation: Long-Range Dependencies
**Problem:** Hidden state must encode everything from the past. Information from long ago gets diluted.

### LSTM (Long Short-Term Memory)
Attempted fix: Add a cell state alongside hidden state with gates to control what to remember/forget. 
*Reality:* Still had limitations with very long sequences.

### RNN Problems Summary
1. Long-range dependency issues (vanishing gradient problem)
2. Slow training (must process tokens sequentially; cannot parallelize)
3. Sequential computation (can't predict next token without processing all previous)

---

## 🎯 Attention: The Key Innovation

### The Core Idea
**Instead of:** Relying only on a hidden state to remember past tokens.
**Do this:** Allow direct connections to past tokens when making predictions.
* When predicting the next word, look directly at all previous words.
* Learn which previous words matter most for current prediction.
* Weight their contributions accordingly.

### Self-Attention: Attention on Steroids
**Self-attention:** Every token attends to every other token in the same sequence.
**Goal:** Compute representation of each token as a function of all other tokens.

### Queries, Keys, Values (Q, K, V)
These are learned projections of the input tokens:

| Term | Intuition | Role |
| :--- | :--- | :--- |
| **Query (Q)** | What am I looking for? | The token seeking information |
| **Key (K)** | What do I contain? | How each token describes itself |
| **Value (V)** | What information do I have? | The actual content to pass along |

**Process:**
1. For token "Teddy bear", create a query.
2. Compare query to all keys in the sequence (dot product).
3. High similarity with "cute" key → "cute" is relevant.
4. Take weighted sum of value vectors.
5. Result: New representation of "Teddy bear" that incorporates "cute".

### The Self-Attention Formula

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Breaking it down:**
1. $QK^T$: Compute all query-key similarities (dot products). Result is an $n \times n$ matrix.
2. $\sqrt{d_k}$: Scale by square root of key dimension to prevent values from getting too large.
3. $\text{softmax}()$: Convert to probabilities (weighted average).
4. $\times V$: Weighted sum of values for new token representations.

---

## 🏗️ The Transformer Architecture

**Why "Attention is All You Need"?**
No recurrence! Just attention + feedforward networks.

### High-Level Structure
1. **Encoder:** Computes meaningful representations of source text
2. **Decoder:** Uses encoder representations to generate translation

### Detailed Component Walkthrough

**1. Input Embedding + Positional Encoding**
* **Problem:** Self-attention doesn't know token order.
* **Solution:** Add positional information (Token embedding + Positional encoding).
* Methods: Learned (can't extrapolate) vs. Fixed sinusoidal (can extrapolate, nearby positions are similar).

**2. Encoder Block (Stacked $N$ times)**
* **a) Multi-Head Self-Attention:** Computes Q, K, V; splits into $h$ heads to learn different relationships; concatenates results.
* **b) Feedforward Neural Network:** Expands to larger dimension ($d_{ff}$), then back. Where most parameters live.
* **c) Residual Connections:** `output = f(x) + x`. Helps with training deep networks/gradients.
* **d) Layer Normalization:** Normalizes activations for convergence.

**3. Decoder Block (Stacked $N$ times)**
* **a) Masked Multi-Head Self-Attention:** Masking prevents attending to future tokens (set to $-\infty$ before softmax).
* **b) Cross-Attention:** Queries from decoder, Keys/Values from encoder output.
* **c) Feedforward / d) Residuals + Layer Norm**

**4. Output Layer**
* Linear projection to vocabulary size → Softmax → Pick next token → Feed back as input (autoregressive).

---

## ⚙️ Important Details and Tricks

### Multi-Head Attention
* Different heads can learn different types of relationships (syntax, semantics).
* Split $d_{model}$ into $h$ chunks, compute attention, concatenate, project back.

### Dimension Flow

| Component | Input Size | Output Size |
| :--- | :--- | :--- |
| **Input embedding** | token index | $(n \times d_{model})$ |
| **Q, K, V projections** | $(n \times d_{model})$ | $(n \times d_q), (n \times d_k), (n \times d_v)$ |
| **Attention output** | $(n \times d_v)$ | $(n \times d_v)$ |
| **Concatenate heads** | $h \times (n \times d_v)$ | $(n \times h \cdot d_v)$ |
| **Final projection** | $(n \times h \cdot d_v)$ | $(n \times d_{model})$ |
| **Feedforward** | $(n \times d_{model})$ | $(n \times d_{ff}) \to (n \times d_{model})$ |

*(Constraint: $d_q = d_k$ for matrix multiplication to work)*

### Training Tricks
* **Dropout:** Randomly set units to 0 during training to improve generalization.
* **Label Smoothing:** Soften the targets (e.g., target word = 90%, others = 10%). Reflects that language has multiple valid completions.

### What Makes Transformers Work?
1. **Attention layers:** Allow tokens to interact directly (capture relationships regardless of distance).
2. **Feedforward networks:** Contain most of the parameters (where actual "learning" happens).

---

## 📝 Summary: From 2010s to Transformers

| Era | Approach | Key Limitation |
| :--- | :--- | :--- |
| **2010s** | RNNs | Sequential, slow, forgets long-range info |
| **2013** | Word2Vec | No context awareness |
| **2017** | Transformers | (Solved many issues, but still evolving) |

**Key Takeaways:**
1. **Tokenization** converts text to numbers; subword (BPE) is most popular.
2. **Embeddings** represent tokens as vectors capturing meaning.
3. **Self-attention** lets tokens interact directly, solving long-range dependency issues.
4. **Transformers** = stacked encoder/decoder blocks with attention + feedforward layers.
5. Training tricks (residuals, layer norm, dropout, label smoothing) make it work.
6. The field is rapidly evolving; this is just the foundation!
