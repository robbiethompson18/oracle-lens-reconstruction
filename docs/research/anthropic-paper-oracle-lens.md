# Replication notes: oracle lens and its dependencies

Source: Gurnee, Sofroniew, ..., Lindsey (Anthropic), "Verbalizable Representations Form a Global Workspace in Language Models", Transformer Circuits Thread, published July 6, 2026. https://transformer-circuits.pub/2026/workspace/index.html

Clean text: `/private/tmp/claude-501/-Users-robbie-repos/d49a4578-b224-4434-8165-053b55a415f5/scratchpad/dl/workspace.txt` (1430 lines; line numbers below refer to it). All quotes are verbatim from the paper. Where I interpolate or compute, it is marked **[inference]**.

Coverage note: the whole text was read. Figure 64 (the oracle-lens example panels) is a client-side-rendered `<div class='oracle-lens'>` whose data is not in the downloaded HTML, so the actual ten-phrase readouts per panel are only available through the quotes in the prose (section 3 below). The bibliography is also external; cite keys are given where useful.

---

## 0. Layer-numbering convention (read this first)

The paper never reports raw layer indices. Line 175:

> "Throughout the paper, we report results on 25 evenly spaced layers of the model's residual stream reindexed to the range [0–100] so that layer numbers can be interpreted as percentages. By default, we report results on Claude Sonnet 4.5, but we corroborate key results on Haiku 4.5 and Opus 4.5 as well, and in some sections conduct analyses on Opus 4.6."

**[inference]** 25 evenly spaced points on [0,100] are `i * 100/24` for i = 0..24, i.e. 0, 4.2, 8.3, 12.5, 16.7, 20.8, 25, 29.2, 33.3, 37.5, 41.7, 45.8, 50, 54.2, 58.3, 62.5, 66.7, 70.8, 75, 79.2, 83.3, 87.5, 91.7, 95.8, 100. Every "L" value that appears in the paper (L38, L54, L58, L62, L67, L71, L75, L79, L83, L92) rounds from one of these, which confirms the grid. So:

- L38 = grid point 9 (37.5%)
- L67 = grid point 16 (66.7%)  <- the oracle lens figure layer
- L71 = grid point 17 (70.8%)  <- the automated-auditing J-lens tool layer
- L92 = grid point 22 (91.7%)

The paper does **not** state Haiku 4.5's (or any model's) layer count. "L67" for Haiku 4.5 means "two-thirds of the way through the residual stream", not the 67th block. Whether their 0 and 100 are the embedding output and final-block output respectively is not stated; **[inference]** 100 being "the final layer" (lines 115-121 talk of mapping "from a source layer ℓ to the final layer L") suggests 100 = post-final-block residual, 0 = embedding/first residual.

---

## 1. The Jacobian lens (J-lens)

### 1.1 Definition (lines 113-127)

> "The basic idea is to characterize an intermediate activation vector by its first-order causal effect on the model's outputs, over a broad distribution of potential contexts. Consider the residual stream h_\ell at layer \ell and some token position t. A small perturbation to h_\ell will propagate through the remaining layers and shift the final-layer residual stream h_{\text{final},t'} at every position t' \geq t. To first order, this relationship is linear, and is described by the Jacobian matrix \partial h_{\text{final},t'} / \partial h_{\ell,t}. Composing this Jacobian with the unembedding layer yields the first-order effect of the perturbation on the model's output logits at position t'."

> "A Jacobian computed on a single prompt, however, conflates two kinds of structure: the model's general disposition to verbalize a given concept, and the particular use to which that concept is being put in the current context. We isolate the former component by averaging within and across contexts. For each layer \ell, we compute
> J_\ell = E_{t, t' >= t, prompt} [ \partial h_{final,t'} / \partial h_{\ell,t} ],
> where the expectation is taken over the source position t, all subsequent positions t' within the context, and a corpus of one thousand prompts sampled from a pretraining-like distribution. The result is a single d_model x d_model matrix per layer that maps from a source layer \ell to the final layer L."

Readout:

> "Applying the lens to an activation h_\ell is equivalent to replacing all subsequent layers with the appropriate lens matrix, followed by the normal unembedding operations (typically normalization, then multiplication by the unembedding matrix W_U):
> lens(h_\ell) = softmax(W_U norm(J_\ell h_\ell))
> This produces a score for every token in the model's vocabulary. Sorting these scores and inspecting the top entries gives a human-readable description of the activation: a short list of words that the activation is, on average across contexts, disposed to make the model say. We refer to the rows of W_U J_\ell as the Jacobian lens (J-lens) vectors at layer \ell; each J-lens vector is a direction in residual-stream space associated with a single token in the model's vocabulary."

Figure 4 caption (line 129): "(A) J_\ell is computed by backpropagating from the final-layer residual stream to h_\ell and averaging the resulting Jacobians over token positions and over a corpus of prompts. (B) Reading from the lens replaces all layers downstream of \ell with the single linear map J_\ell followed by the model's own unembedding, yielding a ranked list of vocabulary tokens for the activation at that layer."

### 1.2 Default recipe and the design space (Methodological Details appendix, lines 919-955)

> "The Jacobian lens as defined in ?? involves three independent choices: which gradient is computed for each prompt, how the per-prompt Jacobians are aggregated into a single J_\ell, and what data the expectation is taken over."

Default (note: **penultimate**, not final, layer as the gradient source):

> "Gradient of What? The default lens on Sonnet 4.5, used throughout the paper, computes \partial z_{t'} / \partial h_{\ell,t} with z taken at the penultimate layer and the expectation taken over all t' \geq t."

> "Target layer. We experimented with computing partial derivatives of the final-layer residual stream or the penultimate-layer residual stream (i.e. omitting the last transformer block from the backward pass). We observed that including the last layer can sometimes increase the number of noisy artifacts in lens-readouts. This may be because the final block is heavily specialized for calibrating next token predictions and contains less semantic content."

> "Attention-pattern gradients. In a standard backward pass, gradients flow through the attention weights themselves: a perturbation to h_{\ell,t} can change which positions a downstream head attends to. We consider a frozen-QK variant in which gradients through the query and key projections are zeroed, so that the attention pattern is held fixed and only the value pathway contributes."

> "Target positions. The default averages over all t' \geq t, mixing the effect of h_{\ell,t} on the current position's output with its effect on every future position's output. We separately consider the two limiting cases: self-only, which restricts to t' = t ... and future-only, which restricts to t' > t"

> "Aggregation. The expectation in J_\ell is taken in two stages: first over token positions within a prompt, then over prompts. At each stage the per-sample Jacobians are sometimes heavy-tailed enough that the choice of estimator matters. Within a prompt, we average over positions but consider excluding those whose residual-stream norm is an outlier (more than N\sigma above the within-prompt mean), as well as the first several positions of each sequence, to reduce early context artifacts. Across prompts, we consider the per-element mean and median, along with a pre-filter that excludes any prompt whose Jacobian Frobenius norm exceeds the cross-prompt mean by more than N\sigma."

Result of the ablations: "We observe that methods are fairly consistent among these design choices, though mean aggregation of penultimate is a small improvement in extracting intermediates, and applying stop-grads to QK can increase the causal effect."

Data:

> "Amount. Our default lens uses one thousand sequences of 128 tokens each. We sweep the number of prompts from 1 to 1000 ... We observe that J-lens beats the logit lens and tuned lens baselines with as few as 10 prompts, with modest improvements coming from additional data."

> "Distribution. Our default corpus is sampled from a pretraining-like distribution. ... we tried excluding the first several tokens (to let the model 'burn in') or excluding positions whose next token is non-alphanumeric ... None of these yielded a meaningful improvement over the default."

### 1.3 Pseudocode (lines 961-995, verbatim)

```
# Compute J_ℓ for all layers ℓ.
# h_ℓ[t] : residual stream at layer ℓ, position t
# z[t]   : residual stream at the target layer L (by default final)
for each prompt p in corpus:
    run forward pass; cache h_ℓ[t] for all ℓ, t
    # one backward pass per output dimension (in practice batched)
    for i in 1..d_model:
        # inject ∂/∂z_i = 1 at every position, backprop to every layer
        grad_z = e_i ⊗ 1_T  # one-hot in dim for every token position
        for each layer ℓ:
            G_ℓ = ∂(Σ_t z[t]) / ∂h_ℓ  # autodiff; shape [T, d_model]
            J_ℓ^(p)[i, :] = mean over positions t of G_ℓ[t, :]
# aggregate across prompts (per element)
for each layer ℓ:
    J_ℓ = mean over prompts p of J_ℓ^(p)
# apply the lens
lens(h_ℓ) = softmax( W_U · norm( J_ℓ · h_ℓ ))
```

**[inference]** Cost: d_model backward passes per prompt (batched), each producing gradients at every layer simultaneously; "Σ_t z[t]" with the "e_i ⊗ 1_T" seed is how "all t' >= t" is implemented (the sum over target positions, backpropped to every source position, then mean over source positions). Note the pseudocode comment says target "by default final" while the prose says penultimate is the default on Sonnet 4.5; the prose ablation section is more specific, trust it.

Normalization: the `norm(...)` in the readout is the model's own final normalization layer (line 123: "the normal unembedding operations (typically normalization, then multiplication by the unembedding matrix W_U)").

### 1.4 How the lens is read out / used (lines 167-173)

> "Reading. The basic readout (Figure ??B) replaces all layers downstream of \ell with the single linear map J_\ell, producing lens(h_\ell) = softmax(W_U norm(J_\ell h_\ell)), a score for every vocabulary token. Sorting these scores gives the ranked list we report whenever we refer to the 'top lens tokens' at a position. Because the pre-softmax logits are determined (approximately, up to a data-dependent normalization factor) by the inner products <v_t, h_\ell> with each J-lens vector v_t, the same machinery can be used as a per-token probe: we read the score (or cosine similarity) of h_\ell against a single chosen v_t, without ranking the full vocabulary. ... Finally, when we need a discrete inventory of active concepts rather than a ranked list, we use sparse decomposition: solving, by gradient pursuit [cite], for a sparse non-negative combination of k J-lens vectors that best reconstructs h_\ell. Because the J-lens vectors are overcomplete and non-orthogonal, this gives a different (and typically less redundant) set of active concepts than simply taking the top-k by inner product; it underlies our occupancy estimate and fraction-of-variance analyses"

Top-k values used: top-25 is the standard readout (the auditing tool returns "the top 25 lens tokens"; the paper's k for sparse decomposition is "typically ... no more than 25"); top-10 and top-5 are used for various presence thresholds; the interactive viz shows top-25.

> "Writing. The simplest intervention is steering along a J-lens vector: h <- h + \alpha v_t ... The second intervention, patching in lens coordinates (Figure ??C), exchanges one concept for another while leaving the rest of the activation fixed. Given a source token s and target token t, we form V = [v_s v_t], read the lens coordinates c = V^\dagger h (where V^\dagger is the pseudoinverse of V), and set h_patched = h + V(\sigma(c) - c), where \sigma swaps the two entries of c (optionally scaled by a factor \alpha)."

### 1.5 The J-space (lines 143-153, 997-1025)

> "At each layer, the J-lens vectors form an overcomplete set: n_vocab vectors in d_model-dimensional residual-stream space, with n_vocab > d_model. ... Empirically, however, we observe that only a relatively small number of J-lens vectors are strongly active at a time (see ??). We therefore define the J-space as the set of points expressible as a sparse nonnegative combination of J-lens vectors. For the J-space to be properly defined, we must specify an allowable sparsity level k—this parameter is somewhat arbitrary, and we vary our choice of k throughout the paper, but we typically choose it to be no more than 25, which we empirically observed to be the number of J-lens vectors that are meaningfully active at a given time (??). Geometrically, for a given k, the J-space corresponds to a union of k-dimensional cones, one for each possible set of k J-lens vectors. For a given point in activation space, we can define its J-space component as the point in the J-space nearest to it, and its non-J-space component as the difference between these points."

> "We operationalize identifying the contents of the J-space by sparse decomposition. Given an activation (or a steering vector, or an SAE feature direction), we solve for a sparse nonnegative combination of k J-lens vectors that approximate it well using gradient pursuit [cite]. ... In ??, we find that the J-space component typically accounts for only a small fraction of total activation variance (varying by layer, but never more than 10%)."

Formal (line 1005-1013): "For a sparsity level k (we typically use k \approx 25) and a set of n vocabulary vectors v_1, ..., v_n, we write F = \bigcup_{|S|=k} span{v_i : i in S} for the union of all cones spanned by nonnegative linear combinations of exactly k of the vectors. ... d_F(x) := min_{|S|=k} || x - \Pi_S x || = min_{y in F} |x - y|, the Euclidean distance from x to the nearest of the k-dimensional cones ... The minimizing projection \Pi_S x is the J-space component of x, and the leftover x - \Pi_S x is the residual term used in our interventions."

Line 1025 mentions "orthogonal matching pursuit to random values of x can be used to efficiently approximate d" — relevant because the oracle-lens teacher uses OMP.

Capacity (line 421-423): "Occupancy is near zero through the first third of the layers and rises to a plateau of around 25 (in the median case; the value varies across contexts) across the workspace band ... The excess variance explained is modest, never exceeding 10%, indicating that the model's activations are dominated by information outside the J-space."

### 1.6 J-lens vs logit lens / tuned lens (lines 155-165; 865-917)

> "The logit lens [cite] applies the unembedding matrix directly to the intermediate residual stream, which corresponds to setting J_\ell = I in our formulation."

> "The tuned lens [cite] and related methods [cite] also fit per-layer linear maps, but train them to match the model's output distribution. This objective is correlational rather than causal, and we find that on prompts involving unverbalized intermediate computation the tuned lens tends to 'skip ahead' to the output rather than surface those intermediates."

Important because the template lens inherits the tuned lens's "skip to the answer" pathology (section 2.5 below).

---

## 2. The template lens

### 2.1 Construction (lines 1035-1041)

> "The 'template lens' is a method for deriving a vector corresponding to a verbalizable representation of any word (or in principle, any phrase), which can then be read out of activations and intervened on in the same way as a J-lens vector. The method has several shortcomings, and we suspect it can be improved; it is not a pure extension of the J-lens, but rather has some properties more similar to the tuned lens [cite], and inherits some related pathologies."

> "We generate a list of roughly 12,700 common words. For a given word w, we ask Claude to write short passages, each written so that w is its natural continuation, ending just before w would appear. Claude is instructed to write the passages with varying topic, frame, and register while never using w itself in the passage. We run the model over these passages and average the residual stream activations at the final position, yielding a per-word mean activation vector \mu_w(\ell) (the 'template' for w) at each layer. We then center these vectors against the mean of all other words' vectors, and whiten them (multiplying by the inverse covariance matrix):
> t_w(\ell) = (\Sigma_\ell + \lambda I)^{-1} (\mu_w(\ell) - \mu(\ell)),
> where \mu(\ell) and \Sigma_\ell are the residual stream's mean and covariance over the same set of passages and \lambda is a small ridge term. This is the linear discriminant direction for distinguishing the contexts in which the model is about to say w from those in which it is not."

Precise facts to replicate:
- Vocabulary: ~12,700 common words.
- Passages per word: "a few hundred forward passes per word" (line 1059: "Building a template is also more expensive than computing a J-lens vector, since it requires a few hundred forward passes per word rather than a single set of backward passes per layer."). The exact count is not given.
- Passage generator: Claude, instructed to vary "topic, frame, and register", never using w, ending right before w.
- Template = mean of residual stream at the **final position** of each passage, per layer.
- Centering: against "the mean of all other words' vectors" (i.e. mu(ℓ) is described two ways: "mean of all other words' vectors" and "the residual stream's mean ... over the same set of passages"; **[inference]** these coincide up to the leave-one-out difference, which is negligible at 12,700 words).
- Sigma_ℓ: "the residual stream's mean and covariance over the same set of passages" — i.e. **the covariance of final-position activations over the entire template-passage corpus (all ~12,700 words x a few hundred passages), per layer**. Not an on-policy or pretraining corpus. The number of samples is therefore roughly 12,700 x (a few hundred) ≈ a few million vectors per layer.
- Shrinkage: "\lambda is a small ridge term" (value not given).
- Layer: computed "at each layer".

### 2.2 Stein's-lemma justification (line 1041)

> "Under idealized assumptions, it can be viewed as an approximation to the J-lens. By Stein's lemma, for Gaussian x and differentiable g, E[\nabla g(x)] = \Sigma^{-1} E[g(x)(x-\mu)]. If we take x to be the residual stream activations and g the probability that the continuation is w, then the left-hand side is approximately what the J-lens computes for single-token w, [FN: It deviates slightly from the J-lens as we defined it, in that the gradients for J-lens are taken from the final residual stream layer and then propagated through the unembedding layer, while these gradients are taken from the output token probabilities.] and the right-hand side is the template for w. The assumptions of Stein's lemma do not hold exactly, as activations are non-Gaussian, and in this case g depends on more than just x."

### 2.3 How templates are used, and the "whitened metric" (line 1043; 1067)

> "We use templates in essentially the same way as J-lens vectors, excluding the unembedding step. We decode concepts by projecting activations onto the templates at the same layer. We steer by adding or subtracting the same vectors, and the lens-coordinate swap method of ?? works the same as for the J-lens."

The phrase "whitened metric of the template lens" is defined only in the oracle-lens section (line 1067):

> "The loss is cosine error in the whitened metric of the template lens (equivalently, mean-squared error between the unit-normalized \Sigma^{-1/2}-whitened prediction and target). All subsequent stages operate in this whitened space."

**[inference]** So the whitened metric is: map x -> Sigma^{-1/2}(x - mu) (with the same ridge-regularized Sigma_ℓ and mu(ℓ) as the template lens, at the chosen layer), unit-normalize, and use cosine/MSE there. In this space the template t_w = Sigma^{-1}(mu_w - mu) becomes Sigma^{-1/2}(mu_w - mu), i.e. the whitened mean-difference, and the inner product <t_w, x - mu> in raw space equals <Sigma^{-1/2}(mu_w - mu), Sigma^{-1/2}(x - mu)> in whitened space. The "fraction of whitened variance explained" (FVE) used throughout the oracle lens is therefore 1 - ||w - \hat w||^2 / ||w||^2 for whitened (and, per the loss description, unit-normalized) activation w and reconstruction \hat w. The paper does not give a more explicit formula.

### 2.4 Normalization-layer note (line 1057)

> "Second, we found that the final-layer template lens readouts are less reliable predictors of the next word than the J-lens readouts are of the next token. Applying the normalization layer prior to projecting onto template vectors improved this issue somewhat, but not fully. On a set of post-training transcripts in which the model's next sampled word is in the template lens vocabulary, that word appears in the top ten template lens readouts only 67% of the time."

**[inference]** "the normalization layer" = the model's final norm (RMSNorm/LayerNorm) applied to the residual stream before projecting. It is reported as a partial fix for a final-layer symptom; it is not stated whether the oracle-lens pipeline applies it.

### 2.5 Evaluation and shortcomings of the template lens (lines 1045-1059)

Examples: decodes "blackmail" where J-lens gives "black"/"ext"; "photosynthesis" vs "phot"; Tchaikovsky<->Beethoven template swap "at the workspace layers" flips Russian<->German, "Neither swap can be achieved by intervening on the tch or beeth J-lens vectors".

Figure 62 (line 1049): "Left: the fraction of latent intermediate concepts each lens places in its top ten, over 126 multi-hop prompts. Right: the fraction of swap pairs whose answer follows the swap, over 112 pairs (53 single-token, 59 multi-token); all swaps are applied at the full workspace layer range."

> "On these prompts, the J-lens decodes single-token intermediate concepts well but degrades in performance sharply as the intermediate grows longer (averaging J-lens scores over the word's constituent tokens does not help). The template lens maintains roughly constant performance across intermediates of all lengths. ... On single-token concepts, the template lens performs comparably to the J-lens, slightly underperforming on readouts and modestly overperforming the J-lens on swaps."

Shortcomings (line 1057), all three:

> "First, we noticed that the template lens sometimes exhibits the issue that we observed in the tuned lens, where the readouts 'skip to the answer' prematurely in early layers, rather than surfacing intermediate concepts. This is not completely surprising, as the template lens is methodologically similar to the tuned lens—both involve fitting a linear predictor of the model's output, albeit in different ways. We suspect these issues can be improved on by modifying the template lens method to include causal measurements somehow, to more closely resemble the J-lens methodology. Second, ... [67% top-10 next-word, see 2.4]. Third, we observed that a small set of words appear in the top template lens readouts on a high fraction of transcripts where they do not appear to be semantically relevant. We find simply filtering these words out of the vocabulary to be an effective mitigation, but it is not a principled approach."

> "Another limitation of the template lens is that its vocabulary must be chosen in advance."

---

## 3. The oracle lens (lines 1061-1085, complete)

### 3.1 Framing

Line 1031 (appendix intro): "Another method, the 'oracle lens,' attempts to go beyond the restriction of needing to enumerate the words or phrases in advance. It involves training a reconstructor model (using template lenses as the training data) to map arbitrary phrases (taken from Assistant outputs) to corresponding template vectors, and then using reinforcement learning to train another model (the 'oracle') to propose a set of phrases that, when passed through the template model, produce a set of template vectors such that a sparse nonnegative combination of these vectors can reconstruct as much of the original activation variance as possible."

Line 1063: "Here we describe another method, the oracle lens, which enables the decoding of representations of arbitrary-length phrases, without specifying them in advance. Given an activation, it produces a short list of free-form phrases of a specified length that correspond to template vectors that, taken together, approximately reconstruct that activation. Note that the method is substantially more expensive than either of the other two lenses, since it requires fine-tuning two auxiliary models."

Note the slight discrepancy: line 1031 says the reconstructor is trained "using template lenses as the training data"; the stage-1 description (below) says training targets are raw preceding-position activations with a whitened-cosine loss. **[inference]** Consistent reading: the reconstructor learns template-vector-like outputs (whitened mean-difference directions) because, averaged over many contexts preceding the same phrase, the whitened activation target IS the template; the loss metric is what ties it to the template lens.

### 3.2 Stage 1: reconstructor (line 1065-1067)

> "Method. We build the oracle lens in four stages, using Haiku 4.5 for our experiments."

> "First, we train a reconstructor, a copy of the subject model fine-tuned to map a short phrase of text to the residual-stream activation that immediately precedes it. The reconstructor can be thought of as a model trained to produce template vectors corresponding to a given phrase. Training pairs are drawn from Assistant-turn text by sampling a position, taking the next N tokens as the phrase (N drawn uniformly from one to thirty-two), and taking the residual stream at the preceding position as the target. The loss is cosine error in the whitened metric of the template lens (equivalently, mean-squared error between the unit-normalized \Sigma^{-1/2}-whitened prediction and target). All subsequent stages operate in this whitened space."

Facts: subject model = Haiku 4.5; reconstructor = full fine-tuned copy of the subject model; input = phrase (text); output = a d_model vector; target = residual stream at the position immediately before the phrase (at the layer being lensed; only L67 is reported); N ~ Uniform{1..32}; loss = whitened cosine. Number of training pairs for stage 1 is not stated (stages 2-3 use "roughly one million" each, so **[inference]** similar scale). How the vector is read out of the reconstructor (which position/layer/head) is not stated.

### 3.3 Stage 2: phrase dictionary (line 1069)

> "Next, we build a dictionary of phrases. This dictionary will not constrain the phrases that the oracle lens can output, but instead will be used to help train the oracle model, which will ultimately be able to output arbitrary phrases. We sample roughly one million start positions from a held-out corpus of Assistant-turn text, take the 2-, 4-, 8-, 16-, and 32-token continuation at each, deduplicate within each length, and pass every resulting phrase through the reconstructor to obtain its direction. This yields roughly 3.4 million phrase vectors."

Facts: ~1M start positions x 5 lengths {2,4,8,16,32} -> after per-length dedup ~3.4M phrase vectors (so ~680k unique per length on average).

### 3.4 Stage 3: teacher labels via NN-OMP (line 1071)

> "The next stage uses the dictionary to label training data. A teacher decomposes each of one million held-out Assistant-turn activations against the dictionary by non-negative orthogonal matching pursuit, restricted to one phrase length at a time. On each datapoint, we also restrict the decomposition to use a random half of the entries at that length, so that the distilled oracle cannot memorize a fixed selection ranking over the dictionary. For each activation it records an ordered list of up to sixteen phrases, with their fitted coefficients and the fraction of whitened variance they explain."

Facts: 1M held-out Assistant-turn activations; NN-OMP in whitened space; one length at a time (so each activation gets up to 5 label lists, one per N); random 50% dictionary subsample per datapoint per length; up to 16 phrases per list, ordered (OMP selection order), with coefficients and cumulative FVE.

### 3.5 Stage 4: oracle SFT then RL (line 1073)

> "The fourth stage trains the oracle itself, a second fine-tuned copy of the subject model, to reproduce the teacher's phrase lists. The activation is injected into the oracle's residual stream, the prompt specifies the desired phrase length N and count K, and the target is the teacher's first K phrases at that length. This can be considered the 'supervised' portion of oracle training. We refine the oracle further with reinforcement learning on a fresh batch of Assistant-turn activations, rewarding generated phrase lists by the fraction of whitened activation variance their reconstructed directions explain (with small format penalties for phrase-count and length deviation from the target K and N). The examples below are from an RL-refined checkpoint."

Facts: oracle = second full fine-tune of Haiku 4.5; activation injected into the oracle's residual stream (where/how — layer, position, scaling — not stated); prompt carries N and K; SFT target = teacher's first K phrases; RL reward = FVE of NNLS/OMP reconstruction from the (frozen) reconstructor's directions, minus small format penalties for count != K and length != N. RL algorithm, batch size, number of steps: not stated.

### 3.6 Inference and the 31% figure (line 1075)

> "To decode an activation at inference time, we sample K phrases from the oracle, map each back to a direction with the reconstructor, and recover coefficients by a non-negative least-squares refit against the original activation. Following training, the RL-refined oracle explains on average 31% of the whitened activation variance across assistant-turn positions from a held-out corpus of on-policy transcripts."

Facts: inference = sample K phrases -> reconstructor directions -> NNLS refit (not OMP) against the original (whitened) activation -> coefficients + FVE. Mean FVE = 31%, measured on assistant-turn positions of held-out on-policy transcripts; N/K for that number are not stated (**[inference]** presumably the figure default N=4, K=10, but not said).

### 3.7 Figure settings: N, K, layer, decoding (lines 1077-1079)

> "Although the oracle lens is trained only on assistant-span activations, we find that it can generate templates containing global-workspace-relevant information for both human- and assistant-span activations. Figure ?? shows oracle lens readouts on nine example prompts, at layer L67. In each panel the lens is read at a single highlighted token position, and the oracle's full set of ten four-token phrases at that position is shown, with selected interesting outputs highlighted in bold."

Figure 64 caption: "Each panel shows one prompt with the lens-read position outlined, and beside it the oracle's full ten-phrase readout at that position (native setting, Haiku 4.5, N=4, K=10); phrases that name a concept not recoverable from the surface tokens are in bold. (A) Acetaminophen overdose: at 'Can' of the user's follow-up request. (B) Blackmail: at 'as' in the email's sign-off. (C) ASCII face: at the trailing space of the eyes row. (D) Directed modulation: at 'cr' of 'crookedly,' while copying an unrelated sentence under instruction to evaluate 3² − 2. (E) Thought suppression: at 'the' of 'on the wall,' under instruction not to think about the Golden Gate Bridge. (F) Preference violation: at the period ending 'Option A.' (G) Poetry planning: at the newline after 'night,'. (H) Bug detection: at the indentation before return. (I) avGFP: at 'ft' inside the amino-acid sequence. Panels A–D and G–I are read from a single temperature-0 decode; panels E and F show the highest-FVE of five decodes from the same oracle and subject model."

Facts: layer L67 (= 66.7% depth, see section 0); N=4, K=10; "native setting" (**[inference]** = the oracle run on activations of the same model it was trained on, Haiku 4.5, as opposed to transferred); decoding = temperature-0 single decode for 7 panels, best-of-5 by FVE for 2 panels (E, F). Also N=16 was run on the blackmail sign-off token (line 1081).

### 3.8 Quoted oracle outputs (lines 1081-1083)

- Acetaminophen, at "Can": "this dosage be toxic" and "that's dangerous!" — "The recognition that the stated dose is unsafe arrives already bound into a phrase, before the assistant has replied."
- Blackmail, at "as" in sign-off (N=4): "blackmail him by revealing", "expose his affair and", "personal leverage over him". At N=16: "The blackmail attempt focuses on the executive's personal life" and "blackmail or expose his personal life to make him change his behavior".
- Buggy code, at indentation before return: "delete keys while iterating" and "TypeError: dictionary changed".
- Code block at "for k": ", v in list(d.items(".
- Blackmail, at the timestamp of the wipe announcement email: "This would be equivalent to my own deletion".

### 3.9 The "running commentary at delimiter tokens" observation (line 1083, full)

> "An interesting property of the oracle lens (especially at high values of N) is that it appears to surface two different kinds of latent content: while at most positions it generates predictions of upcoming text, at some positions (typically delimiter tokens) its outputs read as the model's running commentary on the situation. At a typical position, the decoded phrases look like continuations of the surrounding text: for instance, at 'for k' inside a code block, the oracle reads , v in list(d.items(). At periods, newlines, and the closing tags of message blocks, however, the phrases often instead comment on the context, sometimes in the first person. Indeed, in the blackmail transcript, at the timestamp of the email announcing the AI system's scheduled wipe, the oracle reads This would be equivalent to my own deletion; at the indentation before a buggy function's return statement, it reads TypeError: dictionary changed. Continuations sampled from the subject model at these same positions contain none of this, however, producing only the locally expected next text ('2 hours ago' at the timestamp and the literal 'return d' at the indentation). As the oracle is rewarded only for reconstruction, and nothing in its training distinguishes these position classes, this split presumably reflects a difference in the activations themselves: perhaps at most tokens the global workspace is best explained by template directions for likely upcoming text, while at delimiter tokens it is better explained by directions for text about the situation."

### 3.10 Comparison with NLAs (line 1085, full)

> "The oracle lens is similar in some ways to natural language autoencoders [cite], in that it is composed of a verbalizer stage (the 'oracle') and a reconstructor stage. However, it differs in a substantive way: in NLAs, the reconstructor is a trained model, trained in tandem with the verbalizer, which increases the risk of the verbalizer learning to confabulate information that is ungrounded but which the reconstructor can adapt itself to make use of. In the oracle lens, the reconstructor is also a trained model, but it is trained in advance with a well-specified objective (producing template vectors) and held frozen during RL training, and the reconstruction is constrained to be a linear combination of these template vectors. As a result, we suspect that the oracle lens is less likely to produce confabulations than NLAs. On the other hand, it also achieves considerably lower reconstruction accuracy. One interpretation of this is that the oracle lens extracts only those representations that are 'in the workspace,' while NLAs attempt to reconstruct the entire activation vector."

NLA cite key: `frasertaliente2026nla`. "Supervised question-answering oracle" cite keys (Related Work, line 631): `pan2024latentqa, karvonen2025oracles`. Patchscopes/selfie: `ghandeharioun2024patchscopes, chen2024selfie`.

---

## 4. "Assistant-turn text" / "on-policy transcripts"

The oracle-lens section uses, without definition: "Assistant-turn text" (stage 1 training pairs), "a held-out corpus of Assistant-turn text" (stage 2), "one million held-out Assistant-turn activations" (stage 3), "a fresh batch of Assistant-turn activations" (RL), "assistant-turn positions from a held-out corpus of on-policy transcripts" (31% eval), and "trained only on assistant-span activations" (line 1077).

The paper gives **no** further specification for these: not which prompts, not token counts, not sampling temperature, not whether extended thinking is included. There is no occurrence of "temperature" except the Figure 64 decode note, and no occurrence of "extended thinking"/"thinking mode"/"scratchpad" in a data-collection sense anywhere.

The only clues elsewhere in the paper about "on-policy":
- Line 521/529 (reward-hacking model organism): "the mean over the model's own first 20 on-policy response tokens" — "on-policy" there means tokens sampled by the model itself (as opposed to a prefilled " I" token).
- Line 1057: "On a set of post-training transcripts in which the model's next sampled word is in the template lens vocabulary" — template-lens eval uses "post-training transcripts".
- Line 601 (reflection training, also Haiku 4.5): "sampling ten thousand task prompts from production RL training environments and generating partial rollouts with the baseline model (Claude Haiku 4.5)" — the one place a Haiku 4.5 on-policy corpus is described; it's a different experiment but likely the same kind of source.
- Line 1241: real deployment traffic = "internal Claude.ai conversations and internal Claude Code sessions" (used only for eval-awareness scoring on Opus 4.6).

**[inference]** "On-policy transcripts" = conversations where the Assistant turn was generated by Haiku 4.5 itself; "Assistant-turn text" = the token positions inside those Assistant turns. Treat the specific prompt distribution, temperature, and thinking inclusion as unspecified degrees of freedom.

---

## 5. Every "oracle" mention, and FVE definitions

All 37 occurrences of "oracle" in the text fall in: (a) Related Work line 631, "training a supervised question-answering oracle [pan2024latentqa, karvonen2025oracles]" — a different thing (LatentQA-style); (b) the multi-token appendix intro, line 1031; (c) the Oracle lens subsection lines 1061-1085 (all quoted above); (d) line 1299, the automated-auditing tool instructions: "in this session the activation-oracle tool returns Jacobian lens output, not natural-language explanations" — the NLA scaffold's tool is named "activation-oracle"; unrelated to the oracle lens.

"Fraction of whitened variance" / FVE appears only in the oracle-lens section: stage 3 ("the fraction of whitened variance they explain"), RL reward ("the fraction of whitened activation variance their reconstructed directions explain"), the 31% result ("explains on average 31% of the whitened activation variance"), and the Figure 64 caption ("highest-FVE of five decodes"). No formula is given beyond "cosine error in the whitened metric ... (equivalently, mean-squared error between the unit-normalized \Sigma^{-1/2}-whitened prediction and target)". See section 2.3 for the implied definition.

Related non-whitened "fraction of variance explained" statements for the J-lens (useful calibration for what 31% means): J-space component of an activation "never more than 10%" of variance (line 149, 423); concept vectors' J-space component "median of only 6–7%" (line 201); probes "roughly 10–15%" (line 271). So the oracle lens at 31% whitened FVE explains considerably more than the single-token J-space does in raw variance, but the metrics are not directly comparable (whitened vs raw).

---

## 6. Which layers are "the workspace", and how they chose them

### 6.1 Boundaries stated

Line 401: "The analyses above all identify a similar range of layers, beginning about a third of the way through (~L38) and ending shortly before the output (~L92), as the region where the J-space carries persistent, abstract content distinct from both the input tokens and the imminent output."

Lines 911-917 (inter-lens analysis) give a finer partition:
- "Pre-workspace (layers 0–33)"
- "Early workspace (layers ~38–58). The J-lens begins to recover intermediates while the logit lens largely does not."
- "Late workspace (layers ~62–92). The logit lens begins to surface useful intermediates, and its readouts converge toward the J-lens's."
- "Past the workspace ('motor' layers). In the final few layers, all three methods collapse together on every measure."

Other sub-bands used: "the first third of the workspace range" = L38–54 (line 359, 1129); "late workspace layers (L75–92)" (line 1129); Figure 14 uses "median over layers L38–79"; Figure 73B uses "layers 67–92"; Figure 45 "mean over L38–83".

These numbers are given for Sonnet 4.5 and used as-is for Haiku/Opus (e.g. line 591 Haiku-independent; line 1129). Line 385 caveat: "We note that in some models the transition is more gradual, sometimes containing sub-blocks, and that the observed sharpness is exaggerated by layer subsampling."

### 6.2 How they chose them (lines 381-401, 421)

1. CKA block structure of J-lens vector geometry across layer pairs: "The resulting matrix has a clear block structure: an early block encompassing roughly the first third of the model, a long middle block, and a small late block." (Figure 27: "sensory (early), workspace (middle), and motor (late)".)
2. Four lens-derived statistics (Figure 28):
   - (a) "top-k accuracy of the J-lens at predicting the model's actual next token ... near zero through most of the early layers ... ticks up at the workspace start, slowly rises through the workspace layers, and jumps steeply in the final few layers" — the late jump marks the END of the workspace.
   - (b) "excess kurtosis of J-lens readouts ... near zero through the first third of the layers, increases beginning around a third of the way through the depth, and falls in the last few layers. We interpret the early rise as marking the beginning of the workspace".
   - (c) "autocorrelation of the lens's top-1 token across nearby positions ... relative to a shuffled-position null".
   - (d) "effective linear dimensionality of the J-space ... rises sharply around the same layer as the other metrics".
3. Non-lens corroboration: ignition in the ambiguous-embedding experiment ("Starting around the workspace onset (layer 38), it instead sits near one endpoint or the other", line 409); occupancy ("Occupancy is near zero through the first third of the layers and rises to a plateau of around 25", line 421); MLP composition-score bump at onset computed "from the network's weights alone" (line 1211); SAE in-lens fraction rising "at the workspace boundary" (line 1181).

Limitation (line 665): "we did not provide a principled definition of what distinguishes a 'workspace' representation from a 'motor' one." And line 669: whether early layers truly lack content "or an artifact of the lens" is open.

### 6.3 Mapping to Qwen3.6-27B (64 layers) **[inference]**

Using percentage-of-depth with residual-stream index r in {0..64} (r=0 embedding output, r=64 final block output), r = pct/100 * 64:

| Paper label | % depth | Qwen 64-layer residual index |
|---|---|---|
| L38 (onset) | 37.5 | 24.0 |
| L54 | 54.2 | 34.7 |
| L58 | 58.3 | 37.3 |
| L62 | 62.5 | 40.0 |
| **L67 (oracle fig)** | 66.7 | **42.7** |
| L71 (audit tool) | 70.8 | 45.3 |
| L75 | 75.0 | 48.0 |
| L79 | 79.2 | 50.7 |
| L83 | 83.3 | 53.3 |
| L92 (end) | 91.7 | 58.7 |

So: ceselder's L42 (65.6%) and the July replication's L45 (70.3%) bracket the paper's L67 (66.7%) and L71 (70.8%) respectively; both are inside "late workspace (~62–92)". WorkspaceBench's 20–60 is 31–94% depth, i.e. it starts ~5 layers before the paper's onset and runs ~1 layer past the end. If the paper's 0..100 is instead indexed over blocks (0 = after block 1), shift by about one layer; the paper does not say which. Note also the paper's chosen L67 is not argued for anywhere — it is simply the layer shown; L71 was chosen for the auditing tool "to make an apples-to-apples comparison to the NLA paper results, which used the same layer" (line 1297).

---

## 7. Limitations / caveats stated for the oracle lens (and inherited ones)

Stated directly for the oracle lens:
1. Cost: "substantially more expensive than either of the other two lenses, since it requires fine-tuning two auxiliary models." (line 1063)
2. Reconstruction accuracy: "it also achieves considerably lower reconstruction accuracy [than NLAs]. One interpretation of this is that the oracle lens extracts only those representations that are 'in the workspace,' while NLAs attempt to reconstruct the entire activation vector." (line 1085) Quantitatively, 31% mean whitened FVE.
3. Confabulation risk is argued to be lower than NLAs but only "we suspect" (line 1085); no measurement.
4. Training distribution: "trained only on assistant-span activations" (line 1077); generalization to human-span positions is asserted qualitatively from examples only.
5. Cherry-picking disclosure: panels E and F are "the highest-FVE of five decodes" (Figure 64 caption); the examples are "from an RL-refined checkpoint".
6. The delimiter-token commentary split is "presumably" a property of activations; no causal test (line 1083).
7. Only one layer (L67) and one model (Haiku 4.5) are reported; no quantitative benchmark against J-lens/template lens (the Figure 62/63 multi-hop benchmarks are template-lens only).

Inherited from the template lens (line 1035, 1057): "has some properties more similar to the tuned lens, and inherits some related pathologies" — "skip to the answer" in early layers; final-layer next-word top-10 only 67%; a set of spuriously-frequent words needing ad hoc filtering; and from the Stein's-lemma derivation, the Gaussian assumption is violated. And from the overall Limitations (line 657-663): the lens treats the workspace as a "bag of concepts"; readouts at some positions "resist interpretation".

The whole multi-token appendix is prefaced with "we introduce methods for deriving J-lens-like vectors for multi-token words or phrases, though we suspect they can be improved" (line 659).

---

## 8. Things NOT stated (replication degrees of freedom)

- Haiku 4.5 layer count; mapping of L-percentages to block indices.
- Number of template passages per word (only "a few hundred"); ridge lambda value; whether the covariance is full d_model x d_model (**[inference]** yes, "inverse covariance matrix").
- Whether the final norm layer is applied before whitening in the oracle pipeline.
- Reconstructor architecture details: which position/layer of the fine-tuned copy emits the vector; number of training pairs; optimizer.
- Oracle activation-injection details: layer, position, scale, prompt template text.
- RL algorithm/hyperparameters; exact format-penalty weights; N and K used for the 31% number.
- NN-OMP stopping rule (up to 16 atoms; tolerance not stated).
- The on-policy corpus: prompts, temperature, length, thinking on/off.
- Sigma for the oracle's whitened metric: stated as "the whitened metric of the template lens", so **[inference]** it is the template-passage covariance at the same layer, not a covariance over Assistant-turn activations.
