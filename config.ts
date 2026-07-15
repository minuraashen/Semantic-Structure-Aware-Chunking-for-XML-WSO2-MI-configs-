/**
 * Configuration for the Semantic Structure-Aware XML Chunker
 */
export const config = {
  /**
   * Maximum token limit per chunk embedding text, measured with the embedding
   * model's own tokenizer (tuned for sentence-transformers/all-MiniLM-L6-v2,
   * whose max sequence length is 256).
   */
  maxTokens: 256,

  /**
   * Minimum token mass before the sibling aggregation buffer flushes.
   * Consecutive small siblings are merged until the combined embedding text
   * reaches this size, which prevents one-line noise chunks. Token-based
   * (not line-based) so behavior is independent of source formatting.
   */
  minTokens: 64,

  /**
   * Maximum number of ancestor path entries kept in a chunk's context prefix.
   * Bounds the context so deep nesting cannot crowd out chunk content
   * within the token budget. The root artifact entry is always kept.
   */
  maxContextAncestors: 4,

  /** HuggingFace model ID used for tokenization */
  tokenizerModel: 'sentence-transformers/all-MiniLM-L6-v2',

  /**
   * Model used by the evaluation harness to embed chunks and queries.
   * ONNX export of sentence-transformers/all-MiniLM-L6-v2 (same weights and
   * tokenizer), runnable fully locally via @huggingface/transformers.
   */
  embeddingModel: 'Xenova/all-MiniLM-L6-v2',
};
