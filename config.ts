/**
 * Configuration for the Semantic Structure-Aware XML Chunker
 */
export const config = {
  /** Maximum token limit per chunk (tuned for sentence-transformers/all-MiniLM-L6-v2) */
  maxTokens: 256,

  /** HuggingFace model ID used for tokenization */
  tokenizerModel: 'sentence-transformers/all-MiniLM-L6-v2',
};
