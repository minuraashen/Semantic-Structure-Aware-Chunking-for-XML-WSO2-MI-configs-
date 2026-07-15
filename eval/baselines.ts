import { PreTrainedTokenizer } from '@huggingface/transformers';

/**
 * Baseline chunkers for the retrieval evaluation. All produce the same
 * shape (EvalChunk) as the structural chunker so metrics are comparable.
 */

export interface EvalChunk {
  filePath: string;
  embeddingText: string;
  startLine: number;
  endLine: number;
  /** Cross-artifact references (structural chunker only). */
  references?: string[];
  /** Artifact this chunk belongs to (structural chunker only). */
  artifactName?: string;
  /** True when the chunk is a whole standalone artifact definition. */
  isDefinition?: boolean;
}

function countTokens(tokenizer: PreTrainedTokenizer, text: string): number {
  return tokenizer.encode(text).length;
}

/**
 * Fixed-size baseline: pack consecutive source lines into windows of at most
 * `maxTokens` tokens, with `overlapLines` lines of overlap between windows.
 * The standard "dumb but strong" baseline in code-RAG evaluations.
 */
export function fixedSizeChunks(
  source: string,
  filePath: string,
  tokenizer: PreTrainedTokenizer,
  maxTokens: number,
  overlapLines = 4
): EvalChunk[] {
  const lines = source.split('\n');
  const chunks: EvalChunk[] = [];
  let start = 0;

  while (start < lines.length) {
    let end = start;
    let text = '';
    while (end < lines.length) {
      const candidate = text ? `${text}\n${lines[end]}` : lines[end];
      if (countTokens(tokenizer, candidate) > maxTokens && text !== '') break;
      text = candidate;
      end++;
    }
    if (text.trim().length > 0) {
      chunks.push({
        filePath,
        embeddingText: text,
        startLine: start + 1,
        endLine: end,
      });
    }
    if (end >= lines.length) break;
    start = Math.max(end - overlapLines, start + 1);
  }

  return chunks;
}

/**
 * Recursive splitter baseline (LangChain RecursiveCharacterTextSplitter
 * style): split on progressively finer separators, greedily re-merging
 * adjacent pieces up to the token budget. Separators are XML-oriented, as one
 * would configure for XML input.
 */
export function recursiveSplitChunks(
  source: string,
  filePath: string,
  tokenizer: PreTrainedTokenizer,
  maxTokens: number
): EvalChunk[] {
  const SEPARATORS = ['</', '\n\n', '\n', ' '];

  interface Piece {
    text: string;
    start: number; // char offset in source
  }

  const split = (piece: Piece, sepIndex: number): Piece[] => {
    if (countTokens(tokenizer, piece.text) <= maxTokens) return [piece];
    if (sepIndex >= SEPARATORS.length) return [piece]; // cannot split further

    const sep = SEPARATORS[sepIndex];
    const rawParts: Piece[] = [];
    let cursor = 0;
    for (;;) {
      const idx = piece.text.indexOf(sep, cursor + 1);
      if (idx === -1) {
        rawParts.push({ text: piece.text.slice(cursor), start: piece.start + cursor });
        break;
      }
      rawParts.push({ text: piece.text.slice(cursor, idx), start: piece.start + cursor });
      cursor = idx;
    }
    if (rawParts.length <= 1) return split(piece, sepIndex + 1);

    // Greedy re-merge up to the budget, then recurse on oversized parts.
    const merged: Piece[] = [];
    let current: Piece | null = null;
    for (const part of rawParts) {
      if (current === null) {
        current = part;
        continue;
      }
      const candidate: string = current.text + part.text;
      if (countTokens(tokenizer, candidate) <= maxTokens) {
        current = { text: candidate, start: current.start };
      } else {
        merged.push(current);
        current = part;
      }
    }
    if (current !== null) merged.push(current);

    return merged.flatMap((p) =>
      countTokens(tokenizer, p.text) > maxTokens ? split(p, sepIndex + 1) : [p]
    );
  };

  const lineOfOffset = (offset: number): number =>
    source.slice(0, offset).split('\n').length;

  return split({ text: source, start: 0 }, 0)
    .filter((p) => p.text.trim().length > 0)
    .map((p) => ({
      filePath,
      embeddingText: p.text.trim(),
      startLine: lineOfOffset(p.start),
      endLine: lineOfOffset(p.start + p.text.length - 1),
    }));
}

/** Whole-file baseline: one chunk per artifact (model truncates the excess). */
export function wholeFileChunks(source: string, filePath: string): EvalChunk[] {
  return [
    {
      filePath,
      embeddingText: source,
      startLine: 1,
      endLine: source.split('\n').length,
    },
  ];
}
