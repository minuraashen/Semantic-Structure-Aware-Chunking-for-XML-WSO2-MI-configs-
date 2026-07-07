import * as fs from 'fs';
import * as path from 'path';
import { pipeline, AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import { XMLChunker, ChunkerOptions } from '../chunker';
import { config } from '../config';
import { EvalChunk, fixedSizeChunks, recursiveSplitChunks, wholeFileChunks } from './baselines';
import { queries, EvalQuery } from './queries';

/**
 * Retrieval evaluation: compares the structural chunker (and its ablations)
 * against fixed-size, recursive-split, and whole-file baselines on the
 * labeled query set in queries.ts.
 *
 * Everything runs locally: chunks and queries are embedded with
 * all-MiniLM-L6-v2 (ONNX), ranked by cosine similarity.
 *
 * Metrics (per method, averaged over queries):
 *  - Success@1 / Success@5: ≥1 relevant chunk in top-k
 *  - MRR@10: 1/rank of the first relevant chunk (0 if none in top 10)
 *  - nDCG@10: binary relevance, ideal DCG from the number of relevant
 *    chunks that exist in that method's index (capped at 10)
 *  - LineRecall@5: fraction of gold lines covered by the union of the
 *    top-5 retrieved chunks' spans (rewards retrieving complete answers,
 *    penalizes fragmentation)
 *  - index size and token statistics
 */

interface MethodResult {
  method: string;
  numChunks: number;
  meanTokens: number;
  p95Tokens: number;
  success1: number;
  success5: number;
  mrr10: number;
  ndcg10: number;
  lineRecall5: number;
}

interface RankedChunk extends EvalChunk {
  score: number;
}

function isRelevant(chunk: EvalChunk, q: EvalQuery): boolean {
  return q.gold.some(
    (g) =>
      chunk.filePath === g.file &&
      chunk.startLine <= g.lines[1] &&
      chunk.endLine >= g.lines[0]
  );
}

function lineRecallAt(ranked: RankedChunk[], q: EvalQuery, k: number): number {
  let goldTotal = 0;
  let goldCovered = 0;
  for (const g of q.gold) {
    const covered = new Set<number>();
    for (const chunk of ranked.slice(0, k)) {
      if (chunk.filePath !== g.file) continue;
      const from = Math.max(chunk.startLine, g.lines[0]);
      const to = Math.min(chunk.endLine, g.lines[1]);
      for (let line = from; line <= to; line++) covered.add(line);
    }
    goldTotal += g.lines[1] - g.lines[0] + 1;
    goldCovered += covered.size;
  }
  return goldTotal === 0 ? 0 : goldCovered / goldTotal;
}

function evaluateMethod(
  method: string,
  chunks: EvalChunk[],
  chunkVectors: number[][],
  queryVectors: number[][],
  tokenizer: PreTrainedTokenizer
): MethodResult {
  let success1 = 0;
  let success5 = 0;
  let mrrSum = 0;
  let ndcgSum = 0;
  let lineRecallSum = 0;

  queries.forEach((q, qi) => {
    const qv = queryVectors[qi];
    const ranked: RankedChunk[] = chunks
      .map((c, ci) => ({
        ...c,
        // vectors are L2-normalized → dot product == cosine similarity
        score: dot(qv, chunkVectors[ci]),
      }))
      .sort((a, b) => b.score - a.score);

    const top10 = ranked.slice(0, 10);
    const firstRelevant = top10.findIndex((c) => isRelevant(c, q));

    if (firstRelevant === 0) success1++;
    if (firstRelevant !== -1 && firstRelevant < 5) success5++;
    if (firstRelevant !== -1) mrrSum += 1 / (firstRelevant + 1);

    // Binary nDCG@10
    let dcg = 0;
    top10.forEach((c, i) => {
      if (isRelevant(c, q)) dcg += 1 / Math.log2(i + 2);
    });
    const totalRelevant = chunks.filter((c) => isRelevant(c, q)).length;
    let idcg = 0;
    for (let i = 0; i < Math.min(totalRelevant, 10); i++) idcg += 1 / Math.log2(i + 2);
    ndcgSum += idcg === 0 ? 0 : dcg / idcg;

    lineRecallSum += lineRecallAt(ranked, q, 5);
  });

  const tokenCounts = chunks.map((c) => tokenizer.encode(c.embeddingText).length);
  tokenCounts.sort((a, b) => a - b);
  const n = queries.length;

  return {
    method,
    numChunks: chunks.length,
    meanTokens: round(tokenCounts.reduce((a, b) => a + b, 0) / Math.max(chunks.length, 1)),
    p95Tokens: tokenCounts[Math.floor(tokenCounts.length * 0.95)] ?? 0,
    success1: round(success1 / n),
    success5: round(success5 / n),
    mrr10: round(mrrSum / n),
    ndcg10: round(ndcgSum / n),
    lineRecall5: round(lineRecallSum / n),
  };
}

function dot(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function round(x: number): number {
  return Math.round(x * 1000) / 1000;
}

function collectArtifacts(): string[] {
  const dir = path.join(process.cwd(), 'artifacts');
  const files: string[] = [];
  const scan = (d: string): void => {
    for (const entry of fs.readdirSync(d, { withFileTypes: true })) {
      const p = path.join(d, entry.name);
      if (entry.isDirectory()) scan(p);
      else if (entry.name.endsWith('.xml')) files.push(p);
    }
  };
  scan(dir);
  return files.sort();
}

async function structuralChunks(files: string[], options: ChunkerOptions): Promise<EvalChunk[]> {
  const chunker = new XMLChunker(options);
  const all: EvalChunk[] = [];
  for (const file of files) {
    const relative = path.relative(process.cwd(), file).split(path.sep).join('/');
    const source = await fs.promises.readFile(file, 'utf-8');
    const chunks = await chunker.chunkText(source, relative);
    all.push(
      ...chunks.map((c) => ({
        filePath: c.filePath,
        embeddingText: c.embeddingText,
        startLine: c.startLine,
        endLine: c.endLine,
      }))
    );
  }
  return all;
}

async function main(): Promise<void> {
  console.log('Loading tokenizer and embedding model (local ONNX)...');
  const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
  const embedder = await pipeline('feature-extraction', config.embeddingModel);

  const embed = async (texts: string[]): Promise<number[][]> => {
    const vectors: number[][] = [];
    // Batch to keep memory bounded.
    const BATCH = 16;
    for (let i = 0; i < texts.length; i += BATCH) {
      const batch = texts.slice(i, i + BATCH);
      const output = await embedder(batch, { pooling: 'mean', normalize: true });
      const [rows, dims] = output.dims.length === 2 ? output.dims : [1, output.dims[0]];
      const data = output.data as Float32Array;
      for (let r = 0; r < rows; r++) {
        vectors.push(Array.from(data.slice(r * dims, (r + 1) * dims)));
      }
    }
    return vectors;
  };

  const files = collectArtifacts();
  console.log(`Corpus: ${files.length} artifacts, ${queries.length} labeled queries\n`);

  // Build every method's index.
  console.log('Chunking with all methods...');
  const methods: Array<{ name: string; chunks: EvalChunk[] }> = [];

  methods.push({ name: 'structural (ours)', chunks: await structuralChunks(files, {}) });
  methods.push({ name: 'ours w/o context', chunks: await structuralChunks(files, { includeContext: false }) });
  methods.push({ name: 'ours w/o cleanup', chunks: await structuralChunks(files, { cleanContent: false }) });
  methods.push({ name: 'ours w/o aggregation', chunks: await structuralChunks(files, { aggregate: false }) });

  const fixed: EvalChunk[] = [];
  const recursive: EvalChunk[] = [];
  const whole: EvalChunk[] = [];
  for (const file of files) {
    const relative = path.relative(process.cwd(), file).split(path.sep).join('/');
    const source = fs.readFileSync(file, 'utf-8');
    fixed.push(...fixedSizeChunks(source, relative, tokenizer, config.maxTokens));
    recursive.push(...recursiveSplitChunks(source, relative, tokenizer, config.maxTokens));
    whole.push(...wholeFileChunks(source, relative));
  }
  methods.push({ name: 'fixed-size 256', chunks: fixed });
  methods.push({ name: 'recursive-split 256', chunks: recursive });
  methods.push({ name: 'whole-file', chunks: whole });

  console.log('Embedding queries...');
  const queryVectors = await embed(queries.map((q) => q.query));

  const results: MethodResult[] = [];
  for (const { name, chunks } of methods) {
    console.log(`Embedding ${chunks.length} chunks for: ${name}`);
    const chunkVectors = await embed(chunks.map((c) => c.embeddingText));
    results.push(evaluateMethod(name, chunks, chunkVectors, queryVectors, tokenizer));
  }

  // Report
  const header = ['method', '#chunks', 'meanTok', 'p95Tok', 'S@1', 'S@5', 'MRR@10', 'nDCG@10', 'LineRec@5'];
  const rows = results.map((r) => [
    r.method,
    String(r.numChunks),
    String(r.meanTokens),
    String(r.p95Tokens),
    r.success1.toFixed(3),
    r.success5.toFixed(3),
    r.mrr10.toFixed(3),
    r.ndcg10.toFixed(3),
    r.lineRecall5.toFixed(3),
  ]);
  const widths = header.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)));
  const fmt = (cells: string[]): string => cells.map((c, i) => c.padEnd(widths[i] + 2)).join('');
  console.log('\n=== Retrieval evaluation results ===\n');
  console.log(fmt(header));
  console.log(widths.map((w) => '-'.repeat(w + 2)).join(''));
  rows.forEach((r) => console.log(fmt(r)));

  // Persist for the paper.
  const outDir = path.join(process.cwd(), 'eval');
  fs.writeFileSync(
    path.join(outDir, 'results.json'),
    JSON.stringify(
      {
        date: new Date().toISOString(),
        embeddingModel: config.embeddingModel,
        maxTokens: config.maxTokens,
        minTokens: config.minTokens,
        corpusFiles: files.length,
        numQueries: queries.length,
        results,
      },
      null,
      2
    )
  );

  const md = [
    '| Method | #Chunks | Mean tokens | S@1 | S@5 | MRR@10 | nDCG@10 | LineRec@5 |',
    '|---|---|---|---|---|---|---|---|',
    ...results.map(
      (r) =>
        `| ${r.method} | ${r.numChunks} | ${r.meanTokens} | ${r.success1.toFixed(3)} | ${r.success5.toFixed(3)} | ${r.mrr10.toFixed(3)} | ${r.ndcg10.toFixed(3)} | ${r.lineRecall5.toFixed(3)} |`
    ),
  ].join('\n');
  fs.writeFileSync(path.join(outDir, 'results.md'), md + '\n');

  console.log('\nSaved eval/results.json and eval/results.md');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
