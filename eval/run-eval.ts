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
 * labeled query set in queries.ts — swept across several lightweight
 * embedding models that all run fully locally (ONNX via
 * @huggingface/transformers), matching the privacy constraint of on-prem
 * WSO2 MI deployments.
 *
 * Chunking is held constant across models (token gate uses the MiniLM
 * tokenizer at 256 tokens, a size every swept model accepts), so the sweep
 * isolates the effect of the embedding model from the chunking itself.
 *
 * Metrics (per model × method, averaged over queries):
 *  - Success@1 / Success@5: ≥1 relevant chunk in top-k
 *  - MRR@10: 1/rank of the first relevant chunk (0 if none in top 10)
 *  - nDCG@10: binary relevance, ideal DCG from the number of relevant
 *    chunks that exist in that method's index (capped at 10)
 *  - LineRecall@5: fraction of gold lines covered by the union of the
 *    top-5 retrieved chunks' spans (rewards retrieving complete answers,
 *    penalizes fragmentation)
 */

interface EmbeddingModel {
  id: string;
  /** Short label for tables. */
  label: string;
  /** Some models require task prefixes for asymmetric retrieval. */
  queryPrefix?: string;
  docPrefix?: string;
}

/**
 * Lightweight, fully local embedding models (all ≤33M params, ONNX).
 * These are the realistic candidates for an on-prem WSO2 MI copilot.
 */
const MODELS: EmbeddingModel[] = [
  { id: 'Xenova/all-MiniLM-L6-v2', label: 'all-MiniLM-L6-v2' },
  { id: 'isuruwijesiri/all-MiniLM-L6-v2-code-search-512', label: 'MiniLM-code-search-512' },
  {
    id: 'Xenova/bge-small-en-v1.5',
    label: 'bge-small-en-v1.5',
    queryPrefix: 'Represent this sentence for searching relevant passages: ',
  },
  { id: 'Xenova/e5-small-v2', label: 'e5-small-v2', queryPrefix: 'query: ', docPrefix: 'passage: ' },
  { id: 'Xenova/gte-small', label: 'gte-small' },
];

interface MethodResult {
  model: string;
  method: string;
  numChunks: number;
  meanTokens: number;
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

/**
 * One-hop reference expansion: after dense ranking, each top-5 chunk's
 * cross-artifact references are resolved to their definition chunks, which
 * are injected into the ranking right after the referring chunk. This uses
 * the reference metadata the chunker extracts (e.g. configKey → localEntry)
 * to surface artifacts a query needs but whose text does not match it.
 */
function expandByReferences(ranked: RankedChunk[]): RankedChunk[] {
  const defs = new Map<string, RankedChunk[]>();
  for (const c of ranked) {
    if (c.isDefinition && c.artifactName) {
      const arr = defs.get(c.artifactName) ?? [];
      arr.push(c);
      defs.set(c.artifactName, arr);
    }
  }

  const out: RankedChunk[] = [];
  const seen = new Set<RankedChunk>();
  const push = (c: RankedChunk): void => {
    if (!seen.has(c)) {
      seen.add(c);
      out.push(c);
    }
  };

  ranked.forEach((c, i) => {
    push(c);
    if (i < 5 && c.references) {
      for (const ref of c.references) {
        const name = ref.slice(ref.indexOf(':') + 1);
        for (const d of defs.get(name) ?? []) push(d);
      }
    }
  });
  return out;
}

function evaluateMethod(
  model: string,
  method: string,
  chunks: EvalChunk[],
  chunkVectors: number[][],
  queryVectors: number[][],
  tokenizer: PreTrainedTokenizer,
  expandRefs = false
): MethodResult {
  let success1 = 0;
  let success5 = 0;
  let mrrSum = 0;
  let ndcgSum = 0;
  let lineRecallSum = 0;

  queries.forEach((q, qi) => {
    const qv = queryVectors[qi];
    let ranked: RankedChunk[] = chunks
      .map((c, ci) => ({
        ...c,
        // vectors are L2-normalized → dot product == cosine similarity
        score: dot(qv, chunkVectors[ci]),
      }))
      .sort((a, b) => b.score - a.score);
    if (expandRefs) ranked = expandByReferences(ranked);

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
  const n = queries.length;

  return {
    model,
    method,
    numChunks: chunks.length,
    meanTokens: round(tokenCounts.reduce((a, b) => a + b, 0) / Math.max(chunks.length, 1)),
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
        references: c.referencedSequences,
        artifactName: c.context.artifact.name,
        isDefinition: c.isSequenceDefinition,
      }))
    );
  }
  return all;
}

async function main(): Promise<void> {
  console.log('Loading chunking tokenizer...');
  const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);

  const files = collectArtifacts();
  console.log(`Corpus: ${files.length} artifacts, ${queries.length} labeled queries`);

  // Build every method's index once (chunking is embedding-model independent).
  console.log('Chunking with all methods...');
  const methods: Array<{ name: string; chunks: EvalChunk[]; expandRefs?: boolean }> = [];

  const oursChunks = await structuralChunks(files, {});
  methods.push({ name: 'structural (ours)', chunks: oursChunks });
  methods.push({ name: 'ours + ref-expansion', chunks: oursChunks, expandRefs: true });
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

  const results: MethodResult[] = [];

  for (const model of MODELS) {
    console.log(`\n=== Model: ${model.label} (${model.id}) ===`);
    const embedder = await pipeline('feature-extraction', model.id);

    const embed = async (texts: string[], prefix = ''): Promise<number[][]> => {
      const vectors: number[][] = [];
      const BATCH = 16;
      const prefixed = prefix ? texts.map((t) => prefix + t) : texts;
      for (let i = 0; i < prefixed.length; i += BATCH) {
        const batch = prefixed.slice(i, i + BATCH);
        const output = await embedder(batch, { pooling: 'mean', normalize: true });
        const [rows, dims] = output.dims.length === 2 ? output.dims : [1, output.dims[0]];
        const data = output.data as Float32Array;
        for (let r = 0; r < rows; r++) {
          vectors.push(Array.from(data.slice(r * dims, (r + 1) * dims)));
        }
      }
      return vectors;
    };

    const queryVectors = await embed(
      queries.map((q) => q.query),
      model.queryPrefix ?? ''
    );

    // Cache embeddings per chunk set — the ref-expansion method reuses the
    // structural index, so its chunks are only embedded once.
    const vectorCache = new Map<EvalChunk[], number[][]>();

    for (const { name, chunks, expandRefs } of methods) {
      let chunkVectors = vectorCache.get(chunks);
      if (!chunkVectors) {
        console.log(`  Embedding ${chunks.length} chunks for: ${name}`);
        chunkVectors = await embed(
          chunks.map((c) => c.embeddingText),
          model.docPrefix ?? ''
        );
        vectorCache.set(chunks, chunkVectors);
      }
      const r = evaluateMethod(model.label, name, chunks, chunkVectors, queryVectors, tokenizer, expandRefs);
      results.push(r);
      console.log(
        `    S@1 ${r.success1.toFixed(3)}  S@5 ${r.success5.toFixed(3)}  MRR ${r.mrr10.toFixed(3)}  nDCG ${r.ndcg10.toFixed(3)}  LineRec@5 ${r.lineRecall5.toFixed(3)}`
      );
    }

    // Free the session before loading the next model.
    await (embedder as any).dispose?.();
  }

  // ---- Report ----
  const outDir = path.join(process.cwd(), 'eval');
  fs.writeFileSync(
    path.join(outDir, 'results.json'),
    JSON.stringify(
      {
        date: new Date().toISOString(),
        chunkingTokenizer: config.tokenizerModel,
        maxTokens: config.maxTokens,
        minTokens: config.minTokens,
        models: MODELS.map((m) => m.id),
        corpusFiles: files.length,
        numQueries: queries.length,
        results,
      },
      null,
      2
    )
  );

  const mdSections: string[] = [];
  for (const model of MODELS) {
    const rows = results.filter((r) => r.model === model.label);
    mdSections.push(
      `### ${model.label}\n`,
      '| Method | #Chunks | Mean tokens | S@1 | S@5 | MRR@10 | nDCG@10 | LineRec@5 |',
      '|---|---|---|---|---|---|---|---|',
      ...rows.map(
        (r) =>
          `| ${r.method} | ${r.numChunks} | ${r.meanTokens} | ${r.success1.toFixed(3)} | ${r.success5.toFixed(3)} | ${r.mrr10.toFixed(3)} | ${r.ndcg10.toFixed(3)} | ${r.lineRecall5.toFixed(3)} |`
      ),
      ''
    );
  }
  fs.writeFileSync(path.join(outDir, 'results.md'), mdSections.join('\n'));

  console.log('\nSaved eval/results.json and eval/results.md');

  // Console summary: ours vs best baseline per model.
  console.log('\n=== Summary: structural (ours) vs best baseline, per model ===');
  for (const model of MODELS) {
    const rows = results.filter((r) => r.model === model.label);
    const ours = rows.find((r) => r.method === 'structural (ours)')!;
    const baselines = rows.filter((r) =>
      ['fixed-size 256', 'recursive-split 256', 'whole-file'].includes(r.method)
    );
    const bestBaseline = baselines.reduce((a, b) => (b.ndcg10 > a.ndcg10 ? b : a));
    console.log(
      `${model.label}: ours nDCG ${ours.ndcg10.toFixed(3)} vs best baseline ${bestBaseline.ndcg10.toFixed(3)} (${bestBaseline.method}); LineRec@5 ${ours.lineRecall5.toFixed(3)} vs ${bestBaseline.lineRecall5.toFixed(3)}`
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
