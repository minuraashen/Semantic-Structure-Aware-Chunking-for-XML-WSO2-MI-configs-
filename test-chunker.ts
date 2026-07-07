import * as fs from 'fs';
import * as path from 'path';
import { XMLChunker } from './chunker';
import { AutoTokenizer } from '@huggingface/transformers';
import { config } from './config';

/**
 * Demo script: chunk every XML file in artifacts/ and print a detailed
 * analysis (chunk boundaries, aggregation, cross-references, token sizes).
 * Exports the resulting chunks to test-output/chunks.json.
 *
 * For assertions/regression checks, run the unit tests: `npm test`.
 */

// Color codes for terminal output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  dim: '\x1b[2m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  red: '\x1b[31m',
};

function colorize(text: string, color: keyof typeof colors): string {
  return `${colors[color]}${text}${colors.reset}`;
}

function printSection(title: string): void {
  console.log('\n' + '='.repeat(80));
  console.log(colorize(title, 'bright'));
  console.log('='.repeat(80));
}

function printSubSection(title: string): void {
  console.log('\n' + colorize(title, 'cyan'));
  console.log('-'.repeat(60));
}

function describeAlgorithm(): void {
  printSection('Chunking Algorithm');

  console.log(colorize('✓ Position-annotated parse (sax):', 'green'));
  console.log('  - Every element carries exact source offsets; chunk content is an exact source slice');
  console.log('  - Structure and content come from ONE representation (no re-location step)');

  console.log(colorize('\n✓ Token-gated traversal:', 'green'));
  console.log(`  - Embedding text fits ${config.maxTokens} tokens → chunk; else descend into children`);
  console.log('  - Token counts use the embedding model tokenizer itself (exact gate)');

  console.log(colorize('\n✓ Sibling aggregation:', 'green'));
  console.log(`  - Consecutive small siblings merge until ≥ ${config.minTokens} content tokens`);
  console.log('  - Undersized tails merge backward into the preceding sibling chunk');

  console.log(colorize('\n✓ Context-enriched embedding text:', 'green'));
  console.log('  - Prefix = artifact metadata + capped ancestor path (derived structurally, no LLM)');
  console.log('  - Content = tree linearization with entity decoding + JSON/CDATA protection');
}

function findArtifactFiles(artifactsDir: string): string[] {
  const files: string[] = [];

  function scanDirectory(dir: string): void {
    const entries = fs.readdirSync(dir, { withFileTypes: true });
    for (const entry of entries) {
      const fullPath = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        scanDirectory(fullPath);
      } else if (entry.isFile() && entry.name.endsWith('.xml')) {
        files.push(fullPath);
      }
    }
  }

  scanDirectory(artifactsDir);
  return files;
}

async function generateChunks(files: string[]): Promise<Map<string, any[]>> {
  printSection('XML File Processing & Chunk Generation');

  const chunker = new XMLChunker();
  const allChunks = new Map<string, any[]>();

  for (const file of files) {
    // Store repository-relative paths (portable across machines)
    const relativePath = path.relative(process.cwd(), file).split(path.sep).join('/');
    console.log(colorize(`\n📄 Processing: ${relativePath}`, 'blue'));

    try {
      const source = await fs.promises.readFile(file, 'utf-8');
      const chunks = await chunker.chunkText(source, relativePath);
      allChunks.set(relativePath, chunks);

      console.log(colorize(`  ✓ Generated ${chunks.length} chunks`, 'green'));
      if (chunks.length > 0) {
        const firstChunk = chunks[0];
        console.log(`  - Artifact Type: ${colorize(firstChunk.context.artifact?.type || 'unknown', 'yellow')}`);
        console.log(`  - Artifact Name: ${colorize(firstChunk.context.artifact?.name || 'unknown', 'yellow')}`);
      }
    } catch (error: any) {
      console.log(colorize(`  ✗ Error: ${error?.message || 'Unknown error'}`, 'red'));
    }
  }

  return allChunks;
}

function showChunkDetails(allChunks: Map<string, any[]>): void {
  printSection('Detailed Chunk Analysis');

  let totalChunks = 0;
  const chunkTypeCount = new Map<string, number>();

  for (const [file, chunks] of allChunks.entries()) {
    printSubSection(`File: ${file}`);

    chunks.forEach((chunk, index) => {
      totalChunks++;
      chunkTypeCount.set(chunk.chunkType, (chunkTypeCount.get(chunk.chunkType) || 0) + 1);

      console.log(`\n${colorize(`Chunk #${index + 1}`, 'magenta')}`);
      console.log(`  Type: ${colorize(chunk.chunkType, 'yellow')}${chunk.memberTags ? colorize(` [${chunk.memberTags.join(', ')}]`, 'dim') : ''}${chunk.part ? colorize(` (part ${chunk.part})`, 'dim') : ''}`);
      console.log(`  Lines: ${chunk.startLine}-${chunk.endLine} (${chunk.endLine - chunk.startLine + 1} lines)`);

      if (chunk.context.artifact) {
        console.log(`  Artifact: ${chunk.context.artifact.type} - ${chunk.context.artifact.name}`);
      }
      if (chunk.context.path.length > 0) {
        console.log(`  Path: ${chunk.context.path.map((p: any) => p.tag).join(' > ')}`);
      }
      if (chunk.referencedSequences && chunk.referencedSequences.length > 0) {
        console.log(`  ${colorize('References:', 'green')} ${chunk.referencedSequences.join(', ')}`);
      }
      if (chunk.isSequenceDefinition) {
        console.log(`  ${colorize('✓ Standalone Artifact Definition', 'green')}`);
        console.log(`  Sequence Key: ${chunk.sequenceKey}`);
      }

      const preview = chunk.content.substring(0, 150).replace(/\n/g, ' ').trim();
      console.log(`  ${colorize('Content Preview:', 'dim')}`);
      console.log(`  ${colorize(preview + '...', 'dim')}`);

      const embeddingPreview = chunk.embeddingText.substring(0, 100);
      console.log(`  ${colorize('Embedding Text:', 'dim')}`);
      console.log(`  ${colorize(embeddingPreview + '...', 'dim')}`);
    });
  }

  printSubSection('Summary Statistics');
  console.log(`Total Chunks: ${colorize(totalChunks.toString(), 'bright')}`);

  console.log(colorize('\nChunk Types:', 'cyan'));
  for (const [type, count] of chunkTypeCount.entries()) {
    console.log(`  ${type}: ${count}`);
  }
}

function showCrossReferences(allChunks: Map<string, any[]>): void {
  printSection('Cross-Artifact Reference Detection');

  let totalRefs = 0;
  const refTypes = new Map<string, number>();

  for (const [file, chunks] of allChunks.entries()) {
    const chunksWithRefs = chunks.filter(c => c.referencedSequences && c.referencedSequences.length > 0);

    if (chunksWithRefs.length > 0) {
      printSubSection(`File: ${file}`);

      chunksWithRefs.forEach(chunk => {
        console.log(`  ${colorize(chunk.chunkType, 'yellow')} references:`);
        chunk.referencedSequences.forEach((ref: string) => {
          console.log(`    → ${colorize(ref, 'cyan')}`);
          totalRefs++;
          const refType = ref.split(':')[0];
          refTypes.set(refType, (refTypes.get(refType) || 0) + 1);
        });
      });
    }
  }

  printSubSection('Reference Statistics');
  console.log(`Total References: ${colorize(totalRefs.toString(), 'bright')}`);
  console.log(colorize('\nReference Types:', 'cyan'));
  for (const [type, count] of refTypes.entries()) {
    console.log(`  ${type}: ${count}`);
  }
}

/**
 * Token size verification with the real AutoTokenizer, on embeddingText —
 * the same text the token gate checks and the embedding model receives.
 */
async function verifyTokenSizes(allChunks: Map<string, any[]>): Promise<void> {
  printSection('Token Size Verification (AutoTokenizer)');

  console.log(colorize(`  Gate logic: tokens counted on embeddingText (context prefix + cleaned content)`, 'dim'));

  const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
  const maxTokens = config.maxTokens;
  let oversizedCount = 0;
  let totalTokens = 0;
  let chunkCount = 0;
  let minSeen = Number.POSITIVE_INFINITY;
  let maxSeen = 0;

  for (const [file, chunks] of allChunks.entries()) {
    for (const chunk of chunks) {
      const embeddingTokens = tokenizer.encode(chunk.embeddingText).length;
      totalTokens += embeddingTokens;
      chunkCount++;
      minSeen = Math.min(minSeen, embeddingTokens);
      maxSeen = Math.max(maxSeen, embeddingTokens);

      if (embeddingTokens > maxTokens) {
        oversizedCount++;
        console.log(colorize(`⚠ Oversized chunk in ${file}`, 'yellow'));
        console.log(`  Chunk: ${chunk.chunkType} [${chunk.chunkIndex}]`);
        console.log(`  EmbeddingText tokens: ${colorize(embeddingTokens.toString(), 'red')} (limit: ${maxTokens})`);
        console.log(`  Lines: ${chunk.startLine}-${chunk.endLine}`);
      }
    }
  }

  console.log(`\nChunk token stats: min ${minSeen}, mean ${(totalTokens / Math.max(chunkCount, 1)).toFixed(1)}, max ${maxSeen}`);
  if (oversizedCount === 0) {
    console.log(colorize(`✓ All ${chunkCount} chunks are within the ${maxTokens}-token limit`, 'green'));
    console.log(colorize(`  Model: ${config.tokenizerModel}`, 'dim'));
  } else {
    console.log(colorize(`\n⚠ Found ${oversizedCount} oversized chunks — check the token gate in processSiblings.`, 'yellow'));
  }
}

function exportToJSON(allChunks: Map<string, any[]>): void {
  printSection('Export Chunks to JSON');

  const outputDir = path.join(process.cwd(), 'test-output');
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir);
  }

  const allChunksArray: any[] = [];
  for (const chunks of allChunks.values()) {
    allChunksArray.push(...chunks);
  }

  const outputFile = path.join(outputDir, 'chunks.json');
  fs.writeFileSync(outputFile, JSON.stringify(allChunksArray, null, 2));

  console.log(colorize(`✓ Exported ${allChunksArray.length} chunks to ${outputFile}`, 'green'));
  console.log(`  File size: ${(fs.statSync(outputFile).size / 1024).toFixed(2)} KB`);
}

async function runDemo(): Promise<void> {
  console.log(colorize('\n╔════════════════════════════════════════════════════════════════════════════╗', 'bright'));
  console.log(colorize('║        WSO2 MI XML Chunker - Chunking Demonstration & Analysis            ║', 'bright'));
  console.log(colorize('╚════════════════════════════════════════════════════════════════════════════╝', 'bright'));

  const artifactsDir = path.join(process.cwd(), 'artifacts');

  if (!fs.existsSync(artifactsDir)) {
    console.error(colorize(`\n✗ Error: artifacts directory not found at ${artifactsDir}`, 'red'));
    console.log(colorize('Please ensure the artifacts folder exists with XML files', 'yellow'));
    process.exit(1);
  }

  try {
    describeAlgorithm();

    const files = findArtifactFiles(artifactsDir);
    console.log(colorize(`\n✓ Found ${files.length} XML files in artifacts folder`, 'green'));

    if (files.length === 0) {
      console.log(colorize('⚠ No XML files found. Please add XML files to the artifacts folder.', 'yellow'));
      process.exit(0);
    }

    const allChunks = await generateChunks(files);
    showChunkDetails(allChunks);
    showCrossReferences(allChunks);
    await verifyTokenSizes(allChunks);
    exportToJSON(allChunks);

    printSection('✓ DEMO COMPLETED');
    console.log(colorize('Run `npm test` for the assertion-based unit test suite.', 'green'));
  } catch (error) {
    console.error(colorize('\n✗ Demo execution failed:', 'red'));
    console.error(error);
    process.exit(1);
  }
}

runDemo().catch(console.error);
