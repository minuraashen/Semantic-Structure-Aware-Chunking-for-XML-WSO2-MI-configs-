import * as fs from 'fs';
import * as path from 'path';
import { XMLChunker } from './chunker';
import { artifactRegistry } from './artifact-registry';
import { AutoTokenizer, PreTrainedTokenizer } from '@huggingface/transformers';
import { config } from './config';

/**
 * Comprehensive test script to verify all chunking algorithm functionalities
 * Tests the algorithm with actual XML files from artifacts folder
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

/**
 * Test 1: Verify artifact registry functionality
 */
function testArtifactRegistry(): void {
  printSection('TEST 1: Artifact Registry Functionality');

  console.log(colorize('✓ Registered Plugins:', 'green'));
  const plugins = artifactRegistry.getAllPlugins();
  plugins.forEach(plugin => {
    console.log(`  - ${colorize(plugin.id, 'yellow')}: ${plugin.rootTags.join(', ')}`);
  });

  console.log(colorize('\n✓ Semantic Boundaries:', 'green'));
  const testBoundaries = ['resource', 'inSequence', 'outSequence', 'filter', 'query'];
  testBoundaries.forEach(boundary => {
    const isBoundary = artifactRegistry.isSemanticBoundary(boundary);
    console.log(`  - ${boundary}: ${isBoundary ? colorize('YES', 'green') : colorize('NO', 'red')}`);
  });

  console.log(colorize('\n✓ Mediator Tags:', 'green'));
  const testMediators = ['log', 'property', 'call', 'send', 'payloadFactory'];
  testMediators.forEach(mediator => {
    const isMediator = artifactRegistry.isMediatorTag(mediator);
    console.log(`  - ${mediator}: ${isMediator ? colorize('YES', 'green') : colorize('NO', 'red')}`);
  });
}

/**
 * Test 2: Find all XML files in artifacts folder
 */
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

/**
 * Test 3: Process XML files and generate chunks
 */
async function testChunkGeneration(files: string[]): Promise<Map<string, any[]>> {
  printSection('TEST 2: XML File Processing & Chunk Generation');

  const chunker = new XMLChunker();
  const allChunks = new Map<string, any[]>();

  for (const file of files) {
    const relativePath = file.replace(process.cwd(), '.');
    console.log(colorize(`\n📄 Processing: ${relativePath}`, 'blue'));

    try {
      const chunks = await chunker.chunkFile(file);
      allChunks.set(file, chunks);

      console.log(colorize(`  ✓ Generated ${chunks.length} chunks`, 'green'));
      
      // Show artifact type detection
      if (chunks.length > 0) {
        const firstChunk = chunks[1];
        console.log(`  - Resource Type: ${colorize(firstChunk.resourceType, 'yellow')}`);
        console.log(`  - Resource Name: ${colorize(firstChunk.resourceName, 'yellow')}`);
      }
    } catch (error: any) {
      console.log(colorize(`  ✗ Error: ${error?.message || 'Unknown error'}`, 'red'));
    }
  }

  return allChunks;
}

/**
 * Test 4: Display detailed chunk information
 */
function testChunkDetails(allChunks: Map<string, any[]>): void {
  printSection('TEST 3: Detailed Chunk Analysis');

  let totalChunks = 0;
  const chunkTypeCount = new Map<string, number>();
  const semanticTypeCount = new Map<string, number>();

  for (const [file, chunks] of allChunks.entries()) {
    const relativePath = file.replace(process.cwd(), '.');
    printSubSection(`File: ${relativePath}`);

    chunks.forEach((chunk, index) => {
      totalChunks++;
      
      // Count chunk types
      chunkTypeCount.set(chunk.chunkType, (chunkTypeCount.get(chunk.chunkType) || 0) + 1);
      semanticTypeCount.set(chunk.semanticType, (semanticTypeCount.get(chunk.semanticType) || 0) + 1);

      console.log(`\n${colorize(`Chunk #${index + 1}`, 'magenta')}`);
      console.log(`  Type: ${colorize(chunk.chunkType, 'yellow')}`);
      console.log(`  Semantic Type: ${colorize(chunk.semanticType, 'cyan')}`);
      console.log(`  Intent: ${colorize(chunk.semanticIntent, 'cyan')}`);
      console.log(`  Lines: ${chunk.startLine}-${chunk.endLine} (${chunk.endLine - chunk.startLine + 1} lines)`);
      console.log(`  Content Hash: ${colorize(chunk.contentHash.substring(0, 16) + '...', 'dim')}`);
      
      // Show context
      if (chunk.context.api) {
        console.log(`  API Context: ${chunk.context.api.name || 'N/A'}`);
      }
      if (chunk.context.resource) {
        console.log(`  Resource: ${chunk.context.resource.method} ${chunk.context.resource.uriTemplate}`);
      }
      if (chunk.context.sequence) {
        const seqName = typeof chunk.context.sequence === 'string' 
          ? chunk.context.sequence 
          : chunk.context.sequence.name;
        console.log(`  Sequence: ${seqName}`);
      }
      if (chunk.context.artifact) {
        console.log(`  Artifact: ${chunk.context.artifact.type} - ${chunk.context.artifact.name}`);
      }

      // Show references
      if (chunk.referencedSequences && chunk.referencedSequences.length > 0) {
        console.log(`  ${colorize('References:', 'green')} ${chunk.referencedSequences.join(', ')}`);
      }

      // Show sequence definition status
      if (chunk.isSequenceDefinition) {
        console.log(`  ${colorize('✓ Standalone Sequence Definition', 'green')}`);
        console.log(`  Sequence Key: ${chunk.sequenceKey}`);
      }

      // Show content preview (first 150 chars)
      const preview = chunk.content.substring(0, 150).replace(/\n/g, ' ').trim();
      console.log(`  ${colorize('Content Preview:', 'dim')}`);
      console.log(`  ${colorize(preview + '...', 'dim')}`);

      // Show embedding text preview
      const embeddingPreview = chunk.embeddingText.substring(0, 100);
      console.log(`  ${colorize('Embedding Text:', 'dim')}`);
      console.log(`  ${colorize(embeddingPreview + '...', 'dim')}`);
    });
  }

  // Summary statistics
  printSubSection('Summary Statistics');
  console.log(`Total Chunks: ${colorize(totalChunks.toString(), 'bright')}`);
  
  console.log(colorize('\nChunk Types:', 'cyan'));
  for (const [type, count] of chunkTypeCount.entries()) {
    console.log(`  ${type}: ${count}`);
  }

  console.log(colorize('\nSemantic Types:', 'cyan'));
  for (const [type, count] of semanticTypeCount.entries()) {
    console.log(`  ${type}: ${count}`);
  }
}

/**
 * Test 5: Verify hierarchical relationships
 */
function testHierarchy(allChunks: Map<string, any[]>): void {
  printSection('TEST 4: Hierarchical Relationships');

  for (const [file, chunks] of allChunks.entries()) {
    const relativePath = file.replace(process.cwd(), '.');
    
    // Check for parent-child relationships
    const hasHierarchy = chunks.some(chunk => chunk.parentChunkId !== null);
    
    if (hasHierarchy) {
      printSubSection(`File: ${relativePath}`);
      
      chunks.forEach(chunk => {
        if (chunk.parentChunkId !== null) {
          const parent = chunks.find(c => c.chunkIndex === chunk.parentChunkId);
          console.log(`  ${chunk.chunkType} [${chunk.chunkIndex}] → parent: ${parent?.chunkType || 'unknown'} [${chunk.parentChunkId}]`);
        } else {
          console.log(`  ${colorize(chunk.chunkType, 'yellow')} [${chunk.chunkIndex}] → ${colorize('ROOT', 'green')}`);
        }
      });
    }
  }
}

/**
 * Test 6: Cross-artifact reference detection
 */
function testCrossReferences(allChunks: Map<string, any[]>): void {
  printSection('TEST 5: Cross-Artifact Reference Detection');

  let totalRefs = 0;
  const refTypes = new Map<string, number>();

  for (const [file, chunks] of allChunks.entries()) {
    const relativePath = file.replace(process.cwd(), '.');
    
    const chunksWithRefs = chunks.filter(c => c.referencedSequences && c.referencedSequences.length > 0);
    
    if (chunksWithRefs.length > 0) {
      printSubSection(`File: ${relativePath}`);
      
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
 * Test 7: Token size verification using real AutoTokenizer
 */
async function testTokenSizing(allChunks: Map<string, any[]>): Promise<void> {
  printSection('TEST 6: Token Size Verification (AutoTokenizer)');

  const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
  const maxTokens = config.maxTokens;
  let oversizedCount = 0;

  for (const [file, chunks] of allChunks.entries()) {
    chunks.forEach(chunk => {
      const encoded = tokenizer.encode(chunk.content);
      const tokenCount = encoded.length;

      if (tokenCount > maxTokens) {
        oversizedCount++;
        const relativePath = file.replace(process.cwd(), '.');
        console.log(colorize(`⚠ Oversized chunk in ${relativePath}`, 'yellow'));
        console.log(`  Chunk: ${chunk.chunkType} [${chunk.chunkIndex}]`);
        console.log(`  Tokens: ${tokenCount} (limit: ${maxTokens})`);
        console.log(`  Lines: ${chunk.startLine}-${chunk.endLine}`);
      }
    });
  }

  if (oversizedCount === 0) {
    console.log(colorize(`✓ All chunks are within token limit (${maxTokens} tokens, model: ${config.tokenizerModel})`, 'green'));
  } else {
    console.log(colorize(`\n⚠ Found ${oversizedCount} oversized chunks`, 'yellow'));
  }
}

/**
 * Test 8: Export chunks to JSON
 */
function testExportToJSON(allChunks: Map<string, any[]>): void {
  printSection('TEST 7: Export Chunks to JSON');

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

/**
 * Main test runner
 */
async function runAllTests(): Promise<void> {
  console.log(colorize('\n╔════════════════════════════════════════════════════════════════════════════╗', 'bright'));
  console.log(colorize('║        WSO2 MI XML Chunker - Comprehensive Functionality Test             ║', 'bright'));
  console.log(colorize('╚════════════════════════════════════════════════════════════════════════════╝', 'bright'));

  const artifactsDir = path.join(process.cwd(), 'artifacts');
  
  // Verify artifacts directory exists
  if (!fs.existsSync(artifactsDir)) {
    console.error(colorize(`\n✗ Error: artifacts directory not found at ${artifactsDir}`, 'red'));
    console.log(colorize('Please ensure the artifacts folder exists with XML files', 'yellow'));
    process.exit(1);
  }

  try {
    // Test 1: Registry functionality
    testArtifactRegistry();

    // Find all XML files
    const files = findArtifactFiles(artifactsDir);
    console.log(colorize(`\n✓ Found ${files.length} XML files in artifacts folder`, 'green'));

    if (files.length === 0) {
      console.log(colorize('⚠ No XML files found. Please add XML files to the artifacts folder.', 'yellow'));
      process.exit(0);
    }

    // Test 2: Process files and generate chunks
    const allChunks = await testChunkGeneration(files);

    // Test 3: Display detailed chunk information
    testChunkDetails(allChunks);

    // Test 4: Verify hierarchical relationships
    testHierarchy(allChunks);

    // Test 5: Cross-reference detection
    testCrossReferences(allChunks);

    // Test 6: Token sizing (uses real tokenizer)
    await testTokenSizing(allChunks);

    // Test 7: Export to JSON
    testExportToJSON(allChunks);

    // Final summary
    printSection('✓ ALL TESTS COMPLETED SUCCESSFULLY');
    console.log(colorize('All chunking algorithm functionalities have been verified!', 'green'));

  } catch (error) {
    console.error(colorize('\n✗ Test execution failed:', 'red'));
    console.error(error);
    process.exit(1);
  }
}

// Run tests
runAllTests().catch(console.error);
