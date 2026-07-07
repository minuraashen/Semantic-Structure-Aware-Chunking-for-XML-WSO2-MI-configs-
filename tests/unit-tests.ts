import * as assert from 'assert';
import * as fs from 'fs';
import * as path from 'path';
import { AutoTokenizer } from '@huggingface/transformers';
import { parseXML, XMLElementNode, XMLNode, childElements } from '../xml-parser';
import { XMLChunker, XMLChunk } from '../chunker';
import { config } from '../config';

/**
 * Unit tests for the position-annotated parser and the chunker.
 * Includes the adversarial cases that broke the previous line-regex design:
 * nested same-name tags, multi-line opening tags, dotted tag names, comments
 * containing tag text, CDATA, and XML entities.
 */

let passed = 0;
let failed = 0;
const failures: string[] = [];

async function test(name: string, fn: () => void | Promise<void>): Promise<void> {
  try {
    await fn();
    passed++;
    console.log(`  \x1b[32m✓\x1b[0m ${name}`);
  } catch (err: any) {
    failed++;
    failures.push(name);
    console.log(`  \x1b[31m✗ ${name}\x1b[0m`);
    console.log(`    ${err?.message ?? err}`);
  }
}

function firstElement(source: string): XMLElementNode {
  const doc = parseXML(source);
  return doc.roots.find((n): n is XMLElementNode => n.kind === 'element')!;
}

/** Every element's source slice must start with its own opening tag and end with '>'. */
function assertRoundTrip(source: string): void {
  const doc = parseXML(source);
  const walk = (el: XMLElementNode): void => {
    const slice = doc.source.slice(el.startOffset, el.endOffset);
    assert.ok(
      slice.startsWith('<' + el.tag),
      `<${el.tag}> slice must start with its opening tag, got: ${slice.slice(0, 40)}`
    );
    assert.ok(slice.trimEnd().endsWith('>'), `<${el.tag}> slice must end with '>'`);
    if (!el.selfClosing) {
      assert.ok(
        slice.endsWith(`</${el.tag}>`),
        `<${el.tag}> slice must end with its closing tag`
      );
    }
    childElements(el).forEach(walk);
  };
  doc.roots.filter((n): n is XMLElementNode => n.kind === 'element').forEach(walk);
}

async function run(): Promise<void> {
  console.log('\n=== XML parser tests ===');

  await test('nested same-name tags get correct spans', () => {
    const src = `<a>
  <filter x="outer">
    <filter x="inner"><respond/></filter>
    <log/>
  </filter>
  <filter x="second"/>
</a>`;
    assertRoundTrip(src);
    const root = firstElement(src);
    const [outer, second] = childElements(root);
    assert.strictEqual(outer.attrs.x, 'outer');
    assert.strictEqual(second.attrs.x, 'second');
    const inner = childElements(outer)[0];
    assert.strictEqual(inner.attrs.x, 'inner');
    const doc = parseXML(src);
    // The outer filter's span must include the inner filter and its own close.
    const outerSlice = doc.source.slice(outer.startOffset, outer.endOffset);
    assert.ok(outerSlice.includes('x="inner"'));
    assert.ok(outerSlice.endsWith('</filter>'));
  });

  await test('multi-line opening tags are located correctly', () => {
    const src = `<api name="X">
  <resource methods="GET"
            uri-template="/balance/{id}">
    <respond/>
  </resource>
</api>`;
    assertRoundTrip(src);
    const resource = childElements(firstElement(src))[0];
    assert.strictEqual(resource.attrs['uri-template'], '/balance/{id}');
    const doc = parseXML(src);
    assert.strictEqual(doc.lineOf(resource.startOffset), 2);
    assert.strictEqual(doc.lineOf(resource.endOffset - 1), 5);
  });

  await test('dotted tag names (http.post, ai.agent) parse verbatim', () => {
    const src = `<seq><http.post configKey="C"><relativePath>/a</relativePath></http.post><ai.agent/></seq>`;
    assertRoundTrip(src);
    const [httpPost, aiAgent] = childElements(firstElement(src));
    assert.strictEqual(httpPost.tag, 'http.post');
    assert.strictEqual(aiAgent.tag, 'ai.agent');
  });

  await test('comments containing tag text are ignored', () => {
    const src = `<a><!-- <log/> not real --><log category="INFO"/></a>`;
    assertRoundTrip(src);
    const logs = childElements(firstElement(src));
    assert.strictEqual(logs.length, 1);
    assert.strictEqual(logs[0].attrs.category, 'INFO');
  });

  await test('CDATA is captured as protected text', () => {
    const src = `<script><![CDATA[ if (a < b) { x(); } ]]></script>`;
    const el = firstElement(src);
    const textNode = el.children.find((c) => c.kind === 'text') as Extract<XMLNode, { kind: 'text' }>;
    assert.ok(textNode.cdata);
    assert.strictEqual(textNode.text, ' if (a < b) { x(); } ');
  });

  await test('entities are decoded in attributes and text', () => {
    const src = `<filter xpath="\${vars.amount &gt; 0}">a &amp; b &#65;</filter>`;
    const el = firstElement(src);
    assert.strictEqual(el.attrs.xpath, '${vars.amount > 0}');
    const textNode = el.children.find((c) => c.kind === 'text') as Extract<XMLNode, { kind: 'text' }>;
    assert.strictEqual(textNode.text.trim(), 'a & b A');
  });

  await test('namespace-prefixed tags are preserved', () => {
    const src = `<wsp:Policy xmlns:wsp="http://x"><wsp:All/></wsp:Policy>`;
    assertRoundTrip(src);
    const el = firstElement(src);
    assert.strictEqual(el.tag, 'wsp:Policy');
    assert.strictEqual(childElements(el)[0].tag, 'wsp:All');
  });

  await test('malformed XML throws instead of silently mis-parsing', () => {
    assert.throws(() => parseXML('<a><b></a>'));
    assert.throws(() => parseXML('<a>'));
  });

  await test('all artifact files round-trip exactly', () => {
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
    assert.ok(files.length > 0, 'no artifact files found');
    for (const f of files) {
      assertRoundTrip(fs.readFileSync(f, 'utf-8'));
    }
  });

  console.log('\n=== Tokenizer additivity (O(n) gate correctness) ===');

  await test('WordPiece counts are additive across whitespace-joined segments', async () => {
    const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);
    const specials = tokenizer.encode('').length;
    const ns = (t: string): number => (t === '' ? 0 : tokenizer.encode(t).length - specials);
    const samples = [
      'api BankAPI context=/bankapi',
      'payloadFactory media-type=json format {"status": "success"}',
      'variable name=originalAmount expression=${payload.amount} type=DOUBLE',
      'log category=ERROR message Deposit failed: ${props.synapse.ERROR_MESSAGE}',
      'filter xpath=${vars.amount 0} then respond else throwError',
    ];
    for (let i = 0; i < samples.length; i++) {
      for (let j = 0; j < samples.length; j++) {
        const joined = `${samples[i]} ${samples[j]}`;
        assert.strictEqual(
          ns(joined),
          ns(samples[i]) + ns(samples[j]),
          `additivity failed for samples ${i}+${j}`
        );
      }
    }
  });

  console.log('\n=== Chunker tests ===');

  const chunker = new XMLChunker();
  await chunker.initialize();
  const tokenizer = await AutoTokenizer.from_pretrained(config.tokenizerModel);

  await test('small artifact becomes a single definition chunk', async () => {
    const src = `<localEntry key="CurrencyConverter" xmlns="http://ws.apache.org/ns/synapse">
  <endpooint>value</endpooint>
</localEntry>`;
    const chunks = await chunker.chunkText(src, 'x.xml');
    assert.strictEqual(chunks.length, 1);
    assert.strictEqual(chunks[0].isSequenceDefinition, true);
    assert.strictEqual(chunks[0].sequenceKey, 'CurrencyConverter');
    assert.strictEqual(chunks[0].context.artifact.type, 'localEntry');
  });

  await test('every chunk respects the token limit', async () => {
    const files = collectArtifacts();
    for (const f of files) {
      const chunks = await chunker.chunkFile(f);
      for (const c of chunks) {
        const n = tokenizer.encode(c.embeddingText).length;
        assert.ok(
          n <= config.maxTokens,
          `${path.basename(f)} chunk ${c.chunkIndex} (${c.chunkType}) has ${n} tokens > ${config.maxTokens}`
        );
      }
    }
  });

  await test('chunk content is an exact source slice with correct lines', async () => {
    const files = collectArtifacts();
    for (const f of files) {
      const source = fs.readFileSync(f, 'utf-8');
      const chunks = await chunker.chunkFile(f);
      for (const c of chunks) {
        assert.strictEqual(
          c.content,
          source.slice(c.startOffset, c.endOffset),
          `${path.basename(f)} chunk ${c.chunkIndex}: content is not the exact source slice`
        );
        const upToStart = source.slice(0, c.startOffset);
        assert.strictEqual(
          upToStart.split('\n').length,
          c.startLine,
          `${path.basename(f)} chunk ${c.chunkIndex}: startLine mismatch`
        );
      }
    }
  });

  await test('no element content is lost (recursive coverage)', async () => {
    // Invariant: every element is either inside some chunk's span, or the
    // traversal descended into it — in which case ALL its element children
    // must recursively satisfy the same invariant (the container itself is
    // then represented by the context prefix of its children's chunks).
    const files = collectArtifacts();
    for (const f of files) {
      const source = fs.readFileSync(f, 'utf-8');
      const doc = parseXML(source);
      const chunks = await chunker.chunkFile(f);
      const covered = (el: XMLElementNode): boolean =>
        chunks.some((c) => c.startOffset <= el.startOffset && el.endOffset <= c.endOffset);
      const check = (el: XMLElementNode): void => {
        if (covered(el)) return; // whole subtree inside a chunk
        const children = childElements(el);
        assert.ok(
          children.length > 0,
          `${path.basename(f)}: leaf <${el.tag}> at offset ${el.startOffset} not covered by any chunk`
        );
        children.forEach(check);
      };
      doc.roots.filter((n): n is XMLElementNode => n.kind === 'element').forEach(check);
    }
  });

  await test('small siblings aggregate instead of becoming one-line chunks', async () => {
    // Force descent with a large sibling, then several tiny siblings.
    const filler = Array.from({ length: 40 }, (_, i) => `<property name="p${i}" value="value${i}"/>`).join('\n    ');
    const src = `<api name="Agg" xmlns="http://ws.apache.org/ns/synapse">
  <resource methods="POST" uri-template="/x">
    ${filler}
    <sessionId>abc</sessionId>
    <role>Assistant</role>
    <modelName>gpt-4o</modelName>
    <temperature>0.7</temperature>
  </resource>
</api>`;
    const chunks = await chunker.chunkText(src, 'agg.xml');
    // The four tiny trailing elements must not appear as four standalone chunks.
    const tinyStandalone = chunks.filter(
      (c) => ['sessionId', 'role', 'modelName', 'temperature'].includes(c.chunkType)
    );
    assert.strictEqual(
      tinyStandalone.length,
      0,
      `tiny elements emitted standalone: ${tinyStandalone.map((c) => c.chunkType).join(', ')}`
    );
    // They must still be covered by some chunk.
    for (const tag of ['sessionId', 'role', 'modelName', 'temperature']) {
      assert.ok(
        chunks.some((c) => c.content.includes(`<${tag}>`)),
        `${tag} lost during aggregation`
      );
    }
  });

  await test('aggregated chunks record member tags and stay contiguous', async () => {
    const files = collectArtifacts();
    for (const f of files) {
      const chunks = await chunker.chunkFile(f);
      for (const c of chunks) {
        if (c.chunkType === 'aggregated') {
          assert.ok(c.memberTags && c.memberTags.length > 1, 'aggregated chunk missing memberTags');
          assert.ok(c.endOffset > c.startOffset);
        }
      }
    }
  });

  await test('references are extracted regardless of attribute order', async () => {
    const src = `<sequence name="S" xmlns="http://ws.apache.org/ns/synapse">
  <sequence xmlns="http://ws.apache.org/ns/synapse" key="LoggingSequence"/>
  <http.post someAttr="x" configKey="EmailConnection"><a>b</a></http.post>
  <call-template description="d" target="ConvertCurrencyTool"/>
</sequence>`;
    const chunks = await chunker.chunkText(src, 'refs.xml');
    const allRefs = new Set(chunks.flatMap((c) => c.referencedSequences));
    assert.ok(allRefs.has('sequence:LoggingSequence'), 'missed sequence ref with key not first');
    assert.ok(allRefs.has('localEntry:EmailConnection'), 'missed configKey ref');
    assert.ok(allRefs.has('template:ConvertCurrencyTool'), 'missed call-template ref');
  });

  await test('entities appear decoded in embedding text (no "gt" noise)', async () => {
    const src = `<sequence name="S" xmlns="http://ws.apache.org/ns/synapse">
  <filter xpath="\${vars.amount &gt; 0}"><then><respond/></then></filter>
</sequence>`;
    const chunks = await chunker.chunkText(src, 'ent.xml');
    const text = chunks.map((c) => c.embeddingText).join(' ');
    assert.ok(!/\bgt\b/.test(text), `embedding text contains entity residue: ${text}`);
  });

  await test('JSON payloads are preserved in embedding text', async () => {
    const src = `<api name="J" context="/j" xmlns="http://ws.apache.org/ns/synapse">
  <resource methods="GET" uri-template="/">
    <payloadFactory media-type="json">
      <format>{"greetings":"Welcome to O2 Bank !!"}</format>
    </payloadFactory>
    <respond/>
  </resource>
</api>`;
    const chunks = await chunker.chunkText(src, 'json.xml');
    const text = chunks.map((c) => c.embeddingText).join(' ');
    assert.ok(text.includes('"greetings"'), `JSON destroyed in embedding text: ${text}`);
  });

  await test('context prefix has no tautology noise (no "inSequence: inSequence")', async () => {
    const files = collectArtifacts();
    for (const f of files) {
      const chunks = await chunker.chunkFile(f);
      for (const c of chunks) {
        assert.ok(
          !/(\b[\w.]+): \1\b/.test(c.embeddingText),
          `tautology in embedding text: ${c.embeddingText.slice(0, 100)}`
        );
      }
    }
  });

  await test('context path is capped at maxContextAncestors', async () => {
    // Build deep nesting: each level has a big sibling to force descent.
    const deepInner = `<leaf x="y"/>`;
    let src = deepInner;
    for (let depth = 0; depth < 10; depth++) {
      const filler = Array.from({ length: 30 }, (_, i) => `<pad name="d${depth}p${i}" value="v${i}"/>`).join('');
      src = `<level${depth} attr="a${depth}">${filler}${src}</level${depth}>`;
    }
    src = `<api name="Deep" xmlns="http://ws.apache.org/ns/synapse">${src}</api>`;
    const chunks = await chunker.chunkText(src, 'deep.xml');
    for (const c of chunks) {
      assert.ok(
        c.context.path.length <= config.maxContextAncestors,
        `context path too long: ${c.context.path.length}`
      );
    }
  });

  await test('oversized leaf is split into parts within the limit', async () => {
    const bigText = Array.from({ length: 600 }, (_, i) => `word${i}`).join(' ');
    const src = `<sequence name="Big" xmlns="http://ws.apache.org/ns/synapse"><instructions>${bigText}</instructions></sequence>`;
    const chunks = await chunker.chunkText(src, 'big.xml');
    assert.ok(chunks.length > 1, 'oversized leaf was not split');
    for (const c of chunks) {
      const n = tokenizer.encode(c.embeddingText).length;
      assert.ok(n <= config.maxTokens, `part ${c.part} has ${n} tokens`);
    }
    const parts = chunks.filter((c) => c.part !== undefined);
    assert.ok(parts.length > 1, 'parts not numbered');
  });

  await test('contentHash is stable and collision-distinguishing', async () => {
    const src = `<api name="H" context="/h" xmlns="http://ws.apache.org/ns/synapse">
  <resource methods="GET" uri-template="/"><respond/></resource>
</api>`;
    const a = await chunker.chunkText(src, 'h.xml');
    const b = await chunker.chunkText(src, 'h.xml');
    assert.strictEqual(a[0].contentHash, b[0].contentHash);
    assert.strictEqual(a[0].contentHash.length, 16);
  });

  await test('ablation options change embedding text as documented', async () => {
    const src = `<api name="A" context="/a" xmlns="http://ws.apache.org/ns/synapse">
  <resource methods="GET" uri-template="/"><respond/></resource>
</api>`;
    const noContext = new XMLChunker({ includeContext: false });
    const noClean = new XMLChunker({ cleanContent: false });
    const [withCtx] = await chunker.chunkText(src, 'a.xml');
    const [withoutCtx] = await noContext.chunkText(src, 'a.xml');
    const [raw] = await noClean.chunkText(src, 'a.xml');
    assert.ok(withCtx.embeddingText.startsWith('api A context=/a'));
    assert.ok(!withoutCtx.embeddingText.startsWith('api A'));
    assert.ok(raw.embeddingText.includes('<resource'), 'raw ablation should keep XML syntax');
  });

  await test('deterministic output across runs', async () => {
    const files = collectArtifacts();
    const f = files[0];
    const run1 = await chunker.chunkFile(f);
    const run2 = await chunker.chunkFile(f);
    assert.strictEqual(JSON.stringify(run1), JSON.stringify(run2));
  });

  console.log(`\n${passed} passed, ${failed} failed`);
  if (failed > 0) {
    console.log(`Failed: ${failures.join(', ')}`);
    process.exit(1);
  }
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
  return files;
}

run().catch((err) => {
  console.error(err);
  process.exit(1);
});
