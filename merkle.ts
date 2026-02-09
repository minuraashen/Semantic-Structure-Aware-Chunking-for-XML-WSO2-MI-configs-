import * as crypto from 'crypto';

/**
 * Compute a hash for chunk content and metadata
 * Used for content-based addressing and deduplication
 */
export function computeChunkHash(
  content: string,
  metadata?: Record<string, any>
): string {
  const hash = crypto.createHash('sha256');
  
  // Hash the content
  hash.update(content);
  
  // Include metadata in hash if provided
  if (metadata) {
    hash.update(JSON.stringify(metadata));
  }
  
  return hash.digest('hex');
}
