const DEFAULT_TIMEOUT_MS = {
  ttfb: 30_000,
  total: 120_000,
} as const;

const DEFAULT_RETRY_POLICY = {
  maxRetries: 2,
  retryStatusCodes: [500, 502, 503, 504],
  retryOnNetworkError: true,
  baseDelayMs: 500,
  backoffMultiplier: 2,
  maxDelayMs: 5_000,
  jitterRatio: 0.3,
} as const;

type TimeoutPhase = 'ttfb' | 'total';

type AbortMeta =
  | { type: 'cancel'; source: 'client' | 'upstream'; reason?: unknown }
  | { type: 'timeout'; phase: TimeoutPhase };

export type StreamErrorCode =
  | 'ABORTED'
  | 'TIMEOUT'
  | 'NETWORK'
  | 'HTTP_ERROR'
  | 'PARSER'
  | 'UNKNOWN';

export interface StreamErrorDetails {
  status?: number;
  attempts?: number;
  retryable?: boolean;
  reason?: unknown;
  phase?: TimeoutPhase;
  responseText?: string;
}

export interface StreamRequestError extends Error {
  code: StreamErrorCode;
  details?: StreamErrorDetails;
}

export interface RetryPolicy {
  maxRetries: number;
  retryStatusCodes: number[];
  retryOnNetworkError: boolean;
  baseDelayMs: number;
  backoffMultiplier: number;
  maxDelayMs: number;
  jitterRatio: number;
}

export interface StreamTimeoutOptions {
  ttfb: number;
  total: number;
}

export type StreamParser = 'auto' | 'jsonl' | 'sse' | 'text';

export interface StreamChunk {
  raw: string;
  format: 'jsonl' | 'sse' | 'text';
  json?: unknown;
  eventName?: string | null;
}

export interface StreamRequestOptions extends RequestInit {
  retryPolicy?: Partial<RetryPolicy>;
  timeoutMs?: Partial<StreamTimeoutOptions>;
  parser?: StreamParser;
}

export interface StreamRequestResult extends AsyncIterable<StreamChunk> {
  response: Response;
  cancel: (reason?: unknown) => void;
  attempts: number;
}

const isNumber = (value: unknown): value is number =>
  typeof value === 'number' && Number.isFinite(value);

const randomBetween = (min: number, max: number): number =>
  min + Math.random() * (max - min);

const wait = (ms: number): Promise<void> =>
  ms > 0 ? new Promise((resolve) => setTimeout(resolve, ms)) : Promise.resolve();

const createStreamError = (
  code: StreamErrorCode,
  message: string,
  details?: StreamErrorDetails,
  cause?: unknown,
): StreamRequestError => {
  const error = new Error(message) as StreamRequestError;
  error.name = 'StreamRequestError';
  error.code = code;
  if (details) {
    error.details = details;
  }
  if (cause !== undefined) {
    (error as Error & { cause?: unknown }).cause = cause;
  }
  return error;
};

const mergeRetryPolicy = (policy?: Partial<RetryPolicy>): RetryPolicy => ({
  ...DEFAULT_RETRY_POLICY,
  ...policy,
  retryStatusCodes: policy?.retryStatusCodes ?? [...DEFAULT_RETRY_POLICY.retryStatusCodes],
});

const normalizeTimeouts = (timeouts?: Partial<StreamTimeoutOptions>): StreamTimeoutOptions => ({
  ttfb: isNumber(timeouts?.ttfb) ? Math.max(0, timeouts!.ttfb) : DEFAULT_TIMEOUT_MS.ttfb,
  total: isNumber(timeouts?.total) ? Math.max(0, timeouts!.total) : DEFAULT_TIMEOUT_MS.total,
});

const detectParser = (explicit: StreamParser | undefined, contentType: string | null): Exclude<StreamParser, 'auto'> => {
  if (explicit && explicit !== 'auto') {
    return explicit;
  }

  if (contentType?.includes('text/event-stream')) {
    return 'sse';
  }
  if (contentType?.includes('json')) {
    return 'jsonl';
  }
  return 'text';
};

const tryParseJson = (value: string): unknown | undefined => {
  if (!value) return undefined;
  try {
    return JSON.parse(value);
  } catch (error) {
    console.debug('[streamRequest] JSON parse failed', { value, error });
    return undefined;
  }
};

interface AttemptController {
  controller: AbortController;
  cleanup: () => void;
}

const bindAttemptController = (
  parentSignal: AbortSignal,
): AttemptController => {
  const controller = new AbortController();

  const relay = () => {
    if (!controller.signal.aborted) {
      controller.abort(parentSignal.reason ?? new Error('ParentAbort'));
    }
  };

  if (parentSignal.aborted) {
    controller.abort(parentSignal.reason ?? new Error('ParentAbort'));
  } else {
    parentSignal.addEventListener('abort', relay, { once: true });
  }

  const cleanup = () => parentSignal.removeEventListener('abort', relay);

  return { controller, cleanup };
};

const shouldRetryResponse = (response: Response, attempt: number, policy: RetryPolicy): boolean =>
  attempt <= policy.maxRetries && policy.retryStatusCodes.includes(response.status);

const isNetworkError = (error: unknown): boolean =>
  error instanceof TypeError ||
  (error instanceof Error && error.name === 'FetchError');

const shouldRetryError = (error: unknown, attempt: number, policy: RetryPolicy): boolean =>
  attempt <= policy.maxRetries && policy.retryOnNetworkError && isNetworkError(error);

const computeBackoffDelay = (attemptIndex: number, policy: RetryPolicy): number => {
  const raw = Math.min(policy.maxDelayMs, policy.baseDelayMs * Math.pow(policy.backoffMultiplier, attemptIndex));
  if (policy.jitterRatio <= 0) {
    return raw;
  }
  const jitter = raw * policy.jitterRatio;
  return randomBetween(Math.max(0, raw - jitter), raw + jitter);
};

const parseJsonLines = (buffer: string, flush: boolean): { chunks: StreamChunk[]; remainder: string } => {
  const lines = buffer.split(/\r?\n/);
  let remainder = '';

  if (!flush && !buffer.endsWith('\n') && !buffer.endsWith('\r')) {
    remainder = lines.pop() ?? '';
  }

  const chunks: StreamChunk[] = [];
  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) continue;
    const json = tryParseJson(line);
    chunks.push({ raw: line, json, format: 'jsonl' });
  }

  return { chunks, remainder };
};

const parseSse = (buffer: string, flush: boolean): { chunks: StreamChunk[]; remainder: string } => {
  const events = buffer.split(/\n\n/);
  let remainder = '';

  if (!flush && !buffer.endsWith('\n\n')) {
    remainder = events.pop() ?? '';
  }

  const chunks: StreamChunk[] = [];

  for (const event of events) {
    const trimmed = event.trim();
    if (!trimmed) continue;

    const lines = trimmed.split(/\n/);
    let eventName: string | null = null;
    const dataLines: string[] = [];

    for (const line of lines) {
      if (!line) continue;
      if (line.startsWith('event:')) {
        eventName = line.slice(6).trim() || null;
      } else if (line.startsWith('data:')) {
        dataLines.push(line.slice(5).trimStart());
      }
    }

    const payload = dataLines.join('\n');
    const json = tryParseJson(payload);

    chunks.push({ raw: payload, json, format: 'sse', eventName });
  }

  return { chunks, remainder };
};

const parsePlainText = (buffer: string): { chunks: StreamChunk[]; remainder: string } => {
  if (!buffer) {
    return { chunks: [], remainder: '' };
  }
  return { chunks: [{ raw: buffer, format: 'text' }], remainder: '' };
};

const parseBuffer = (
  buffer: string,
  parser: Exclude<StreamParser, 'auto'>,
  flush: boolean,
): { chunks: StreamChunk[]; remainder: string } => {
  switch (parser) {
    case 'sse':
      return parseSse(buffer, flush);
    case 'text':
      return parsePlainText(buffer);
    case 'jsonl':
    default:
      return parseJsonLines(buffer, flush);
  }
};

export const isStreamRequestError = (value: unknown): value is StreamRequestError =>
  typeof value === 'object' && value !== null && 'code' in value && 'message' in value;

export async function streamRequest(
  url: string,
  options: StreamRequestOptions = {},
): Promise<StreamRequestResult> {
  const retryPolicy = mergeRetryPolicy(options.retryPolicy);
  const timeouts = normalizeTimeouts(options.timeoutMs);

  const cancelController = new AbortController();
  let abortMeta: AbortMeta | undefined;

  const cancelWithMeta = (meta: AbortMeta) => {
    if (!cancelController.signal.aborted) {
      abortMeta = meta;
      cancelController.abort(meta);
    }
  };

  if (options.signal) {
    if (options.signal.aborted) {
      cancelWithMeta({ type: 'cancel', source: 'upstream', reason: options.signal.reason });
    } else {
      options.signal.addEventListener(
        'abort',
        () => {
          cancelWithMeta({ type: 'cancel', source: 'upstream', reason: options.signal?.reason });
        },
        { once: true },
      );
    }
  }

  const requestInit: RequestInit = { ...options };
  delete (requestInit as StreamRequestOptions).retryPolicy;
  delete (requestInit as StreamRequestOptions).timeoutMs;
  delete (requestInit as StreamRequestOptions).parser;
  requestInit.signal = undefined;

  let attempts = 0;
  let lastError: unknown;
  let response: Response | null = null;

  while (attempts <= retryPolicy.maxRetries) {
    const { controller, cleanup } = bindAttemptController(cancelController.signal);
    attempts += 1;
    try {
      response = await fetch(url, { ...requestInit, signal: controller.signal });

      if (!response.ok) {
        if (shouldRetryResponse(response, attempts, retryPolicy)) {
          const delay = computeBackoffDelay(attempts - 1, retryPolicy);
          await wait(delay);
          continue;
        }

        const bodyText = await response.text().catch(() => undefined);
        throw createStreamError(
          response.status >= 500 ? 'NETWORK' : 'HTTP_ERROR',
          `HTTP ${response.status}`,
          {
            status: response.status,
            attempts,
            retryable: response.status >= 500 && attempts <= retryPolicy.maxRetries,
            responseText: bodyText,
          },
        );
      }

      break;
    } catch (error) {
      lastError = error;
      if (cancelController.signal.aborted) {
        break;
      }
      if (shouldRetryError(error, attempts, retryPolicy)) {
        const delay = computeBackoffDelay(attempts - 1, retryPolicy);
        await wait(delay);
        continue;
      }
      break;
    } finally {
      cleanup();
    }
  }

  if (!response || !response.ok) {
    if (cancelController.signal.aborted) {
      const meta = (cancelController.signal.reason as AbortMeta | undefined) ?? abortMeta;
      if (meta?.type === 'timeout') {
        throw createStreamError(
          'TIMEOUT',
          meta.phase === 'ttfb' ? '等待服务器响应超时' : '请求执行时间过长',
          { attempts, phase: meta.phase },
        );
      }
      const cancelReason = meta?.type === 'cancel' ? meta.reason : undefined;
      if (meta?.type === 'cancel') {
        throw createStreamError('ABORTED', '请求已取消', { attempts, reason: cancelReason });
      }
      throw createStreamError('ABORTED', '请求已终止', { attempts, reason: cancelReason });
    }

    if (lastError instanceof Error && 'code' in lastError && (lastError as StreamRequestError).code) {
      throw lastError;
    }

    if (isNetworkError(lastError)) {
      throw createStreamError('NETWORK', '网络请求失败', { attempts }, lastError);
    }

    throw createStreamError('UNKNOWN', '流式请求失败', { attempts }, lastError);
  }

  if (!response.body) {
    throw createStreamError('NETWORK', '响应不包含可读的数据流', { attempts });
  }

  const parser = detectParser(options.parser, response.headers.get('content-type'));
  const reader = response.body.getReader();
  const decoder = new TextDecoder();

  let ttfbTimer: ReturnType<typeof setTimeout> | undefined;
  let totalTimer: ReturnType<typeof setTimeout> | undefined;
  let firstChunk = false;

  const clearTimers = () => {
    if (ttfbTimer) {
      clearTimeout(ttfbTimer);
      ttfbTimer = undefined;
    }
    if (totalTimer) {
      clearTimeout(totalTimer);
      totalTimer = undefined;
    }
  };

  if (timeouts.ttfb > 0) {
    ttfbTimer = setTimeout(() => {
      cancelWithMeta({ type: 'timeout', phase: 'ttfb' });
    }, timeouts.ttfb);
  }

  const generator = async function* (): AsyncGenerator<StreamChunk, void, void> {
    let buffer = '';
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) {
          break;
        }

        if (!firstChunk) {
          firstChunk = true;
          if (ttfbTimer) {
            clearTimeout(ttfbTimer);
            ttfbTimer = undefined;
          }
          if (timeouts.total > 0) {
            totalTimer = setTimeout(() => {
              cancelWithMeta({ type: 'timeout', phase: 'total' });
            }, timeouts.total);
          }
        }

        if (value) {
          buffer += decoder.decode(value, { stream: true });
          const { chunks, remainder } = parseBuffer(buffer, parser, false);
          buffer = remainder;
          for (const chunk of chunks) {
            yield chunk;
          }
        }
      }

      buffer += decoder.decode();
      const { chunks } = parseBuffer(buffer, parser, true);
      for (const chunk of chunks) {
        yield chunk;
      }
    } catch (error) {
      if (cancelController.signal.aborted) {
        const meta = (cancelController.signal.reason as AbortMeta | undefined) ?? abortMeta;
        if (meta?.type === 'timeout') {
          throw createStreamError(
            'TIMEOUT',
            meta.phase === 'ttfb' ? '等待服务器响应超时' : '请求执行时间过长',
            { attempts, phase: meta.phase },
            error,
          );
        }
        const cancelReason = meta?.type === 'cancel' ? meta.reason : undefined;
        if (meta?.type === 'cancel') {
          throw createStreamError('ABORTED', '请求已取消', { attempts, reason: cancelReason }, error);
        }
        throw createStreamError('ABORTED', '请求已终止', { attempts, reason: cancelReason }, error);
      }

      throw createStreamError('NETWORK', '读取数据流时发生错误', { attempts }, error);
    } finally {
      clearTimers();
      reader.releaseLock();
    }
  };

  let consumed = false;

  const iterable: StreamRequestResult = {
    response,
    attempts,
    cancel: (reason?: unknown) => {
      cancelWithMeta({ type: 'cancel', source: 'client', reason });
    },
    [Symbol.asyncIterator]() {
      if (consumed) {
        throw createStreamError('UNKNOWN', '流已被消费');
      }
      consumed = true;
      return generator();
    },
  };

  return iterable;
}
