import { describe, it, expect, vi, afterEach } from 'vitest';
import { streamRequest, isStreamRequestError } from '@/services/streamRequest';

type FetchType = typeof global.fetch;

const originalFetch: FetchType = global.fetch;

afterEach(() => {
  global.fetch = originalFetch;
});

describe('streamRequest', () => {
  it('retries on network error and resolves successfully', async () => {
    const encoder = new TextEncoder();
    const bodyStream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoder.encode('{"event":"init"}\n'));
        controller.enqueue(encoder.encode('{"stock_code":"AAA"}\n'));
        controller.close();
      },
    });

    const fetchMock = vi
      .fn<Parameters<FetchType>, ReturnType<FetchType>>()
      .mockRejectedValueOnce(new TypeError('Network error'))
      .mockResolvedValueOnce(
        new Response(bodyStream, {
          headers: {
            'Content-Type': 'application/json',
          },
        }),
      );

    global.fetch = fetchMock as FetchType;

    const stream = await streamRequest('http://example.com/stream', {
      retryPolicy: {
        maxRetries: 1,
        retryStatusCodes: [],
        retryOnNetworkError: true,
        baseDelayMs: 0,
        backoffMultiplier: 1,
        maxDelayMs: 0,
        jitterRatio: 0,
      },
      timeoutMs: {
        ttfb: 1_000,
        total: 2_000,
      },
    });

    const received: unknown[] = [];
    for await (const chunk of stream) {
      received.push(chunk.json ?? chunk.raw);
    }

    expect(received).toEqual([{ event: 'init' }, { stock_code: 'AAA' }]);
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it('cancels an in-flight stream and throws an aborted error', async () => {
    let rejectRead: ((reason?: any) => void) | null = null;

    const mockReader = {
      read: vi.fn(() => {
        return new Promise<IteratorResult<Uint8Array>>((_, reject) => {
          rejectRead = reject;
        });
      }),
      releaseLock: vi.fn(),
    };

    const fetchMock = vi.fn<Parameters<FetchType>, ReturnType<FetchType>>((_, init) => {
      const signal = init?.signal as AbortSignal | undefined;
      if (signal) {
        signal.addEventListener(
          'abort',
          () => {
            rejectRead?.(new DOMException('Aborted', 'AbortError'));
          },
          { once: true },
        );
      }

      return Promise.resolve({
        ok: true,
        body: {
          getReader: () => mockReader,
        },
        headers: new Headers({ 'Content-Type': 'application/json' }),
      } as unknown as Response);
    });

    global.fetch = fetchMock as FetchType;

    const stream = await streamRequest('http://example.com/stream', {
      timeoutMs: {
        ttfb: 5_000,
        total: 5_000,
      },
    });

    const iterator = stream[Symbol.asyncIterator]();
    const nextPromise = iterator.next();

    stream.cancel('user');

    await expect(nextPromise).rejects.toSatisfy((error: unknown) => {
      return isStreamRequestError(error) && error.code === 'ABORTED';
    });
    expect(mockReader.releaseLock).toHaveBeenCalled();
  });
});
