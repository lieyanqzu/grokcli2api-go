package grok

import (
	"context"
	"crypto/tls"
	"log/slog"
	"net/http/httptrace"
	"strings"
	"sync"
	"time"
)

type timingContextKey struct{}

// RequestTiming records low-cardinality latency milestones without retaining
// request bodies, credentials, affinity keys, or user identifiers.
type RequestTiming struct {
	mu      sync.Mutex
	started time.Time
	route   string

	decode, prepare, acquireWait, refreshWait time.Duration
	dns, connect, tls                         time.Duration
	wroteRequest, upstreamHeaders             time.Duration
	firstBody, firstEvent, firstUpstreamText  time.Duration
	firstDownstreamFlush, firstDownstreamText time.Duration
	hasWroteRequest, hasUpstreamHeaders       bool
	hasFirstBody, hasFirstEvent               bool
	hasFirstUpstreamText                      bool
	hasFirstDownstreamFlush                   bool
	hasFirstDownstreamText                    bool
	attempts, retries                         int
	connectionReused                          bool
	contentEncoding                           string
	finished                                  bool
}

func NewRequestTiming(route string) *RequestTiming {
	return &RequestTiming{started: time.Now(), route: route}
}

func WithRequestTiming(ctx context.Context, timing *RequestTiming) context.Context {
	if timing == nil {
		return ctx
	}
	return context.WithValue(ctx, timingContextKey{}, timing)
}

func RequestTimingFromContext(ctx context.Context) *RequestTiming {
	timing, _ := ctx.Value(timingContextKey{}).(*RequestTiming)
	return timing
}

func (t *RequestTiming) elapsed() time.Duration { return time.Since(t.started) }

func (t *RequestTiming) MarkDecode(duration time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.decode += duration
	t.mu.Unlock()
}

func (t *RequestTiming) MarkPrepare(duration time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.prepare += duration
	t.mu.Unlock()
}

func (t *RequestTiming) MarkAcquire(duration time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.acquireWait += duration
	t.mu.Unlock()
}

func (t *RequestTiming) MarkRefresh(duration time.Duration) {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.refreshWait += duration
	t.mu.Unlock()
}

func (t *RequestTiming) MarkAttempt() {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.attempts++
	t.mu.Unlock()
}

func (t *RequestTiming) MarkRetry() {
	if t == nil {
		return
	}
	t.mu.Lock()
	t.retries++
	t.mu.Unlock()
}

func (t *RequestTiming) MarkUpstreamHeaders(encoding string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if !t.hasUpstreamHeaders {
		t.upstreamHeaders = t.elapsed()
		t.hasUpstreamHeaders = true
	}
	switch strings.ToLower(strings.TrimSpace(encoding)) {
	case "", "identity":
		t.contentEncoding = "identity"
	case "gzip":
		t.contentEncoding = "gzip"
	default:
		t.contentEncoding = "other"
	}
	t.mu.Unlock()
}

func (t *RequestTiming) MarkFirstBodyByte() {
	if t == nil {
		return
	}
	t.mu.Lock()
	if !t.hasFirstBody {
		t.firstBody = t.elapsed()
		t.hasFirstBody = true
	}
	t.mu.Unlock()
}

func (t *RequestTiming) MarkFirstEvent() {
	if t == nil {
		return
	}
	t.mu.Lock()
	if !t.hasFirstEvent {
		t.firstEvent = t.elapsed()
		t.hasFirstEvent = true
	}
	t.mu.Unlock()
}

func (t *RequestTiming) MarkFirstUpstreamText() {
	if t == nil {
		return
	}
	t.mu.Lock()
	if !t.hasFirstUpstreamText {
		t.firstUpstreamText = t.elapsed()
		t.hasFirstUpstreamText = true
	}
	t.mu.Unlock()
}

func (t *RequestTiming) MarkDownstreamFlush(text bool) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if !t.hasFirstDownstreamFlush {
		t.firstDownstreamFlush = t.elapsed()
		t.hasFirstDownstreamFlush = true
	}
	if text && !t.hasFirstDownstreamText {
		t.firstDownstreamText = t.elapsed()
		t.hasFirstDownstreamText = true
	}
	t.mu.Unlock()
}

func (t *RequestTiming) ClientTrace(onWrote func()) *httptrace.ClientTrace {
	if t == nil {
		return &httptrace.ClientTrace{WroteRequest: func(httptrace.WroteRequestInfo) { onWrote() }}
	}
	var traceMu sync.Mutex
	var dnsStart, connectStart, tlsStart time.Time
	return &httptrace.ClientTrace{
		DNSStart: func(httptrace.DNSStartInfo) {
			traceMu.Lock()
			dnsStart = time.Now()
			traceMu.Unlock()
		},
		DNSDone: func(httptrace.DNSDoneInfo) {
			traceMu.Lock()
			started := dnsStart
			traceMu.Unlock()
			if !started.IsZero() {
				t.addNetwork(&t.dns, time.Since(started))
			}
		},
		ConnectStart: func(_, _ string) {
			traceMu.Lock()
			connectStart = time.Now()
			traceMu.Unlock()
		},
		ConnectDone: func(_, _ string, _ error) {
			traceMu.Lock()
			started := connectStart
			traceMu.Unlock()
			if !started.IsZero() {
				t.addNetwork(&t.connect, time.Since(started))
			}
		},
		TLSHandshakeStart: func() {
			traceMu.Lock()
			tlsStart = time.Now()
			traceMu.Unlock()
		},
		TLSHandshakeDone: func(_ tls.ConnectionState, _ error) {
			traceMu.Lock()
			started := tlsStart
			traceMu.Unlock()
			if !started.IsZero() {
				t.addNetwork(&t.tls, time.Since(started))
			}
		},
		GotConn: func(info httptrace.GotConnInfo) {
			t.mu.Lock()
			t.connectionReused = info.Reused
			t.mu.Unlock()
		},
		WroteRequest: func(httptrace.WroteRequestInfo) {
			onWrote()
			t.mu.Lock()
			if !t.hasWroteRequest {
				t.wroteRequest = t.elapsed()
				t.hasWroteRequest = true
			}
			t.mu.Unlock()
		},
	}
}

func (t *RequestTiming) addNetwork(target *time.Duration, duration time.Duration) {
	t.mu.Lock()
	*target += duration
	t.mu.Unlock()
}

func (t *RequestTiming) Finish(outcome string) {
	if t == nil {
		return
	}
	t.mu.Lock()
	if t.finished {
		t.mu.Unlock()
		return
	}
	t.finished = true
	total := t.elapsed()
	if t.attempts > 1 && t.retries < t.attempts-1 {
		t.retries = t.attempts - 1
	}
	fields := []any{
		"route", t.route, "outcome", outcome, "total", total,
		"decode", t.decode, "prepare", t.prepare, "account_wait", t.acquireWait,
		"refresh_wait", t.refreshWait, "dns", t.dns, "connect", t.connect,
		"tls", t.tls, "connection_reused", t.connectionReused,
		"wrote_request", t.wroteRequest, "upstream_headers", t.upstreamHeaders,
		"first_body", t.firstBody, "first_event", t.firstEvent,
		"first_upstream_text", t.firstUpstreamText,
		"has_upstream_text", t.hasFirstUpstreamText,
		"first_downstream_flush", t.firstDownstreamFlush,
		"first_downstream_text", t.firstDownstreamText,
		"has_downstream_text", t.hasFirstDownstreamText,
		"attempts", t.attempts, "retries", t.retries,
		"content_encoding", t.contentEncoding,
	}
	t.mu.Unlock()
	slog.Debug("generation timing", fields...)
}
