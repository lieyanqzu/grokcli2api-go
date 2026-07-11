package server

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Futureppo/grokcli2api-go/internal/config"
)

func BenchmarkStreamingProxyTTFT(b *testing.B) {
	origin := time.Now()
	var upstreamTextAt atomic.Int64
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		upstreamTextAt.Store(time.Since(origin).Nanoseconds())
		if r.URL.Path == "/v1/chat/completions" {
			_, _ = io.WriteString(w, "data: {\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\ndata: [DONE]\n\n")
			return
		}
		_, _ = io.WriteString(w, "event: response.output_text.delta\ndata: {\"type\":\"response.output_text.delta\",\"item_id\":\"item-1\",\"delta\":\"hello\"}\n\nevent: response.completed\ndata: {\"type\":\"response.completed\",\"response\":{\"status\":\"completed\"}}\n\n")
	}))
	defer upstream.Close()

	dir := b.TempDir()
	writeCredentialFile(b, dir, "benchmark", "token")
	cfg := config.Config{
		ChatProxyBaseURL: upstream.URL, ChatProxyVersion: "v1", AuthsDir: dir,
		AuthsReloadInterval: time.Hour, AuthRefreshConcurrency: 1, AccountMaxInflight: 16,
		AffinityTTL: time.Hour, AffinityMaxEntries: 1024, RetryMaxAttempts: 1,
		RetryBaseDelay: time.Millisecond, RateLimitCooldown: time.Minute, QuotaCooldown: 24 * time.Hour,
		ClientName: "grok-shell", ClientVersion: "0.2.93", ClientSurface: "tui",
		ClientIdentifier: "grok-shell", TokenAuth: "xai-grok-cli", StreamCompression: "identity",
	}
	app, err := New(cfg)
	if err != nil {
		b.Fatal(err)
	}
	defer app.Close()
	downstream := httptest.NewServer(app.Handler())
	defer downstream.Close()
	transport := &http.Transport{MaxIdleConns: 16, MaxIdleConnsPerHost: 16, IdleConnTimeout: time.Minute}
	client := &http.Client{Transport: transport}
	defer transport.CloseIdleConnections()

	routes := []struct {
		name, path string
		payload    func(string) map[string]any
	}{
		{name: "responses", path: "/v1/responses", payload: func(text string) map[string]any {
			return map[string]any{"model": "grok-4", "input": text, "stream": true}
		}},
		{name: "chat", path: "/v1/chat/completions", payload: func(text string) map[string]any {
			return map[string]any{"model": "grok-4", "messages": []any{map[string]any{"role": "user", "content": text}}, "stream": true}
		}},
		{name: "anthropic", path: "/v1/messages", payload: func(text string) map[string]any {
			return map[string]any{"model": "grok-4", "max_tokens": 64, "messages": []any{map[string]any{"role": "user", "content": text}}, "stream": true}
		}},
	}
	sizes := []struct {
		name string
		size int
	}{{"1KB", 1 << 10}, {"100KB", 100 << 10}, {"1MB", 1 << 20}}
	for _, route := range routes {
		for _, size := range sizes {
			b.Run(fmt.Sprintf("%s/%s", route.name, size.name), func(b *testing.B) {
				payload, err := json.Marshal(route.payload(strings.Repeat("x", size.size)))
				if err != nil {
					b.Fatal(err)
				}
				b.ReportAllocs()
				b.SetBytes(int64(len(payload)))
				var totalTTFT time.Duration
				var totalProxyText time.Duration
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					started := time.Now()
					request, err := http.NewRequest(http.MethodPost, downstream.URL+route.path, bytes.NewReader(payload))
					if err != nil {
						b.Fatal(err)
					}
					request.Header.Set("Content-Type", "application/json")
					response, err := client.Do(request)
					if err != nil {
						b.Fatal(err)
					}
					scanner := bufio.NewScanner(response.Body)
					seen := false
					var ttft time.Duration
					for scanner.Scan() {
						if !seen && strings.Contains(scanner.Text(), "hello") {
							seen = true
							ttft = time.Since(started)
							totalProxyText += time.Since(origin) - time.Duration(upstreamTextAt.Load())
						}
					}
					totalTTFT += ttft
					_ = response.Body.Close()
					if err := scanner.Err(); err != nil {
						b.Fatal(err)
					}
					if !seen {
						b.Fatal("stream ended without visible text")
					}
				}
				b.ReportMetric(float64(totalTTFT.Nanoseconds())/float64(b.N), "ns/ttft")
				b.ReportMetric(float64(totalProxyText.Nanoseconds())/float64(b.N), "ns/proxy_text")
			})
		}
	}
}
