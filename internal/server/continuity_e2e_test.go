package server

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Futureppo/grokcli2api-go/internal/auth"
	"github.com/Futureppo/grokcli2api-go/internal/config"
	"github.com/Futureppo/grokcli2api-go/internal/modelcatalog"
)

func TestOpaqueStateOwnershipFailsClosedBeforeUpstream(t *testing.T) {
	var inferenceCalls atomic.Int32
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeTestCatalog(w, modelcatalog.BackendResponses)
			return
		}
		inferenceCalls.Add(1)
		writeJSON(w, http.StatusInternalServerError, map[string]any{"error": "unexpected inference"})
	}))
	defer upstream.Close()

	s := newContinuityE2EServer(t, upstream.URL, []string{"account-a", "account-b"})
	accounts := s.pool.AccountIDs()
	if len(accounts) != 2 {
		t.Fatalf("accounts=%d, want 2", len(accounts))
	}
	tenantA := s.pool.TenantID("tenant-a-key")
	tenantB := s.pool.TenantID("tenant-b-key")

	// A Messages thinking signature owned by tenant A must not be forwarded by
	// tenant B when the selected account preserves it on a Responses backend.
	s.continuity.BindStateToken(tenantA, "foreign-signature", accounts[0], "grok-4", modelcatalog.BackendResponses, "session-a", 1)
	assertStateRejected(t, s, "tenant-b-key", "/v1/messages", `{
		"model":"grok-4","max_tokens":8,
		"messages":[
			{"role":"assistant","content":[{"type":"thinking","thinking":"summary","signature":"foreign-signature"}]},
			{"role":"user","content":"continue"}
		]
	}`, "", http.StatusNotFound)

	// Checking every retained item prevents an owned-first/foreign-second
	// encrypted reasoning sequence from bypassing tenant isolation.
	s.continuity.BindStateToken(tenantB, "owned-ciphertext", accounts[0], "grok-4", modelcatalog.BackendResponses, "session-b", 2)
	s.continuity.BindStateToken(tenantA, "foreign-ciphertext", accounts[0], "grok-4", modelcatalog.BackendResponses, "session-a", 2)
	assertStateRejected(t, s, "tenant-b-key", "/v1/responses", `{
		"model":"grok-4","input":[
			{"type":"reasoning","encrypted_content":"owned-ciphertext"},
			{"type":"reasoning","encrypted_content":"foreign-ciphertext"},
			{"type":"message","role":"user","content":[{"type":"input_text","text":"continue"}]}
		]
	}`, "", http.StatusNotFound)

	// An explicit client session cannot hide a stolen opaque token behind its
	// higher affinity priority.
	sessionAffinity := auth.Affinity{Tenant: tenantB, Key: "session:client-session", Mode: auth.AffinityHard}
	s.continuity.Bind(continuityKey(tenantB, sessionAffinity, "grok-4"), sessionAffinity, accounts[0], "grok-4", modelcatalog.BackendResponses, "session-b", 3)
	assertStateRejected(t, s, "tenant-b-key", "/v1/responses", `{
		"model":"grok-4","input":[
			{"type":"reasoning","encrypted_content":"foreign-ciphertext"},
			{"type":"message","role":"user","content":"continue"}
		]
	}`, "client-session", http.StatusNotFound)

	// Two valid handles for the same tenant still cannot be combined when they
	// belong to different upstream accounts/sessions.
	s.continuity.BindStateToken(tenantB, "route-one", accounts[0], "grok-4", modelcatalog.BackendResponses, "session-one", 4)
	s.continuity.BindStateToken(tenantB, "route-two", accounts[1], "grok-4", modelcatalog.BackendResponses, "session-two", 4)
	assertStateRejected(t, s, "tenant-b-key", "/v1/responses", `{
		"model":"grok-4","input":[
			{"type":"reasoning","encrypted_content":"route-one"},
			{"type":"reasoning","encrypted_content":"route-two"},
			{"type":"message","role":"user","content":"continue"}
		]
	}`, "", http.StatusServiceUnavailable)

	if got := inferenceCalls.Load(); got != 0 {
		t.Fatalf("unsafe state reached upstream %d times", got)
	}
}

func TestDroppedStateStartsFreshWithoutFalseOwnershipError(t *testing.T) {
	var mu sync.Mutex
	var inferenceBodies []map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			writeTestCatalog(w, modelcatalog.BackendResponses)
			return
		}
		var body map[string]any
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
			t.Errorf("decode inference: %v", err)
			return
		}
		mu.Lock()
		inferenceBodies = append(inferenceBodies, body)
		mu.Unlock()
		switch r.URL.Path {
		case "/v1/responses":
			writeJSON(w, http.StatusOK, map[string]any{
				"id": "resp-fresh", "object": "response", "status": "completed", "output": []any{},
				"usage": map[string]any{"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
			})
		case "/v1/chat/completions":
			writeJSON(w, http.StatusOK, map[string]any{"choices": []any{map[string]any{
				"index": 0, "finish_reason": "stop", "message": map[string]any{"role": "assistant", "content": "ok"},
			}}})
		default:
			t.Errorf("unexpected inference path %s", r.URL.Path)
		}
	}))
	defer upstream.Close()

	s := newContinuityE2EServer(t, upstream.URL, []string{"account"})

	// A lost store:false tool replay deliberately removes previous_response_id
	// before the immutable plan is built, so the ownership guard must not reject
	// the resulting fresh request.
	replayBody := `{
		"model":"grok-4","previous_response_id":"lost-store-false","store":false,
		"input":[
			{"type":"function_call_output","call_id":"lost","output":"x"},
			{"type":"message","role":"user","content":"continue"}
		]
	}`
	replay := httptest.NewRecorder()
	replayReq := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(replayBody))
	replayReq.Header.Set("Authorization", "Bearer tenant-b-key")
	s.Handler().ServeHTTP(replay, replayReq)
	if replay.Code != http.StatusOK {
		t.Fatalf("store:false replay status=%d body=%s", replay.Code, replay.Body.String())
	}

	// When the only account's selected backend cannot express Responses state,
	// previous_response_id is silently removed and a new conversation is sent.
	accountID := s.pool.AccountIDs()[0]
	if err := s.pool.UpdateModelDescriptors(accountID, []modelcatalog.ModelDescriptor{{
		ID: "grok-4", WireModel: "grok-4-chat", Backend: modelcatalog.BackendChatCompletions, SupportedInAPI: true,
	}}, "", time.Now()); err != nil {
		t.Fatal(err)
	}
	dropped := httptest.NewRecorder()
	droppedReq := httptest.NewRequest(http.MethodPost, "/v1/responses", strings.NewReader(`{
		"model":"grok-4","previous_response_id":"unknown-but-dropped","input":"hello"
	}`))
	droppedReq.Header.Set("Authorization", "Bearer tenant-b-key")
	s.Handler().ServeHTTP(dropped, droppedReq)
	if dropped.Code != http.StatusOK {
		t.Fatalf("dropped-state status=%d body=%s", dropped.Code, dropped.Body.String())
	}

	mu.Lock()
	defer mu.Unlock()
	if len(inferenceBodies) != 2 {
		t.Fatalf("inference bodies=%d, want 2", len(inferenceBodies))
	}
	for index, body := range inferenceBodies {
		if _, leaked := body["previous_response_id"]; leaked {
			t.Fatalf("request %d leaked previous_response_id: %#v", index, body)
		}
	}
}

func assertStateRejected(t *testing.T, s *Server, apiKey, path, body, session string, wantStatus int) {
	t.Helper()
	recorder := httptest.NewRecorder()
	request := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	request.Header.Set("Authorization", "Bearer "+apiKey)
	if session != "" {
		request.Header.Set("X-Grok-Session-ID", session)
	}
	s.Handler().ServeHTTP(recorder, request)
	if recorder.Code != wantStatus {
		t.Fatalf("path=%s status=%d want=%d body=%s", path, recorder.Code, wantStatus, recorder.Body.String())
	}
}

func newContinuityE2EServer(t *testing.T, upstream string, subjects []string) *Server {
	t.Helper()
	dir := t.TempDir()
	for index, subject := range subjects {
		writeCredentialFileModels(t, dir, subject, fmt.Sprintf("token-%d", index), []string{"grok-4"})
	}
	cfg := config.Config{
		ChatProxyBaseURL: upstream, ChatProxyVersion: "v1", AuthsDir: dir,
		AuthsReloadInterval: time.Hour, AuthRefreshConcurrency: 1, AccountMaxInflight: 2,
		ModelsRefreshInterval: 6 * time.Hour, RetryMaxAttempts: 3, RetryBaseDelay: time.Millisecond,
		RateLimitCooldown: time.Minute, QuotaCooldown: 24 * time.Hour,
		AffinityTTL: time.Hour, AffinityMaxEntries: 128,
		ClientName: "grok-cli", ClientVersion: "0.2.102", ClientSurface: "headless",
		ClientIdentifier: "grok-cli", TokenAuth: "xai-grok-cli", StreamCompression: "identity",
		APIKeys: []string{"tenant-a-key", "tenant-b-key"},
	}
	s, err := New(cfg)
	if err != nil {
		t.Fatal(err)
	}
	for _, accountID := range s.pool.AccountIDs() {
		if err := s.pool.UpdateModelDescriptors(accountID, []modelcatalog.ModelDescriptor{{
			ID: "grok-4", WireModel: "grok-4", Backend: modelcatalog.BackendResponses, SupportedInAPI: true,
		}}, "", time.Now()); err != nil {
			s.Close()
			t.Fatal(err)
		}
	}
	t.Cleanup(s.Close)
	return s
}

func writeTestCatalog(w http.ResponseWriter, backend modelcatalog.Backend) {
	writeJSON(w, http.StatusOK, map[string]any{"data": []any{map[string]any{
		"id": "grok-4", "model": "grok-4", "apiBackend": string(backend), "supportedInApi": true,
	}}})
}
