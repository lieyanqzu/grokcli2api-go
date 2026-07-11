package config

import (
	"strings"
	"testing"
)

func TestLoadValidatesStreamCompression(t *testing.T) {
	t.Setenv("GROK_STREAM_COMPRESSION", "brotli")
	_, err := Load()
	if err == nil || !strings.Contains(err.Error(), "GROK_STREAM_COMPRESSION") {
		t.Fatalf("Load() error = %v, want stream compression validation error", err)
	}
}
