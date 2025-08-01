package config

import (
	"encoding/json"
	"testing"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/env"
	"github.com/stretchr/testify/require"
)

func TestOllamaProviderConfiguration(t *testing.T) {
	ollamaConfig := `{
		"providers": {
			"ollama": {
				"type": "openai",
				"base_url": "http://localhost:11434/v1",
				"api_key": "",
				"models": [
					{
						"id": "codellama:34b",
						"name": "Code Llama 34B",
						"cost_per_1m_in": 0,
						"cost_per_1m_out": 0,
						"cost_per_1m_in_cached": 0,
						"cost_per_1m_out_cached": 0,
						"context_window": 16384,
						"default_max_tokens": 4096,
						"can_reason": false,
						"has_reasoning_efforts": false,
						"supports_attachments": false
					},
					{
						"id": "codellama:7b", 
						"name": "Code Llama 7B",
						"cost_per_1m_in": 0,
						"cost_per_1m_out": 0,
						"cost_per_1m_in_cached": 0,
						"cost_per_1m_out_cached": 0,
						"context_window": 16384,
						"default_max_tokens": 4096,
						"can_reason": false,
						"has_reasoning_efforts": false,
						"supports_attachments": false
					}
				]
			}
		}
	}`

	var config Config
	err := json.Unmarshal([]byte(ollamaConfig), &config)
	require.NoError(t, err, "Should be able to parse Ollama configuration")

	// Set defaults and configure the provider properly
	config.setDefaults("/tmp")
	
	// Create mock environment with no special variables needed for Ollama
	env := env.NewFromMap(map[string]string{})
	resolver := NewEnvironmentVariableResolver(env)
	
	// Configure providers (this will set the ID field and validate)
	err = config.configureProviders(env, resolver, []catwalk.Provider{})
	require.NoError(t, err, "Should be able to configure Ollama provider")

	// Validate that the provider was processed correctly
	ollamaProvider, exists := config.Providers.Get("ollama")
	require.True(t, exists, "Ollama provider should exist after configuration")
	require.Equal(t, "ollama", ollamaProvider.ID)
	require.Equal(t, catwalk.TypeOpenAI, ollamaProvider.Type)
	require.Equal(t, "http://localhost:11434/v1", ollamaProvider.BaseURL)
	require.Equal(t, "", ollamaProvider.APIKey)
	require.Len(t, ollamaProvider.Models, 2)

	// Validate the first model
	firstModel := ollamaProvider.Models[0]
	require.Equal(t, "codellama:34b", firstModel.ID)
	require.Equal(t, "Code Llama 34B", firstModel.Name)
	require.Equal(t, float64(0), firstModel.CostPer1MIn)
	require.Equal(t, float64(0), firstModel.CostPer1MOut)
	require.Equal(t, int64(16384), firstModel.ContextWindow)
	require.Equal(t, int64(4096), firstModel.DefaultMaxTokens)

	// Validate the second model
	secondModel := ollamaProvider.Models[1]
	require.Equal(t, "codellama:7b", secondModel.ID)
	require.Equal(t, "Code Llama 7B", secondModel.Name)
	require.Equal(t, int64(16384), secondModel.ContextWindow)
}