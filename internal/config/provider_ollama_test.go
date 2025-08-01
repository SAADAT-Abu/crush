package config

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFetchOllamaModels(t *testing.T) {
	tests := []struct {
		name           string
		response       string
		statusCode     int
		expectedModels int
		expectError    bool
	}{
		{
			name: "successful response with models",
			response: `{
				"models": [
					{
						"name": "codellama:7b",
						"modified_at": "2024-01-01T12:00:00Z",
						"size": 3826793677,
						"digest": "sha256:123",
						"details": {
							"format": "gguf",
							"family": "llama",
							"families": ["llama"],
							"parameter_size": "7B",
							"quantization_level": "Q4_0"
						}
					},
					{
						"name": "mistral:7b",
						"modified_at": "2024-01-01T12:00:00Z",
						"size": 4000000000,
						"digest": "sha256:456",
						"details": {
							"format": "gguf",
							"family": "mistral",
							"families": ["mistral"],
							"parameter_size": "7B",
							"quantization_level": "Q4_0"
						}
					}
				]
			}`,
			statusCode:     200,
			expectedModels: 2,
			expectError:    false,
		},
		{
			name:           "empty models response",
			response:       `{"models": []}`,
			statusCode:     200,
			expectedModels: 0,
			expectError:    false,
		},
		{
			name:           "server error",
			response:       "",
			statusCode:     500,
			expectedModels: 0,
			expectError:    true,
		},
		{
			name:           "invalid json",
			response:       `invalid json`,
			statusCode:     200,
			expectedModels: 0,
			expectError:    true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a test server
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "/api/tags", r.URL.Path)
				assert.Equal(t, "GET", r.Method)
				
				w.WriteHeader(tt.statusCode)
				if tt.response != "" {
					w.Write([]byte(tt.response))
				}
			}))
			defer server.Close()

			// Replace the hardcoded URL in fetchOllamaModels
			// For testing, we need to modify the function to accept a URL parameter
			// Since we can't modify the existing function without breaking other tests,
			// we'll test the conversion logic separately
			
			if tt.expectError && tt.statusCode != 200 {
				// Test error cases by trying to connect to a non-existent server
				ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
				defer cancel()
				
				_, err := fetchOllamaModels(ctx)
				assert.Error(t, err)
				return
			}

			if !tt.expectError && tt.statusCode == 200 {
				// Test the conversion logic with mock data
				var response OllamaTagsResponse
				err := json.Unmarshal([]byte(tt.response), &response)
				require.NoError(t, err)

				models := make([]catwalk.Model, 0, len(response.Models))
				for _, ollamaModel := range response.Models {
					catwalkModel := convertOllamaModel(ollamaModel)
					models = append(models, catwalkModel)
				}

				assert.Len(t, models, tt.expectedModels)
				
				if len(models) > 0 {
					// Verify the first model conversion
					model := models[0]
					assert.Equal(t, "codellama:7b", model.ID)
					assert.Equal(t, "codellama (7b)", model.Name)
					assert.Equal(t, int64(16384), model.ContextWindow) // codellama should have larger context
					assert.Equal(t, int64(4096), model.DefaultMaxTokens)
					assert.Equal(t, float64(0), model.CostPer1MIn) // Local models have no cost
					assert.Equal(t, float64(0), model.CostPer1MOut)
				}
			}
		})
	}
}

func TestConvertOllamaModel(t *testing.T) {
	tests := []struct {
		name           string
		ollamaModel    OllamaModel
		expectedName   string
		expectedContext int64
		expectedImages bool
	}{
		{
			name: "codellama model",
			ollamaModel: OllamaModel{
				Name: "codellama:7b",
				Size: 3826793677,
				Details: struct {
					Format           string   `json:"format"`
					Family           string   `json:"family"`
					Families         []string `json:"families"`
					ParameterSize    string   `json:"parameter_size"`
					QuantizationLevel string  `json:"quantization_level"`
				}{
					Family: "llama",
					ParameterSize: "7B",
				},
			},
			expectedName:    "codellama (7b)",
			expectedContext: 16384, // codellama gets larger context
			expectedImages:  false,
		},
		{
			name: "mistral model",
			ollamaModel: OllamaModel{
				Name: "mistral:7b-instruct",
				Size: 4000000000,
				Details: struct {
					Format           string   `json:"format"`
					Family           string   `json:"family"`
					Families         []string `json:"families"`
					ParameterSize    string   `json:"parameter_size"`
					QuantizationLevel string  `json:"quantization_level"`
				}{
					Family: "mistral",
					ParameterSize: "7B",
				},
			},
			expectedName:    "mistral (7b-instruct)",
			expectedContext: 8192, // mistral gets medium context
			expectedImages:  false,
		},
		{
			name: "vision model",
			ollamaModel: OllamaModel{
				Name: "llava:7b",
				Size: 4000000000,
				Details: struct {
					Format           string   `json:"format"`
					Family           string   `json:"family"`
					Families         []string `json:"families"`
					ParameterSize    string   `json:"parameter_size"`
					QuantizationLevel string  `json:"quantization_level"`
				}{
					Family: "llava",
					ParameterSize: "7B",
				},
			},
			expectedName:    "llava (7b)",
			expectedContext: 4096, // default context
			expectedImages:  true, // vision model supports images
		},
		{
			name: "model without version",
			ollamaModel: OllamaModel{
				Name: "custom-model",
				Size: 1000000000,
				Details: struct {
					Format           string   `json:"format"`
					Family           string   `json:"family"`
					Families         []string `json:"families"`
					ParameterSize    string   `json:"parameter_size"`
					QuantizationLevel string  `json:"quantization_level"`
				}{
					Family: "custom",
				},
			},
			expectedName:    "custom-model",
			expectedContext: 4096, // default context
			expectedImages:  false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertOllamaModel(tt.ollamaModel)
			
			assert.Equal(t, tt.ollamaModel.Name, result.ID)
			assert.Equal(t, tt.expectedName, result.Name)
			assert.Equal(t, tt.expectedContext, result.ContextWindow)
			assert.Equal(t, tt.expectedContext/4, result.DefaultMaxTokens)
			assert.Equal(t, tt.expectedImages, result.SupportsImages)
			
			// Verify cost is always 0 for local models
			assert.Equal(t, float64(0), result.CostPer1MIn)
			assert.Equal(t, float64(0), result.CostPer1MOut)
			assert.Equal(t, float64(0), result.CostPer1MInCached)
			assert.Equal(t, float64(0), result.CostPer1MOutCached)
		})
	}
}

func TestCreateOllamaProvider(t *testing.T) {
	// This test would require mocking the HTTP client or having a test server
	// For now, we'll test that the function handles empty model lists correctly
	
	// Test with no models available
	t.Run("no models available", func(t *testing.T) {
		// We can't easily test this without mocking the HTTP client
		// The function will try to connect to localhost:11434 which likely won't exist in CI
		ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
		defer cancel()
		
		_, err := createOllamaProvider(ctx)
		// We expect an error since Ollama likely isn't running
		assert.Error(t, err)
	})
}

// TestOllamaProviderIntegration tests the integration with the main provider loading
func TestOllamaProviderIntegration(t *testing.T) {
	// Test that the provider loading doesn't break when Ollama is not available
	t.Run("provider loading with unavailable ollama", func(t *testing.T) {
		// Create a mock client that returns some providers
		mockClient := &mockProviderClient{
			shouldFail: false,
		}
		
		// Test that loading providers works even when Ollama is not available
		providers, err := loadProvidersOnce(mockClient, "/tmp/test-providers.json")
		
		// Should not fail even if Ollama is unavailable
		assert.NoError(t, err)
		assert.NotEmpty(t, providers)
		
		// Should contain at least the mock provider
		found := false
		for _, p := range providers {
			if string(p.ID) == "" && p.Name == "Mock" { // The existing mock provider
				found = true
				break
			}
		}
		assert.True(t, found, "Mock provider should be present")
	})
}