package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
	"github.com/charmbracelet/crush/internal/config"
)

// Copy of the structures from provider.go
type OllamaModel struct {
	Name       string    `json:"name"`
	ModifiedAt time.Time `json:"modified_at"`
	Size       int64     `json:"size"`
	Digest     string    `json:"digest"`
	Details    struct {
		Format           string   `json:"format"`
		Family           string   `json:"family"`
		Families         []string `json:"families"`
		ParameterSize    string   `json:"parameter_size"`
		QuantizationLevel string  `json:"quantization_level"`
	} `json:"details"`
}

type OllamaTagsResponse struct {
	Models []OllamaModel `json:"models"`
}

// Copy of the functions from provider.go
func fetchOllamaModels(ctx context.Context) ([]catwalk.Model, error) {
	client := &http.Client{
		Timeout: 5 * time.Second,
	}

	req, err := http.NewRequestWithContext(ctx, "GET", "http://localhost:11434/api/tags", nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	resp, err := client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to connect to Ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Ollama API returned status %d", resp.StatusCode)
	}

	var tagsResp OllamaTagsResponse
	if err := json.NewDecoder(resp.Body).Decode(&tagsResp); err != nil {
		return nil, fmt.Errorf("failed to decode Ollama response: %w", err)
	}

	models := make([]catwalk.Model, 0, len(tagsResp.Models))
	for _, ollamaModel := range tagsResp.Models {
		catwalkModel := convertOllamaModel(ollamaModel)
		models = append(models, catwalkModel)
	}

	return models, nil
}

func convertOllamaModel(ollamaModel OllamaModel) catwalk.Model {
	// Extract a more user-friendly display name
	displayName := ollamaModel.Name
	if strings.Contains(displayName, ":") {
		parts := strings.Split(displayName, ":")
		if len(parts) >= 2 {
			displayName = fmt.Sprintf("%s (%s)", parts[0], parts[1])
		}
	}

	// Estimate context window based on model family/name
	contextWindow := int64(4096) // Default context window
	if strings.Contains(strings.ToLower(ollamaModel.Name), "llama") {
		contextWindow = 8192
	}
	if strings.Contains(strings.ToLower(ollamaModel.Name), "codellama") {
		contextWindow = 16384
	}
	if strings.Contains(strings.ToLower(ollamaModel.Name), "mistral") {
		contextWindow = 8192
	}

	// Estimate default max tokens (typically 25% of context window)
	defaultMaxTokens := contextWindow / 4

	// Determine if model supports images (very basic heuristic)
	supportsImages := strings.Contains(strings.ToLower(ollamaModel.Name), "vision") ||
		strings.Contains(strings.ToLower(ollamaModel.Name), "llava")

	return catwalk.Model{
		ID:               ollamaModel.Name,
		Name:             displayName,
		ContextWindow:    contextWindow,
		DefaultMaxTokens: defaultMaxTokens,
		SupportsImages:   supportsImages,
		// Local models have no API costs
		CostPer1MIn:        0,
		CostPer1MOut:       0,
		CostPer1MInCached:  0,
		CostPer1MOutCached: 0,
	}
}

func createOllamaProvider(ctx context.Context) (*catwalk.Provider, error) {
	models, err := fetchOllamaModels(ctx)
	if err != nil {
		return nil, fmt.Errorf("failed to fetch Ollama models: %w", err)
	}

	if len(models) == 0 {
		return nil, fmt.Errorf("no models found in local Ollama installation")
	}

	// Choose default models based on available models
	var defaultLargeModelID, defaultSmallModelID string
	
	// Look for common "large" models first
	for _, model := range models {
		modelName := strings.ToLower(model.Name)
		if strings.Contains(modelName, "70b") || strings.Contains(modelName, "13b") {
			if defaultLargeModelID == "" {
				defaultLargeModelID = model.ID
			}
		}
	}
	
	// Look for common "small" models
	for _, model := range models {
		modelName := strings.ToLower(model.Name)
		if strings.Contains(modelName, "7b") || strings.Contains(modelName, "3b") {
			if defaultSmallModelID == "" {
				defaultSmallModelID = model.ID
			}
		}
	}

	// If no specific size models found, use the first and second models
	if defaultLargeModelID == "" && len(models) > 0 {
		defaultLargeModelID = models[0].ID
	}
	if defaultSmallModelID == "" && len(models) > 1 {
		defaultSmallModelID = models[1].ID
	} else if defaultSmallModelID == "" {
		defaultSmallModelID = defaultLargeModelID
	}

	return &catwalk.Provider{
		Name:                "Ollama (Local)",
		ID:                  "ollama",
		APIKey:              "", // Ollama doesn't require API key
		APIEndpoint:         "http://localhost:11434/v1",
		Type:                catwalk.TypeOpenAI, // Ollama is OpenAI-compatible
		DefaultLargeModelID: defaultLargeModelID,
		DefaultSmallModelID: defaultSmallModelID,
		Models:              models,
	}, nil
}

func main() {
	fmt.Println("Testing Ollama provider...")
	
	ctx := context.Background()
	
	// Test fetching models
	models, err := fetchOllamaModels(ctx)
	if err != nil {
		log.Printf("Error fetching models: %v", err)
		return
	}
	
	fmt.Printf("Found %d models:\n", len(models))
	for _, model := range models {
		fmt.Printf("- %s: %s (context: %d, max_tokens: %d, supports_images: %t)\n", 
			model.ID, model.Name, model.ContextWindow, model.DefaultMaxTokens, model.SupportsImages)
	}
	
	// Test creating provider
	provider, err := createOllamaProvider(ctx)
	if err != nil {
		log.Printf("Error creating provider: %v", err)
		return
	}
	
	fmt.Printf("\nCreated Ollama provider:\n")
	fmt.Printf("- Name: %s\n", provider.Name)
	fmt.Printf("- ID: %s\n", provider.ID)
	fmt.Printf("- Type: %s\n", provider.Type)
	fmt.Printf("- Endpoint: %s\n", provider.APIEndpoint)
	fmt.Printf("- Models: %d\n", len(provider.Models))
	fmt.Printf("- Default Large Model: %s\n", provider.DefaultLargeModelID)
	fmt.Printf("- Default Small Model: %s\n", provider.DefaultSmallModelID)
	
	fmt.Println("\nTesting provider integration...")
	
	// Test provider loading
	providers, err := config.Providers()
	if err != nil {
		log.Printf("Error loading providers: %v", err)
		return
	}
	
	fmt.Printf("Total providers loaded: %d\n", len(providers))
	for _, p := range providers {
		fmt.Printf("- %s (%s): %d models\n", p.Name, p.ID, len(p.Models))
		if string(p.ID) == "ollama" {
			fmt.Printf("  Found Ollama provider! ✅\n")
			for i, model := range p.Models {
				if i < 3 { // Show first 3 models
					fmt.Printf("    * %s (%s)\n", model.Name, model.ID)
				}
			}
			if len(p.Models) > 3 {
				fmt.Printf("    ... and %d more models\n", len(p.Models)-3)
			}
		}
	}
}