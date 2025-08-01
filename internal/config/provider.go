package config

import (
	"cmp"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/catwalk/pkg/catwalk"
)

type ProviderClient interface {
	GetProviders() ([]catwalk.Provider, error)
}

var (
	providerOnce sync.Once
	providerList []catwalk.Provider
)

// file to cache provider data
func providerCacheFileData() string {
	xdgDataHome := os.Getenv("XDG_DATA_HOME")
	if xdgDataHome != "" {
		return filepath.Join(xdgDataHome, appName, "providers.json")
	}

	// return the path to the main data directory
	// for windows, it should be in `%LOCALAPPDATA%/crush/`
	// for linux and macOS, it should be in `$HOME/.local/share/crush/`
	if runtime.GOOS == "windows" {
		localAppData := os.Getenv("LOCALAPPDATA")
		if localAppData == "" {
			localAppData = filepath.Join(os.Getenv("USERPROFILE"), "AppData", "Local")
		}
		return filepath.Join(localAppData, appName, "providers.json")
	}

	return filepath.Join(os.Getenv("HOME"), ".local", "share", appName, "providers.json")
}

func saveProvidersInCache(path string, providers []catwalk.Provider) error {
	slog.Info("Saving cached provider data", "path", path)
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return fmt.Errorf("failed to create directory for provider cache: %w", err)
	}

	data, err := json.MarshalIndent(providers, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal provider data: %w", err)
	}

	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("failed to write provider data to cache: %w", err)
	}
	return nil
}

func loadProvidersFromCache(path string) ([]catwalk.Provider, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read provider cache file: %w", err)
	}

	var providers []catwalk.Provider
	if err := json.Unmarshal(data, &providers); err != nil {
		return nil, fmt.Errorf("failed to unmarshal provider data from cache: %w", err)
	}
	return providers, nil
}

func Providers() ([]catwalk.Provider, error) {
	catwalkURL := cmp.Or(os.Getenv("CATWALK_URL"), defaultCatwalkURL)
	client := catwalk.NewWithURL(catwalkURL)
	path := providerCacheFileData()
	return loadProvidersOnce(client, path)
}

func loadProvidersOnce(client ProviderClient, path string) ([]catwalk.Provider, error) {
	var err error
	providerOnce.Do(func() {
		providerList, err = loadProviders(client, path)
	})
	if err != nil {
		return nil, err
	}
	return providerList, nil
}

func loadProviders(client ProviderClient, path string) (providerList []catwalk.Provider, err error) {
	// if cache is not stale, load from it
	stale, exists := isCacheStale(path)
	if !stale {
		slog.Info("Using cached provider data", "path", path)
		providerList, err = loadProvidersFromCache(path)
		if len(providerList) > 0 && err == nil {
			go func() {
				slog.Info("Updating provider cache in background")
				updated, uerr := client.GetProviders()
				if len(updated) > 0 && uerr == nil {
					// Add dynamic Ollama provider to the updated list
					if ollamaProvider, ollamaErr := createOllamaProvider(context.Background()); ollamaErr == nil {
						updated = append(updated, *ollamaProvider)
					}
					_ = saveProvidersInCache(path, updated)
				}
			}()
			// Try to add dynamic Ollama provider to cached list
			if ollamaProvider, ollamaErr := createOllamaProvider(context.Background()); ollamaErr == nil {
				providerList = append(providerList, *ollamaProvider)
			}
			return
		}
	}

	slog.Info("Getting live provider data")
	providerList, err = client.GetProviders()
	
	// Add dynamic Ollama provider if available
	if ollamaProvider, ollamaErr := createOllamaProvider(context.Background()); ollamaErr == nil {
		slog.Info("Adding Ollama provider with models", "model_count", len(ollamaProvider.Models))
		providerList = append(providerList, *ollamaProvider)
	} else {
		slog.Debug("Ollama provider not available", "error", ollamaErr)
	}
	
	if len(providerList) > 0 && err == nil {
		err = saveProvidersInCache(path, providerList)
		return
	}
	if !exists {
		err = fmt.Errorf("failed to load providers")
		return
	}
	providerList, err = loadProvidersFromCache(path)
	// Try to add dynamic Ollama provider to fallback cached list
	if ollamaProvider, ollamaErr := createOllamaProvider(context.Background()); ollamaErr == nil {
		providerList = append(providerList, *ollamaProvider)
	}
	return
}

func isCacheStale(path string) (stale, exists bool) {
	info, err := os.Stat(path)
	if err != nil {
		return true, false
	}
	return time.Since(info.ModTime()) > 24*time.Hour, true
}

// OllamaModel represents a model returned by Ollama's /api/tags endpoint
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

// OllamaTagsResponse represents the response from Ollama's /api/tags endpoint
type OllamaTagsResponse struct {
	Models []OllamaModel `json:"models"`
}

// fetchOllamaModels calls Ollama's /api/tags endpoint to get locally available models
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

// convertOllamaModel converts an Ollama model to a catwalk.Model
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

// createOllamaProvider creates a dynamic Ollama provider with locally available models
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
