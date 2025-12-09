package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net/http"
	"sync"
	"time"
)

const (
	qdrantBaseURL    = "http://localhost:6333"
	articlesColl     = "articles"            // koleksi untuk artikel dimensi 4
	embeddingsColl   = "articles_embeddings" // koleksi untuk embedding dimensi 128
	articlesVecDim   = 4
	embeddingsVecDim = 128
	batchSize        = 100
)

var httpClient = &http.Client{
	Timeout: 30 * time.Second,
}

type VectorParams struct {
	Size     int    `json:"size"`
	Distance string `json:"distance"`
}

type CreateCollectionBody struct {
	Vectors interface{} `json:"vectors"`
}

type Point struct {
	ID      int64                  `json:"id"`
	Vector  []float32              `json:"vector"`
	Payload map[string]interface{} `json:"payload,omitempty"`
}

type UpsertPointsBody struct {
	Points []Point `json:"points"`
}

type SearchRequest struct {
	Vector      []float32 `json:"vector"`
	Limit       int       `json:"limit"`
	WithPayload bool      `json:"with_payload"`
}

type SearchResponse struct {
	Result []struct {
		ID      int64                  `json:"id"`
		Payload map[string]interface{} `json:"payload"`
		Score   *float64               `json:"score,omitempty"` // nilai similarity
	} `json:"result"`
}

type CollectionInfoResponse struct {
	Result struct {
		Config struct {
			Params struct {
				Vectors struct {
					Size int `json:"size"`
				} `json:"vectors"`
			} `json:"params"`
		} `json:"config"`
		PointsCount int `json:"points_count"` // jumlah total data dalam koleksi
	} `json:"result"`
	Status string  `json:"status"`
	Time   float64 `json:"time"`
}

// createCollectionIfNotExists akan membuat collection jika belum ada.
// Jika sudah ada, dicek apakah ukuran vektornya cocok.
func createCollectionIfNotExists(ctx context.Context, collection string, vecSize int) error {

	// cek apakah collection sudah ada
	info, err := getCollectionInfo(ctx, collection)
	if err == nil {
		// jika sudah ada pastikan ukuran vector sesuai
		if info.Result.Config.Params.Vectors.Size == vecSize {
			return nil
		}
		return fmt.Errorf(
			"koleksi %q sudah ada dengan ukuran %d (harus %d)",
			collection,
			info.Result.Config.Params.Vectors.Size,
			vecSize,
		)
	}

	// jika belum ada buat collection baru
	url := fmt.Sprintf("%s/collections/%s", qdrantBaseURL, collection)
	body := CreateCollectionBody{
		Vectors: VectorParams{
			Size:     vecSize,
			Distance: "Cosine",
		},
	}
	data, _ := json.Marshal(body)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	b, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("gagal membuat koleksi %s: %s", collection, string(b))
}

// mengambil info collection
func getCollectionInfo(ctx context.Context, collection string) (*CollectionInfoResponse, error) {
	url := fmt.Sprintf("%s/collections/%s", qdrantBaseURL, collection)

	req, _ := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		var info CollectionInfoResponse
		if err := json.NewDecoder(resp.Body).Decode(&info); err != nil {
			return nil, err
		}
		return &info, nil
	}

	b, _ := io.ReadAll(resp.Body)
	return nil, fmt.Errorf("gagal getCollectionInfo: %s", string(b))
}

// upsert data ke Qdrant
func upsertPoints(ctx context.Context, collection string, points []Point) error {
	url := fmt.Sprintf("%s/collections/%s/points", qdrantBaseURL, collection)

	body := UpsertPointsBody{Points: points}
	data, _ := json.Marshal(body)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPut, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 200 && resp.StatusCode < 300 {
		return nil
	}

	b, _ := io.ReadAll(resp.Body)
	return fmt.Errorf("gagal upsert ke %s: %s", collection, string(b))
}

// mencari vector paling mirip
func searchNearest(ctx context.Context, collection string, query []float32, limit int) (string, error) {
	url := fmt.Sprintf("%s/collections/%s/points/search", qdrantBaseURL, collection)

	reqBody := SearchRequest{
		Vector:      query,
		Limit:       limit,
		WithPayload: true,
	}
	data, _ := json.Marshal(reqBody)

	req, _ := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(data))
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		b, _ := io.ReadAll(resp.Body)
		return "", fmt.Errorf("search gagal: %s", string(b))
	}

	var sr SearchResponse
	if err := json.NewDecoder(resp.Body).Decode(&sr); err != nil {
		return "", err
	}

	if len(sr.Result) == 0 {
		return "", errors.New("hasil pencarian kosong")
	}

	// ambil title dari payload hasil pertama
	title, ok := sr.Result[0].Payload["title"].(string)
	if !ok {
		return "", errors.New("payload tidak memiliki title")
	}

	return title, nil
}

// upsert concurrent dengan batch
func BulkInsert(ctx context.Context, collection string, vectors [][]float32, payloads []map[string]interface{}) error {

	if len(vectors) != len(payloads) {
		return errors.New("jumlah vector dan payload tidak sama")
	}

	total := len(vectors)
	if total == 0 {
		return nil
	}

	// cek ukuran vector collection
	info, err := getCollectionInfo(ctx, collection)
	if err != nil {
		return fmt.Errorf("gagal mengambil info collection: %w", err)
	}

	expectedSize := info.Result.Config.Params.Vectors.Size
	if len(vectors[0]) != expectedSize {
		return fmt.Errorf("dimensi vector tidak cocok (harus %d)", expectedSize)
	}

	// bagi data menjadi batch
	type batch struct{ start, end int }
	var batches []batch
	for i := 0; i < total; i += batchSize {
		j := i + batchSize
		if j > total {
			j = total
		}
		batches = append(batches, batch{start: i, end: j})
	}

	var wg sync.WaitGroup
	errCh := make(chan error, len(batches))
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	startingID := int64(1000)

	for bi, b := range batches {
		wg.Add(1)

		go func(batchIndex int, b batch) {
			defer wg.Done()

			// siapkan data untuk batch ini
			points := make([]Point, 0, b.end-b.start)
			for i := b.start; i < b.end; i++ {

				// clone payload agar tidak bentrok antar goroutine
				var pl map[string]interface{}
				if payloads[i] != nil {
					pl = make(map[string]interface{}, len(payloads[i]))
					for k, v := range payloads[i] {
						pl[k] = v
					}
				}

				points = append(points, Point{
					ID:      startingID + int64(i),
					Vector:  vectors[i],
					Payload: pl,
				})
			}

			// upsert dengan retry sederhana
			var lastErr error
			maxRetries := 2

			for attempt := 0; attempt <= maxRetries; attempt++ {
				if ctx.Err() != nil {
					return
				}

				if err := upsertPoints(ctx, collection, points); err != nil {
					lastErr = err
					time.Sleep(time.Duration(200*(attempt+1)) * time.Millisecond)
					continue
				}

				log.Printf("Batch %d berhasil upsert (%d data)", batchIndex+1, len(points))
				return
			}

			select {
			case errCh <- fmt.Errorf("batch %d error: %w", batchIndex+1, lastErr):
			default:
			}

			cancel()

		}(bi, b)
	}

	wg.Wait()
	close(errCh)

	if err, ok := <-errCh; ok {
		return err
	}

	return nil
}

func main() {
	ctx := context.Background()

	// 1. membuat koleksi artikel dimensi 4
	log.Printf("Menyiapkan koleksi %q (dim=%d)...", articlesColl, articlesVecDim)
	if err := createCollectionIfNotExists(ctx, articlesColl, articlesVecDim); err != nil {
		log.Fatalf("Gagal membuat koleksi articles: %v", err)
	}
	log.Println("Koleksi articles siap.")

	// upsert 3 artikel contoh
	articles := []Point{
		{ID: 1, Vector: []float32{0.1, 0.2, 0.3, 0.4}, Payload: map[string]interface{}{"title": "Go qdrant test"}},
		{ID: 2, Vector: []float32{0.9, 0.1, 0.2, 0.3}, Payload: map[string]interface{}{"title": "Vektor data"}},
		{ID: 3, Vector: []float32{0.2, 0.2, 0.2, 0.2}, Payload: map[string]interface{}{"title": "Random artikel"}},
	}
	log.Println("Mengirim 3 artikel contoh...")
	if err := upsertPoints(ctx, articlesColl, articles); err != nil {
		log.Fatalf("Gagal menginsert 3 artikel: %v", err)
	}
	log.Println("Artikel berhasil disimpan.")

	// cari artikel paling mirip
	query := []float32{0.1, 0.2, 0.3, 0.4}
	title, err := searchNearest(ctx, articlesColl, query, 1)
	if err != nil {
		log.Fatalf("Pencarian gagal: %v", err)
	}
	log.Printf("Artikel paling mirip: %s", title)

	// siapkan koleksi embedding 128 dimensi
	log.Printf("Menyiapkan koleksi %q (dim=%d)...", embeddingsColl, embeddingsVecDim)
	if err := createCollectionIfNotExists(ctx, embeddingsColl, embeddingsVecDim); err != nil {
		log.Fatalf("Gagal membuat koleksi embedding: %v", err)
	}
	log.Println("Koleksi embedding siap.")

	// generate 1000 embedding acak
	n := 1000
	vectors := make([][]float32, n)
	payloads := make([]map[string]interface{}, n)

	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := 0; i < n; i++ {
		vec := make([]float32, embeddingsVecDim)
		for j := 0; j < embeddingsVecDim; j++ {
			vec[j] = rnd.Float32()
		}
		vectors[i] = vec
		payloads[i] = map[string]interface{}{
			"title": fmt.Sprintf("bulk-vec-%d", i),
			"meta":  map[string]interface{}{"index": i},
		}
	}

	log.Printf("Memulai BulkInsert (concurrent, batch=%d)...", batchSize)
	start := time.Now()

	if err := BulkInsert(ctx, embeddingsColl, vectors, payloads); err != nil {
		log.Fatalf("BulkInsert gagal: %v", err)
	}

	log.Printf("BulkInsert selesai dalam %s", time.Since(start))

	// tampilkan jumlah total data di koleksi embedding
	info, err := getCollectionInfo(ctx, embeddingsColl)
	if err == nil {
		log.Printf("Total data dalam koleksi embedding: %d", info.Result.PointsCount)
	}
}
