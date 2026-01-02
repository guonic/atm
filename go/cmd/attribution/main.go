package main

import (
	"context"
	"database/sql"
	"flag"
	"fmt"
	"log"
	"net"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/nexusquant/nq/go/internal/analysis/attribution"
	_ "github.com/lib/pq" // PostgreSQL driver
	"google.golang.org/grpc"
	"google.golang.org/grpc/reflection"

	pb "github.com/nexusquant/nq/pkg/proto/backtest_attribution"
)

var (
	port     = flag.Int("port", 50051, "gRPC server port")
	dbHost   = flag.String("db-host", "localhost", "Database host")
	dbPort   = flag.Int("db-port", 5432, "Database port")
	dbUser   = flag.String("db-user", "quant", "Database user")
	dbPass   = flag.String("db-pass", "", "Database password")
	dbName   = flag.String("db-name", "quant_db", "Database name")
)

// AttributionServer implements the gRPC AttributionService.
type AttributionServer struct {
	pb.UnimplementedAttributionServiceServer
	service *attribution.AttributionService
}

// NewAttributionServer creates a new attribution server.
func NewAttributionServer(service *attribution.AttributionService) *AttributionServer {
	return &AttributionServer{service: service}
}

// GetBasicStats implements the GetBasicStats RPC.
func (s *AttributionServer) GetBasicStats(
	ctx context.Context,
	req *pb.AnalysisRequest,
) (*pb.BasicStatsResponse, error) {
	var startDate, endDate *time.Time

	if req.StartDate != "" {
		t, err := time.Parse(time.RFC3339, req.StartDate)
		if err != nil {
			return &pb.BasicStatsResponse{
				ExpId: req.ExpId,
				Error: fmt.Sprintf("invalid start_date: %v", err),
			}, nil
		}
		startDate = &t
	}

	if req.EndDate != "" {
		t, err := time.Parse(time.RFC3339, req.EndDate)
		if err != nil {
			return &pb.BasicStatsResponse{
				ExpId: req.ExpId,
				Error: fmt.Sprintf("invalid end_date: %v", err),
			}, nil
		}
		endDate = &t
	}

	stats, err := s.service.GetBasicStats(ctx, req.ExpId, startDate, endDate)
	if err != nil {
		return &pb.BasicStatsResponse{
			ExpId: req.ExpId,
			Error: err.Error(),
		}, nil
	}

	return &pb.BasicStatsResponse{
		ExpId: req.ExpId,
		Stats: &pb.BasicStats{
			WinRate:          stats.WinRate,
			ProfitFactor:     stats.ProfitFactor,
			SharpeRatio:      stats.SharpeRatio,
			MaxDrawdown:      stats.MaxDrawdown,
			TotalReturn:      stats.TotalReturn,
			AnnualizedReturn: stats.AnnualizedReturn,
			TotalTrades:      stats.TotalTrades,
			WinningTrades:    stats.WinningTrades,
			LosingTrades:    stats.LosingTrades,
			AvgProfit:       stats.AvgProfit,
			AvgLoss:         stats.AvgLoss,
			ProfitLossRatio:  stats.ProfitLossRatio,
		},
	}, nil
}

// GetTurnoverAttribution implements the GetTurnoverAttribution RPC.
func (s *AttributionServer) GetTurnoverAttribution(
	ctx context.Context,
	req *pb.AnalysisRequest,
) (*pb.TurnoverAttributionResponse, error) {
	var startDate, endDate *time.Time

	if req.StartDate != "" {
		t, err := time.Parse(time.RFC3339, req.StartDate)
		if err != nil {
			return &pb.TurnoverAttributionResponse{
				ExpId: req.ExpId,
				Error: fmt.Sprintf("invalid start_date: %v", err),
			}, nil
		}
		startDate = &t
	}

	if req.EndDate != "" {
		t, err := time.Parse(time.RFC3339, req.EndDate)
		if err != nil {
			return &pb.TurnoverAttributionResponse{
				ExpId: req.ExpId,
				Error: fmt.Sprintf("invalid end_date: %v", err),
			}, nil
		}
		endDate = &t
	}

	attr, err := s.service.GetTurnoverAttribution(ctx, req.ExpId, startDate, endDate)
	if err != nil {
		return &pb.TurnoverAttributionResponse{
			ExpId: req.ExpId,
			Error: err.Error(),
		}, nil
	}

	reasonBreakdown := make(map[string]float64)
	for k, v := range attr.ReasonBreakdown {
		reasonBreakdown[k] = v
	}

	return &pb.TurnoverAttributionResponse{
		ExpId: req.ExpId,
		Attribution: &pb.TurnoverAttribution{
			TotalTurnover:      attr.TotalTurnover,
			RankEdgeTurnover:   attr.RankEdgeTurnover,
			StopLossTurnover:   attr.StopLossTurnover,
			RankOutTurnover:    attr.RankOutTurnover,
			ReasonBreakdown:       reasonBreakdown,
			RankEdgeCount:         attr.RankEdgeCount,
			SignalNoiseRatio:   attr.SignalNoiseRatio,
		},
	}, nil
}

// GetStructuralAnalysis implements the GetStructuralAnalysis RPC.
func (s *AttributionServer) GetStructuralAnalysis(
	ctx context.Context,
	req *pb.StructuralRequest,
) (*pb.StructuralAnalysisResponse, error) {
	date, err := time.Parse("2006-01-02", req.Date)
	if err != nil {
		return &pb.StructuralAnalysisResponse{
			Error: fmt.Sprintf("invalid date: %v", err),
		}, nil
	}

	analysis, err := s.service.GetStructuralAnalysis(
		ctx,
		req.ExpId,
		date,
		req.Symbol,
		req.LinkType,
		req.TopK,
	)
	if err != nil {
		return &pb.StructuralAnalysisResponse{
			ExpId: req.ExpId,
			Error: err.Error(),
		}, nil
	}

	var neighbors []*pb.Neighbor
	for _, n := range analysis.Neighbors {
		neighbors = append(neighbors, &pb.Neighbor{
			Symbol:   n.Symbol,
			Weight:   n.Weight,
			LinkType: n.LinkType,
		})
	}

	return &pb.StructuralAnalysisResponse{
		ExpId: req.ExpId,
		Analysis: &pb.StructuralAnalysis{
			Symbol:                      analysis.Symbol,
			Date:                        analysis.Date.Format("2006-01-02"),
			Neighbors:                   neighbors,
			NeighborWeightConcentration: analysis.NeighborWeightConcentration,
			ContagionScore:              analysis.ContagionScore,
			HighWeightNeighbors:          analysis.HighWeightNeighbors,
		},
	}, nil
}

// GetExperimentInfo implements the GetExperimentInfo RPC.
func (s *AttributionServer) GetExperimentInfo(
	ctx context.Context,
	req *pb.ExperimentRequest,
) (*pb.ExperimentInfoResponse, error) {
	// TODO: Implement experiment info retrieval
	return &pb.ExperimentInfoResponse{
		ExpId: req.ExpId,
		Error: "not implemented",
	}, nil
}

func main() {
	flag.Parse()

	// Connect to database
	dsn := fmt.Sprintf(
		"host=%s port=%d user=%s password=%s dbname=%s sslmode=disable",
		*dbHost, *dbPort, *dbUser, *dbPass, *dbName,
	)

	db, err := sql.Open("postgres", dsn)
	if err != nil {
		log.Fatalf("Failed to connect to database: %v", err)
	}
	defer db.Close()

	if err := db.Ping(); err != nil {
		log.Fatalf("Failed to ping database: %v", err)
	}

	// Create attribution service
	service := attribution.NewAttributionService(db)

	// Create gRPC server
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer()
	pb.RegisterAttributionServiceServer(grpcServer, NewAttributionServer(service))

	// Enable reflection for testing
	reflection.Register(grpcServer)

	// Graceful shutdown
	go func() {
		sigint := make(chan os.Signal, 1)
		signal.Notify(sigint, os.Interrupt, syscall.SIGTERM)
		<-sigint

		log.Println("Shutting down server...")
		grpcServer.GracefulStop()
	}()

	log.Printf("Attribution service listening on :%d", *port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

