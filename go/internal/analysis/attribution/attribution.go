// Package attribution provides backtest attribution analysis.
// Supports basic statistics, turnover attribution, and structural analysis.
package attribution

import (
	"context"
	"database/sql"
	"fmt"
	"math"
	"time"

	_ "github.com/lib/pq" // PostgreSQL driver
)

// AttributionService provides attribution analysis functionality.
type AttributionService struct {
	db *sql.DB
}

// NewAttributionService creates a new attribution service.
func NewAttributionService(db *sql.DB) *AttributionService {
	return &AttributionService{db: db}
}

// BasicStats contains basic backtest statistics.
type BasicStats struct {
	WinRate            float64
	ProfitFactor       float64
	SharpeRatio        float64
	MaxDrawdown        float64
	TotalReturn        float64
	AnnualizedReturn   float64
	TotalTrades        int32
	WinningTrades      int32
	LosingTrades       int32
	AvgProfit          float64
	AvgLoss            float64
	ProfitLossRatio    float64
}

// GetBasicStats calculates basic statistics for an experiment.
func (s *AttributionService) GetBasicStats(
	ctx context.Context,
	expID string,
	startDate, endDate *time.Time,
) (*BasicStats, error) {
	// Calculate win rate and profit factor from trades
	winRate, profitFactor, totalTrades, winningTrades, losingTrades, avgProfit, avgLoss, err :=
		s.calculateTradeStats(ctx, expID, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate trade stats: %w", err)
	}

	// Calculate Sharpe ratio and drawdown from ledger
	sharpeRatio, maxDrawdown, totalReturn, annualizedReturn, err :=
		s.calculatePerformanceMetrics(ctx, expID, startDate, endDate)
	if err != nil {
		return nil, fmt.Errorf("failed to calculate performance metrics: %w", err)
	}

	profitLossRatio := 0.0
	if avgLoss != 0 {
		profitLossRatio = avgProfit / math.Abs(avgLoss)
	}

	return &BasicStats{
		WinRate:          winRate,
		ProfitFactor:     profitFactor,
		SharpeRatio:      sharpeRatio,
		MaxDrawdown:      maxDrawdown,
		TotalReturn:      totalReturn,
		AnnualizedReturn: annualizedReturn,
		TotalTrades:      totalTrades,
		WinningTrades:    winningTrades,
		LosingTrades:     losingTrades,
		AvgProfit:        avgProfit,
		AvgLoss:          avgLoss,
		ProfitLossRatio:  profitLossRatio,
	}, nil
}

// calculateTradeStats calculates trade statistics.
func (s *AttributionService) calculateTradeStats(
	ctx context.Context,
	expID string,
	startDate, endDate *time.Time,
) (winRate, profitFactor float64, totalTrades, winningTrades, losingTrades int32, avgProfit, avgLoss float64, err error) {
	query := `
		SELECT 
			COUNT(*) as total_trades,
			COUNT(*) FILTER (WHERE pnl_ratio > 0) as winning_trades,
			COUNT(*) FILTER (WHERE pnl_ratio < 0) as losing_trades,
			COALESCE(SUM(pnl_ratio) FILTER (WHERE pnl_ratio > 0), 0) as total_profit,
			COALESCE(ABS(SUM(pnl_ratio) FILTER (WHERE pnl_ratio < 0)), 0) as total_loss,
			COALESCE(AVG(pnl_ratio) FILTER (WHERE pnl_ratio > 0), 0) as avg_profit,
			COALESCE(AVG(pnl_ratio) FILTER (WHERE pnl_ratio < 0), 0) as avg_loss
		FROM edios.bt_trades
		WHERE exp_id = $1 AND pnl_ratio IS NOT NULL
	`
	args := []interface{}{expID}

	if startDate != nil {
		query += " AND deal_time >= $2"
		args = append(args, *startDate)
		if endDate != nil {
			query += " AND deal_time <= $3"
			args = append(args, *endDate)
		}
	} else if endDate != nil {
		query += " AND deal_time <= $2"
		args = append(args, *endDate)
	}

	var total, winning, losing int32
	var totalProfit, totalLoss, avgP, avgL sql.NullFloat64

	err = s.db.QueryRowContext(ctx, query, args...).Scan(
		&total, &winning, &losing, &totalProfit, &totalLoss, &avgP, &avgL,
	)
	if err != nil {
		if err == sql.ErrNoRows {
			return 0, 0, 0, 0, 0, 0, 0, nil
		}
		return 0, 0, 0, 0, 0, 0, 0, err
	}

	totalTrades = total
	winningTrades = winning
	losingTrades = losing

	if total > 0 {
		winRate = float64(winning) / float64(total)
	}

	if totalLoss.Float64 > 0 {
		profitFactor = totalProfit.Float64 / totalLoss.Float64
	}

	if avgP.Valid {
		avgProfit = avgP.Float64
	}
	if avgL.Valid {
		avgLoss = avgL.Float64
	}

	return winRate, profitFactor, totalTrades, winningTrades, losingTrades, avgProfit, avgLoss, nil
}

// calculatePerformanceMetrics calculates performance metrics from ledger.
func (s *AttributionService) calculatePerformanceMetrics(
	ctx context.Context,
	expID string,
	startDate, endDate *time.Time,
) (sharpeRatio, maxDrawdown, totalReturn, annualizedReturn float64, err error) {
	query := `
		SELECT date, nav
		FROM edios.bt_ledger
		WHERE exp_id = $1
	`
	args := []interface{}{expID}

	if startDate != nil {
		query += " AND date >= $2"
		args = append(args, *startDate)
		if endDate != nil {
			query += " AND date <= $3"
			args = append(args, *endDate)
		}
	} else if endDate != nil {
		query += " AND date <= $2"
		args = append(args, *endDate)
	}

	query += " ORDER BY date ASC"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return 0, 0, 0, 0, err
	}
	defer rows.Close()

	var dates []time.Time
	var navs []float64
	var firstNav, lastNav float64

	for rows.Next() {
		var date time.Time
		var nav sql.NullFloat64
		if err := rows.Scan(&date, &nav); err != nil {
			return 0, 0, 0, 0, err
		}
		if nav.Valid {
			dates = append(dates, date)
			navs = append(navs, nav.Float64)
			if len(navs) == 1 {
				firstNav = nav.Float64
			}
			lastNav = nav.Float64
		}
	}

	if len(navs) == 0 {
		return 0, 0, 0, 0, nil
	}

	// Calculate total return
	if firstNav > 0 {
		totalReturn = (lastNav - firstNav) / firstNav
	}

	// Calculate annualized return
	if len(dates) > 1 {
		days := dates[len(dates)-1].Sub(dates[0]).Hours() / 24
		if days > 0 {
			annualizedReturn = math.Pow(1+totalReturn, 365.0/days) - 1
		}
	}

	// Calculate Sharpe ratio (simplified, using daily returns)
	if len(navs) > 1 {
		var returns []float64
		for i := 1; i < len(navs); i++ {
			if navs[i-1] > 0 {
				ret := (navs[i] - navs[i-1]) / navs[i-1]
				returns = append(returns, ret)
			}
		}

		if len(returns) > 0 {
			mean := 0.0
			for _, r := range returns {
				mean += r
			}
			mean /= float64(len(returns))

			variance := 0.0
			for _, r := range returns {
				variance += (r - mean) * (r - mean)
			}
			variance /= float64(len(returns))
			stdDev := math.Sqrt(variance)

			if stdDev > 0 {
				// Annualized Sharpe ratio (assuming 252 trading days)
				sharpeRatio = mean * math.Sqrt(252) / stdDev
			}
		}
	}

	// Calculate maximum drawdown
	maxDrawdown = 0.0
	peak := navs[0]
	for _, nav := range navs {
		if nav > peak {
			peak = nav
		}
		drawdown := (peak - nav) / peak
		if drawdown > maxDrawdown {
			maxDrawdown = drawdown
		}
	}

	return sharpeRatio, maxDrawdown, totalReturn, annualizedReturn, nil
}

// TurnoverAttribution contains turnover attribution metrics.
type TurnoverAttribution struct {
	TotalTurnover      float64
	RankEdgeTurnover   float64
	StopLossTurnover   float64
	RankOutTurnover    float64
	ReasonBreakdown    map[string]float64
	RankEdgeCount      int32
	SignalNoiseRatio   float64
}

// GetTurnoverAttribution analyzes turnover attribution.
func (s *AttributionService) GetTurnoverAttribution(
	ctx context.Context,
	expID string,
	startDate, endDate *time.Time,
) (*TurnoverAttribution, error) {
	// Get turnover breakdown by reason
	query := `
		SELECT 
			reason,
			COUNT(*) as count,
			SUM(ABS(amount * price)) as turnover
		FROM edios.bt_trades
		WHERE exp_id = $1 AND side = -1 AND reason IS NOT NULL
	`
	args := []interface{}{expID}

	if startDate != nil {
		query += " AND deal_time >= $2"
		args = append(args, *startDate)
		if endDate != nil {
			query += " AND deal_time <= $3"
			args = append(args, *endDate)
		}
	} else if endDate != nil {
		query += " AND deal_time <= $2"
		args = append(args, *endDate)
	}

	query += " GROUP BY reason"

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query turnover: %w", err)
	}
	defer rows.Close()

	reasonBreakdown := make(map[string]float64)
	var totalTurnover, rankEdgeTurnover float64
	var rankEdgeCount int32

	for rows.Next() {
		var reason sql.NullString
		var count int32
		var turnover sql.NullFloat64

		if err := rows.Scan(&reason, &count, &turnover); err != nil {
			return nil, err
		}

		if reason.Valid && turnover.Valid {
			reasonBreakdown[reason.String] = turnover.Float64
			totalTurnover += turnover.Float64

			// Check for rank edge cases (e.g., "rank_out" with rank close to threshold)
			if reason.String == "rank_out" || reason.String == "rank_edge" {
				rankEdgeTurnover += turnover.Float64
				rankEdgeCount += count
			}
		}
	}

	// Get total turnover from ledger
	ledgerQuery := `
		SELECT SUM(deal_amount) as total_deal_amount
		FROM edios.bt_ledger
		WHERE exp_id = $1
	`
	ledgerArgs := []interface{}{expID}

	if startDate != nil {
		ledgerQuery += " AND date >= $2"
		ledgerArgs = append(ledgerArgs, *startDate)
		if endDate != nil {
			ledgerQuery += " AND date <= $3"
			ledgerArgs = append(ledgerArgs, *endDate)
		}
	} else if endDate != nil {
		ledgerQuery += " AND date <= $2"
		ledgerArgs = append(ledgerArgs, *endDate)
	}

	var totalDealAmount sql.NullFloat64
	err = s.db.QueryRowContext(ctx, ledgerQuery, ledgerArgs...).Scan(&totalDealAmount)
	if err != nil && err != sql.ErrNoRows {
		return nil, fmt.Errorf("failed to query ledger: %w", err)
	}

	// Calculate signal noise ratio (simplified)
	signalNoiseRatio := 0.0
	if rankEdgeTurnover > 0 && totalTurnover > 0 {
		signalNoiseRatio = 1.0 - (rankEdgeTurnover / totalTurnover)
	}

	return &TurnoverAttribution{
		TotalTurnover:    totalTurnover,
		RankEdgeTurnover: rankEdgeTurnover,
		StopLossTurnover: reasonBreakdown["stop_loss"],
		RankOutTurnover:  reasonBreakdown["rank_out"],
		ReasonBreakdown:  reasonBreakdown,
		RankEdgeCount:    rankEdgeCount,
		SignalNoiseRatio: signalNoiseRatio,
	}, nil
}

// StructuralAnalysis contains structural analysis metrics.
type StructuralAnalysis struct {
	Symbol                    string
	Date                     time.Time
	Neighbors                []Neighbor
	NeighborWeightConcentration float64
	ContagionScore           float64
	HighWeightNeighbors      []string
}

// Neighbor represents a neighbor node.
type Neighbor struct {
	Symbol    string
	Weight    float64
	LinkType  string
}

// GetStructuralAnalysis analyzes structural sensitivity.
func (s *AttributionService) GetStructuralAnalysis(
	ctx context.Context,
	expID string,
	date time.Time,
	symbol string,
	linkType string,
	topK int32,
) (*StructuralAnalysis, error) {
	query := `
		SELECT source, target, weight, link_type
		FROM edios.bt_model_links
		WHERE exp_id = $1 AND date = $2 AND (source = $3 OR target = $3)
	`
	args := []interface{}{expID, date, symbol}

	if linkType != "" {
		query += " AND link_type = $4"
		args = append(args, linkType)
	}

	query += " ORDER BY weight DESC"

	if topK > 0 {
		query += fmt.Sprintf(" LIMIT %d", topK)
	}

	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query links: %w", err)
	}
	defer rows.Close()

	var neighbors []Neighbor
	var totalWeight float64
	var highWeightNeighbors []string

	for rows.Next() {
		var source, target, linkType sql.NullString
		var weight sql.NullFloat64

		if err := rows.Scan(&source, &target, &weight, &linkType); err != nil {
			return nil, err
		}

		if weight.Valid {
			neighborSymbol := target.String
			if target.String == symbol {
				neighborSymbol = source.String
			}

			neighbors = append(neighbors, Neighbor{
				Symbol:   neighborSymbol,
				Weight:   weight.Float64,
				LinkType: linkType.String,
			})

			totalWeight += weight.Float64
			if weight.Float64 > 0.1 { // Threshold for high weight
				highWeightNeighbors = append(highWeightNeighbors, neighborSymbol)
			}
		}
	}

	// Calculate weight concentration (Herfindahl index)
	concentration := 0.0
	if totalWeight > 0 {
		for _, n := range neighbors {
			share := n.Weight / totalWeight
			concentration += share * share
		}
	}

	// Contagion score (simplified: based on number of high-weight neighbors)
	contagionScore := float64(len(highWeightNeighbors)) / float64(len(neighbors)+1)

	return &StructuralAnalysis{
		Symbol:                      symbol,
		Date:                        date,
		Neighbors:                   neighbors,
		NeighborWeightConcentration: concentration,
		ContagionScore:              contagionScore,
		HighWeightNeighbors:         highWeightNeighbors,
	}, nil
}

