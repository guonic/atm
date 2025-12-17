-- quant_db 数据库初始化脚本
-- 此脚本仅在数据库首次初始化时执行（通过 /docker-entrypoint-initdb.d/ 目录）

-- 连接到 quant_db 数据库（在初始化时自动执行）
\c quant_db

-- 对日线表启用压缩策略（30天前的数据自动压缩）
-- 注意：如果表不存在，此命令会失败，但不会影响数据库初始化
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'stock_kline_daily'
    ) THEN
        PERFORM add_compression_policy('stock_kline_daily', INTERVAL '30 days');
        RAISE NOTICE '已为 stock_kline_daily 表添加压缩策略（30天）';
    ELSE
        RAISE NOTICE 'stock_kline_daily 表不存在，跳过压缩策略设置';
    END IF;
END $$;

-- 对Tick表设置90天保留期，自动删除过期数据（节省Mac存储）
-- 注意：如果表不存在，此命令会失败，但不会影响数据库初始化
DO $$
BEGIN
    IF EXISTS (
        SELECT FROM information_schema.tables 
        WHERE table_schema = 'public' 
        AND table_name = 'stock_tick'
    ) THEN
        PERFORM add_retention_policy('stock_tick', INTERVAL '90 days');
        RAISE NOTICE '已为 stock_tick 表添加保留策略（90天）';
    ELSE
        RAISE NOTICE 'stock_tick 表不存在，跳过保留策略设置';
    END IF;
END $$;

-- 验证压缩策略（如果表存在）
SELECT 
    hypertable_name,
    compress_after,
    schedule_interval
FROM timescaledb_information.compression_policies
WHERE hypertable_name IN ('stock_kline_daily', 'stock_tick');

-- 验证保留策略（如果表存在）
SELECT 
    hypertable_name,
    drop_after,
    schedule_interval
FROM timescaledb_information.retention_policies
WHERE hypertable_name IN ('stock_kline_daily', 'stock_tick');

