"use client";

import { useEffect, useRef, useImperativeHandle, forwardRef } from "react";
import {
  createChart,
  IChartApi,
  ISeriesApi,
  CandlestickSeries,
  HistogramSeries,
  CrosshairMode,
  LineStyle,
  ColorType,
} from "lightweight-charts";

export interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

interface TvChartProps {
  candles: CandleData[];
  symbol?: string;
  height?: number;
  showVolume?: boolean;
  overlays?: {
    bb_up?: number[];
    bb_dn?: number[];
    ma?: number[];
    zscore?: number[];
  };
}

export interface TvChartRef {
  fitContent: () => void;
}

const TvChart = forwardRef<TvChartRef, TvChartProps>(function TvChart(
  { candles, symbol, height = 520, showVolume = true, overlays },
  ref
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null);

  useImperativeHandle(ref, () => ({
    fitContent: () => chartRef.current?.timeScale().fitContent(),
  }));

  useEffect(() => {
    if (!containerRef.current) return;

    const chart = createChart(containerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "#94a3b8",
        fontFamily: '"Space Mono", monospace',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "rgba(51, 65, 85, 0.4)" },
        horzLines: { color: "rgba(51, 65, 85, 0.4)" },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: "rgba(147, 51, 234, 0.5)",
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: "#1e1b4b",
        },
        horzLine: {
          color: "rgba(147, 51, 234, 0.5)",
          width: 1,
          style: LineStyle.Dashed,
          labelBackgroundColor: "#1e1b4b",
        },
      },
      rightPriceScale: {
        borderColor: "rgba(51, 65, 85, 0.5)",
        textColor: "#94a3b8",
      },
      timeScale: {
        borderColor: "rgba(51, 65, 85, 0.5)",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 5,
      },
      width: containerRef.current.clientWidth,
      height: showVolume ? height - 100 : height,
    });

    chartRef.current = chart;

    // Candlestick series
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#10b981",
      downColor: "#f43f5e",
      borderUpColor: "#10b981",
      borderDownColor: "#f43f5e",
      wickUpColor: "#10b981",
      wickDownColor: "#f43f5e",
    });
    candleSeriesRef.current = candleSeries;

    // Volume series
    if (showVolume) {
      const volSeries = chart.addSeries(HistogramSeries, {
        color: "rgba(59, 130, 246, 0.3)",
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      });
      chart.priceScale("volume").applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      });
      volumeSeriesRef.current = volSeries;
    }

    const ro = new ResizeObserver((entries) => {
      const { width } = entries[0].contentRect;
      chart.applyOptions({ width });
    });
    ro.observe(containerRef.current);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [height, showVolume]);

  // Update data whenever candles change
  useEffect(() => {
    if (!candleSeriesRef.current || !candles?.length) return;

    // Sort by time ascending
    const sorted = [...candles].sort((a, b) => a.time - b.time);
    candleSeriesRef.current.setData(sorted as any);

    if (volumeSeriesRef.current) {
      const volData = sorted.map((c) => ({
        time: c.time,
        value: c.volume ?? 0,
        color:
          c.close >= c.open
            ? "rgba(16, 185, 129, 0.3)"
            : "rgba(244, 63, 94, 0.3)",
      }));
      volumeSeriesRef.current.setData(volData as any);
    }

    chartRef.current?.timeScale().fitContent();
  }, [candles]);

  return (
    <div className="w-full" style={{ height }}>
      <div ref={containerRef} style={{ width: "100%", height: "100%" }} />
    </div>
  );
});

export default TvChart;
