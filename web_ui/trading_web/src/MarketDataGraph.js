import React, { useEffect, useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
import 'chartjs-adapter-date-fns';
import { TimeScale } from 'chart.js'; // TimeScaleのインポート

// 必要なコンポーネントをChart.jsに登録
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  TimeScale, // TimeScaleを登録
  Title,
  Tooltip,
  Legend
);


const MarketDataGraph = () => {
  const [marketData, setMarketData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/read_db_market_data/', {
          num_rows: 50
        });
        setMarketData(response.data);
      } catch (error) {
        console.error("Error fetching data: ", error);
      }
    };

    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const data = {
    labels: marketData.map(data => data.start_at),
    datasets: [
      {
        label: 'Close',
        data: marketData.map(data => data.close),
        borderColor: 'rgb(255, 99, 132)',
        borderWidth: 2,
        tension: 0.1
      },
      {
        label: 'Upper2',
        data: marketData.map(data => data.upper2),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      },
      {
        label: 'Middle',
        data: marketData.map(data => data.middle),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      },
      {
        label: 'Lower2',
        data: marketData.map(data => data.lower2),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      }
    ],
  };

  const options = {
    scales: {
        x: {
            type: 'time',
            time: {
              parser: 'yyyy-MM-dd\'T\'HH:mm:ss', // 必要に応じてカスタマイズ
              unit: 'hour',
              stepSize: 1,
              displayFormats: {
                hour: 'HH:mm'
              },
              tooltipFormat: 'yyyy-MM-dd HH:mm'
            },
            ticks: {
              source: 'data', // 'labels'または'data'を使用してラベルの生成元を指定
              autoSkip: true,
              maxTicksLimit: 20 // 表示する最大のティック数を制限
            },
            title: {
              display: true,
              text: 'Time'
            }
          },
      y: {
        beginAtZero: false
      }
    },
    responsive: true,
    maintainAspectRatio: false,
  };

  return (
    <div className="container mt-5">
      <div style={{ height: "500px" }}>
        <Line data={data} options={options} />
      </div>
    </div>
  );
};

export default MarketDataGraph;
