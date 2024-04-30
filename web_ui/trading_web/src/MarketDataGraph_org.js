// MarketDataGraph.js
import React, { useEffect, useState } from 'react';
import axios from 'axios';



import 'bootstrap/dist/css/bootstrap.min.css';

import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend,
  } from 'chart.js';
  import { Line } from 'react-chartjs-2';
  import 'chartjs-adapter-date-fns'; // date-fnsアダプターをインポート

  // Chart.jsコンポーネントの登録
  ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
  );


const MarketDataGraph = () => {
  const [marketData, setMarketData] = useState([]);

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

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const data = {
    labels: marketData.map((data) => data.start_at),
    datasets: [
      {
        label: 'Close',
        data: marketData.map((data) => data.close),
        borderColor: 'rgb(255, 99, 132)',
        borderWidth: 2,
        tension: 0.1
      },
      {
        label: 'Upper2',
        data: marketData.map((data) => data.upper2),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      },
      {
        label: 'Middle',
        data: marketData.map((data) => data.middle),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      },
      {
        label: 'Lower2',
        data: marketData.map((data) => data.lower2),
        borderColor: 'rgb(75, 192, 192)',
        borderWidth: 1,
        tension: 0.1
      }
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: false
      }
    }
  };

  return (
    <div className="container mt-5">
      <h2>Market Data Graph</h2>
      <Line data={data} options={options} />
    </div>
  );
};

export default MarketDataGraph;
