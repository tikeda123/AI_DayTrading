import React, { useState, useEffect } from 'react';
import 'bootstrap/dist/css/bootstrap.min.css';
import axios from 'axios';

function App() {
  const [marketData, setMarketData] = useState([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.post('http://localhost:8000/read_db_market_data/', {
          num_rows: 30 // 仕様に従った設定
        });
        setMarketData(response.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
    const intervalId = setInterval(fetchData, 10000); // 10秒ごとにデータを再取得

    return () => clearInterval(intervalId); // コンポーネントのクリーンアップ
  }, []);

  return (
    <div className="container mt-5">
      <h2>Market Data</h2>
      <table className="table table-striped">
        <thead>
          <tr>
            <th>Start At</th>
            <th>Open</th>
            <th>High</th>
            <th>Low</th>
            <th>Close</th>
            <th>Volume</th>
            <th>Upper2</th>
            <th>Middle</th>
            <th>Lower2</th>
          </tr>
        </thead>
        <tbody>
          {marketData.map((data, index) => (
            <tr key={index}>
              <td>{data.start_at}</td>
              <td>{data.open}</td>
              <td>{data.high}</td>
              <td>{data.low}</td>
              <td>{data.close}</td>
              <td>{data.volume}</td>
              <td>{data.upper2}</td>
              <td>{data.middle}</td>
              <td>{data.lower2}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default App;


