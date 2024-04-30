import React, { useEffect, useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

function AccountLog() {
  const [data, setData] = useState([]);

  // データを取得する関数
  const fetchData = async () => {
    try {
      const response = await axios.post('http://localhost:8000/read_account_data/', {
        num_rows: -1,
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
      });
      setData(response.data);
    } catch (error) {
      console.error("エラーが発生しました:", error);
    }
  };

  useEffect(() => {
    // コンポーネントがマウントされた時にデータを一度取得
    fetchData();
    // その後、5分ごとにデータを取得
    const interval = setInterval(fetchData, 300000); // 300000ms = 5分
    return () => clearInterval(interval); // コンポーネントのアンマウント時にインターバルをクリア
  }, []);

  return (
    <div className="container mt-5">
      <table className="table" style={{ fontSize: '0.8rem' }}>
        <thead>
          <tr>
            <th>Serial</th>
            <th>Date</th>
            <th>Cash In</th>
            <th>Cash Out</th>
            <th>Amount</th>
          </tr>
        </thead>
        <tbody>
          {data.map((item) => (
            <tr key={item.serial}>
              <td>{item.serial}</td>
              <td>{item.date}</td>
              <td>{item.cash_in}</td>
              <td>{item.cash_out}</td>
              <td>{item.amount}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}


export default AccountLog;
