import React, { useEffect, useState } from 'react';
import axios from 'axios';
import 'bootstrap/dist/css/bootstrap.min.css';

const FxTransaction = () => {
  const [transactions, setTransactions] = useState([]);

  // 最初の起動時と1.5分ごとにデータをフェッチするための関数
  const fetchData = async () => {
    try {
      const response = await axios.post('http://localhost:8000/read_fxtransaction_data/', {
        num_rows: -1 // 仕様に基づき、num_rowsを-1に設定
      }, {
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (response.data && response.data.length > 0) {
        setTransactions(response.data);
      }
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  useEffect(() => {
    fetchData(); // コンポーネントマウント時にデータをフェッチ

    const intervalId = setInterval(fetchData,  300000); // 5分(300秒)ごとにデータを更新

    return () => clearInterval(intervalId); // コンポーネントのクリーンアップ
  }, []); // 空の依存配列を指定して、コンポーネントマウント時のみ実行

  return (
    <div className="container mt-5" style={{ fontSize: '0.8rem' }}>
      <table className="table">
        <thead>
          <tr>
            <th>Serial</th>
            <th>Initial Equity</th>
            <th>Equity</th>
            <th>Leverage</th>
            <th>Contract</th>
            <th>Quantity</th>
            <th>Entry Price</th>
            <th>Exit Price</th>
            <th>PL</th>
            <th>Prediction</th>
            <th>Trade Type</th>
            <th>Entry Time</th>
            <th>Exit Time</th>
            <th>Direction</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((transaction, index) => (
            <tr key={index}>
              <td>{transaction.serial}</td>
              <td>{transaction.init_equity}</td>
              <td>{transaction.equity}</td>
              <td>{transaction.leverage}</td>
              <td>{transaction.contract}</td>
              <td>{transaction.qty}</td>
              <td>{transaction.entry_price}</td>
              <td>{transaction.exit_price}</td>
              <td>{transaction.pl}</td>
              <td>{transaction.pred}</td>
              <td>{transaction.tradetype}</td>
              <td>{transaction.entrytime}</td>
              <td>{transaction.exittime}</td>
              <td>{transaction.direction}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default FxTransaction;


